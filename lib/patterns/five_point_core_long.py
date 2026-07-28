"""five_point_core_long — ядро для 5 LONG-паттернов (зеркало five_point_core.py).

Геометрия (i1<i2<i3<i4<i5, хронологически):
  i1,i3,i5 = Williams N=2 FH (сопротивление)   ← зеркало: теперь HIGH-фракталы
  i2,i4    = Williams N=2 FL (опора)             ← зеркало: теперь LOW-фракталы
  BR = первое закрытие ВЫШЕ сопротивления после i5 (если опора не пробита раньше)

Линии:
  Сопротивление: через (i1,H1) и (i3,H3) → res_at(j) = H1 + res_slope*(j-i1)
  Опора:         через (i2,L2) и (i4,L4) → sup_at(j) = L2 + sup_slope*(j-i2)

5 режимов (все LONG):
  ascending_triangle:  сопротивление плоское, опора растёт (зеркало descending_triangle)
  sym_triangle_long:   сопротивление падает, опора растёт (зеркало sym_triangle SHORT)
  falling_wedge:       обе линии падают, сопр. круче, narrow_frac > 10%
  falling_channel:     обе линии падают, ~параллельны, narrow_frac <= 10%
  rectangle_bottom:    обе линии плоские (зеркало rectangle_top)

TG (зеркало five_point_core.py):
  sym_triangle_long:  TG = c[BR] + (H1 - L2)
  ascending_triangle: TG = res_at(BR) + (H1 - min(L2,L4))
  falling_wedge/channel: TG = max(H1,H3,H5)  ("top of the wedge")
  rectangle_bottom:   TG = res_at(BR) + (res_at(BR) - sup_at(BR))

FVG(2)/MA(2a,2c,2d) — LONG-фильтр block5_common.apply_fvg_ma_filter_long,
bullish FVG (direction=long), окно FVG = (i4, BR].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import TF_1H_MS
from block5_common import fractals, load_hma_wma, apply_fvg_ma_filter_long

MODES_LONG = ('ascending_triangle', 'sym_triangle_long', 'falling_wedge',
              'falling_channel', 'rectangle_bottom')


@dataclass
class FivePointLongParams:
    max_span:             int   = 96    # часов, i1 -> i5
    max_confirm:          int   = 24    # часов, i5 -> BR
    line_tol_pct:         float = 0.35  # допуск касания линии, % от цены
    flat_tol_pct_per_bar: float = 0.05  # порог "плоскости" сопр./опоры
    narrow_frac_thr:      float = 0.10  # falling_wedge >, falling_channel <=
    n_workers:            int   = 30


def _empty_stats() -> dict:
    return {
        'n_fh': 0, 'n_fl': 0, 'passed_geometry': 0,
        'rej_fvg': 0, 'rej_hma_wma': 0, 'passed_canon': 0,
    }


def _process_i2_chunk_long(mode: str, i2_chunk: list[int], h: np.ndarray, l: np.ndarray,
                             c: np.ndarray, fh_list: list[int], fl_list: list[int],
                             n_bars: int, p: FivePointLongParams) -> list[dict]:
    """Обрабатывает чанк FL-фракталов i2 (первая точка опоры).

    Порядок поиска:
      outer loop: i2 (FL, первое касание опоры)
        i1_cands: FH до i2 (первое касание сопротивления)
        i4_cands: FL после i2 (второе касание опоры)
          i3_cands: FH между i2 и i4 (второе касание сопротивления)
            i5_cands: FH после i4 (третье касание сопротивления)
    """
    out: list[dict] = []
    FLAT = p.flat_tol_pct_per_bar

    for i2 in i2_chunk:
        i2 = int(i2)
        L2 = l[i2]

        # i1 = первое касание сопротивления (FH до i2)
        i1_cands = [i for i in fh_list if i2 - p.max_span <= i < i2]
        if not i1_cands:
            continue

        # i4 = второе касание опоры (FL после i2)
        if mode in ('ascending_triangle', 'sym_triangle_long'):
            # опора растёт
            i4_cands = [i for i in fl_list if i2 < i <= i2 + p.max_span and l[i] > L2]
        elif mode == 'rectangle_bottom':
            near_l2 = L2 * FLAT * 6 / 100
            i4_cands = [i for i in fl_list if i2 < i <= i2 + p.max_span and abs(l[i] - L2) <= near_l2]
        else:  # falling_wedge, falling_channel — опора падает
            i4_cands = [i for i in fl_list if i2 < i <= i2 + p.max_span and l[i] < L2]
        if not i4_cands:
            continue

        for i1 in i1_cands:
            H1 = h[i1]

            for i4 in i4_cands:
                L4 = l[i4]

                # i3 = второе касание сопротивления (FH между i2 и i4)
                if mode in ('ascending_triangle', 'rectangle_bottom'):
                    near = H1 * FLAT * 6 / 100
                    i3_cands = [i for i in fh_list if i2 < i < i4 and abs(h[i] - H1) <= near]
                elif mode == 'sym_triangle_long':
                    i3_cands = [i for i in fh_list if i2 < i < i4 and h[i] <= H1]
                else:  # falling_wedge, falling_channel
                    i3_cands = [i for i in fh_list if i2 < i < i4 and h[i] < H1]
                if not i3_cands:
                    continue

                for i3 in i3_cands:
                    H3 = h[i3]

                    # i5 = третье касание сопротивления (FH после i4)
                    if mode in ('ascending_triangle', 'rectangle_bottom'):
                        near = H1 * FLAT * 6 / 100
                        i5_cands = [i for i in fh_list if i4 < i <= i2 + p.max_span and abs(h[i] - H1) <= near]
                    elif mode == 'sym_triangle_long':
                        i5_cands = [i for i in fh_list if i4 < i <= i2 + p.max_span and h[i] <= H3]
                    else:  # falling_wedge, falling_channel
                        i5_cands = [i for i in fh_list if i4 < i <= i2 + p.max_span and h[i] < H3]
                    if not i5_cands:
                        continue

                    # Наклоны линий
                    res_slope = (H3 - H1) / (i3 - i1)
                    res_slope_pct = res_slope * 100 / H1
                    sup_slope = (L4 - L2) / (i4 - i2)
                    sup_slope_pct = sup_slope * 100 / L2

                    # Проверки наклона по режиму
                    if mode == 'sym_triangle_long':
                        if res_slope_pct >= -FLAT:
                            continue  # сопр. должно падать
                        if sup_slope_pct <= FLAT:
                            continue  # опора должна расти
                    elif mode == 'ascending_triangle':
                        if abs(res_slope_pct) > FLAT:
                            continue  # сопр. плоское
                        if sup_slope_pct <= FLAT:
                            continue  # опора должна расти
                    elif mode == 'rectangle_bottom':
                        if abs(res_slope_pct) > FLAT:
                            continue  # сопр. плоское
                        if abs(sup_slope_pct) > FLAT:
                            continue  # опора плоская
                    else:  # falling_wedge, falling_channel
                        if res_slope >= 0:
                            continue  # сопр. должно падать
                        if sup_slope >= 0:
                            continue  # опора должна падать
                        if res_slope >= sup_slope:
                            continue  # сопр. должно падать быстрее (конвергенция)

                    def res_at(i, H1=H1, res_slope=res_slope, i1=i1):
                        return H1 + res_slope * (i - i1)

                    def sup_at(i, L2=L2, sup_slope=sup_slope, i2=i2):
                        return L2 + sup_slope * (i - i2)

                    # narrow_frac для falling_wedge/falling_channel
                    gap_ref = None
                    if mode in ('falling_wedge', 'falling_channel'):
                        gap_ref = res_at(i1) - sup_at(i1)
                        if gap_ref <= 0:
                            continue

                    for i5 in i5_cands:
                        H5 = h[i5]
                        tol = p.line_tol_pct / 100.0 * H5
                        if abs(H5 - res_at(i5)) > tol * 3:
                            continue

                        if mode in ('falling_wedge', 'falling_channel'):
                            gap_i5 = res_at(i5) - sup_at(i5)
                            if gap_i5 <= 0:
                                continue
                            narrow_frac = 1 - gap_i5 / gap_ref
                            if mode == 'falling_wedge' and narrow_frac <= p.narrow_frac_thr:
                                continue
                            if mode == 'falling_channel' and narrow_frac > p.narrow_frac_thr:
                                continue

                        # Инвалидация между i1 и i5:
                        #   опора не пробита (l[j] < sup_at(j) * (1-tol))
                        #   сопр. не пробито раньше времени (h[j] > res_at(j) * (1+tol))
                        invalid = False
                        for j in range(i1, i5 + 1):
                            if l[j] < sup_at(j) * (1 - p.line_tol_pct / 100.0):
                                invalid = True
                                break
                            if j > i1 and j < i5 and h[j] > res_at(j) * (1 + p.line_tol_pct / 100.0):
                                invalid = True
                                break
                        if invalid:
                            continue

                        # BR: первое закрытие ВЫШЕ сопротивления после i5
                        # (если опора пробита ниже — стоп, не BR)
                        max_scan = min(n_bars, i5 + p.max_confirm + 1)
                        i_br = None
                        for j in range(i5 + 1, max_scan):
                            if l[j] < sup_at(j):
                                break
                            if c[j] > res_at(j):
                                i_br = j
                                break
                        if i_br is None:
                            continue

                        # TG (зеркало five_point_core.py)
                        if mode == 'sym_triangle_long':
                            height = H1 - L2
                            if height <= 0:
                                continue
                            TG = c[i_br] + height
                        elif mode == 'ascending_triangle':
                            base = min(L2, L4)
                            height = H1 - base
                            if height <= 0:
                                continue
                            TG = res_at(i_br) + height
                        elif mode == 'rectangle_bottom':
                            height = res_at(i_br) - sup_at(i_br)
                            if height <= 0:
                                continue
                            TG = res_at(i_br) + height
                        else:  # falling_wedge, falling_channel
                            TG = max(H1, H3, H5)

                        out.append({
                            'i1': i1, 'i2': i2, 'i3': i3, 'i4': i4, 'i5': i5,
                            'i_breakout': i_br,
                            'H1': float(H1), 'L2': float(L2), 'H3': float(H3),
                            'L4': float(L4), 'H5': float(H5),
                            'breakout_close': float(c[i_br]), 'TG': float(TG),
                            'width_bars': int(i5 - i1),
                        })
    return out


def detect_five_point_long_geometry(mode: str, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                                     ts: np.ndarray,
                                     params: Optional[FivePointLongParams] = None,
                                     ) -> tuple[list[dict], dict]:
    assert mode in MODES_LONG, f"unknown five_point_long mode: {mode}"
    if params is None:
        params = FivePointLongParams()

    n_bars = len(c)
    stats = _empty_stats()
    if n_bars < 20:
        return [], stats

    h = h.astype(np.float64); l = l.astype(np.float64); c = c.astype(np.float64)

    FH = fractals(h, 2, 'high')
    FL = fractals(l, 2, 'low')
    stats['n_fh'] = len(FH); stats['n_fl'] = len(FL)
    fh_list = FH.tolist(); fl_list = FL.tolist()

    # Внешний цикл по FL (i2) — зеркало SHORT (внешний цикл по FH/i2)
    n_workers = max(1, min(params.n_workers, os.cpu_count() or 4))
    i2_chunks = np.array_split(FL, n_workers)
    i2_chunks = [chunk.tolist() for chunk in i2_chunks if len(chunk) > 0]

    if len(i2_chunks) > 1:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_i2_chunk_long)(mode, chunk, h, l, c, fh_list, fl_list, n_bars, params)
            for chunk in i2_chunks
        )
    else:
        chunk_results = [_process_i2_chunk_long(mode, chunk, h, l, c, fh_list, fl_list, n_bars, params)
                          for chunk in i2_chunks]

    patterns: list[dict] = []
    for res in chunk_results:
        patterns.extend(res)

    for pat in patterns:
        pat["ts_i1"] = int(ts[pat["i1"]]); pat["ts_i2"] = int(ts[pat["i2"]])
        pat["ts_i3"] = int(ts[pat["i3"]]); pat["ts_i4"] = int(ts[pat["i4"]])
        pat["ts_i5"] = int(ts[pat["i5"]])
        pat['ts_breakout'] = int(ts[pat['i_breakout']])

    # dedup по ts_breakout
    seen: set[int] = set(); uniq: list[dict] = []
    for pat in sorted(patterns, key=lambda p: p['ts_breakout']):
        if pat['ts_breakout'] not in seen:
            seen.add(pat['ts_breakout']); uniq.append(pat)
    stats['passed_geometry'] = len(uniq)

    return uniq, stats


def compute_five_point_long(mode: str, pattern_name: str, df_1h: pd.DataFrame, symbol: str,
                             params: Optional[FivePointLongParams] = None,
                             ) -> tuple[pd.DataFrame, dict]:
    if params is None:
        params = FivePointLongParams()

    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    geo, stats = detect_five_point_long_geometry(mode, h, l, c, t_arr, params)

    n_bars = len(h)
    hma_mhull, hma_shull, wma50 = load_hma_wma(symbol, t_arr)

    # Per-pattern LONG filter config (зеркало SHORT _filter_cfg):
    #   ascending_triangle: 2a✓ 2c✓ 2d✗ (flat top → MA death-cross mirror редко)
    #   sym_triangle_long:  2a✗ 2c✗ 2d✓ (MA ещё нейтральны при пробое)
    #   falling_wedge:      2a✓ 2c✓ 2d✓
    #   falling_channel:    2a✓ 2c✗ 2d✓ (HMA ещё падает во время канала)
    #   rectangle_bottom:   2a✓ 2c✓ 2d✓
    _filter_cfg = {
        'ascending_triangle': dict(use_2a=True,  use_2c=True,  use_2d=False),
        'sym_triangle_long':  dict(use_2a=False, use_2c=False, use_2d=True),
        'falling_wedge':      dict(use_2a=True,  use_2c=True,  use_2d=True),
        'falling_channel':    dict(use_2a=True,  use_2c=False, use_2d=True),
        'rectangle_bottom':   dict(use_2a=True,  use_2c=True,  use_2d=True),
    }
    fcfg = _filter_cfg.get(mode, dict(use_2a=True, use_2c=True, use_2d=True))
    final = apply_fvg_ma_filter_long(geo, t_arr, c, fvg_from_key='i4', i_br_key='i_breakout',
                                      h_arr=h, l_arr=l, n_bars=n_bars, hma_mhull=hma_mhull,
                                      hma_shull=hma_shull, wma50=wma50, stats=stats, **fcfg)

    rows = []
    for pat in final:
        rows.append({
            "signal_ts": pat['ts_breakout'] + TF_1H_MS,
            "direction": "long",
            "pattern": pattern_name,
            "ts_breakout": pat['ts_breakout'],
            "ts_i1": pat["ts_i1"], "ts_i2": pat["ts_i2"], "ts_i3": pat["ts_i3"],
            "ts_i4": pat["ts_i4"], "ts_i5": pat["ts_i5"],
            "H1": pat['H1'], "L2": pat['L2'], "H3": pat['H3'],
            "L4": pat['L4'], "H5": pat['H5'],
            "breakout_close": pat['breakout_close'], "target": pat['TG'],
            "width_bars": pat['width_bars'],
            "status": "CONFIRMED",
        })
    _COLS = ["signal_ts","direction","pattern","ts_breakout",
             "ts_i1","ts_i2","ts_i3","ts_i4","ts_i5",
             "H1","L2","H3","L4","H5",
             "breakout_close","target","width_bars","status"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_COLS), stats


def print_stats(stats: dict, label: str) -> None:
    import sys
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry   n={stats['passed_geometry']}", file=sys.stderr, flush=True)
    print(f"    rej_fvg (2)              {stats['rej_fvg']}", file=sys.stderr, flush=True)
    print(f"    rej_hma_wma (2a/2c/2d)   {stats['rej_hma_wma']}", file=sys.stderr, flush=True)
    print(f"  passed_canon (full)   n={stats['passed_canon']}  ← {label}", file=sys.stderr, flush=True)
