"""five_point_core — общее ядро для 5 паттернов с геометрией i1..i5 (пять
чередующихся касаний): Symmetrical Triangle, Descending Triangle, Rising
Wedge, Rising Channel, Rectangle Top. Портировано verbatim из G:\\Claude\\
patterns\\{sym_triangle,descending_triangle,rising_wedge,rising_channel,
rectangle_top}\\scan_*.py на ASVK-portable контракт (df_1h передаётся
вызывающим кодом, FVG — events_e12d, HMA/WMA — level-1 shared indicators),
см. block5_common.py.

Геометрия (i1<i2<i3<i4<i5, хронологически):
  i1,i3,i5 = Williams N=2 FL (опора)
  i2,i4    = Williams N=2 FH (сопротивление)
  BR = первое закрытие ниже опоры после i5 (если сопротивление не пробито раньше)

Различие между 5 режимами — ТОЛЬКО в направлении/крутизне опоры и
сопротивления + формула TG (см. _process_i2_chunk и MODE-специфика внутри);
инвалидация/BR-поиск идентичны во всех 5 (verbatim из scan_triangle.py):

  sym_triangle:        опора растёт (>FLAT_TOL), сопротивление падает (Ch.66)
                        TG = close[BR] - (H2-L1)
  descending_triangle:  опора плоская (|slope|<=FLAT_TOL),
                        сопротивление падает заметно (Ch.65)
                        TG = sup_at(BR) - (H2-min(L1,L3,L5))
  rising_wedge:         обе линии растут, опора круче, narrow_frac > 0.10 (Ch.74)
                        TG = min(L1,L3,L5)  ("target is the bottom of the wedge")
  rising_channel:       обе линии растут, опора круче, narrow_frac <= 0.10
                        (non-canon, дополнение rising_wedge)
                        TG = min(L1,L3,L5)  (по аналогии, не из книги)
  rectangle_top:        ОБЕ линии плоские (|slope|<=FLAT_TOL) — i4 ищется
                        допуском около H2 (не "ниже"), в отличие от
                        descending_triangle (Ch.52)
                        TG = sup_at(BR) - (res_at(BR)-sup_at(BR))

FVG(2)/MA(2a,2c,2d) — общий SHORT-фильтр block5_common.apply_fvg_ma_filter,
окно FVG = (i4, BR] (verbatim граница из scan_*.py: range(i4, i_br+1)).
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
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — level-1
from common import TF_1H_MS
from block5_common import fractals, load_hma_wma, apply_fvg_ma_filter

MODES = ('sym_triangle', 'descending_triangle', 'rising_wedge', 'rising_channel', 'rectangle_top')


@dataclass
class FivePointParams:
    max_span_15:          int   = 96    # часов, i1 -> i5
    max_confirm:          int   = 24    # часов, i5 -> BR
    line_tol_pct:         float = 0.35  # допуск касания линии, % от цены
    flat_tol_pct_per_bar: float = 0.05  # порог "плоскости" опоры/сопротивления
    narrow_frac_thr:      float = 0.10  # rising_wedge >, rising_channel <=
    n_workers:             int  = 30


def _empty_stats() -> dict:
    return {
        'n_fh': 0, 'n_fl': 0, 'passed_geometry': 0,
        'rej_fvg': 0, 'rej_hma_wma': 0, 'passed_canon': 0,
    }


def _process_i2_chunk(mode: str, i2_chunk: list[int], h: np.ndarray, l: np.ndarray,
                       c: np.ndarray, fh_list: list[int], fl_list: list[int],
                       n_bars: int, p: FivePointParams) -> list[dict]:
    out: list[dict] = []
    FLAT = p.flat_tol_pct_per_bar

    for i2 in i2_chunk:
        i2 = int(i2)
        H2 = h[i2]

        i1_cands = [i for i in fl_list if i2 - p.max_span_15 <= i < i2]
        if not i1_cands:
            continue

        if mode in ('sym_triangle', 'descending_triangle'):
            i4_cands = [i for i in fh_list if i2 < i <= i2 + p.max_span_15 and h[i] < H2]
        elif mode == 'rectangle_top':
            near_h2 = H2 * FLAT * 6 / 100
            i4_cands = [i for i in fh_list if i2 < i <= i2 + p.max_span_15 and abs(h[i] - H2) <= near_h2]
        else:  # rising_wedge, rising_channel — сопротивление тоже растёт
            i4_cands = [i for i in fh_list if i2 < i <= i2 + p.max_span_15 and h[i] > H2]
        if not i4_cands:
            continue

        for i1 in i1_cands:
            L1 = l[i1]
            for i4 in i4_cands:
                H4 = h[i4]

                if mode in ('descending_triangle', 'rectangle_top'):
                    near = L1 * FLAT * 6 / 100
                    i3_cands = [i for i in fl_list if i2 < i < i4 and abs(l[i] - L1) <= near]
                else:
                    i3_cands = [i for i in fl_list if i2 < i < i4 and l[i] >= L1]
                if not i3_cands:
                    continue

                for i3 in i3_cands:
                    L3 = l[i3]

                    if mode in ('descending_triangle', 'rectangle_top'):
                        near = L1 * FLAT * 6 / 100
                        i5_cands = [i for i in fl_list if i4 < i <= i1 + p.max_span_15 and abs(l[i] - L1) <= near]
                    else:
                        i5_cands = [i for i in fl_list if i4 < i <= i1 + p.max_span_15 and l[i] >= L3]
                    if not i5_cands:
                        continue

                    sup_slope = (L3 - L1) / (i3 - i1)
                    sup_slope_pct = sup_slope * 100 / L1
                    res_slope = (H4 - H2) / (i4 - i2)
                    res_slope_pct = res_slope * 100 / H2

                    if mode == 'sym_triangle':
                        if sup_slope_pct <= FLAT:
                            continue
                        if res_slope >= 0:
                            continue
                    elif mode == 'descending_triangle':
                        if abs(sup_slope_pct) > FLAT:
                            continue
                        if res_slope_pct >= -FLAT:
                            continue
                    elif mode == 'rectangle_top':
                        if abs(sup_slope_pct) > FLAT:
                            continue
                        if abs(res_slope_pct) > FLAT:
                            continue
                    else:  # rising_wedge / rising_channel
                        if sup_slope <= 0:
                            continue
                        if res_slope <= 0:
                            continue
                        if sup_slope <= res_slope:
                            continue

                    def sup_at(i, L1=L1, sup_slope=sup_slope, i1=i1):
                        return L1 + sup_slope * (i - i1)

                    def res_at(i, H2=H2, res_slope=res_slope, i2=i2):
                        return H2 + res_slope * (i - i2)

                    gap_i1 = None
                    if mode in ('rising_wedge', 'rising_channel'):
                        gap_i1 = res_at(i1) - sup_at(i1)
                        if gap_i1 <= 0:
                            continue

                    for i5 in i5_cands:
                        L5 = l[i5]
                        tol = p.line_tol_pct / 100.0 * L5
                        if abs(L5 - sup_at(i5)) > tol * 3:
                            continue

                        if mode in ('rising_wedge', 'rising_channel'):
                            gap_i5 = res_at(i5) - sup_at(i5)
                            narrow_frac = 1 - gap_i5 / gap_i1
                            if mode == 'rising_wedge' and narrow_frac <= p.narrow_frac_thr:
                                continue
                            if mode == 'rising_channel' and narrow_frac > p.narrow_frac_thr:
                                continue

                        # инвалидация: между i1 и i5 сопротивление/опора не пробиты раньше времени
                        invalid = False
                        for j in range(i1, i5 + 1):
                            if h[j] > res_at(j) * (1 + p.line_tol_pct / 100.0):
                                invalid = True
                                break
                            if j > i1 and j < i5 and l[j] < sup_at(j) * (1 - p.line_tol_pct / 100.0):
                                invalid = True
                                break
                        if invalid:
                            continue

                        # BR: первое закрытие ниже опоры после i5 (пробой верха раньше — не считается)
                        max_scan = min(n_bars, i5 + p.max_confirm + 1)
                        i_br = None
                        for j in range(i5 + 1, max_scan):
                            if h[j] > res_at(j):
                                break
                            if c[j] < sup_at(j):
                                i_br = j
                                break
                        if i_br is None:
                            continue

                        if mode == 'sym_triangle':
                            height = H2 - L1
                            if height <= 0:
                                continue
                            TG = c[i_br] - height
                        elif mode == 'descending_triangle':
                            A = min(L1, L3, L5)
                            height = H2 - A
                            if height <= 0:
                                continue
                            TG = sup_at(i_br) - height
                        elif mode == 'rectangle_top':
                            height = res_at(i_br) - sup_at(i_br)
                            if height <= 0:
                                continue
                            TG = sup_at(i_br) - height
                        else:  # rising_wedge, rising_channel
                            TG = min(L1, L3, L5)

                        out.append({
                            'i1': i1, 'i2': i2, 'i3': i3, 'i4': i4, 'i5': i5,
                            'i_breakout': i_br,
                            'L1': float(L1), 'H2': float(H2), 'L3': float(L3),
                            'H4': float(H4), 'L5': float(L5),
                            'breakout_close': float(c[i_br]), 'TG': float(TG),
                            'width_bars': int(i5 - i1),
                        })
    return out


def detect_five_point_geometry(mode: str, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                                ts: np.ndarray,
                                params: Optional[FivePointParams] = None) -> tuple[list[dict], dict]:
    assert mode in MODES, f"unknown five_point mode: {mode}"
    if params is None:
        params = FivePointParams()

    n_bars = len(c)
    stats = _empty_stats()
    if n_bars < 20:
        return [], stats

    h = h.astype(np.float64); l = l.astype(np.float64); c = c.astype(np.float64)

    FH = fractals(h, 2, 'high')
    FL = fractals(l, 2, 'low')
    stats['n_fh'] = len(FH); stats['n_fl'] = len(FL)
    fh_list = FH.tolist(); fl_list = FL.tolist()

    n_workers = max(1, min(params.n_workers, os.cpu_count() or 4))
    i2_chunks = np.array_split(FH, n_workers)
    i2_chunks = [chunk.tolist() for chunk in i2_chunks if len(chunk) > 0]

    if len(i2_chunks) > 1:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_i2_chunk)(mode, chunk, h, l, c, fh_list, fl_list, n_bars, params)
            for chunk in i2_chunks
        )
    else:
        chunk_results = [_process_i2_chunk(mode, chunk, h, l, c, fh_list, fl_list, n_bars, params)
                          for chunk in i2_chunks]

    patterns: list[dict] = []
    for res in chunk_results:
        patterns.extend(res)

    for p in patterns:
        p["ts_i1"] = int(ts[p["i1"]]); p["ts_i2"] = int(ts[p["i2"]]); p["ts_i3"] = int(ts[p["i3"]]); p["ts_i5"] = int(ts[p["i5"]])
        p['ts_i4'] = int(ts[p['i4']])
        p['ts_breakout'] = int(ts[p['i_breakout']])

    # dedup по ts_breakout (одна i2-голова может дать несколько валидных комбо)
    seen: set[int] = set(); uniq: list[dict] = []
    for p in sorted(patterns, key=lambda p: p['ts_breakout']):
        if p['ts_breakout'] not in seen:
            seen.add(p['ts_breakout']); uniq.append(p)
    stats['passed_geometry'] = len(uniq)

    return uniq, stats


def compute_five_point(mode: str, pattern_name: str, df_1h: pd.DataFrame, symbol: str,
                        params: Optional[FivePointParams] = None) -> tuple[pd.DataFrame, dict]:
    if params is None:
        params = FivePointParams()

    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    geo, stats = detect_five_point_geometry(mode, h, l, c, t_arr, params)

    n_bars = len(h)
    hma_mhull, hma_shull, wma50 = load_hma_wma(symbol, t_arr)

    # Per-pattern filter config (audit 2026-07-26: pass-rates per filter)
    # descending:  2a=94% 2c=89% 2d=42% → drop 2d
    # sym_tri:     2a=50% 2c=47% 2d=75% → drop 2a,2c
    # rising_wedge:2a=37% 2c=25% 2d=78% → keep all (all-4 gives target ~25/yr/asset)
    # rising_channel: 2a=64% 2c=38% 2d=78% → drop 2c
    _filter_cfg = {
        'descending_triangle': dict(use_2a=True,  use_2c=True,  use_2d=False),
        'sym_triangle':        dict(use_2a=False, use_2c=False, use_2d=True),
        'rising_wedge':        dict(use_2a=True,  use_2c=True,  use_2d=True),
        'rising_channel':      dict(use_2a=True,  use_2c=False, use_2d=True),
        'rectangle_top':       dict(use_2a=True,  use_2c=True,  use_2d=True),
    }
    fcfg = _filter_cfg.get(mode, dict(use_2a=True, use_2c=True, use_2d=True))
    final = apply_fvg_ma_filter(geo, t_arr, c, fvg_from_key='i4', i_br_key='i_breakout',
                                 h_arr=h, l_arr=l, n_bars=n_bars, hma_mhull=hma_mhull,
                                 hma_shull=hma_shull, wma50=wma50, stats=stats, **fcfg)

    rows = []
    for p in final:
        rows.append({
            "signal_ts": p['ts_breakout'] + TF_1H_MS,
            "direction": "short",
            "pattern": pattern_name,
            "ts_breakout": p['ts_breakout'],
            "ts_i1": p["ts_i1"], "ts_i2": p["ts_i2"], "ts_i3": p["ts_i3"], "ts_i4": p["ts_i4"], "ts_i5": p["ts_i5"],
            "L1": p['L1'], "H2": p['H2'], "L3": p['L3'], "H4": p['H4'], "L5": p['L5'],
            "breakout_close": p['breakout_close'], "target": p['TG'],
            "width_bars": p['width_bars'],
            "status": "CONFIRMED",
        })
    return pd.DataFrame(rows), stats


def print_stats(stats: dict, label: str) -> None:
    import sys
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry   n={stats['passed_geometry']}", file=sys.stderr, flush=True)
    print(f"    rej_fvg (2)              {stats['rej_fvg']}", file=sys.stderr, flush=True)
    print(f"    rej_hma_wma (2a/2c/2d)   {stats['rej_hma_wma']}", file=sys.stderr, flush=True)
    print(f"  passed_canon (full)   n={stats['passed_canon']}  ← {label}", file=sys.stderr, flush=True)
