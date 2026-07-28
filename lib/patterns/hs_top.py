"""Head & Shoulders TOP — канон Bulkowski (Encyclopedia 3rd Ed., Ch.41).

Геометрия (LS/LA/H/RA/RS + breakout) портирована verbatim из canon-верифи­
цированного G:\\Claude\\patterns\\hs_top\\scan_hs_top.py (91 паттерн на
BTCUSDT 1h, 2020-01 → 2026-07) — ЗАМЕНА прежней scipy.find_peaks/ATR-
prominence геометрии, которая не была фракталами Уильямса (аудит 2026-07-24,
решение принято пользователем 2026-07-25: заменить на верную версию).

Anchors = Williams N=2 fractals (5-свечной пивот) для highs (FH) и lows (FL).
Голова H — самый высокий FH, RS — самый высокий FH между H и BR (не
"последний"). Neckline = линия LA→RA. BR = первый close < neckline после RS.

Canon-условия (важные_условия.md), не менявшиеся при этом переносе:
  1a) intermediate FH не касается условных линий LS→H и H→RS
  1d) RS→BR ≤ MAX_CONFIRM свечей (свежий пробой)
  4)  span LS→RS ≤ MAX_SPAN часов
  5)  |neckline slope| ≤ max_nl_slope_pct_per_bar %/бар (отсекает клинья/треугольники)
  2)  Bearish FVG обязателен в (RS, BR]        — events_e12d (element=fvg, tf=1h,
                                                   direction=short, kind=born)
  2a) WMA(50)[BR] > close[BR] И HMA(78)[BR] > close[BR]
  2c) HMA(78)[BR] < HMA(78)[BR-2]              (трендлайн падает)
  2d) HMA(78)[BR] > WMA(50)[BR]
  2b) traveled/distance < max_path_traveled_pct, distance = NL_BR − TG

TG (Bulkowski Ch.41, verified — см. scan_hs_top.py docstring):
  height = H_p − NL(H_time)                        # всегда
  base   = RA_p  если neckline down-sloping (slope<0)
         = close[BR]  если neckline up/flat
  TG     = base − height
(Раньше в ASVK-версии base был ВСЕГДА close[BR] — неверно для down-sloping
neckline; исправлено при этом переносе.)

HMA(78)/WMA(50) — level-1 shared indicators на 1h (lib/trendline.py --tf 1h
--length 78, lib/wma.py). FVG/2a/2c/2d/2b-слой (apply_canon_filters) не
менялся — уже был верно закодирован по canon 2026-07-23, менялась только
геометрия и TG-формула выше неё.

Только H&S TOP (SHORT). H&S Bottom (LONG, зеркальный паттерн) не закодирован.

Reads:
  G:\\ASVK\\data\\{SYM}USDT_1m.csv
  G:\\ASVK\\data\\events\\events_e12d_{SYM}_*.parquet        (FVG, element=fvg)
  G:\\ASVK\\data\\trendline\\trendline_{SYM}_1h78_*.parquet  (HMA-78/1h, 2a/2c/2d)
  G:\\ASVK\\data\\wma\\wma_{SYM}_*.parquet                    (WMA-50/1h, 2a/2d)
Writes:
  G:\\ASVK\\data\\patterns\\hs_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — level-1
from common import load_1m, agg_1h, latest_events_path, DATA_OUT, TF_1H_MS
from trendline import latest_trendline_path
from wma import latest_wma_path
from block5_common import fractals


# ============================================================================
# Параметры (verbatim из scan_hs_top.py, оформлены как dataclass)
# ============================================================================

@dataclass
class HSTopParams:
    max_span:                 int   = 72     # LS→H и LS→RS ≤ 72 бара (часа)
    max_confirm_bars:         int   = 7      # RS→BR ≤ 7 баров (1d canon)
    max_nl_slope_pct_per_bar: float = 0.05   # |neckline slope| ≤ 0.05%/бар (canon 5)
    n_workers:                int   = 30

    # --- канон-доп: 2b path-check (важные_условия.md) ----------------------
    max_path_traveled_pct: float = 0.60


# ============================================================================
# Helpers
# ============================================================================

def _line_value_at(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def _atr14_nolookahead(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    out = np.zeros_like(tr)
    cum = 0.0
    for i in range(len(tr)):
        cum += tr[i]
        if i >= 14:
            cum -= tr[i - 14]
            out[i] = cum / 14
        else:
            out[i] = cum / (i + 1)
    return out


def _empty_stats() -> dict:
    return {
        'n_fh': 0, 'n_fl': 0,
        'rej_invalidated': 0, 'rej_no_breakout': 0,
        'passed_geometry': 0,
        # доп-канон (apply_canon_filters)
        'rej_fvg': 0, 'rej_hma_wma': 0, 'rej_path_2b': 0,
        'passed_canon': 0,
    }


# ============================================================================
# Геометрия паттерна — Williams N=2 fractals, verbatim из scan_hs_top.py
# (_process_head_chunk), портировано на joblib+numpy без файлового I/O.
# ============================================================================

def _process_head_chunk(head_chunk: list[int], h: np.ndarray, l: np.ndarray,
                         c: np.ndarray, FH: np.ndarray, FL: np.ndarray,
                         fh_set: set[int], n_bars: int,
                         params: HSTopParams) -> tuple[list[dict], dict]:
    MAX_SPAN = params.max_span
    MAX_CONFIRM = params.max_confirm_bars
    local_stats = {'rej_invalidated': 0, 'rej_no_breakout': 0}
    out: list[dict] = []

    for i_h in head_chunk:
        i_h = int(i_h)
        if i_h < 6 or i_h > n_bars - MAX_CONFIRM - 2:
            continue
        H_p = h[i_h]

        ls_cands = [i for i in FH if i_h - MAX_SPAN <= i < i_h and h[i] < H_p]
        if not ls_cands:
            continue

        for i_ls in ls_cands:
            LS_p = h[i_ls]
            left_fls = [i for i in FL if i_ls < i < i_h]
            if not left_fls:
                continue
            i_la = min(left_fls, key=lambda i: l[i])
            LA_p = l[i_la]

            max_scan = min(n_bars, i_ls + MAX_SPAN + MAX_CONFIRM + 1)
            i_br = i_rs = i_ra = None
            RS_p = RA_p = slope = None
            invalidated = False

            for j in range(i_h + 3, max_scan):
                if c[j] > H_p:
                    invalidated = True
                    break
                rs_cands_j = [i for i in fh_set if i_h < i < j and h[i] < H_p]
                if not rs_cands_j:
                    continue
                # RS = самый ВЫСОКИЙ FH между H и BR (не latest)
                cur_i_rs = max(rs_cands_j, key=lambda i: h[i])
                if cur_i_rs - i_ls > MAX_SPAN:
                    continue
                cur_RS_p = h[cur_i_rs]
                # 1d: между RS и BR ≤ MAX_CONFIRM баров
                if j - cur_i_rs > MAX_CONFIRM:
                    continue
                if j > cur_i_rs + 1:
                    if float(h[cur_i_rs + 1:j + 1].max()) >= cur_RS_p:
                        continue
                right_fls = [i for i in FL if i_h < i < cur_i_rs]
                if not right_fls:
                    continue
                cur_i_ra = min(right_fls, key=lambda i: l[i])
                cur_RA_p = l[cur_i_ra]
                # 1a: intermediate FH не касается LS→H / H→RS
                cond_fail = False
                for i_int in FH:
                    i_int = int(i_int)
                    if i_ls < i_int < i_h:
                        if h[i_int] >= _line_value_at(i_ls, LS_p, i_h, H_p, i_int):
                            cond_fail = True
                            break
                    elif i_h < i_int < cur_i_rs:
                        if h[i_int] >= _line_value_at(i_h, H_p, cur_i_rs, cur_RS_p, i_int):
                            cond_fail = True
                            break
                if cond_fail:
                    continue
                cur_slope = (cur_RA_p - LA_p) / (cur_i_ra - i_la)
                nl_j = LA_p + cur_slope * (j - i_la)
                if c[j] < nl_j:
                    i_br = j; i_rs = cur_i_rs; i_ra = cur_i_ra
                    RS_p = cur_RS_p; RA_p = cur_RA_p; slope = cur_slope
                    break

            if invalidated:
                local_stats['rej_invalidated'] += 1
                continue
            if i_br is None:
                local_stats['rej_no_breakout'] += 1
                continue

            # 5: |neckline slope %/бар| ≤ max_nl_slope_pct_per_bar
            nl_slope_pct_per_bar = slope * 100 / LA_p
            if abs(nl_slope_pct_per_bar) > params.max_nl_slope_pct_per_bar:
                continue

            nl_h = LA_p + slope * (i_h - i_la)
            nl_br = LA_p + slope * (i_br - i_la)
            pattern_height = H_p - nl_h
            up_sloping = slope >= 0
            c_br = c[i_br]
            base = c_br if up_sloping else RA_p
            target_full = base - pattern_height
            target_half = base - 0.5 * pattern_height
            target_2x = base - 2.0 * pattern_height

            out.append({
                'i_ls': i_ls, 'i_la': i_la, 'i_h': i_h, 'i_ra': i_ra, 'i_rs': i_rs,
                'i_breakout': i_br,
                'ls_p': float(LS_p), 'la_p': float(LA_p), 'h_p': float(H_p),
                'ra_p': float(RA_p), 'rs_p': float(RS_p),
                'ts_ls': 0, 'ts_la': 0, 'ts_h': 0, 'ts_ra': 0, 'ts_rs': 0,   # заполняется выше (нужен t_arr)
                'ts_breakout': 0,
                'pattern_height': float(pattern_height),
                'breakout_close': float(c_br),
                'breakout_base': float(base),
                'target_half': float(target_half),
                'target_full': float(target_full),
                'target_2x': float(target_2x),
                'stop': float(RS_p * 1.005),
                'nl_at_br': float(nl_br),
                'up_sloping_neckline': bool(up_sloping),
                'neckline_at_head': float(nl_h),
                'width_bars': int(i_rs - i_ls),
            })
            # НЕ break: другие (LS) комбо для этой же головы допустимы

    return out, local_stats


def detect_hs_top_geometry(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                            ts: np.ndarray,
                            params: Optional[HSTopParams] = None) -> tuple[list[dict], dict]:
    if params is None:
        params = HSTopParams()

    n_bars = len(c)
    stats = _empty_stats()
    if n_bars < 20:
        return [], stats

    h = h.astype(np.float64); l = l.astype(np.float64); c = c.astype(np.float64)

    FH = fractals(h, 2, 'high')
    FL = fractals(l, 2, 'low')
    stats['n_fh'] = len(FH); stats['n_fl'] = len(FL)
    fh_set = set(int(i) for i in FH)

    n_workers = max(1, min(params.n_workers, os.cpu_count() or 4))
    head_chunks = np.array_split(FH, n_workers)
    head_chunks = [chunk.tolist() for chunk in head_chunks if len(chunk) > 0]

    if len(head_chunks) > 1:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_head_chunk)(chunk, h, l, c, FH, FL, fh_set, n_bars, params)
            for chunk in head_chunks
        )
    else:
        chunk_results = [_process_head_chunk(chunk, h, l, c, FH, FL, fh_set, n_bars, params)
                          for chunk in head_chunks]

    patterns: list[dict] = []
    for res, local_stats in chunk_results:
        patterns.extend(res)
        for k, v in local_stats.items():
            stats[k] += v

    # ts_* заполняем после сборки (ts массив общий для всех воркеров)
    for p in patterns:
        p['ts_ls'] = int(ts[p['i_ls']]); p['ts_la'] = int(ts[p['i_la']])
        p['ts_h']  = int(ts[p['i_h']]);  p['ts_ra'] = int(ts[p['i_ra']])
        p['ts_rs'] = int(ts[p['i_rs']]); p['ts_breakout'] = int(ts[p['i_breakout']])

    # dedup by ts_breakout (одна голова может дать несколько валидных BR через разные LS)
    seen: set[int] = set(); uniq: list[dict] = []
    for p in patterns:
        if p['ts_breakout'] not in seen:
            seen.add(p['ts_breakout']); uniq.append(p)
    stats['passed_geometry'] = len(uniq)

    # ATR14 — только для информационного height_atr в выходной схеме
    atr14 = _atr14_nolookahead(h, l, c)
    for p in uniq:
        atr_h = float(atr14[p['i_h']])
        p['atr14_at_head'] = atr_h
        p['height_atr'] = float(p['pattern_height'] / atr_h) if atr_h > 0 else 0.0

    return uniq, stats


# ============================================================================
# Канон-доп: FVG(2) + HMA/WMA(2a/2c/2d) + path(2b) — НЕ менялось при переносе
# ============================================================================

def apply_canon_filters(geo_patterns: list[dict], t_arr: np.ndarray,
                         h_arr: np.ndarray, l_arr: np.ndarray,
                         hma78_mhull: np.ndarray, hma78_shull: np.ndarray,
                         wma50: np.ndarray, stats: dict,
                         params: HSTopParams) -> list[dict]:
    """Добавляет правила 2/2a/2b/2c/2d (важные_условия.md)."""
    n_bars = len(h_arr)
    out = []
    for p in geo_patterns:
        i_br = p['i_breakout']
        i_rs = p['i_rs']

        # 2) Bearish FVG inline: low[i-1] > high[i+1] в окне (RS, BR) — строго до BR
        fvg_ok = False
        for i_fvg in range(i_rs + 1, min(n_bars - 1, i_br)):
            if 0 < i_fvg < n_bars - 1:
                if l_arr[i_fvg - 1] > h_arr[i_fvg + 1]:
                    fvg_ok = True; break
        if not fvg_ok:
            stats['rej_fvg'] += 1
            continue

        hma_br = hma78_mhull[i_br]
        shull_br = hma78_shull[i_br]
        wma_br = wma50[i_br]
        close_br = p['breakout_close']

        if any(np.isnan(x) for x in (hma_br, shull_br, wma_br)):
            stats['rej_hma_wma'] += 1
            continue

        cond_2a = (wma_br > close_br) and (hma_br > close_br)
        cond_2c = close_br < shull_br  # смена цвета: close < mhull[i-2]
        cond_2d = hma_br > wma_br
        if not (cond_2a and cond_2c and cond_2d):
            stats['rej_hma_wma'] += 1
            continue

        # 2b) traveled/distance < max_path_traveled_pct, TG = target_full
        distance = p['nl_at_br'] - p['target_full']
        traveled = p['nl_at_br'] - close_br
        path_ok = True
        if distance > 0:
            path_ok = (traveled / distance) < params.max_path_traveled_pct
        if not path_ok:
            stats['rej_path_2b'] += 1
            continue

        stats['passed_canon'] += 1
        p2 = dict(p)
        p2['fvg_ok'] = True
        p2['cond_2a'] = True
        p2['cond_2c'] = True
        p2['cond_2d'] = True
        p2['cond_2b'] = True
        out.append(p2)
    return out


# ============================================================================
# Orchestration
# ============================================================================

def compute_hs(df_1h: pd.DataFrame, symbol: str,
               params: Optional[HSTopParams] = None) -> tuple[pd.DataFrame, dict]:
    if params is None:
        params = HSTopParams()

    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    geo, stats = detect_hs_top_geometry(h, l, c, t_arr, params)


    tl = pd.read_parquet(latest_trendline_path(symbol, variant="1h78"),
                          columns=["ts", "mhull", "shull"])
    hma_mhull = pd.Series(tl["mhull"].to_numpy(), index=tl["ts"].to_numpy()).reindex(t_arr).to_numpy()
    hma_shull = pd.Series(tl["shull"].to_numpy(), index=tl["ts"].to_numpy()).reindex(t_arr).to_numpy()

    wm = pd.read_parquet(latest_wma_path(symbol), columns=["ts", "wma50"])
    wma50 = pd.Series(wm["wma50"].to_numpy(), index=wm["ts"].to_numpy()).reindex(t_arr).to_numpy()

    final = apply_canon_filters(geo, t_arr, h, l, hma_mhull, hma_shull, wma50, stats, params)

    rows = []
    for p in final:
        rows.append({
            "signal_ts": p['ts_breakout'] + TF_1H_MS,   # born = close BR + tf_ms
            "direction": "short",
            "pattern": "hs_top",
            "ts_breakout": p['ts_breakout'],
            "ts_ls": p['ts_ls'], "ts_h": p['ts_h'], "ts_rs": p['ts_rs'],
            "h_p": p['h_p'], "ls_p": p['ls_p'], "rs_p": p['rs_p'],
            "breakout_close": p['breakout_close'],
            "target_half": p['target_half'], "target_full": p['target_full'],
            "target_2x": p['target_2x'], "stop": p['stop'],
            "width_bars": p['width_bars'], "height_atr": p['height_atr'],
            "status": "CONFIRMED",   # detect_hs_top_geometry возвращает только realized breakouts
        })
    return pd.DataFrame(rows), stats


def print_stats(stats: dict) -> None:
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"    rej_invalidated          {stats['rej_invalidated']}", file=sys.stderr, flush=True)
    print(f"    rej_no_breakout          {stats['rej_no_breakout']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry (1,1a,1d,4,5)   n={stats['passed_geometry']}", file=sys.stderr, flush=True)
    print(f"    rej_fvg (2)                {stats['rej_fvg']}", file=sys.stderr, flush=True)
    print(f"    rej_hma_wma (2a/2c/2d)     {stats['rej_hma_wma']}", file=sys.stderr, flush=True)
    print(f"    rej_path_2b                {stats['rej_path_2b']}", file=sys.stderr, flush=True)
    print(f"  passed_canon (full)   n={stats['passed_canon']}  ← HS_TOP", file=sys.stderr, flush=True)


def main() -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-23")
    args = ap.parse_args()

    print(f"hs_top (Williams-fractal canon): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    df_1m = load_1m(args.symbol)
    df_1h = agg_1h(df_1m)
    print(f"  1h bars: {len(df_1h):,}", file=sys.stderr, flush=True)

    hits, stats = compute_hs(df_1h, args.symbol)
    print_stats(stats)

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  in window [{args.start}, {args.end}): {len(hits):,} signals", file=sys.stderr, flush=True)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"hs_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
