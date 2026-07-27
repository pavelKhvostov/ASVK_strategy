"""Double Bottom (Adam&Adam variant, LONG) — зеркало Double Top (Bulkowski Ch.29).

Геометрия (i1<i2<i3<i4, хронологически):
  i1,i3 = Williams N=2 FL (два дна, ~одна цена, допуск SAME_PRICE_TOL_PCT)
  i2,i4 = Williams N=2 FH (пики, ~одна цена)
  Инвалидация — per-leg true extremum (L1/H2/L3/H4 — настоящий экстремум своего
  отрезка) + потолок max(H2,H4) не пробит снизу, между i2 и i4 пол не пробит.
  BR = первое закрытие ВЫШЕ линии сопротивления (i2->i4, экстраполирована) после i4.
  MAX_SPAN i1->i4 = 96ч, MAX_CONFIRM i4->BR = 24ч.

TG (зеркало Double Top Ch.30):
  height = max(H2,H4) - min(L1,L3)
  TG = max(H2,H4) + height

FVG(2)/MA(2a_long,2c_long,2d_long) — LONG-фильтр apply_fvg_ma_filter_long,
bullish FVG (direction=long), окно FVG = (i4, BR].
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
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import load_1m, agg_1h, DATA_OUT, TF_1H_MS
from block5_common import fractals, load_hma_wma, apply_fvg_ma_filter_long

PATTERN_NAME = "double_bottom"


@dataclass
class DoubleBottomParams:
    max_span:           int   = 96
    max_confirm:        int   = 24
    line_tol_pct:       float = 0.35
    same_price_tol_pct: float = 3.0
    n_workers:          int   = 30


def _empty_stats() -> dict:
    return {'n_fh': 0, 'n_fl': 0, 'passed_geometry': 0,
            'rej_fvg': 0, 'rej_hma_wma': 0, 'passed_canon': 0}


def _process_chunk(i1_chunk: list[int], h: np.ndarray, l: np.ndarray, c: np.ndarray,
                    fh_list: list[int], fl_list: list[int], n_bars: int,
                    p: DoubleBottomParams) -> list[dict]:
    out: list[dict] = []
    TOL = p.same_price_tol_pct

    for i1 in i1_chunk:
        i1 = int(i1)
        L1 = l[i1]  # первое дно

        i2_cands = [i for i in fh_list if i1 < i <= i1 + p.max_span]
        if not i2_cands:
            continue

        for i2 in i2_cands:
            H2 = h[i2]
            # i3 = второе дно (~такая же цена, как L1)
            i3_cands = [i for i in fl_list
                        if i2 < i <= i1 + p.max_span
                        and abs(l[i] - L1) * 100 / L1 <= TOL]
            if not i3_cands:
                continue

            # per-leg integrity от i1 до i2
            if h[i1:i2 + 1].max() > H2:
                continue
            if l[i1:i2 + 1].min() < L1:
                continue

            for i3 in i3_cands:
                L3 = l[i3]
                floor_val = min(L1, L3)

                # между i2 и i3: close не поднимается выше H2
                invalid = False
                for j in range(i2, i3):
                    if c[j] > H2:
                        invalid = True
                        break
                if invalid:
                    continue

                # per-leg integrity от i2 до i3
                if l[i2:i3 + 1].min() < L3:
                    continue
                if h[i2:i3 + 1].max() > H2:
                    continue

                # i4 = второй пик (~такая же цена, как H2)
                i4_cands = [i for i in fh_list
                            if i3 < i <= i1 + p.max_span
                            and abs(h[i] - H2) * 100 / H2 <= TOL]
                if not i4_cands:
                    continue

                for i4 in i4_cands:
                    H4 = h[i4]
                    # per-leg integrity от i3 до i4
                    if h[i3:i4 + 1].max() > H4:
                        continue
                    if l[i3:i4 + 1].min() < L3:
                        continue

                    res_slope = (H4 - H2) / (i4 - i2)

                    def res_at(i, H2=H2, i2=i2, res_slope=res_slope):
                        return H2 + res_slope * (i - i2)

                    # Инвалидация: пол floor_val не пробит; сопр. не пробито раньше времени
                    invalid2 = False
                    for j in range(i1, i4 + 1):
                        if c[j] < floor_val * (1 - p.line_tol_pct / 100.0):
                            invalid2 = True
                            break
                        if j > i2 and j < i4 and c[j] > res_at(j) * (1 + p.line_tol_pct / 100.0):
                            invalid2 = True
                            break
                    if invalid2:
                        continue

                    # BR: первое закрытие ВЫШЕ линии сопротивления после i4
                    max_scan = min(n_bars, i4 + p.max_confirm + 1)
                    i_br = None
                    for j in range(i4 + 1, max_scan):
                        if c[j] < floor_val:
                            break
                        if c[j] > res_at(j):
                            i_br = j
                            break
                    if i_br is None:
                        continue

                    ceiling = max(H2, H4)
                    base = min(L1, L3)
                    height = ceiling - base
                    if height <= 0:
                        continue
                    TG = ceiling + height

                    out.append({
                        'i1': i1, 'i2': i2, 'i3': i3, 'i4': i4, 'i_breakout': i_br,
                        'L1': float(L1), 'H2': float(H2), 'L3': float(L3), 'H4': float(H4),
                        'breakout_close': float(c[i_br]), 'TG': float(TG),
                        'width_bars': int(i4 - i1),
                    })
    return out


def detect_double_bottom_geometry(h: np.ndarray, l: np.ndarray, c: np.ndarray, ts: np.ndarray,
                                   params: Optional[DoubleBottomParams] = None,
                                   ) -> tuple[list[dict], dict]:
    if params is None:
        params = DoubleBottomParams()

    n_bars = len(c)
    stats = _empty_stats()
    if n_bars < 20:
        return [], stats

    h = h.astype(np.float64); l = l.astype(np.float64); c = c.astype(np.float64)

    FH = fractals(h, 2, 'high')
    FL = fractals(l, 2, 'low')
    stats['n_fh'] = len(FH); stats['n_fl'] = len(FL)
    fh_list = FH.tolist(); fl_list = FL.tolist()

    # Внешний цикл по FL (i1 = первое дно) — зеркало double_top (цикл по FH)
    n_workers = max(1, min(params.n_workers, os.cpu_count() or 4))
    chunks = np.array_split(FL, n_workers)
    chunks = [chunk.tolist() for chunk in chunks if len(chunk) > 0]

    if len(chunks) > 1:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_chunk)(chunk, h, l, c, fh_list, fl_list, n_bars, params)
            for chunk in chunks
        )
    else:
        chunk_results = [_process_chunk(chunk, h, l, c, fh_list, fl_list, n_bars, params)
                          for chunk in chunks]

    patterns: list[dict] = []
    for res in chunk_results:
        patterns.extend(res)

    for pat in patterns:
        pat['ts_i4'] = int(ts[pat['i4']])
        pat['ts_breakout'] = int(ts[pat['i_breakout']])

    seen: set[int] = set(); uniq: list[dict] = []
    for pat in sorted(patterns, key=lambda p: p['ts_breakout']):
        if pat['ts_breakout'] not in seen:
            seen.add(pat['ts_breakout']); uniq.append(pat)
    stats['passed_geometry'] = len(uniq)

    return uniq, stats


def compute_double_bottom(df_1h: pd.DataFrame, symbol: str,
                           params: Optional[DoubleBottomParams] = None,
                           ) -> tuple[pd.DataFrame, dict]:
    if params is None:
        params = DoubleBottomParams()

    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    geo, stats = detect_double_bottom_geometry(h, l, c, t_arr, params)

    n_bars = len(h)
    hma_mhull, hma_shull, wma50 = load_hma_wma(symbol, t_arr)

    final = apply_fvg_ma_filter_long(geo, t_arr, c, fvg_from_key='i4', i_br_key='i_breakout',
                                      h_arr=h, l_arr=l, n_bars=n_bars, hma_mhull=hma_mhull,
                                      hma_shull=hma_shull, wma50=wma50, stats=stats)

    rows = []
    for pat in final:
        rows.append({
            "signal_ts": pat['ts_breakout'] + TF_1H_MS,
            "direction": "long",
            "pattern": PATTERN_NAME,
            "ts_breakout": pat['ts_breakout'], "ts_i4": pat['ts_i4'],
            "L1": pat['L1'], "H2": pat['H2'], "L3": pat['L3'], "H4": pat['H4'],
            "breakout_close": pat['breakout_close'], "target": pat['TG'],
            "width_bars": pat['width_bars'],
            "status": "CONFIRMED",
        })
    return pd.DataFrame(rows), stats


def print_stats(stats: dict) -> None:
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry   n={stats['passed_geometry']}", file=sys.stderr, flush=True)
    print(f"    rej_fvg (2)              {stats['rej_fvg']}", file=sys.stderr, flush=True)
    print(f"    rej_hma_wma (2a/2c/2d)   {stats['rej_hma_wma']}", file=sys.stderr, flush=True)
    print(f"  passed_canon (full)   n={stats['passed_canon']}  ← DOUBLE_BOTTOM", file=sys.stderr, flush=True)


def main() -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-26")
    args = ap.parse_args()

    print(f"{PATTERN_NAME}: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    df_1h = agg_1h(load_1m(args.symbol))
    hits, stats = compute_double_bottom(df_1h, args.symbol)
    print_stats(stats)

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  in window: {len(hits):,} signals", file=sys.stderr, flush=True)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"{PATTERN_NAME}_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
