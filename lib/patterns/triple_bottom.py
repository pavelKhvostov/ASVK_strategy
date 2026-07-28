"""Triple Bottom (LONG) — зеркало Triple Top (SHORT). Bulkowski Ch.69 "Triple Bottoms".

Геометрия (i1<i2<i3<i4<i5<i6, по аналогии с Triple Top — 6 точек):
  i1,i3,i5 = Williams N=2 FL (впадины, ~одна цена, допуск SAME_PRICE_TOL_PCT)
  i2,i4,i6 = Williams N=2 FH (пики, ~одна цена, линия сопротивления через i2->i6)
  Граница с H&S Bottom (СТРОГО без допуска):
    NOT(L3 < L1 and L3 < L5) — иначе это уже H&S Bottom, не Triple Bottom.
  Инвалидация — per-leg true extremum на каждом из 5 отрезков + floor min(L1,L3,L5)
  не пробит между i1 и i6 (с допуском одной свечи).
  i4 между уровнями i2,i6 и i1,i3,i5 (ниже линии сопротивления).
  BR = первое закрытие ВЫШЕ линии сопротивления (i2->i6, экстраполирована) после i6.
  MAX_SPAN i1->i6 = 96ч, MAX_CONFIRM i6->BR = 24ч.

TG (зеркало Bulkowski Triple Top):
  height = max(H2,H4,H6) - min(L1,L3,L5)
  TG = max(H2,H4,H6) + height

FVG(2)/MA(2a,2c,2d) — LONG-фильтр block5_common.apply_fvg_ma_filter_long,
окно FVG = (i6, BR].
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

PATTERN_NAME = "triple_bottom"


@dataclass
class TripleBottomParams:
    max_span:           int   = 96
    max_confirm:        int   = 24
    line_tol_pct:       float = 0.35
    same_price_tol_pct: float = 1.0
    n_workers:          int   = 30


def _empty_stats() -> dict:
    return {'n_fh': 0, 'n_fl': 0, 'passed_geometry': 0,
            'rej_fvg': 0, 'rej_hma_wma': 0, 'passed_canon': 0}


def _process_chunk(i1_chunk: list[int], h: np.ndarray, l: np.ndarray, c: np.ndarray,
                    fh_list: list[int], fl_list: list[int], n_bars: int,
                    p: TripleBottomParams) -> list[dict]:
    out: list[dict] = []
    TOL = p.same_price_tol_pct

    for i1 in i1_chunk:
        i1 = int(i1)
        L1 = l[i1]  # первая впадина (support)

        i2_cands = [i for i in fh_list if i1 < i <= i1 + p.max_span]
        if not i2_cands:
            continue

        for i2 in i2_cands:
            H2 = h[i2]
            # true extremum i1..i2: L1 — настоящий минимум, H2 — настоящий максимум
            if l[i1:i2 + 1].min() < L1:
                continue
            if h[i1:i2 + 1].max() > H2:
                continue

            i3_cands = [i for i in fl_list
                        if i2 < i <= i1 + p.max_span
                        and abs(l[i] - L1) * 100 / L1 <= TOL]
            if not i3_cands:
                continue

            for i3 in i3_cands:
                L3 = l[i3]
                # true extremum i2..i3
                if l[i2:i3 + 1].min() < L3:
                    continue
                if h[i2:i3 + 1].max() > H2:
                    continue

                i4_cands = [i for i in fh_list
                            if i3 < i <= i1 + p.max_span
                            and h[i] <= H2 * (1 + TOL / 100.0)]
                # i4 между уровнями i1,i3,i5 и i2,i6 (ниже линии сопротивления)
                if not i4_cands:
                    continue

                for i4 in i4_cands:
                    H4 = h[i4]
                    # true extremum i3..i4
                    if h[i3:i4 + 1].max() > H4:
                        continue
                    if l[i3:i4 + 1].min() < L3:
                        continue

                    i5_cands = [i for i in fl_list
                                if i4 < i <= i1 + p.max_span
                                and abs(l[i] - L1) * 100 / L1 <= TOL
                                and abs(l[i] - L3) * 100 / L3 <= TOL]
                    if not i5_cands:
                        continue

                    for i5 in i5_cands:
                        L5 = l[i5]
                        # граница с H&S Bottom: центр НЕ должен строго превышать обе внешние впадины
                        if L3 < L1 and L3 < L5:
                            continue  # это уже H&S Bottom, не Triple Bottom
                        # true extremum i4..i5
                        if l[i4:i5 + 1].min() < L5:
                            continue
                        if h[i4:i5 + 1].max() > H4:
                            continue

                        i6_cands = [i for i in fh_list
                                    if i5 < i <= i1 + p.max_span
                                    and abs(h[i] - H2) * 100 / H2 <= TOL]
                        # i6 ≈ H2: линия сопротивления через точки 2 и 6
                        if not i6_cands:
                            continue

                        support = min(L1, L3, L5)

                        for i6 in i6_cands:
                            H6 = h[i6]
                            # true extremum i5..i6
                            if h[i5:i6 + 1].max() > H6:
                                continue
                            if l[i5:i6 + 1].min() < L5:
                                continue

                            res_slope = (H6 - H2) / (i6 - i2)

                            def res_at(i, H2=H2, i2=i2, res_slope=res_slope):
                                return H2 + res_slope * (i - i2)

                            # i4 должна быть ниже линии сопротивления в своей позиции
                            if H4 > res_at(i4) * (1 + p.line_tol_pct / 100.0):
                                continue

                            # Инвалидация: floor не пробит между i1 и i6
                            invalid = False
                            for j in range(i1, i6 + 1):
                                if c[j] < support * (1 - p.line_tol_pct / 100.0):
                                    if j + 1 <= i6 and c[j + 1] >= support * (1 - p.line_tol_pct / 100.0):
                                        continue  # вернулась — ок
                                    invalid = True
                                    break
                            if invalid:
                                continue

                            max_scan = min(n_bars, i6 + p.max_confirm + 1)
                            i_br = None
                            for j in range(i6 + 1, max_scan):
                                if c[j] < support:
                                    break  # пробой поддержки вниз — паттерн отменён
                                if c[j] > res_at(j):
                                    i_br = j
                                    break
                            if i_br is None:
                                continue

                            ceiling = max(H2, H4, H6)
                            height = ceiling - support
                            if height <= 0:
                                continue
                            TG = ceiling + height

                            out.append({
                                'i1': i1, 'i2': i2, 'i3': i3, 'i4': i4, 'i5': i5, 'i6': i6,
                                'i_breakout': i_br,
                                'L1': float(L1), 'H2': float(H2), 'L3': float(L3),
                                'H4': float(H4), 'L5': float(L5), 'H6': float(H6),
                                'breakout_close': float(c[i_br]), 'TG': float(TG),
                                'width_bars': int(i6 - i1),
                            })
    return out


def detect_triple_bottom_geometry(h: np.ndarray, l: np.ndarray, c: np.ndarray, ts: np.ndarray,
                                   params: Optional[TripleBottomParams] = None) -> tuple[list[dict], dict]:
    if params is None:
        params = TripleBottomParams()

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
    # Внешний цикл по FL (впадинам) — зеркало Triple Top (FH)
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

    for p in patterns:
        p['ts_i6'] = int(ts[p['i6']])
        p['ts_breakout'] = int(ts[p['i_breakout']])

    seen: set[int] = set(); uniq: list[dict] = []
    for p in sorted(patterns, key=lambda p: p['ts_breakout']):
        if p['ts_breakout'] not in seen:
            seen.add(p['ts_breakout']); uniq.append(p)
    stats['passed_geometry'] = len(uniq)

    return uniq, stats


def compute_triple_bottom(df_1h: pd.DataFrame, symbol: str,
                           params: Optional[TripleBottomParams] = None) -> tuple[pd.DataFrame, dict]:
    if params is None:
        params = TripleBottomParams()

    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    geo, stats = detect_triple_bottom_geometry(h, l, c, t_arr, params)

    n_bars = len(h)
    hma_mhull, hma_shull, wma50 = load_hma_wma(symbol, t_arr)

    final = apply_fvg_ma_filter_long(geo, t_arr, c, fvg_from_key='i6', i_br_key='i_breakout',
                                      h_arr=h, l_arr=l, n_bars=n_bars, hma_mhull=hma_mhull,
                                      hma_shull=hma_shull, wma50=wma50, stats=stats)

    rows = []
    for p in final:
        rows.append({
            "signal_ts":      p['ts_breakout'] + TF_1H_MS,
            "direction":      "long",
            "pattern":        PATTERN_NAME,
            "ts_breakout":    p['ts_breakout'],
            "ts_i6":          p['ts_i6'],
            "L1": p['L1'], "H2": p['H2'], "L3": p['L3'],
            "H4": p['H4'], "L5": p['L5'], "H6": p['H6'],
            "breakout_close": p['breakout_close'],
            "target":         p['TG'],
            "width_bars":     p['width_bars'],
            "status":         "CONFIRMED",
        })
    _COLS = ["signal_ts","direction","pattern","ts_breakout","ts_i6",
             "L1","H2","L3","H4","L5","H6",
             "breakout_close","target","width_bars","status"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_COLS), stats


def print_stats(stats: dict) -> None:
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry   n={stats['passed_geometry']}", file=sys.stderr, flush=True)
    print(f"    rej_fvg (2)              {stats['rej_fvg']}", file=sys.stderr, flush=True)
    print(f"    rej_hma_wma (2a/2c/2d)   {stats['rej_hma_wma']}", file=sys.stderr, flush=True)
    print(f"  passed_canon (full)   n={stats['passed_canon']}  ← TRIPLE_BOTTOM", file=sys.stderr, flush=True)


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
    hits, stats = compute_triple_bottom(df_1h, args.symbol)
    print_stats(stats)

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms   = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  in window: {len(hits):,} signals", file=sys.stderr, flush=True)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"{PATTERN_NAME}_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
