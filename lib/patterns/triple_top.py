"""Triple Top (SHORT) — Bulkowski Ch.68 "Triple Tops". Геометрия портирована
verbatim из канон-верифицированного G:\\Claude\\patterns\\triple_top\\
scan_triple_top.py (460 уникальных паттернов, BTCUSDT 1h 2020-01 → 2026-07)
на ASVK-portable контракт (df_1h передаётся вызывающим кодом, FVG/MA —
block5_common.apply_fvg_ma_filter).

Геометрия (i1<i2<i3<i4<i5<i6, по cheat-sheet tradingaxe.com — 6 точек, не 5):
  i1,i3,i5 = Williams N=2 FH (вершины, ~одна цена, допуск SAME_PRICE_TOL_PCT)
  i2,i4,i6 = Williams N=2 FL (впадины, ~одна цена, линия может быть слегка
    наклонной через i2->i6)
  Граница с H&S (важные_условия.md п.1b, СТРОГО без допуска):
    NOT(H3 > H1 and H3 > H5)  — иначе это уже H&S TOP, не Triple Top.
  Инвалидация — per-leg true extremum на каждом из 5 отрезков (i1..i2,
  i2..i3, i3..i4, i4..i5, i5..i6), как в Double Top, + потолок max(H1,H3,H5)
  не пробит между i1 и i6.
  BR = первое закрытие ниже линии впадин (i2->i6, экстраполирована) после i6.
  MAX_SPAN i1->i6 = 96ч, MAX_CONFIRM i6->BR = 24ч.

TG (Bulkowski Ch.68, Table 68.10, сверено по PDF: "subtract the lowest low
from the highest high... subtract the height from the lowest low"):
  height = max(H1,H3,H5) - min(L2,L4,L6)
  TG = min(L2,L4,L6) - height

FVG(2)/MA(2a,2c,2d) — общий SHORT-фильтр block5_common.apply_fvg_ma_filter,
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
from block5_common import fractals, load_hma_wma, apply_fvg_ma_filter

PATTERN_NAME = "triple_top"


@dataclass
class TripleTopParams:
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
                    p: TripleTopParams) -> list[dict]:
    out: list[dict] = []
    TOL = p.same_price_tol_pct

    for i1 in i1_chunk:
        i1 = int(i1)
        H1 = h[i1]

        i2_cands = [i for i in fl_list if i1 < i <= i1 + p.max_span]
        if not i2_cands:
            continue

        for i2 in i2_cands:
            L2 = l[i2]
            if h[i1:i2 + 1].max() > H1:
                continue
            if l[i1:i2 + 1].min() < L2:
                continue

            i3_cands = [i for i in fh_list
                        if i2 < i <= i1 + p.max_span
                        and abs(h[i] - H1) * 100 / H1 <= TOL]
            if not i3_cands:
                continue

            for i3 in i3_cands:
                H3 = h[i3]
                if h[i2:i3 + 1].max() > H3:
                    continue
                if l[i2:i3 + 1].min() < L2:
                    continue

                i4_cands = [i for i in fl_list
                            if i3 < i <= i1 + p.max_span
                            and l[i] >= L2 * (1 - TOL / 100.0)]
                # i4 между уровнями 2,6 и 1,3,5 (выше NL)
                if not i4_cands:
                    continue

                for i4 in i4_cands:
                    L4 = l[i4]
                    if l[i3:i4 + 1].min() < L4:
                        continue
                    if h[i3:i4 + 1].max() > H3:
                        continue

                    i5_cands = [i for i in fh_list
                                if i4 < i <= i1 + p.max_span
                                and abs(h[i] - H1) * 100 / H1 <= TOL
                                and abs(h[i] - H3) * 100 / H3 <= TOL]
                    if not i5_cands:
                        continue

                    for i5 in i5_cands:
                        H5 = h[i5]
                        if H3 > H1 and H3 > H5:
                            continue  # это уже H&S, не Triple Top
                        if h[i4:i5 + 1].max() > H5:
                            continue
                        if l[i4:i5 + 1].min() < L4:
                            continue

                        i6_cands = [i for i in fl_list
                                    if i5 < i <= i1 + p.max_span
                                    and abs(l[i] - L2) * 100 / L2 <= TOL]
                        # i6 ≈ L2: нклайн через точки 2 и 6. Точка 4 не участвует.
                        if not i6_cands:
                            continue

                        ceiling = max(H1, H3, H5)

                        for i6 in i6_cands:
                            L6 = l[i6]
                            if l[i5:i6 + 1].min() < L6:
                                continue
                            if h[i5:i6 + 1].max() > H5:
                                continue

                            sup_slope = (L6 - L2) / (i6 - i2)

                            def sup_at(i, L2=L2, i2=i2, sup_slope=sup_slope):
                                return L2 + sup_slope * (i - i2)

                            # i4 должна быть выше neckline в своей позиции
                            if L4 < sup_at(i4) * (1 - p.line_tol_pct / 100.0):
                                continue

                            invalid = False
                            for j in range(i1, i6 + 1):
                                if c[j] > ceiling * (1 + p.line_tol_pct / 100.0):
                                    # Одна свеча может пробить — проверяем возврат на следующей
                                    if j + 1 <= i6 and c[j + 1] <= ceiling * (1 + p.line_tol_pct / 100.0):
                                        continue  # вернулась — ок
                                    invalid = True
                                    break
                            if invalid:
                                continue

                            max_scan = min(n_bars, i6 + p.max_confirm + 1)
                            i_br = None
                            for j in range(i6 + 1, max_scan):
                                if c[j] > ceiling:
                                    break
                                if c[j] < sup_at(j):
                                    i_br = j
                                    break
                            if i_br is None:
                                continue

                            base = min(L2, L4, L6)
                            height = ceiling - base
                            if height <= 0:
                                continue
                            TG = base - height

                            out.append({
                                'i1': i1, 'i2': i2, 'i3': i3, 'i4': i4, 'i5': i5, 'i6': i6,
                                'i_breakout': i_br,
                                'H1': float(H1), 'L2': float(L2), 'H3': float(H3),
                                'L4': float(L4), 'H5': float(H5), 'L6': float(L6),
                                'breakout_close': float(c[i_br]), 'TG': float(TG),
                                'width_bars': int(i6 - i1),
                            })
    return out


def detect_triple_top_geometry(h: np.ndarray, l: np.ndarray, c: np.ndarray, ts: np.ndarray,
                                params: Optional[TripleTopParams] = None) -> tuple[list[dict], dict]:
    if params is None:
        params = TripleTopParams()

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
    chunks = np.array_split(FH, n_workers)
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


def compute_triple_top(df_1h: pd.DataFrame, symbol: str,
                        params: Optional[TripleTopParams] = None) -> tuple[pd.DataFrame, dict]:
    if params is None:
        params = TripleTopParams()

    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    geo, stats = detect_triple_top_geometry(h, l, c, t_arr, params)

    n_bars = len(h)
    hma_mhull, hma_shull, wma50 = load_hma_wma(symbol, t_arr)

    final = apply_fvg_ma_filter(geo, t_arr, c, fvg_from_key='i6', i_br_key='i_breakout',
                                 h_arr=h, l_arr=l, n_bars=n_bars, hma_mhull=hma_mhull,
                                 hma_shull=hma_shull, wma50=wma50, stats=stats)

    rows = []
    for p in final:
        rows.append({
            "signal_ts": p['ts_breakout'] + TF_1H_MS,
            "direction": "short",
            "pattern": PATTERN_NAME,
            "ts_breakout": p['ts_breakout'], "ts_i6": p['ts_i6'],
            "H1": p['H1'], "L2": p['L2'], "H3": p['H3'],
            "L4": p['L4'], "H5": p['H5'], "L6": p['L6'],
            "breakout_close": p['breakout_close'], "target": p['TG'],
            "width_bars": p['width_bars'],
            "status": "CONFIRMED",
        })
    return pd.DataFrame(rows), stats


def print_stats(stats: dict) -> None:
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry   n={stats['passed_geometry']}", file=sys.stderr, flush=True)
    print(f"    rej_fvg (2)              {stats['rej_fvg']}", file=sys.stderr, flush=True)
    print(f"    rej_hma_wma (2a/2c/2d)   {stats['rej_hma_wma']}", file=sys.stderr, flush=True)
    print(f"  passed_canon (full)   n={stats['passed_canon']}  ← TRIPLE_TOP", file=sys.stderr, flush=True)


def main() -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-23")
    args = ap.parse_args()

    print(f"{PATTERN_NAME}: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    df_1h = agg_1h(load_1m(args.symbol))
    hits, stats = compute_triple_top(df_1h, args.symbol)
    print_stats(stats)

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  in window: {len(hits):,} signals", file=sys.stderr, flush=True)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"{PATTERN_NAME}_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
