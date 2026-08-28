"""b7_money_hands — B7 «умные деньги»: климактический объём + поглощение (ASVK-portable).

Реализует блок B7, зарезервированный в каноне ASVK, но не написанный (structure_BTC.png:
«B7 Money Hands — planned»). Материал: neeffektivnosti.pdf / vse-o-likvidnosti.pdf —
крупный игрок оставляет след аномальным объёмом на развороте (climax + absorption:
большой объём при отбое = поглощение противоположной стороны).

Причинно на баре пивота i (12h):
    vol_ratio = volume[i] / mean(volume[i-N .. i-1])     — объём относительно СРЕДНЕГО ДО i
    climax    = vol_ratio >= K_CLIMAX
    reject    = закрытие в сторону разворота (поглощение):
                SHORT close[i] <  (high[i]+low[i])/2      (закрылись в нижней половине)
                LONG  close[i] >  (high[i]+low[i])/2      (в верхней половине)

Sub-условия:
  b7c1  money hands   = climax AND reject
  b7c2  strong climax = vol_ratio >= 2*K_CLIMAX (без требования reject — экстремальный объём)
  b7_hit = b7c1   — production (след умных денег = объём + поглощение).

Причинность: ✅ средний объём по барам ДО i, всё остальное внутри i.
Pool: A1 (все пивоты).

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/{SYM}USDT_1m.csv
Writes:
  data/fractal12h/b7_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_1m, agg_12h, DATA_OUT


VOL_N = 20            # окно среднего объёма: 20 × 12h = 10 дней
K_CLIMAX = 1.5        # объём >= 1.5× среднего = климакс


def compute_b7(a_cand: pd.DataFrame, df_12h: pd.DataFrame) -> pd.DataFrame:
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    c = df_12h["close"].to_numpy()
    v = df_12h["volume"].to_numpy()
    ts = df_12h["ts"].to_numpy()
    T = len(ts)

    # причинный средний объём по [i-N .. i-1]
    avg_v = np.full(T, np.nan)
    for i in range(1, T):
        a = max(0, i - VOL_N)
        if i > a:
            avg_v[i] = v[a:i].mean()
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}

    rows = []
    for _, row in a_cand[a_cand["a1_pre_w"]].iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        i = ts_to_idx.get(ts_pivot)
        if i is None:
            continue
        ratio = v[i] / avg_v[i] if (np.isfinite(avg_v[i]) and avg_v[i] > 0) else 0.0
        climax = bool(ratio >= K_CLIMAX)
        strong = bool(ratio >= 2 * K_CLIMAX)
        mid = (h[i] + l[i]) / 2.0
        reject = bool(c[i] < mid) if direction == "short" else bool(c[i] > mid)
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "b7_vol_ratio": float(ratio),
            "b7c1":   bool(climax and reject),
            "b7c2":   strong,
            "b7_hit": bool(climax and reject),
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b7c1", "b7c2", "b7_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        mark = "  ← B7 (production)" if col == "b7_hit" else ""
        print(f"  {col:8s} n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{mark}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"b7_money_hands (climax vol N={VOL_N} K={K_CLIMAX}): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {int(a_cand['a1_pre_w'].sum()):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_b7(a_cand, df_12h)
    print_stats(hits)

    out = DATA_OUT / f"b7_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
