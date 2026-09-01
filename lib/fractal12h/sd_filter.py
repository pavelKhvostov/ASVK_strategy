"""sd_filter — режим Standard Deviation (сила движения) на 12h (ASVK-portable).

Материал: sistema.pdf Арденского — система №3 требует SD ≥ 1 (импульс сильный, торгуем
по движению), система №4 работает при SD ~0.5 (вялый рынок, контр-режим). Это фильтр
РЕЖИМА, не подтверждение направления: он говорит, сильное ли движение сформировало пивот.

SD здесь = размах бара пивота в единицах ATR (насколько импульсным был бар относительно
недавней волатильности):
    sd = (high[i] - low[i]) / ATR14[i]        — причинно (ATR по барам до i)

Sub-условия:
  sd_ge1    sd >= 1.0     — сильный импульсный бар (режим тренда, система №3)
  sd_mid    0.5 <= sd < 1 — умеренный (система №4)
  sd_hit  = sd_ge1        — production-режим «сильное движение».

Причинность: ✅ ATR по True Range баров до i, размах внутри i.
Pool: A1 (все пивоты).

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/{SYM}USDT_1m.csv
Writes:
  data/fractal12h/sd_hits_{SYM}_{start}_{end}.parquet
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


ATR_N = 14
SD_STRONG = 1.0
SD_MID = 0.5


def wilder_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = ATR_N) -> np.ndarray:
    """ATR Уайлдера, причинно (NaN на первых n барах). ATR[i] использует TR до i включительно."""
    T = len(h)
    prev_c = np.empty(T)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = np.full(T, np.nan)
    if T <= n:
        return atr
    atr[n] = tr[1:n + 1].mean()
    for i in range(n + 1, T):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def compute_sd(a_cand: pd.DataFrame, df_12h: pd.DataFrame) -> pd.DataFrame:
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    c = df_12h["close"].to_numpy()
    ts = df_12h["ts"].to_numpy()

    atr = wilder_atr(h, l, c, ATR_N)
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}

    rows = []
    for _, row in a_cand[a_cand["a1_pre_w"]].iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        i = ts_to_idx.get(ts_pivot)
        if i is None:
            continue
        sd = (h[i] - l[i]) / atr[i] if (np.isfinite(atr[i]) and atr[i] > 0) else 0.0
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        row["direction"],
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "sd_value": float(sd),
            "sd_ge1":  bool(sd >= SD_STRONG),
            "sd_mid":  bool(SD_MID <= sd < SD_STRONG),
            "sd_hit":  bool(sd >= SD_STRONG),
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["sd_ge1", "sd_mid", "sd_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        mark = "  ← SD (production)" if col == "sd_hit" else ""
        print(f"  {col:8s} n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{mark}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"sd_filter (range/ATR{ATR_N}): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {int(a_cand['a1_pre_w'].sum()):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_sd(a_cand, df_12h)
    print_stats(hits)

    out = DATA_OUT / f"sd_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
