"""b6_divergence — B6 RSI-дивергенция на 12h (ASVK-portable).

Реализует блок B6, который в каноне ASVK был зарезервирован, но не написан
(див. structure_BTC.png: «B6 RSI — planned»). Материал: divergence.pdf Арденского
(осцилляторы RSI/Stochastic/MACD/Cumulative Delta; применение: намёк на завершение
тренда, подтверждение сигнала, подтверждение ложного выноса).

Регулярная дивергенция = цена делает новый экстремум, а осциллятор — НЕ делает
(импульс ослаб на выносе). Считается причинно на баре пивота i: RSI(14, Wilder) по
закрытиям 12h до i включительно, сравнение с предыдущим одноимённым фракталом.

Sub-условия (short = пивот FH, long = пивот FL):
  b6c1  regular divergence:
        SHORT h[i] > prev_FH.high  AND  rsi[i] < rsi[prev_FH]   (выше по цене, ниже по RSI)
        LONG  l[i] < prev_FL.low   AND  rsi[i] > rsi[prev_FL]
  b6c2  экстремум осциллятора на пивоте:
        SHORT rsi[i] >= 70   LONG rsi[i] <= 30
  b6_hit = b6c1   — production-подтверждение (дивергенция, как в методичке).

Причинность: ✅ RSI по данным до i, prev_FH/prev_FL строго до i.
Pool: A1 (все пивоты из a_candidates), как B1/B2/B9.

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/{SYM}USDT_1m.csv
Writes:
  data/fractal12h/b6_hits_{SYM}_{start}_{end}.parquet
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
import a1_filter


RSI_N = 14
RSI_OB = 70.0
RSI_OS = 30.0


def wilder_rsi(close: np.ndarray, n: int = RSI_N) -> np.ndarray:
    """RSI Уайлдера (причинно, NaN на первых n барах)."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    T = len(close)
    avg_g = np.full(T, np.nan)
    avg_l = np.full(T, np.nan)
    if T <= n:
        return np.full(T, np.nan)
    avg_g[n] = gain[1:n + 1].mean()
    avg_l[n] = loss[1:n + 1].mean()
    for i in range(n + 1, T):
        avg_g[i] = (avg_g[i - 1] * (n - 1) + gain[i]) / n
        avg_l[i] = (avg_l[i - 1] * (n - 1) + loss[i]) / n
    rs = np.where(avg_l > 0, avg_g / avg_l, np.inf)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _prev_swing_idx(is_swing: np.ndarray) -> np.ndarray:
    """Для каждого бара i → индекс последнего swing СТРОГО до i (-1 если нет)."""
    T = len(is_swing)
    out = np.full(T, -1, dtype=np.int64)
    last = -1
    for i in range(T):
        out[i] = last
        if is_swing[i]:
            last = i
    return out


def compute_b6(a_cand: pd.DataFrame, df_12h: pd.DataFrame) -> pd.DataFrame:
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    c = df_12h["close"].to_numpy()
    ts = df_12h["ts"].to_numpy()

    rsi = wilder_rsi(c, RSI_N)
    a1 = a1_filter.compute_a1(df_12h)
    prev_fh = _prev_swing_idx(a1["short"])
    prev_fl = _prev_swing_idx(a1["long"])
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}

    rows = []
    for _, row in a_cand[a_cand["a1_pre_w"]].iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        i = ts_to_idx.get(ts_pivot)
        if i is None:
            continue
        r_i = rsi[i]
        if direction == "short":
            j = prev_fh[i]
            div = bool(j >= 0 and np.isfinite(r_i) and np.isfinite(rsi[j])
                       and h[i] > h[j] and r_i < rsi[j])
            ext = bool(np.isfinite(r_i) and r_i >= RSI_OB)
        else:
            j = prev_fl[i]
            div = bool(j >= 0 and np.isfinite(r_i) and np.isfinite(rsi[j])
                       and l[i] < l[j] and r_i > rsi[j])
            ext = bool(np.isfinite(r_i) and r_i <= RSI_OS)
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "b6c1":   div,
            "b6c2":   ext,
            "b6_hit": div,   # production = регулярная дивергенция
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b6c1", "b6c2", "b6_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        mark = "  ← B6 (production)" if col == "b6_hit" else ""
        print(f"  {col:8s} n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{mark}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"b6_divergence (RSI-14, A1 pool): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {int(a_cand['a1_pre_w'].sum()):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_b6(a_cand, df_12h)
    print_stats(hits)

    out = DATA_OUT / f"b6_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
