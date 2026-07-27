"""B3 Fractal Liquidity — sub-basket B3C1 (maxV sweep i-1).

Портировано из ~/smc-warehouse/scripts/фрактал-12h/b3_fractal_liquidity.py (WSL,
read-only источник). Пока одно sub-условие (канон: B3C2..B3C6 запланированы, но
не реализованы — как и в WSL-источнике).

Canonical semantic:
  SHORT (FH): h12[i] > maxV(i-1) AND c12[i] < maxV(i-1)  (pierce up + close down)
  LONG  (FL): l12[i] < maxV(i-1) AND c12[i] > maxV(i-1)  (pierce down + close up)

БЕЗ depth-фильтра (в отличие от B9C2 = тот же sweep + depth/ATR14 >= 0.7).
Pool: A1+A2+A4, без A3 (a_cand[a124_pool], см. a_cascade.py) — по явной инструкции
пользователя (A3 исключён из домена). До 2026-07-24 здесь ошибочно стоял
a4_body_wick — кумулятивный A1+A2+A3+A4, тащивший A3 внутрь домена вопреки команде.

maxV — level-1 shared indicator, см. G:\\ASVK\\lib\\maxv.py (MAXV_LTF_MIN=90,
walk-forward validated на BTC/ETH/SOL 2026-07-23).

Reads:
  G:\\ASVK\\data\\fractal12h\\a_candidates_{SYM}_{start}_{end}.parquet
  G:\\ASVK\\data\\maxv\\maxv_{SYM}_*.parquet (последний по дате)
Writes:
  G:\\ASVK\\data\\fractal12h\\b3_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — для maxv (level-1)
from common import load_1m, agg_12h, DATA_OUT
from maxv import latest_maxv_path


def load_maxv(symbol: str, t12: np.ndarray) -> np.ndarray:
    mv = pd.read_parquet(latest_maxv_path(symbol), columns=["ts", "maxv"])
    s = pd.Series(mv["maxv"].to_numpy(), index=mv["ts"].to_numpy())
    return s.reindex(t12).to_numpy()


def compute_b3(a_cand: pd.DataFrame, df_12h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    t12 = df_12h["ts"].to_numpy()
    h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy()
    c12 = df_12h["close"].to_numpy()
    n12 = len(t12)

    maxv = load_maxv(symbol, t12)

    sw_short = np.zeros(n12, dtype=bool)
    sw_long = np.zeros(n12, dtype=bool)
    mv_prev = np.roll(maxv, 1); mv_prev[0] = np.nan
    valid = ~np.isnan(mv_prev)
    sw_short[valid] = (h12[valid] > mv_prev[valid]) & (c12[valid] < mv_prev[valid])
    sw_long[valid] = (l12[valid] < mv_prev[valid]) & (c12[valid] > mv_prev[valid])

    ts_to_idx = {int(t): k for k, t in enumerate(t12)}
    pool = a_cand[a_cand["a124_pool"]].copy()
    rows = []
    for _, row in pool.iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        idx = ts_to_idx.get(ts_pivot)
        if idx is None:
            continue
        hit = bool(sw_short[idx]) if direction == "short" else bool(sw_long[idx])
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction": direction,
            "confirmable": bool(row["confirmable"]),
            "confirmed": bool(row["confirmed"]),
            "b3c1": hit,
            "b3_hit": hit,   # пока B3 = B3C1 (единственный sub)
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b3c1", "b3_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        marker = "  ← B3" if col == "b3_hit" else ""
        print(f"  {col:8s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{marker}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-08")
    args = ap.parse_args()

    print(f"b3_fractal_liquidity (ASVK-portable): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1+A2+A4 candidates domain: {int(a_cand['a124_pool'].sum()):,}",
          file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_b3(a_cand, df_12h, args.symbol)
    print_stats(hits)

    out = DATA_OUT / f"b3_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
