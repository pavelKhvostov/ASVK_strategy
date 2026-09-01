"""b_ote — Premium/Discount + OTE-позиция в дилинг-рейндже на 12h (ASVK-portable).

Материал: premium-discount-markets.pdf Арденского. Правило направления: шорт брать
только из Premium (верхняя половина диапазона, дорого), лонг — только из Discount
(нижняя половина, дёшево); зона входа OTE = 0.62–0.79 коррекции. Это НАПРАВЛЕННЫЙ
фильтр, не подтверждение: он отбраковывает сигналы с «неправильной» стороны диапазона.

Дилинг-рейндж строится причинно как rolling max/min последних RANGE_N закрытых 12h
баров (по умолчанию 60 = 30 дней). Позиция пивота:
    pos = (close[i] - range_lo) / (range_hi - range_lo)   ∈ [0,1]

Sub-условия (short = пивот FH, long = пивот FL):
  ote_side : SHORT pos >= 0.5   LONG pos <= 0.5           — правильная половина (Premium/Discount)
  ote_deep : SHORT pos >= 0.705 LONG pos <= 0.295         — глубоко в OTE-зоне
  ote_hit  = ote_side   — production-фильтр направления.

Причинность: ✅ диапазон по барам до i включительно, pos по close[i].
Pool: A1 (все пивоты).

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/{SYM}USDT_1m.csv
Writes:
  data/fractal12h/ote_hits_{SYM}_{start}_{end}.parquet
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


RANGE_N = 60          # дилинг-рейндж: 60 × 12h = 30 дней
SIDE_THR = 0.5        # Premium/Discount раздел
DEEP_HI = 0.705       # OTE верх (short)
DEEP_LO = 0.295       # OTE низ (long)


def _rolling_range(h: np.ndarray, l: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """range_hi[i] = max(high[i-n+1..i]), range_lo[i] = min(low[i-n+1..i]) — причинно."""
    T = len(h)
    rhi = np.full(T, np.nan)
    rlo = np.full(T, np.nan)
    for i in range(T):
        a = max(0, i - n + 1)
        rhi[i] = h[a:i + 1].max()
        rlo[i] = l[a:i + 1].min()
    return rhi, rlo


def compute_ote(a_cand: pd.DataFrame, df_12h: pd.DataFrame) -> pd.DataFrame:
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    c = df_12h["close"].to_numpy()
    ts = df_12h["ts"].to_numpy()

    rhi, rlo = _rolling_range(h, l, RANGE_N)
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}

    rows = []
    for _, row in a_cand[a_cand["a1_pre_w"]].iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        i = ts_to_idx.get(ts_pivot)
        if i is None:
            continue
        rng = rhi[i] - rlo[i]
        pos = (c[i] - rlo[i]) / rng if rng > 0 else 0.5
        if direction == "short":
            side = bool(pos >= SIDE_THR)
            deep = bool(pos >= DEEP_HI)
        else:
            side = bool(pos <= SIDE_THR)
            deep = bool(pos <= DEEP_LO)
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "ote_pos":  float(pos),
            "ote_side": side,
            "ote_deep": deep,
            "ote_hit":  side,   # production-фильтр направления
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["ote_side", "ote_deep", "ote_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        mark = "  ← OTE (production)" if col == "ote_hit" else ""
        print(f"  {col:8s} n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{mark}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"b_ote (Premium/Discount, range_N={RANGE_N}): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {int(a_cand['a1_pre_w'].sum()):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_ote(a_cand, df_12h)
    print_stats(hits)

    out = DATA_OUT / f"ote_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
