"""b_structure — B-структура: BOS / CHoCH / sweep структуры на 12h (ASVK-portable).

Закрывает главный пробел ASVK: в проекте нет ни одного детектора рыночной структуры,
хотя «слом структуры» — ядро метода Арденского (structure-analysis.pdf: swing HH/HL vs
LH/LL, BOS = продолжение, CHoCH = смена характера, sweep = ложный вынос уровня).

Работает поверх тех же A1-пивотов (фракталы Уильямса-2), что и B1..B9, и в той же
булевой логике «сработало на баре пивота или нет». Причинность строгая: всё считается
по данным ДО и ВКЛючая бар пивота i — никакого заглядывания вперёд (исход по-прежнему
меряется отдельным confirmed = Williams n=2).

Swing-структура строится из самих фракталов 12h (a1_filter): каждый FH — swing high,
каждый FL — swing low. Для пивота на баре i берём последний swing ДО i.

Sub-условия (short = пивот FH, long = пивот FL; для long всё зеркально):
  bstruct_sweep : SHORT h[i] > prev_swing_high   — вынос структурной ликвидности вверх
                  LONG  l[i] < prev_swing_low     — вынос вниз («ложный вынос» уровня)
  bstruct_choch : SHORT close[i] < prev_swing_low — закрытие пробило последний HL вниз (разворот)
                  LONG  close[i] > prev_swing_high— закрытие пробило последний LH вверх
  bstruct_bos   : sweep И close закрылось ОБРАТНО в диапазон (rejection):
                  SHORT h[i]>prev_SH AND close[i] < prev_SH  (снял ликвидность и вернулся)
                  LONG  l[i]<prev_SL AND close[i] > prev_SL

  bstruct_hit = bstruct_bos   — production-триггер: sweep структуры + rejection-close.
                (чистый sweep без rejection слишком широк для FH/FL; bos = «ложный вынос
                 структурного уровня с возвратом» — прямой аналог Arden PoT-триггера.)

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/{SYM}USDT_1m.csv
Writes:
  data/fractal12h/bstruct_hits_{SYM}_{start}_{end}.parquet
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


def compute_bstruct(a_cand: pd.DataFrame, df_12h: pd.DataFrame) -> pd.DataFrame:
    ts = df_12h["ts"].to_numpy()
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    c = df_12h["close"].to_numpy()

    a1 = a1_filter.compute_a1(df_12h)         # {"short": FH mask, "long": FL mask} — причинно
    prev_fh = _prev_swing_idx(a1["short"])    # последний swing-high до i
    prev_fl = _prev_swing_idx(a1["long"])     # последний swing-low до i
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}

    rows = []
    for _, row in a_cand[a_cand["a1_pre_w"]].iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        i = ts_to_idx.get(ts_pivot)
        if i is None:
            continue
        if direction == "short":
            j_sh, j_sl = prev_fh[i], prev_fl[i]
            sweep = bool(j_sh >= 0 and h[i] > h[j_sh])
            choch = bool(j_sl >= 0 and c[i] < l[j_sl])
            bos   = bool(sweep and c[i] < h[j_sh])
        else:  # long
            j_sh, j_sl = prev_fh[i], prev_fl[i]
            sweep = bool(j_sl >= 0 and l[i] < l[j_sl])
            choch = bool(j_sh >= 0 and c[i] > h[j_sh])
            bos   = bool(sweep and c[i] > l[j_sl])
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "bstruct_sweep":    sweep,
            "bstruct_choch":    choch,
            "bstruct_bos":      bos,
            "bstruct_hit":      bos,   # production-триггер
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["bstruct_sweep", "bstruct_choch", "bstruct_bos", "bstruct_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        mark = "  ← B-struct (production)" if col == "bstruct_hit" else ""
        print(f"  {col:14s} n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{mark}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"b_structure (BOS/CHoCH/sweep, A1 pool): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {int(a_cand['a1_pre_w'].sum()):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_bstruct(a_cand, df_12h)
    print_stats(hits)

    out = DATA_OUT / f"bstruct_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
