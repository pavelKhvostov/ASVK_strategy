"""Basket — OR-union B1 ∪ B2 ∪ B3 ∪ B4 ∪ B5 ∪ B8 ∪ B9 (ASVK-portable).

Портировано из ~/smc-warehouse/scripts/фрактал-12h/basket.py (WSL, read-only источник),
BLOCKS = ["b1", "b2", "b3", "b4", "b5", "b8", "b9"] — B2 = только B2C1 (B2C2 теперь
считается отдельно в b2_ob.py, но в b2_hit/basket не входит, см. b2_ob.py докстринг),
B4 = B4C1..B4C6 (развёрнуто 2026-07-25), B8 = только B8C1.

Reads:
  G:\\ASVK\\data\\fractal12h\\a_candidates_{SYM}_{start}_{end}.parquet
  G:\\ASVK\\data\\fractal12h\\b{1,2,3,4,5,8,9}_hits_{SYM}_{start}_{end}.parquet

Writes:
  G:\\ASVK\\data\\fractal12h\\basket_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import DATA_OUT


BLOCKS = ["b1", "b2", "b3", "b4", "b5", "b8", "b9"]
HIT_COLS = [f"{b}_hit" for b in BLOCKS]


def load_block_hits(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Загружает hits каждого B-блока, merge по (pivot_ts, direction)."""
    dfs = []
    for block in BLOCKS:
        path = DATA_OUT / f"{block}_hits_{symbol}_{start}_{end}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_parquet(path)
        # держим и сам _hit, и все под-условия (bXcY) — нужны для отображения в TUI,
        # какое именно условие сработало на конкретный сигнал (не только блок целиком)
        cols_keep = ["pivot_open_ts_ms", "direction", "confirmable", "confirmed"] + \
                    [c for c in df.columns if c.startswith(block)]
        dfs.append(df[cols_keep])

    base = dfs[0]
    for df in dfs[1:]:
        base = base.merge(df.drop(columns=["confirmable", "confirmed"]),
                          on=["pivot_open_ts_ms", "direction"], how="outer")
    for col in HIT_COLS:
        base[col] = base[col].fillna(False)
    return base


def compute_basket(hits: pd.DataFrame) -> pd.DataFrame:
    """basket_hit = OR по всем B*_hit колонкам."""
    hits["basket_hit"] = hits[HIT_COLS].any(axis=1)
    return hits


def print_stats(hits: pd.DataFrame) -> None:
    print(f"\n=== Per-block recap ===", file=sys.stderr, flush=True)
    for col in HIT_COLS:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        print(f"  {col:12s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%",
              file=sys.stderr, flush=True)

    print(f"\n=== BASKET (B1∪B2∪B3∪B4∪B5∪B8∪B9) ===", file=sys.stderr, flush=True)
    m = hits["basket_hit"]
    n = int(m.sum())
    cm = m & hits["confirmable"]
    n_c = int(cm.sum())
    n_conf = int(hits.loc[cm, "confirmed"].sum())
    wr = 100.0 * n_conf / n_c if n_c else 0.0
    base_m = hits["confirmable"]
    base_wr = 100.0 * int(hits.loc[base_m, "confirmed"].sum()) / int(base_m.sum())
    print(f"  basket        n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%  Δ={wr-base_wr:+5.2f}pp",
          file=sys.stderr, flush=True)
    print(f"  A1 baseline   n={int(base_m.sum()):>4,}  WR={base_wr:5.2f}%",
          file=sys.stderr, flush=True)
    print(f"  selectivity   {n}/{int(base_m.sum())} = {100*n/int(base_m.sum()):.1f}%",
          file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-08")
    args = ap.parse_args()

    print(f"basket assembly (B1∪B2∪B3∪B4∪B5∪B8∪B9): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    hits = load_block_hits(args.symbol, args.start, args.end)
    print(f"  merged {len(hits):,} rows (all blocks)", file=sys.stderr, flush=True)

    hits = compute_basket(hits)
    print_stats(hits)

    out = DATA_OUT / f"basket_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"\nwritten: {out.name}  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
