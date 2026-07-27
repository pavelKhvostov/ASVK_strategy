"""Crab (BEARISH harmonic, SHORT) — X-A-B-C-D, D ВЫХОДИТ ЗА X (самый глубокий
экстеншн из всех 4 паттернов). Тонкая обёртка над harmonic_core.py.

Carney canonical: AB/XA=0.382-0.618, D/XA≈1.618 (первичный идентификатор).
Relaxed: ab_hi=0.680, cd_hi=4.000, dxa=[1.500,1.750].
CD/BC верхняя граница 3.618 (canonical Carney) математически отсекает 84%
честных крабов при малом BC — поэтому расширена до 4.000. D/XA=[1.50,1.75]
является основным фильтром.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import load_1m, agg_1h, DATA_OUT
from harmonic_core import HarmonicParams, compute_harmonic, print_stats

PATTERN_NAME = "crab"

PARAMS = HarmonicParams(
    ab_lo=0.350, ab_hi=0.680,
    bc_lo=0.300, bc_hi=0.886,
    cd_lo=2.000, cd_hi=4.000,
    d_xa_lo=1.500, d_xa_hi=1.750,
    d_beyond_x=True,
)


def compute_crab(df_1h, symbol: str):
    return compute_harmonic(PATTERN_NAME, PARAMS, df_1h, symbol)


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
    hits, stats = compute_crab(df_1h, args.symbol)
    print_stats(stats, PATTERN_NAME.upper())

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
