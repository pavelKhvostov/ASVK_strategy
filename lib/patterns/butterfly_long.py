"""Butterfly BULLISH (LONG) — зеркало butterfly.py.
X=FL, A=FH, B=FL, C=FH, D=FL (PRZ = дно, вход LONG).
AB/XA~0.786, D/XA~1.272, D НИЖЕ X (экстеншн).
Коэффициенты идентичны SHORT Butterfly.
"""
from __future__ import annotations
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import load_1m, agg_1h, DATA_OUT
from harmonic_core_long import HarmonicLongParams, compute_harmonic_long, print_stats

PATTERN_NAME = "butterfly_long"

PARAMS = HarmonicLongParams(
    ab_lo=0.730, ab_hi=0.840,
    bc_lo=0.382, bc_hi=0.886,
    cd_lo=1.618, cd_hi=2.618,
    d_xa_lo=1.150, d_xa_hi=1.400,
    d_beyond_x=True,
)


def compute_butterfly_long(df_1h, symbol: str):
    return compute_harmonic_long(PATTERN_NAME, PARAMS, df_1h, symbol)


def main() -> None:
    from datetime import datetime, timezone
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-26")
    args = ap.parse_args()
    t0 = time.time()
    df_1h = agg_1h(load_1m(args.symbol))
    hits, stats = compute_butterfly_long(df_1h, args.symbol)
    print_stats(stats, PATTERN_NAME.upper())
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
