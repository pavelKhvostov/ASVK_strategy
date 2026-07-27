"""Bat (BEARISH harmonic, SHORT) — X-A-B-C-D, D остаётся МЕЖДУ A и X
(ретрейс, глубже Gartley). Тонкая обёртка над harmonic_core.py, см.
канон-верифицированный G:\\Claude\\patterns\\bat\\scan_bat.py (77 уникальных
паттернов, BTCUSDT 1h 2020-01 → 2026-07, "полный эксперимент" 2026-07-25).

Коэффициенты: AB/XA=0.382-0.50, D/XA=0.85-0.92 (цель 0.886).

Вход в точке D, без FVG/BR, без MA-фильтра (см. harmonic_core.py docstring).
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

PATTERN_NAME = "bat"

PARAMS = HarmonicParams(
    ab_lo=0.382, ab_hi=0.600,
    bc_lo=0.382, bc_hi=0.920,
    cd_lo=1.500, cd_hi=2.750,
    d_xa_lo=0.840, d_xa_hi=0.930,
    d_beyond_x=False,
)


def compute_bat(df_1h, symbol: str):
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
    hits, stats = compute_bat(df_1h, args.symbol)
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
