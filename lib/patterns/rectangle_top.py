"""Rectangle Top (BEARISH, SHORT) — Bulkowski Ch.52 "Rectangle Tops". Тонкая
обёртка над five_point_core.py, см. канон-верифицированный
G:\\Claude\\patterns\\rectangle_top\\scan_rectangle_top.py (103 уникальных
паттерна, BTCUSDT 1h 2020-01 → 2026-07).

Геометрия: ОБЕ линии (опора i1,i3,i5 и сопротивление i2,i4) почти
горизонтальны (|slope|<=0.05%/бар каждая) — в отличие от Descending Triangle,
где горизонтальна только опора. Касания упрощены до строгого чередования
1-2-3-4-5 (книга допускает не-чередующиеся касания — задокументированное
упрощение). TG (Table 52.10): height=res_at(BR)-sup_at(BR) (обе трендлинии
на момент пробоя, не точки-цены), TG=sup_at(BR)-height.

Reads/Writes: см. block5_common.py / five_point_core.py.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import load_1m, agg_1h, DATA_OUT
from five_point_core import compute_five_point, print_stats

PATTERN_NAME = "rectangle_top"


def compute_rectangle_top(df_1h, symbol: str):
    return compute_five_point('rectangle_top', PATTERN_NAME, df_1h, symbol)


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
    hits, stats = compute_rectangle_top(df_1h, args.symbol)
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
