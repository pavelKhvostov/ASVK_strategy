"""vwap_anchors — D/W Williams n=2 фракталы + W-alignment, level-1 shared indicator
(ASVK-standalone).

НЕ сам VWAP — VWAP anchor-dependent (считается на лету, inline, в потребителях типа
lib/fractal12h/b5_vwap.py), а список ЯКОРЕЙ для него: Daily-фракталы, подтверждённые
совпадением с Weekly-фракталом (W-aligned). Формула verbatim из WSL
~/smc-warehouse/scripts/фрактал-12h/b5_vwap.py (detect_fractals/mark_w_aligned).

Williams n=2 fractal (симметричный, ±1/±2 бара):
    FH: high[i] > high[i-2], high[i-1], high[i+1], high[i+2]
    FL: low[i]  < low[i-2], low[i-1], low[i+1], low[i+2]

W-aligned: D-фрактал совпал с W-фракталом того же side, чей W-бар содержит момент
D-фрактала (w.ts <= d.ts < w.ts + 7d), и уровень совпадает с допуском 0.01.

ready_ms = d.ts + 3 дня (canon-слак после Williams n=2 подтверждения на D).

Depends only on:
    data/{SYMBOL}USDT_1m.csv   (сам daemon, автономно)
Writes:
    data/vwap_anchors/vwap_anchors_{SYMBOL}_{start}_{end}.parquet
    columns: ts, side (FH/FL), level, aligned_W, ready_ms

Usage:
    python vwap_anchors.py --symbol BTC --start 2018-01-01 --end 2026-07-24
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

WAREHOUSE = pathlib.Path(__file__).resolve().parent.parent  # ASVK-standalone
DATA_DIR = WAREHOUSE / "data"
VWAP_ANCHORS_DIR = DATA_DIR / "vwap_anchors"

TF_1D_MS = 24 * 60 * 60 * 1000
TF_1W_MS = 7 * 24 * 60 * 60 * 1000
MON_ANCHOR_MS = 1_483_315_200_000  # 2017-01-02 UTC (Monday) — Pine weekly convention

N_FRACTAL  = 2      # Williams n=2 (±1, ±2 checks)
LEVEL_TOL_PCT = 0.005  # W-aligned: ±0.5% price tolerance
READY_DAYS = 3       # ready_ms = d.ts + 3*TF_1D_MS


def load_1m(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}USDT_1m.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"loading {path.name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    df = pd.read_csv(path, dtype={"open": "float64", "high": "float64",
                                   "low": "float64", "close": "float64",
                                   "volume": "float64"})
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("ts").drop_duplicates("ts", keep="first").reset_index(drop=True)
    print(f"  {len(df):,} 1m bars in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return df


def agg_tf(df_1m: pd.DataFrame, tf_ms: int, anchor_ms: int = 0) -> pd.DataFrame:
    ts = df_1m["ts"].values
    if anchor_ms == 0:
        buckets = (ts // tf_ms) * tf_ms
    else:
        buckets = ((ts - anchor_ms) // tf_ms) * tf_ms + anchor_ms
    g = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    return g.rename(columns={"bucket": "ts"})


def agg_1d(df_1m: pd.DataFrame) -> pd.DataFrame:
    return agg_tf(df_1m, TF_1D_MS, 0)


def agg_1w(df_1m: pd.DataFrame) -> pd.DataFrame:
    return agg_tf(df_1m, TF_1W_MS, MON_ANCHOR_MS)


def detect_fractals(bars: pd.DataFrame) -> list[dict]:
    """Williams n=2 fractals: FH (high) и FL (low). Symmetric ±N checks."""
    ts = bars["ts"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    n = len(bars)
    out = []
    for i in range(N_FRACTAL, n - N_FRACTAL):
        h_i, l_i = h[i], l[i]
        if (h_i > h[i-2] and h_i > h[i-1]
                and h_i > h[i+1] and h_i > h[i+2]):
            out.append({"ts": int(ts[i]), "side": "FH", "level": float(h_i)})
        if (l_i < l[i-2] and l_i < l[i-1]
                and l_i < l[i+1] and l_i < l[i+2]):
            out.append({"ts": int(ts[i]), "side": "FL", "level": float(l_i)})
    return out


def mark_w_aligned(fr_D: list[dict], fr_W: list[dict]) -> None:
    """Проставить aligned_W/ready_ms на каждом D-фрактале (in-place)."""
    for d in fr_D:
        aligned = any(
            w["side"] == d["side"]
            and w["ts"] <= d["ts"] < w["ts"] + TF_1W_MS
            and abs(w["level"] - d["level"]) / max(d["level"], 1.0) < LEVEL_TOL_PCT
            for w in fr_W
        )
        d["aligned_W"] = aligned
        d["ready_ms"] = d["ts"] + READY_DAYS * TF_1D_MS


def latest_vwap_anchors_path(symbol: str) -> pathlib.Path:
    """Последний (по mtime) vwap_anchors_{symbol}_*.parquet."""
    candidates = sorted(VWAP_ANCHORS_DIR.glob(f"vwap_anchors_{symbol}_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no vwap_anchors_{symbol}_*.parquet in {VWAP_ANCHORS_DIR}")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-24")
    args = ap.parse_args()

    print(f"vwap_anchors (D/W fractals + W-alignment): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    df_1m = load_1m(args.symbol)
    df_1d = agg_1d(df_1m)
    df_1w = agg_1w(df_1m)
    print(f"  D bars: {len(df_1d):,}   W bars: {len(df_1w):,}", file=sys.stderr, flush=True)

    fr_D = detect_fractals(df_1d)
    fr_W = detect_fractals(df_1w)
    mark_w_aligned(fr_D, fr_W)
    n_aligned = sum(1 for d in fr_D if d["aligned_W"])
    print(f"  D fractals: {len(fr_D):,}   W fractals: {len(fr_W):,}   D W-aligned: {n_aligned:,}",
          file=sys.stderr, flush=True)

    out_df = pd.DataFrame(fr_D, columns=["ts", "side", "level", "aligned_W", "ready_ms"])

    VWAP_ANCHORS_DIR.mkdir(parents=True, exist_ok=True)
    out = VWAP_ANCHORS_DIR / f"vwap_anchors_{args.symbol}_{args.start}_{args.end}.parquet"
    out_df.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(out_df):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
