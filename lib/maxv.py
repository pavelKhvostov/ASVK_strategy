"""maxv — per-12h-bar maxV level + ATR14, level-1 shared indicator (ASVK-standalone).

Раньше maxV/ATR14 считались приватно внутри lib/fractal12h/b1_fvg.py, а b9_others.py
тянул их оттуда импортом — неверная архитектура: это foundational-индикатор уровня
"сырые данные" (как e12d/s7d), а не деталь одной конкретной B-стратегии. Читают его
несколько независимых условий (B9C2 напрямую; WIDE-фильтр во всей B1-семье; при
будущем расширении — любые новые условия). Поэтому считается один раз здесь и
пишется на диск, как e12d/s7d.

maxV(k) = close LTF-бара (агрегат MAXV_LTF_MIN минут) с абсолютным max объёмом
          (любое направление, вкл. doji) внутри 12h бара k.
ATR14(k) = SMA(TR, 14) на 12h-серии (canon: НЕ Wilder EWM).

MAXV_LTF_MIN = 90 (2026-07-23): walk-forward исследование на 11 активах, train
2020-2025 / test 2025-now — на BTC/ETH/SOL (единственные активы в ASVK) 90m
даёт лучший результат СРАЗУ по обоим критериям (mean WR 75.70% и худший из
трёх активов WR 74.04% на test, против 72.01%/70.71% на прежнем 1m). Раньше
здесь стоял 1m (голый close 1m-бара с max объёмом в bull/bear группе) — сама
формула агрегации изменилась, не только длина окна: теперь LTF-бар — это
агрегат нескольких 1m-баров (сумма volume, close последнего), без разбивки
на bull/bear (доджи больше не исключаются).

Depends only on:
    data/{SYMBOL}USDT_1m.csv   (сам daemon, автономно)
Writes:
    data/maxv/maxv_{SYMBOL}_{start}_{end}.parquet   — columns: ts, atr14, maxv

Usage:
    python maxv.py --symbol BTC --start 2018-01-01 --end 2026-07-24
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

WAREHOUSE = pathlib.Path(__file__).resolve().parent.parent  # ASVK-standalone
DATA_DIR = WAREHOUSE / "data"
MAXV_DIR = DATA_DIR / "maxv"

TF_12H_MS = 12 * 60 * 60 * 1000
ATR_N = 14
MAXV_LTF_MIN = 90


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


def agg_12h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Отбрасывает последний бар, если он ещё не закрылся — см.
    lib/fractal12h/common.py:agg_12h() (тот же принцип, для консистентности
    level-1 индикаторов)."""
    ts = df_1m["ts"].values
    buckets = (ts // TF_12H_MS) * TF_12H_MS
    g = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    g = g.rename(columns={"bucket": "ts"})
    if len(g) == 0:
        return g
    last_1m_close_ms = int(df_1m["ts"].to_numpy()[-1]) + 60_000
    last_bucket_close_ms = int(g["ts"].to_numpy()[-1]) + TF_12H_MS
    if last_1m_close_ms < last_bucket_close_ms:
        g = g.iloc[:-1].reset_index(drop=True)
    return g


def atr14_sma(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """ATR14 = SMA(TR, 14) — canon-совместимо (не Wilder EWM)."""
    n = len(h)
    c_prev = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - c_prev), np.abs(l - c_prev)])
    tr[0] = 0.0
    csum = np.cumsum(tr)
    atr = np.zeros(n)
    atr[ATR_N:] = (csum[ATR_N:] - csum[:n - ATR_N]) / ATR_N
    return atr


def compute_maxv_ltf(df_1m: pd.DataFrame, t12: np.ndarray, ltf_min: int = MAXV_LTF_MIN) -> np.ndarray:
    """maxV(k) = close LTF-бара (агрегат ltf_min минут — сумма volume, close последнего
    1m-бара внутри LTF-окна) с абсолютным max объёмом внутри 12h бара k. Направление
    (bull/bear) не учитывается — просто какое окно собрало больше объёма."""
    ltf_ms = ltf_min * 60_000
    ts = df_1m["ts"].to_numpy()
    buckets = (ts // ltf_ms) * ltf_ms
    ltf = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        close=("close", "last"), volume=("volume", "sum"))
    ltf["bar12"] = (ltf["bucket"] // TF_12H_MS) * TF_12H_MS
    idx = ltf.groupby("bar12")["volume"].idxmax()
    winners = ltf.loc[idx, ["bar12", "close"]].set_index("bar12")["close"]
    return winners.reindex(t12).to_numpy()


def latest_maxv_path(symbol: str) -> pathlib.Path:
    """Последний (по mtime) maxv_{symbol}_*.parquet — как latest_events_path в fractal12h/common.py."""
    candidates = sorted(MAXV_DIR.glob(f"maxv_{symbol}_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no maxv_{symbol}_*.parquet in {MAXV_DIR}")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-24")
    args = ap.parse_args()

    print(f"maxv: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)
    t12 = df_12h["ts"].to_numpy()
    h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy()
    c12 = df_12h["close"].to_numpy()
    print(f"  12h bars: {len(df_12h):,}", file=sys.stderr, flush=True)

    atr14 = atr14_sma(h12, l12, c12)
    maxv = compute_maxv_ltf(df_1m, t12, MAXV_LTF_MIN)
    n_have = int(np.sum(~np.isnan(maxv)))
    print(f"  LTF={MAXV_LTF_MIN}m  {n_have:,}/{len(t12):,} bars with maxV", file=sys.stderr, flush=True)

    out_df = pd.DataFrame({"ts": t12, "atr14": atr14, "maxv": maxv})

    MAXV_DIR.mkdir(parents=True, exist_ok=True)
    out = MAXV_DIR / f"maxv_{args.symbol}_{args.start}_{args.end}.parquet"
    out_df.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(out_df):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
