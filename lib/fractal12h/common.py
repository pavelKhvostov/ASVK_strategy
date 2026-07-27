"""common — общие утилиты для проекта «фрактал-12h» (ASVK-portable).

Портировано из ~/smc-warehouse/scripts/фрактал-12h/common.py (WSL, read-only источник).
Автономно: читает ТОЛЬКО G:\\ASVK\\data\\, пишет в G:\\ASVK\\data\\fractal12h\\.
"""
from __future__ import annotations
import pathlib
import sys
import time

import numpy as np
import pandas as pd


BASE = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "data"
EVENTS_DIR = DATA_DIR / "events"
DATA_OUT = DATA_DIR / "fractal12h"

TF_15M_MS = 15 * 60 * 1000       # 900_000
TF_12H_MS = 12 * 60 * 60 * 1000  # 43_200_000
TF_1D_MS  = 24 * 60 * 60 * 1000  # 86_400_000
TF_1W_MS  = 7 * 24 * 60 * 60 * 1000
MON_ANCHOR_MS = 1_483_315_200_000  # 2017-01-02 UTC (Monday) — только для 1w


def load_1m(symbol: str) -> pd.DataFrame:
    """Загрузить 1m OHLC из G:\\ASVK\\data\\{SYM}USDT_1m.csv (пишет asvk.py daemon).

    Returns DataFrame columns: ts (int ms UTC), open, high, low, close, volume.
    """
    path = DATA_DIR / f"{symbol}USDT_1m.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[common] loading {path.name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    df = pd.read_csv(path, dtype={"open": "float64", "high": "float64",
                                   "low": "float64", "close": "float64",
                                   "volume": "float64"})
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("ts").drop_duplicates("ts", keep="first").reset_index(drop=True)
    print(f"[common]   {len(df):,} 1m bars in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return df


def latest_events_path(symbol: str) -> pathlib.Path:
    """Последний (по mtime) events_e12d_{symbol}_*.parquet — имя файла меняется
    каждый день (--end двигается), поэтому ищем самый свежий, а не фиксированный."""
    candidates = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no events_e12d_{symbol}_*.parquet in {EVENTS_DIR}")
    return candidates[-1]


def agg_tf(df_1m: pd.DataFrame, tf_ms: int, anchor_ms: int = 0) -> pd.DataFrame:
    """Generic aggregation 1m → любой TF. Правило anchor как в e12d (CLAUDE.md канон).

    anchor_ms=0 → epoch-floor (для 15m..3d)
    anchor_ms>0 → shift-anchor (для 1w: MON_ANCHOR_MS)
    """
    ts = df_1m["ts"].values
    if anchor_ms == 0:
        buckets = (ts // tf_ms) * tf_ms
    else:
        buckets = ((ts - anchor_ms) // tf_ms) * tf_ms + anchor_ms
    g = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return g.rename(columns={"bucket": "ts"})


def agg_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """1m → 15m (epoch-floor)."""
    return agg_tf(df_1m, TF_15M_MS, 0)


def agg_12h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """1m → 12h (epoch-floor). Совпадает с TV BINANCE:BTCUSDT.

    Отбрасывает последний бар, если он ещё не закрылся (не хватает 1m-данных
    до конца его 12h-окна). До закрытия бара i понятия сигнала не существует —
    все A/B-условия читают h/l/close бара i напрямую, поэтому на ещё
    формирующемся баре они посчитаны на "плывущих" данных и не значат ничего
    (см. разбор ASVK B5C1 2026-07-24: сигнал появился и пропал за 8.5ч из-за
    незакрытого бара, а не из-за исхода через 24ч)."""
    df_12h = agg_tf(df_1m, TF_12H_MS, 0)
    if len(df_12h) == 0:
        return df_12h
    last_1m_close_ms = int(df_1m["ts"].to_numpy()[-1]) + 60_000
    last_bucket_close_ms = int(df_12h["ts"].to_numpy()[-1]) + TF_12H_MS
    if last_1m_close_ms < last_bucket_close_ms:
        df_12h = df_12h.iloc[:-1].reset_index(drop=True)
    return df_12h


def agg_1d(df_1m: pd.DataFrame) -> pd.DataFrame:
    """1m → 1d (epoch-floor)."""
    return agg_tf(df_1m, TF_1D_MS, 0)


def agg_1w(df_1m: pd.DataFrame) -> pd.DataFrame:
    """1m → 1w (Monday-anchor). Совпадает с TV weekly."""
    return agg_tf(df_1m, TF_1W_MS, MON_ANCHOR_MS)
