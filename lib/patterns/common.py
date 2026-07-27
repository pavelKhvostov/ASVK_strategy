"""common — общие утилиты для проекта «паттерны» (Block 4, ASVK-portable).

Портировано по аналогии с lib/fractal12h/common.py. Автономно: читает ТОЛЬКО
G:\\ASVK\\data\\, пишет в G:\\ASVK\\data\\patterns\\. Работает на 1h TF (в отличие
от фрактал-12h, который на 12h) — источник: ~/smc-warehouse/паттерны/{голова и
плечи,клин}/ (WSL, read-only).
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
DATA_OUT = DATA_DIR / "patterns"

TF_1H_MS = 60 * 60 * 1000


def load_1m(symbol: str) -> pd.DataFrame:
    """Загрузить 1m OHLC из G:\\ASVK\\data\\{SYM}USDT_1m.csv (пишет asvk.py daemon)."""
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


def agg_1h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """1m → 1h (epoch-floor).

    Отбрасывает последний бар, если он ещё не закрылся (не хватает 1m-данных
    до конца его 1h-окна) — та же причина, что и в lib/fractal12h/common.py:
    hs_top.py/wedge.py читают high/low/close бара напрямую (breakout-скан,
    HMA/WMA born-детекция), и на ещё формирующемся баре эти значения "плывут"."""
    ts = df_1m["ts"].values
    buckets = (ts // TF_1H_MS) * TF_1H_MS
    g = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    g = g.rename(columns={"bucket": "ts"})
    if len(g) == 0:
        return g
    last_1m_close_ms = int(df_1m["ts"].to_numpy()[-1]) + 60_000
    last_bucket_close_ms = int(g["ts"].to_numpy()[-1]) + TF_1H_MS
    if last_1m_close_ms < last_bucket_close_ms:
        g = g.iloc[:-1].reset_index(drop=True)
    return g


def latest_events_path(symbol: str) -> pathlib.Path:
    """Последний (по mtime) events_e12d_{symbol}_*.parquet — как в lib/fractal12h/common.py."""
    candidates = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no events_e12d_{symbol}_*.parquet in {EVENTS_DIR}")
    return candidates[-1]
