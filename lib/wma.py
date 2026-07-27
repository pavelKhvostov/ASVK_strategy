"""wma — WMA(50) на 1h, level-1 shared indicator (ASVK-standalone).

WMA(50) на 1h — вторая половина пары "HMA(78)/WMA(50)", используемой канон-условиями
паттернов (см. lib/patterns/hs_top.py — правила 2a/2d; lib/patterns/wedge.py — born).
HMA(78) на 1h считается отдельно, см. lib/trendline.py (--tf 1h --length 78).

Формула: WMA(src, n) — взвешенное скользящее с линейными весами (1..n, последний бар
весит больше всех). Векторизовано через np.convolve (эквивалентно циклу в
trend_line_asvk.wma(), но быстрее на длинных 1h-сериях).

Depends only on:
    data/{SYMBOL}USDT_1m.csv   (сам daemon, автономно)
Writes:
    data/wma/wma_{SYMBOL}_{start}_{end}.parquet
    columns: ts, wma50

Usage:
    python wma.py --symbol BTC --start 2018-01-01 --end 2026-07-24
"""
from __future__ import annotations
import argparse
import io
import pathlib
import sys
import time

import numpy as np
import pandas as pd

WAREHOUSE = pathlib.Path(__file__).resolve().parent.parent  # ASVK-standalone
DATA_DIR = WAREHOUSE / "data"
WMA_DIR = DATA_DIR / "wma"

TF_1H_MS = 60 * 60 * 1000
WMA_LEN = 50
# incremental mode: сколько часов 1m-хвоста тянуть с запасом (WMA_LEN=50 + margin на
# разрывы/gap в данных). Больше, чем нужно математически, но дёшево — это только
# tail-seek, не полный файл.
TAIL_MARGIN_HOURS = 200
TAIL_1M_LINES = TAIL_MARGIN_HOURS * 60 + 120


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


def _tail_lines(path: pathlib.Path, n_lines: int) -> list[bytes]:
    """Последние n_lines строк файла — seek с конца блоками по 64KB (без чтения всего
    файла), тот же приём, что dashboard/server.py:tail_lines()."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        block_size = 65536
        blocks: list[bytes] = []
        pos = file_size
        newline_count = 0
        while pos > 0 and newline_count <= n_lines:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            block = f.read(read_size)
            blocks.append(block)
            newline_count += block.count(b"\n")
        data = b"".join(reversed(blocks))
    lines = data.split(b"\n")
    lines = [ln for ln in lines if ln.strip()]
    return lines[-n_lines:] if len(lines) > n_lines else lines


def load_1m_tail(symbol: str, n_lines: int = TAIL_1M_LINES) -> pd.DataFrame:
    """Последние n_lines строк CSV (не весь файл) — для incremental-режима."""
    path = DATA_DIR / f"{symbol}USDT_1m.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        hdr = f.readline().strip()
    body = _tail_lines(path, n_lines)
    raw = b"\n".join([hdr, *body])
    df = pd.read_csv(io.BytesIO(raw), dtype={"open": "float64", "high": "float64",
                                              "low": "float64", "close": "float64",
                                              "volume": "float64"})
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("ts").drop_duplicates("ts", keep="first").reset_index(drop=True)
    return df


def agg_1h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Отбрасывает последний бар, если он ещё не закрылся — см.
    lib/fractal12h/common.py:agg_12h() (тот же принцип, для консистентности
    level-1 индикаторов; WMA на последнем баре тоже "плывёт", пока бар открыт)."""
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


def wma(values: np.ndarray, n: int = WMA_LEN) -> np.ndarray:
    """WMA с линейными весами (1..n). np.convolve эквивалент цикла в trend_line_asvk.wma()."""
    weights = np.arange(1, n + 1, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < n:
        return out
    conv = np.convolve(values, weights[::-1], mode="valid") / weights.sum()
    out[n - 1:] = conv
    return out


def latest_wma_path(symbol: str) -> pathlib.Path:
    """Последний (по mtime) wma_{symbol}_*.parquet — как latest_maxv_path в lib/maxv.py."""
    candidates = sorted(WMA_DIR.glob(f"wma_{symbol}_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no wma_{symbol}_*.parquet in {WMA_DIR}")
    return candidates[-1]


def _full_recompute(symbol: str) -> pd.DataFrame:
    df_1m = load_1m(symbol)
    df_1h = agg_1h(df_1m)
    t_arr = df_1h["ts"].to_numpy()
    print(f"  1h bars: {len(df_1h):,}", file=sys.stderr, flush=True)
    wma50 = wma(df_1h["close"].to_numpy(), WMA_LEN)
    n_have = int(np.sum(~np.isnan(wma50)))
    print(f"  WMA-{WMA_LEN} (1h): {n_have:,}/{len(t_arr):,} bars with value", file=sys.stderr, flush=True)
    return pd.DataFrame({"ts": t_arr, "wma50": wma50})


def _incremental_recompute(symbol: str) -> pd.DataFrame | None:
    """Досчитать только хвост (последние TAIL_MARGIN_HOURS часов) и смёржить поверх
    существующего parquet. Возвращает None, если инкремент невозможен/небезопасен
    (нет прежнего файла, или между старой историей и новым хвостом обнаружен разрыв) —
    вызывающий код в этом случае обязан откатиться на полный пересчёт."""
    try:
        existing_path = latest_wma_path(symbol)
    except FileNotFoundError:
        print("  no existing wma parquet — incremental not possible, falling back to full",
              file=sys.stderr, flush=True)
        return None
    existing_df = pd.read_parquet(existing_path)
    if existing_df.empty:
        return None

    df_1m_tail = load_1m_tail(symbol)
    df_1h_tail = agg_1h(df_1m_tail)
    print(f"  tail: {len(df_1m_tail):,} 1m bars → {len(df_1h_tail):,} 1h bars", file=sys.stderr, flush=True)

    wma50_tail = wma(df_1h_tail["close"].to_numpy(), WMA_LEN)
    tail_df = pd.DataFrame({"ts": df_1h_tail["ts"].to_numpy(), "wma50": wma50_tail})
    valid_tail = tail_df[tail_df["wma50"].notna()].reset_index(drop=True)
    if valid_tail.empty:
        print("  tail too short for a full WMA window — falling back to full", file=sys.stderr, flush=True)
        return None

    cutoff = int(valid_tail["ts"].min())
    last_existing_ts = int(existing_df["ts"].max())
    if cutoff > last_existing_ts + TF_1H_MS:
        # разрыв между тем, что уже посчитано, и тем, что можем пересчитать из хвоста —
        # запаса TAIL_MARGIN_HOURS не хватило (напр. daemon долго не запускался).
        print(f"  gap detected between existing data (up to {last_existing_ts}) and "
              f"tail (from {cutoff}) — falling back to full", file=sys.stderr, flush=True)
        return None

    merged = pd.concat([existing_df[existing_df["ts"] < cutoff], valid_tail], ignore_index=True)
    merged = merged.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    n_have = int(merged["wma50"].notna().sum())
    print(f"  incremental merge: {len(existing_df):,} existing + {len(valid_tail):,} recomputed tail "
          f"→ {len(merged):,} rows ({n_have:,} with value)", file=sys.stderr, flush=True)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--incremental", action="store_true",
                     help="Досчитать только хвост поверх существующего parquet вместо полного "
                          "пересчёта всей истории. Если прежнего файла нет — тихо откатывается "
                          "на полный пересчёт (bootstrap).")
    args = ap.parse_args()

    print(f"wma (WMA-{WMA_LEN} 1h): {args.symbol} {args.start} → {args.end}"
          f"{' [incremental]' if args.incremental else ''}", file=sys.stderr, flush=True)
    t0 = time.time()

    out_df = None
    if args.incremental:
        out_df = _incremental_recompute(args.symbol)
    if out_df is None:
        out_df = _full_recompute(args.symbol)

    WMA_DIR.mkdir(parents=True, exist_ok=True)
    out = WMA_DIR / f"wma_{args.symbol}_{args.start}_{args.end}.parquet"
    out_df.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(out_df):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
