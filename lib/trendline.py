"""trendline — MA(length, mode) на заданном TF + band, level-1 shared indicator (ASVK-standalone).

Canon-варианты используются в проекте (variant = "{tf}{length}" для mode=Hma,
иначе "{tf}{length}{mode}" — Hma не помечается суффиксом ради обратной
совместимости с уже существующими файлами):
  D200      — HMA-200 на Daily. B4C2 (lib/fractal12h/b4_hma.py, FULL_DISP).
  12h78     — HMA-78 на 12h. B4C1 (union с D78, multi-TF).
  D78       — HMA-78 на Daily. B4C1.
  1h78      — HMA-78 на 1h. Детекторы паттернов (lib/patterns/hs_top.py, wedge.py).
  12h9Thma  — THMA-9 на 12h. B4C3 (research-находка, MA-family scan 2026-07-25).
  D9Thma    — THMA-9 на Daily. B4C5.
  D50Wma    — WMA-50 на Daily (плоская, БЕЗ Hull-обёртки). B4C4.
  D20Ehma   — EHMA-20 на Daily. B4C6.

Режимы (--mode):
  Hma/Ehma/Thma — Hull-семейство через trend_line_asvk() (2*MA(n/2)-MA(n), сглажено
                  WMA/EMA(round(sqrt(n)))). formula verbatim из trend_line_asvk.py.
  Wma/Sma/Ema   — ПЛОСКИЕ MA напрямую (без Hull-обёртки), той же length. Обёрнуты в
                  ту же выходную схему (mhull/shull/upper/lower/color) для
                  единообразия читателей — shull здесь просто mhull[i-2], тот же
                  сдвиг, что и у Hull-варианта.

  HMA(src, n) = WMA(2*WMA(src, n/2) − WMA(src, n), round(√n))
  MHULL(i) = MA(close, n)[i]             — центральная линия (Hull или плоская)
  SHULL(i) = MHULL(i-2)                  — сдвиг на 2 бара назад
  upper(i) = max(MHULL(i), SHULL(i))     — band верх
  lower(i) = min(MHULL(i), SHULL(i))     — band низ
  color(i) = 'up' if close[i] > SHULL(i) else 'down'

Depends only on:
    data/{SYMBOL}USDT_1m.csv   (сам daemon, автономно)
    trend_line_asvk.py         (level-1 формула, тот же lib/)
Writes:
    data/trendline/trendline_{SYMBOL}_{variant}_{start}_{end}.parquet
    columns: ts, mhull, shull, upper, lower, color

Usage:
    python trendline.py --symbol BTC --tf D  --length 200 --mode Hma  --start 2018-01-01 --end 2026-07-24
    python trendline.py --symbol BTC --tf D  --length 50  --mode Wma  --start 2018-01-01 --end 2026-07-24
    python trendline.py --symbol BTC --tf 12h --length 9  --mode Thma --start 2018-01-01 --end 2026-07-24
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from trend_line_asvk import trend_line_asvk, sma, wma, ema_series

FLAT_MODES = {"Sma": sma, "Wma": wma, "Ema": ema_series}
HULL_MODES = {"Hma", "Ehma", "Thma"}

WAREHOUSE = pathlib.Path(__file__).resolve().parent.parent  # ASVK-standalone
DATA_DIR = WAREHOUSE / "data"
TRENDLINE_DIR = DATA_DIR / "trendline"

TF_MS = {
    "1h": 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "D": 24 * 60 * 60 * 1000,
}


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


def agg_tf(df_1m: pd.DataFrame, tf_ms: int) -> pd.DataFrame:
    """Отбрасывает последний бар, если он ещё не закрылся — см.
    lib/fractal12h/common.py:agg_12h() (тот же принцип, для консистентности
    level-1 индикаторов)."""
    ts = df_1m["ts"].values
    buckets = (ts // tf_ms) * tf_ms
    g = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    g = g.rename(columns={"bucket": "ts"})
    if len(g) == 0:
        return g
    last_1m_close_ms = int(df_1m["ts"].to_numpy()[-1]) + 60_000
    last_bucket_close_ms = int(g["ts"].to_numpy()[-1]) + tf_ms
    if last_1m_close_ms < last_bucket_close_ms:
        g = g.iloc[:-1].reset_index(drop=True)
    return g


def latest_trendline_path(symbol: str, variant: str) -> pathlib.Path:
    """Последний (по mtime) trendline_{symbol}_{variant}_*.parquet.

    variant обязателен — без него можно случайно подхватить не тот TF/length
    (напр. B4C2 подхватит 1h78 вместо D200).
    """
    candidates = sorted(TRENDLINE_DIR.glob(f"trendline_{symbol}_{variant}_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no trendline_{symbol}_{variant}_*.parquet in {TRENDLINE_DIR}")
    return candidates[-1]


def _flat_result(closes: list[float], mode: str, length: int) -> dict:
    """Плоская MA (Sma/Wma/Ema), обёрнутая в ту же схему, что trend_line_asvk()."""
    mhull = FLAT_MODES[mode](closes, length)
    shull: list[float | None] = [None] * len(closes)
    for i in range(2, len(closes)):
        shull[i] = mhull[i - 2]
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)
    color: list[str | None] = [None] * len(closes)
    for i in range(len(closes)):
        m, s = mhull[i], shull[i]
        if m is None or s is None:
            continue
        upper[i] = max(m, s)
        lower[i] = min(m, s)
        color[i] = "up" if closes[i] > s else "down"
    return {"mhull": mhull, "shull": shull, "upper": upper, "lower": lower, "color": color}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--tf", default="D", choices=list(TF_MS))
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--mode", default="Hma", choices=["Hma", "Ehma", "Thma", "Wma", "Sma", "Ema"])
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-07-24")
    args = ap.parse_args()
    variant = f"{args.tf}{args.length}" if args.mode == "Hma" else f"{args.tf}{args.length}{args.mode}"

    print(f"trendline ({args.mode}-{args.length} {args.tf} + band): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    df_1m = load_1m(args.symbol)
    df_tf = agg_tf(df_1m, TF_MS[args.tf])
    t_arr = df_tf["ts"].to_numpy()
    print(f"  {args.tf} bars: {len(df_tf):,}", file=sys.stderr, flush=True)

    closes = list(df_tf["close"])
    if args.mode in HULL_MODES:
        res = trend_line_asvk(closes, length=args.length, length_mult=1.0, mode=args.mode)
    else:
        res = _flat_result(closes, args.mode, args.length)
    n_have = sum(1 for x in res["mhull"] if x is not None)
    print(f"  {args.mode}-{args.length} ({args.tf}): {n_have:,}/{len(t_arr):,} bars with value",
          file=sys.stderr, flush=True)

    out_df = pd.DataFrame({
        "ts": t_arr,
        "mhull": np.array([x if x is not None else np.nan for x in res["mhull"]]),
        "shull": np.array([x if x is not None else np.nan for x in res["shull"]]),
        "upper": np.array([x if x is not None else np.nan for x in res["upper"]]),
        "lower": np.array([x if x is not None else np.nan for x in res["lower"]]),
        "color": [x if x is not None else "" for x in res["color"]],
    })

    TRENDLINE_DIR.mkdir(parents=True, exist_ok=True)
    out = TRENDLINE_DIR / f"trendline_{args.symbol}_{variant}_{args.start}_{args.end}.parquet"
    out_df.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(out_df):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
