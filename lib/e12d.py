"""e12d — тонкий оркестратор поиска зон интереса.

Модель: график + каноны элементов → журнал.

Поток:
    1. Читаем 1m OHLCV из склада (график/{symbol}_1m.csv)
    2. Агрегируем в 10 TF (15m..3d)
    3. Для каждого (element, tf) вызываем elements/{element}.detect(candles)
    4. Собираем все события, ренумеруем zone_id глобально
    5. Сортируем по ts → parquet в data/events/

Никакой inline детекции. Все каноны — в модулях элементов.
Direction canon: long/short во всех событиях.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


WAREHOUSE = pathlib.Path(__file__).resolve().parent.parent  # ASVK-standalone
ELEMENTS_DIR = WAREHOUSE / "lib" / "детекторы"  # ASVK-standalone
sys.path.insert(0, str(ELEMENTS_DIR))

import ob, fvg, marubozu, rb, ob_liq, rdrb, i_rdrb, block_orders, fractal  # noqa: E402
import breaker_block, mitigation_block, i_fvg, ob_vc  # noqa: E402


UTC = timezone.utc

# TF grid — 11 TF, все агрегируются из 1m
# Anchor rule: все ТФ используют epoch-floor (anchor=0), КРОМЕ 1w:
#   weekly = Monday-anchored (TV BINANCE:BTCUSDT convention, MON_ANCHOR = 2017-01-02 UTC).
# См. CLAUDE.md канон tf-склейка.
TF_MS: dict[str, int] = {
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h":  60 * 60_000,
    "2h":  2 * 60 * 60_000,
    "4h":  4 * 60 * 60_000,
    "6h":  6 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d":  86_400_000,
    "2d":  2 * 86_400_000,
    "3d":  3 * 86_400_000,
    "1w":  7 * 86_400_000,
}

# Anchor offsets (ms). Default 0 (epoch-floor).
# TV BINANCE:BTCUSDT anchor conventions verified 2026-07-10:
#   1w = Monday-anchored (2017-01-02 UTC)
#   2d = shifted +1 day от epoch (Fri-aligned; verified TV bar ts=1763683200 Fri 2025-11-21)
#   3d = TBD (verify)
TF_ANCHORS: dict[str, int] = {
    "1w": 1_483_315_200_000,  # 2017-01-02 00:00 UTC (Monday)
    "2d": 86_400_000,         # +1 day shift (Fri-aligned per TV BINANCE:BTCUSDT)
}

# Одноклеточные (простые) детекторы — вызываются одинаково
SIMPLE_DETECTORS = {
    "ob":                ob,
    "fvg":               fvg,
    "marubozu":          marubozu,
    "rb":                rb,
    "ob_liq":            ob_liq,
    "rdrb":              rdrb,
    "i_rdrb":            i_rdrb,
    "block_orders":      block_orders,
    "fractal":           fractal,
    "breaker_block":     breaker_block,
    "mitigation_block":  mitigation_block,
    "i_fvg":             i_fvg,
}

# ob_vc — специальный, требует LTF данные
OB_VC_HTFS = ("1h", "2h", "4h", "6h", "12h", "1d", "2d", "3d")


# ─── I/O ───────────────────────────────────────────────────────────

def load_1m(csv_path: pathlib.Path) -> pd.DataFrame:
    """Загрузить 1m CSV в DataFrame. Векторизация pandas ~15× быстрее csv.reader loop.

    Returns DataFrame columns: ts (int ms), open, high, low, close, volume.
    """
    print(f"Loading 1m from {csv_path.name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    df = pd.read_csv(csv_path, dtype={"open": "float64", "high": "float64",
                                       "low": "float64", "close": "float64",
                                       "volume": "float64"})
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    n_raw = len(df)
    df = df.sort_values("ts").drop_duplicates("ts", keep="first").reset_index(drop=True)
    if len(df) != n_raw:
        print(f"  deduped: {n_raw:,} → {len(df):,}", file=sys.stderr, flush=True)
    print(f"  loaded {len(df):,} 1m bars in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return df


def aggregate_arrays(df_1m: pd.DataFrame, tf_ms: int, anchor_ms: int = 0) -> tuple:
    """1m DataFrame → TF numpy arrays (o, h, l, c, ts) через groupby.

    anchor_ms=0 → epoch-floor (canonical для всех ТФ кроме 1w).
    anchor_ms>0 → shift-anchored (для 1w: Monday 2017-01-02 UTC).
    """
    ts = df_1m["ts"].values
    if anchor_ms == 0:
        buckets = (ts // tf_ms) * tf_ms
    else:
        buckets = ((ts - anchor_ms) // tf_ms) * tf_ms + anchor_ms
    grouped = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    return (
        grouped["open"].to_numpy(dtype=np.float64),
        grouped["high"].to_numpy(dtype=np.float64),
        grouped["low"].to_numpy(dtype=np.float64),
        grouped["close"].to_numpy(dtype=np.float64),
        grouped["bucket"].to_numpy(dtype=np.int64),
    )


def filter_range(df_1m: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Slice DataFrame по ts диапазону (векторизованно)."""
    mask = (df_1m["ts"] >= start_ms) & (df_1m["ts"] < end_ms)
    return df_1m[mask].reset_index(drop=True)


# ─── Detector dispatch ──────────────────────────────────────────────

def run_simple_arrays(element: str, tf: str, arrays: tuple) -> list[dict]:
    """Воркер: детектор напрямую на numpy (o,h,l,c,ts). Fast pickle через buffer protocol."""
    o_a, h_a, l_a, c_a, ts_a = arrays
    module = SIMPLE_DETECTORS[element]
    events = module.detect(o_a, h_a, l_a, c_a, ts_a, tf_ms=TF_MS[tf])
    for ev in events:
        ev["element"] = element
        ev["tf"] = tf
    return events


def run_ob_vc(htf: str, htf_arrays: tuple, ltf_arrays: dict[str, tuple]) -> list[dict]:
    o, h, l, c, ts = htf_arrays
    events = ob_vc.detect(
        o, h, l, c, ts,
        tf_ms=TF_MS[htf], htf=htf,
        ltf_arrays=ltf_arrays,
        ltf_ms_map=TF_MS,
    )
    for ev in events:
        ev["element"] = "ob_vc"
        ev["tf"] = htf
    return events


# ─── Main pipeline ──────────────────────────────────────────────────

def process_symbol(symbol: str, start: str, end: str, n_workers: int = 30) -> pd.DataFrame:
    csv_path = WAREHOUSE / f"data/{symbol}USDT_1m.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df_1m = load_1m(csv_path)
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
    df_1m = filter_range(df_1m, start_ms, end_ms)
    print(f"  {symbol}: {len(df_1m):,} 1m bars in range", file=sys.stderr, flush=True)

    # Parallel aggregate → numpy arrays per TF
    print(f"  aggregating {len(TF_MS)} TFs in parallel...", file=sys.stderr, flush=True)
    t0 = time.time()

    def _agg_one(tf, tf_ms, anchor_ms):
        return tf, aggregate_arrays(df_1m, tf_ms, anchor_ms)

    agg_results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_agg_one)(tf, tf_ms, TF_ANCHORS.get(tf, 0)) for tf, tf_ms in TF_MS.items()
    )
    tf_arrays: dict[str, tuple] = dict(agg_results)
    for tf in TF_MS:
        print(f"  {tf}: {len(tf_arrays[tf][0]):,} bars", file=sys.stderr, flush=True)
    print(f"  aggregation done in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    # ─── Merged dispatch: ВСЕ 140 jobs (132 simple + 8 ob_vc) в один Pool ───
    # Heaviest first: ob_vc 1h/2h (long-running) стартуют раньше, occupy workers.
    # Simple detectors заполняют оставшиеся slots. 30 workers всегда заняты.
    OB_VC_ORDER = ["1h", "2h", "4h", "6h", "12h", "1d", "2d", "3d"]   # 1h heaviest
    ob_vc_jobs = [("ob_vc", htf) for htf in OB_VC_ORDER if htf in tf_arrays]
    simple_jobs = [("simple", elem, tf) for tf in TF_MS for elem in SIMPLE_DETECTORS]

    def _run_job(job):
        if job[0] == "ob_vc":
            _, htf = job
            return run_ob_vc(htf, tf_arrays[htf], tf_arrays)
        else:
            _, elem, tf = job
            return run_simple_arrays(elem, tf, tf_arrays[tf])

    # Heaviest first: ob_vc jobs → simple jobs (heaviest simple TFs first from 15m)
    all_jobs = ob_vc_jobs + simple_jobs
    print(f"  running {len(all_jobs)} jobs ({len(ob_vc_jobs)} ob_vc + {len(simple_jobs)} simple) on {n_workers} workers...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=n_workers, backend="loky", batch_size=1)(
        delayed(_run_job)(job) for job in all_jobs
    )
    print(f"  all detectors done in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    # ─── Renumber zone_id globally ───
    global_zid = 0
    for group in results:
        # каждый group имеет свой набор локальных zone_id 1..N
        seen_local = {}
        for ev in group:
            local = ev["zone_id"]
            if local not in seen_local:
                global_zid += 1
                seen_local[local] = global_zid
            ev["zone_id"] = seen_local[local]
            ev["symbol"] = symbol

    all_events = [ev for group in results for ev in group]
    print(f"  {symbol}: {len(all_events):,} total events, {global_zid:,} unique zones",
          file=sys.stderr, flush=True)

    # ─── Materialize meta as JSON-string (parquet-friendly) ───
    df = pd.DataFrame(all_events)
    if not df.empty:
        df["meta"] = df["meta"].apply(str)  # dict → str для parquet
        df = df.sort_values("ts").reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-07-07")
    parser.add_argument("--n-workers", type=int, default=30)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    out_path = args.out or (WAREHOUSE / f"data/events/events_e12d_{args.symbol}_{args.start}_{args.end}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"e12d: {args.symbol} {args.start}..{args.end} → {out_path}", file=sys.stderr, flush=True)
    t0 = time.time()
    df = process_symbol(args.symbol, args.start, args.end, n_workers=args.n_workers)
    print(f"total time: {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    df.to_parquet(out_path, index=False, compression="zstd", compression_level=9)
    print(f"written: {out_path}  ({len(df):,} rows)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
