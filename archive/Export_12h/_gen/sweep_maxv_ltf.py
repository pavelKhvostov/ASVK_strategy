"""B3C1 (maxV sweep, без depth) — поиск универсального LTF-параметра maxV.

Пул: A1 + A2 + A4 (skip A3, per предыдущий анализ).
Train: 2020-01-01 .. 2025-01-01 (калибровка/справочно).
Test:  2025-01-01 .. конец данных (главный критерий отбора, out-of-sample).

maxV(LTF) = close LTF-бара с АБСОЛЮТНЫМ макс объёмом (любого направления, incl.
doji) внутри родительского 12h бара i-1 — тот же алгоритм, что VIC ASVK
(индикаторы/vic_asvk.py), но векторизовано через pandas (быстрее для sweep
по многим LTF-вариантам × 11 активам).

Read-only: только читает ~/smc-warehouse/график/*.csv. Параллелизация по
активам (joblib, 11 задач) — каждый worker грузит 1m один раз и считает
все LTF-варианты сразу (дешевле, чем перезагружать данные на каждый вариант).
"""
from __future__ import annotations
import pathlib
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

GRAPHIC = pathlib.Path.home() / "smc-warehouse" / "график"
SYMBOLS = ["BTC", "ETH", "SOL", "ADA", "AVAX", "BNB", "DOGE", "DOT", "LINK", "LTC", "XRP"]

TF_12H_MS = 12 * 60 * 60 * 1000
LEFT_EXT_N = 5
BODY_MAX = 0.80
WICK_MIN = 0.03

LTF_VARIANTS_MIN = sorted(set(
    list(range(1, 11)) +                      # 1..10, шаг 1
    [12, 14, 16, 18, 20] +                     # 10..20, шаг 2
    [25, 30, 35, 40, 45, 50, 55, 60] +         # 20..60, шаг 5
    [70, 80, 90, 100, 110, 120] +              # 60..120, шаг 10
    [135, 150, 165, 180] +                     # 120..180, шаг 15
    [200, 220, 240] +                          # 180..240, шаг 20
    [270, 300, 330, 360]                       # 240..360, шаг 30
))  # 40 точек, плотнее в зоне 60-240m (где предыдущий грубый sweep нашёл интересное)

TRAIN_START = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
TRAIN_END = TEST_START = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def load_1m(symbol: str) -> pd.DataFrame:
    path = GRAPHIC / f"{symbol}USDT_1m.csv"
    df = pd.read_csv(path, dtype={"open": "float64", "high": "float64",
                                   "low": "float64", "close": "float64",
                                   "volume": "float64"})
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    return df.sort_values("ts").drop_duplicates("ts", keep="first").reset_index(drop=True)


def agg_12h(df_1m: pd.DataFrame) -> pd.DataFrame:
    ts = df_1m["ts"].values
    buckets = (ts // TF_12H_MS) * TF_12H_MS
    g = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    return g.rename(columns={"bucket": "ts"})


def compute_a_pool(df_12h: pd.DataFrame) -> pd.DataFrame:
    """A1 + A2 + A4(body/wick), skip A3 — как в предыдущем анализе."""
    o = df_12h["open"].to_numpy(); h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy(); c = df_12h["close"].to_numpy()
    ts = df_12h["ts"].to_numpy(); T = len(df_12h)

    hl = h - l
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    body_pct = np.divide(body, hl, out=np.zeros_like(body), where=(hl > 0))
    up_wick_pct = np.divide(upper_wick, hl, out=np.zeros_like(upper_wick), where=(hl > 0))
    lo_wick_pct = np.divide(lower_wick, hl, out=np.zeros_like(lower_wick), where=(hl > 0))

    a1_fh = np.zeros(T, dtype=bool); a1_fh[2:] = (h[2:] > h[1:-1]) & (h[2:] > h[:-2])
    left5_max = pd.Series(h).rolling(LEFT_EXT_N).max().shift(1).to_numpy()
    a2_fh = a1_fh & np.where(np.isnan(left5_max), False, h > left5_max)
    conf_fh = np.zeros(T, dtype=bool); conf_fh[:-2] = (h[1:-1] < h[:-2]) & (h[2:] < h[:-2])

    a1_fl = np.zeros(T, dtype=bool); a1_fl[2:] = (l[2:] < l[1:-1]) & (l[2:] < l[:-2])
    left5_min = pd.Series(l).rolling(LEFT_EXT_N).min().shift(1).to_numpy()
    a2_fl = a1_fl & np.where(np.isnan(left5_min), False, l < left5_min)
    conf_fl = np.zeros(T, dtype=bool); conf_fl[:-2] = (l[1:-1] > l[:-2]) & (l[2:] > l[:-2])

    confirmable = np.arange(T) <= T - 3

    frames = []
    for direction, a1, a2, conf, body_p, wick_p in [
        ("short", a1_fh, a2_fh, conf_fh, body_pct, up_wick_pct),
        ("long", a1_fl, a2_fl, conf_fl, body_pct, lo_wick_pct),
    ]:
        m = a2 & (body_pct <= BODY_MAX) & (wick_p >= WICK_MIN)  # A1+A2+A4(body/wick), skip A3
        idx = np.flatnonzero(a1 & m)
        frames.append(pd.DataFrame({
            "pivot_open_ts_ms": ts[idx].astype(np.int64), "direction": direction,
            "confirmable": confirmable[idx], "confirmed": conf[idx] & confirmable[idx],
        }))
    return pd.concat(frames, ignore_index=True).sort_values("pivot_open_ts_ms").reset_index(drop=True)


def compute_maxv_ltf(df_1m: pd.DataFrame, t12: np.ndarray, ltf_min: int) -> np.ndarray:
    """maxV(k) = close LTF-бара с абсолютным max объёмом (любое направление, incl. doji)
    внутри 12h бара k. Векторизовано: агрегируем 1m->LTF, потом idxmax(volume) per 12h bucket."""
    ltf_ms = ltf_min * 60_000
    ts = df_1m["ts"].to_numpy()
    buckets = (ts // ltf_ms) * ltf_ms
    ltf = df_1m.assign(bucket=buckets).groupby("bucket", sort=True, as_index=False).agg(
        close=("close", "last"), volume=("volume", "sum"))
    ltf["bar12"] = (ltf["bucket"] // TF_12H_MS) * TF_12H_MS
    idx = ltf.groupby("bar12")["volume"].idxmax()
    winners = ltf.loc[idx, ["bar12", "close"]].set_index("bar12")["close"]
    return winners.reindex(t12).to_numpy()


def stats_for_pool(pool: pd.DataFrame, sw_short, sw_long, ts_to_idx) -> tuple:
    n_hit = 0; n_c = 0; n_conf = 0
    for row in pool.itertuples():
        idx = ts_to_idx.get(int(row.pivot_open_ts_ms))
        if idx is None: continue
        hit = sw_short[idx] if row.direction == "short" else sw_long[idx]
        if hit:
            n_hit += 1
            if row.confirmable:
                n_c += 1
                if row.confirmed: n_conf += 1
    wr = 100.0 * n_conf / n_c if n_c else 0.0
    return n_hit, n_conf, n_c, wr


def process_symbol(symbol: str) -> dict:
    t0 = time.time()
    df_1m = load_1m(symbol)
    df_12h = agg_12h(df_1m)
    t12 = df_12h["ts"].to_numpy(); h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy(); c12 = df_12h["close"].to_numpy()
    n12 = len(t12)

    pool = compute_a_pool(df_12h)
    train_pool = pool[(pool["pivot_open_ts_ms"] >= TRAIN_START) & (pool["pivot_open_ts_ms"] < TRAIN_END)]
    test_pool = pool[pool["pivot_open_ts_ms"] >= TEST_START]
    full_pool = pool  # ветка 2: весь период без train/test разбивки (сколько есть данных)

    ts_to_idx = {int(t): k for k, t in enumerate(t12)}
    out = {"symbol": symbol, "variants": {}, "load_s": round(time.time() - t0, 1)}

    for ltf_min in LTF_VARIANTS_MIN:
        maxv = compute_maxv_ltf(df_1m, t12, ltf_min)
        mv_prev = np.roll(maxv, 1); mv_prev[0] = np.nan
        valid = ~np.isnan(mv_prev)
        sw_short = np.zeros(n12, dtype=bool); sw_long = np.zeros(n12, dtype=bool)
        sw_short[valid] = (h12[valid] > mv_prev[valid]) & (c12[valid] < mv_prev[valid])
        sw_long[valid] = (l12[valid] < mv_prev[valid]) & (c12[valid] > mv_prev[valid])

        tr = stats_for_pool(train_pool, sw_short, sw_long, ts_to_idx)
        te = stats_for_pool(test_pool, sw_short, sw_long, ts_to_idx)
        fu = stats_for_pool(full_pool, sw_short, sw_long, ts_to_idx)
        out["variants"][ltf_min] = {"train": tr, "test": te, "full": fu}

    out["total_s"] = round(time.time() - t0, 1)
    return out


def main():
    t0 = time.time()
    results = Parallel(n_jobs=len(SYMBOLS), backend="loky")(
        delayed(process_symbol)(sym) for sym in SYMBOLS
    )
    print(f"all symbols done in {time.time()-t0:.1f}s", flush=True)

    by_symbol = {r["symbol"]: r for r in results}
    for sym in SYMBOLS:
        r = by_symbol[sym]
        print(f"  {sym}: loaded in {r['load_s']}s, total {r['total_s']}s", flush=True)

    PROD_SYMBOLS = ["BTC", "ETH", "SOL"]  # только они реально в ASVK-демоне

    def summarize(period_key, symbol_set, title):
        print("\n" + "=" * 110, flush=True)
        print(title, flush=True)
        print(f"{'LTF':>5s} {'WR mean':>9s} {'WR min':>8s} {'WR max':>8s} {'total n':>9s}", flush=True)
        rows = []
        for ltf_min in LTF_VARIANTS_MIN:
            wrs = [by_symbol[s]["variants"][ltf_min][period_key][3] for s in symbol_set]
            total_n = sum(by_symbol[s]["variants"][ltf_min][period_key][0] for s in symbol_set)
            row = {"ltf": ltf_min, "mean": np.mean(wrs), "min": np.min(wrs),
                  "max": np.max(wrs), "total_n": total_n}
            rows.append(row)
            print(f"{ltf_min:>5d} {row['mean']:>8.2f}% {row['min']:>7.2f}% "
                  f"{row['max']:>7.2f}% {row['total_n']:>9d}", flush=True)
        best = max(rows, key=lambda r: r["mean"])
        univ = max(rows, key=lambda r: r["min"])
        print(f"Лучший по mean: LTF={best['ltf']}m ({best['mean']:.2f}%)   "
              f"Лучший по min: LTF={univ['ltf']}m ({univ['min']:.2f}%)", flush=True)
        return rows, best, univ

    rows1, best1, univ1 = summarize("test", SYMBOLS, "ВЕТКА 1 — TEST (2025-now), 11 активов")
    rows1p, best1p, univ1p = summarize("test", PROD_SYMBOLS, "ВЕТКА 1p — TEST (2025-now), ТОЛЬКО BTC/ETH/SOL")
    rows2, best2, univ2 = summarize("full", SYMBOLS, "ВЕТКА 2 — весь период, 11 активов")
    rows2p, best2p, univ2p = summarize("full", PROD_SYMBOLS, "ВЕТКА 2p — весь период, ТОЛЬКО BTC/ETH/SOL")

    print("\n" + "=" * 110, flush=True)
    print("СРАВНЕНИЕ веток (топ-кандидаты)", flush=True)
    candidates = sorted({best1["ltf"], univ1["ltf"], best2["ltf"], univ2["ltf"],
                         best1p["ltf"], univ1p["ltf"], best2p["ltf"], univ2p["ltf"]})
    print(f"Кандидаты для per-symbol разбора: {candidates}", flush=True)

    for ltf_show in candidates:
        print("\n" + "=" * 110, flush=True)
        print(f"Per-symbol breakdown для LTF={ltf_show}m", flush=True)
        print(f"{'symbol':8s} {'train n':>8s} {'train WR':>9s} {'test n':>7s} {'test WR':>8s} "
              f"{'full n':>7s} {'full WR':>8s}", flush=True)
        for sym in SYMBOLS:
            tr = by_symbol[sym]["variants"][ltf_show]["train"]
            te = by_symbol[sym]["variants"][ltf_show]["test"]
            fu = by_symbol[sym]["variants"][ltf_show]["full"]
            print(f"{sym:8s} {tr[0]:>8d} {tr[3]:>8.2f}% {te[0]:>7d} {te[3]:>7.2f}% "
                  f"{fu[0]:>7d} {fu[3]:>7.2f}%", flush=True)


if __name__ == "__main__":
    main()
