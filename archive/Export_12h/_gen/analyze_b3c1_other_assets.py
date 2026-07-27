"""Разовый анализ: B3C1 (maxV sweep, без depth) на пуле A1+A2+A4 (skip A3),
для активов, которых нет в ASVK (только BTC/ETH/SOL заведены в демон).

Читает ТОЛЬКО ~/smc-warehouse/график/{SYM}USDT_1m.csv (read-only, ничего не пишет
обратно в WSL warehouse). Self-contained, повторяет логику a_cascade.py + maxv.py
один в один (без импортов из ASVK, чтобы не зависеть от Windows-путей).
"""
from __future__ import annotations
import pathlib
import sys

import numpy as np
import pandas as pd

GRAPHIC = pathlib.Path.home() / "smc-warehouse" / "график"

TF_12H_MS = 12 * 60 * 60 * 1000
LEFT_EXT_N = 5
BODY_MAX = 0.80
WICK_MIN = 0.03
ATR_N = 14

SYMBOLS = ["ADA", "AVAX", "BNB", "DOGE", "DOT", "LINK", "LTC", "XRP"]


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
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    return g.rename(columns={"bucket": "ts"})


def atr14_sma(h, l, c):
    n = len(h)
    c_prev = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - c_prev), np.abs(l - c_prev)])
    tr[0] = 0.0
    csum = np.cumsum(tr)
    atr = np.zeros(n)
    atr[ATR_N:] = (csum[ATR_N:] - csum[:n - ATR_N]) / ATR_N
    return atr


def compute_maxv(df_1m, t12):
    ts_1m = df_1m["ts"].to_numpy(); o_1m = df_1m["open"].to_numpy()
    c_1m = df_1m["close"].to_numpy(); v_1m = df_1m["volume"].to_numpy()
    is_bull = c_1m > o_1m; is_bear = c_1m < o_1m
    bar_of_1m = ((ts_1m // TF_12H_MS) * TF_12H_MS).astype(np.int64)
    n12 = len(t12)
    maxv = np.full(n12, np.nan)
    ts_to_idx = {int(t): k for k, t in enumerate(t12)}
    df1 = pd.DataFrame({'bar_ts': bar_of_1m, 'c': c_1m, 'v': v_1m, 'bull': is_bull, 'bear': is_bear})
    for bar_ts, g in df1.groupby('bar_ts'):
        idx = ts_to_idx.get(int(bar_ts))
        if idx is None: continue
        bulls = g[g.bull]; bears = g[g.bear]
        mb = bulls.v.max() if len(bulls) else 0
        mr = bears.v.max() if len(bears) else 0
        if mb == 0 and mr == 0: continue
        row = bulls.loc[bulls.v.idxmax()] if mb >= mr else bears.loc[bears.v.idxmax()]
        maxv[idx] = row.c
    return maxv


def compute_a_cascade(df_12h):
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

    a1_fh = np.zeros(T, dtype=bool)
    a1_fh[2:] = (h[2:] > h[1:-1]) & (h[2:] > h[:-2])
    h_ser = pd.Series(h)
    left5_max = h_ser.rolling(LEFT_EXT_N).max().shift(1).to_numpy()
    a2_fh = a1_fh & np.where(np.isnan(left5_max), False, h > left5_max)
    conf_fh = np.zeros(T, dtype=bool)
    conf_fh[:-2] = (h[1:-1] < h[:-2]) & (h[2:] < h[:-2])

    a1_fl = np.zeros(T, dtype=bool)
    a1_fl[2:] = (l[2:] < l[1:-1]) & (l[2:] < l[:-2])
    l_ser = pd.Series(l)
    left5_min = l_ser.rolling(LEFT_EXT_N).min().shift(1).to_numpy()
    a2_fl = a1_fl & np.where(np.isnan(left5_min), False, l < left5_min)
    conf_fl = np.zeros(T, dtype=bool)
    conf_fl[:-2] = (l[1:-1] > l[:-2]) & (l[2:] > l[:-2])

    confirmable = np.arange(T) <= T - 3

    frames = []
    for direction, a1, a2, conf, wick_pct in [
        ("short", a1_fh, a2_fh, conf_fh, up_wick_pct),
        ("long", a1_fl, a2_fl, conf_fl, lo_wick_pct),
    ]:
        idx = np.flatnonzero(a1)
        frames.append(pd.DataFrame({
            "pivot_open_ts_ms": ts[idx].astype(np.int64), "direction": direction,
            "a2_ext_5": a2[idx], "confirmable": confirmable[idx],
            "confirmed": conf[idx] & confirmable[idx],
            "body_pct": body_pct[idx], "wick_pct": wick_pct[idx],
        }))
    return pd.concat(frames, ignore_index=True).sort_values("pivot_open_ts_ms").reset_index(drop=True)


def analyze(symbol: str) -> dict:
    df_1m = load_1m(symbol)
    df_12h = agg_12h(df_1m)
    t12 = df_12h["ts"].to_numpy(); h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy(); c12 = df_12h["close"].to_numpy()

    cand = compute_a_cascade(df_12h)
    pool = cand[cand["a2_ext_5"] & (cand["body_pct"] <= BODY_MAX) & (cand["wick_pct"] >= WICK_MIN)]

    maxv = compute_maxv(df_1m, t12)
    n12 = len(t12)
    mv_prev = np.roll(maxv, 1); mv_prev[0] = np.nan
    valid = ~np.isnan(mv_prev)
    sw_short = np.zeros(n12, dtype=bool); sw_long = np.zeros(n12, dtype=bool)
    sw_short[valid] = (h12[valid] > mv_prev[valid]) & (c12[valid] < mv_prev[valid])
    sw_long[valid] = (l12[valid] < mv_prev[valid]) & (c12[valid] > mv_prev[valid])

    ts_to_idx = {int(t): k for k, t in enumerate(t12)}
    n_hit = 0; n_c = 0; n_conf = 0
    for _, row in pool.iterrows():
        idx = ts_to_idx.get(int(row["pivot_open_ts_ms"]))
        if idx is None: continue
        hit = sw_short[idx] if row["direction"] == "short" else sw_long[idx]
        if hit:
            n_hit += 1
            if row["confirmable"]:
                n_c += 1
                if row["confirmed"]: n_conf += 1
    wr = 100.0 * n_conf / n_c if n_c else 0.0
    return {"symbol": symbol, "pool_n": len(pool), "n": n_hit, "conf": n_conf, "n_c": n_c, "wr": wr}


def main():
    print(f"{'symbol':8s} {'pool_n':>7s} {'n':>5s} {'conf/n_c':>10s} {'WR':>7s}", file=sys.stderr)
    results = []
    for sym in SYMBOLS:
        try:
            r = analyze(sym)
            results.append(r)
            print(f"{r['symbol']:8s} {r['pool_n']:>7d} {r['n']:>5d} "
                  f"{r['conf']:>4d}/{r['n_c']:<5d} {r['wr']:>6.2f}%", file=sys.stderr)
        except Exception as e:
            print(f"{sym:8s} ERROR: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
