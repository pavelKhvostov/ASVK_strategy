import ast, sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / "smc-warehouse/scripts/фрактал-12h"))
import numpy as np
import pandas as pd
from common import load_1m, agg_12h, WAREHOUSE, DATA_OUT

OB_TFS = ("12h", "1d", "2d", "3d", "1w")
EVENTS_END = "2026-07-19"
SYMBOLS = ["ETH", "SOL", "ADA", "AVAX", "BNB", "DOGE", "DOT", "LINK", "LTC", "XRP"]


def load_block_orders_zones(symbol):
    p = WAREHOUSE / f"data/events/events_e12d_{symbol}_2018-01-01_{EVENTS_END}.parquet"
    df = pd.read_parquet(p)
    df = df[(df["element"] == "block_orders") & (df["kind"] == "born") & df["tf"].isin(OB_TFS)]
    g = (df.groupby("zone_id", as_index=False).first()
           [["zone_id", "ts", "tf", "direction", "zone_lo", "zone_hi", "meta"]]
           .rename(columns={"ts": "born_ts"}))

    def _L(m):
        try:
            d = ast.literal_eval(m) if isinstance(m, str) else m
            return d["last_bar_idx"] - d["preceding_idx"] + 1
        except Exception:
            return 999

    g["_L"] = g["meta"].map(_L)
    return g[g["_L"] <= 8].drop(columns=["meta", "_L"]).reset_index(drop=True)


def first_sweep50_idx(zone_lo, zone_hi, direction, born_ts, t12, h12, l12, c12):
    if not np.isfinite(zone_lo) or not np.isfinite(zone_hi) or zone_hi - zone_lo <= 0:
        return None
    sp = int(np.searchsorted(t12, int(born_ts), side="left"))
    if sp >= len(t12):
        return None
    mid = (zone_lo + zone_hi) / 2.0
    hh, ll, cc = h12[sp:], l12[sp:], c12[sp:]
    mask = (hh >= mid) & (cc < zone_lo) if direction == "short" else (ll <= mid) & (cc > zone_hi)
    if not mask.any():
        return None
    return int(sp + int(np.argmax(mask)))


def eval_b2c1(zones, t12, h12, l12, c12):
    fires = set()
    for z in zones.to_dict(orient="records"):
        fs = first_sweep50_idx(z["zone_lo"], z["zone_hi"], z["direction"], z["born_ts"], t12, h12, l12, c12)
        if fs is not None:
            fires.add((fs, z["direction"]))
    return fires


print(f"{'symbol':8s} {'A1_n':>6s} {'bo_zones':>9s} {'b2c1_n':>7s} {'conf':>6s} {'WR':>7s} | "
      f"{'long_n':>7s} {'long_WR':>8s} | {'short_n':>7s} {'short_WR':>8s}")
for sym in SYMBOLS:
    a_files = sorted(DATA_OUT.glob(f"a_candidates_{sym}_2020-01-01_*.parquet"))
    if not a_files:
        print(f"{sym:8s}  NO a_candidates")
        continue
    a_cand = pd.read_parquet(a_files[-1])
    bo_zones = load_block_orders_zones(sym)

    df_1m = load_1m(sym)
    df_12h = agg_12h(df_1m)
    t12 = df_12h["ts"].to_numpy()
    h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy()
    c12 = df_12h["close"].to_numpy()

    B2C1 = eval_b2c1(bo_zones, t12, h12, l12, c12)

    ts_to_idx = {int(t): k for k, t in enumerate(t12)}
    a1 = a_cand[a_cand["a1_pre_w"]].copy()
    rows = []
    for _, row in a1.iterrows():
        idx = ts_to_idx.get(int(row["pivot_open_ts_ms"]))
        if idx is None:
            continue
        key = (idx, row["direction"])
        rows.append({"direction": row["direction"], "confirmable": bool(row["confirmable"]),
                     "confirmed": bool(row["confirmed"]), "b2c1": key in B2C1})
    hits = pd.DataFrame(rows)

    def stats(sub):
        m = sub["b2c1"]
        n = int(m.sum())
        cm = m & sub["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(sub.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        return n, n_conf, n_c, wr

    n, n_conf, n_c, wr = stats(hits)
    ln, lconf, lnc, lwr = stats(hits[hits["direction"] == "long"])
    sn, sconf, snc, swr = stats(hits[hits["direction"] == "short"])
    print(f"{sym:8s} {len(a1):6,} {len(bo_zones):9,} {n:7,} {n_conf:3,}/{n_c:<3,} {wr:6.2f}% | "
          f"{ln:7,} {lwr:7.2f}% | {sn:7,} {swr:7.2f}%")
