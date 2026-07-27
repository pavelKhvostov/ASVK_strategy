import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "smc-warehouse/scripts/фрактал-12h"))
from common import load_1m, agg_12h, WAREHOUSE, TF_12H_MS
from a_cascade import LEFT_EXT_N, BODY_MAX, WICK_MIN

SYMBOLS = ["ETH", "SOL", "ADA", "AVAX", "BNB", "DOGE", "DOT", "LINK", "LTC", "XRP"]
START_MS = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
END_MS = int(pd.Timestamp("2026-07-19", tz="UTC").timestamp() * 1000)

# ── B8 force framework constants (verbatim из b8_power_zone.py) ──
TF_WEIGHT = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24, "2d": 48, "3d": 72}
CLASS_MAP = {
    "ob": "block", "ob_vc": "block", "block_orders": "block",
    "breaker_block": "block", "mitigation_block": "block", "rb": "block",
    "fvg": "inefficiency", "i_fvg": "inefficiency",
    "rdrb": "inefficiency", "i_rdrb": "inefficiency", "marubozu": "inefficiency",
    "fractal": "liquidity", "ob_liq": "liquidity",
}
CLASS_W = {"block": 3, "inefficiency": 2, "liquidity": 1}
MIT_MODEL_W = {"sweep": 0.5, "first_touch": 1.0, "wick_fill": 0.7}
ELEMENT_MIT_MODEL = {"fractal": "sweep", "marubozu": "sweep", "rb": "first_touch", "ob_liq": "first_touch"}
PROXIMITY_PCT = 3.0
PROXIMITY_FLOOR = 0.3
THR_NET_FL = -1000
THR_NET_FH = +500
THR_NET_W2 = -2000


def independent_masks(df12: pd.DataFrame) -> dict:
    o, h = df12["open"].to_numpy(), df12["high"].to_numpy()
    l, c = df12["low"].to_numpy(), df12["close"].to_numpy()
    T = len(df12)
    hl = h - l
    body = np.abs(c - o)
    up_wick = h - np.maximum(o, c)
    lo_wick = np.minimum(o, c) - l
    color = np.where(c > o, 1, np.where(c < o, -1, 0))
    non_doji = color != 0
    body_pct = np.divide(body, hl, out=np.zeros_like(body), where=(hl > 0))
    up_pct = np.divide(up_wick, hl, out=np.zeros_like(up_wick), where=(hl > 0))
    lo_pct = np.divide(lo_wick, hl, out=np.zeros_like(lo_wick), where=(hl > 0))

    opp = np.zeros(T, dtype=bool)
    opp[1:] = (color[1:] != color[:-1]) & non_doji[1:] & non_doji[:-1]
    three = np.zeros(T, dtype=bool)
    three[2:] = ((color[2:] == color[1:-1]) & (color[2:] == color[:-2])
                 & non_doji[2:] & non_doji[1:-1] & non_doji[:-2])
    a3_mask = opp | three

    out = {}
    for d, arr, wick_pct in (("short", h, up_pct), ("long", l, lo_pct)):
        a1 = np.zeros(T, dtype=bool)
        if d == "short":
            a1[2:] = (h[2:] > h[1:-1]) & (h[2:] > h[:-2])
            ext = pd.Series(h).rolling(LEFT_EXT_N).max().shift(1).to_numpy()
            a2 = np.where(np.isnan(ext), False, h > ext)
        else:
            a1[2:] = (l[2:] < l[1:-1]) & (l[2:] < l[:-2])
            ext = pd.Series(l).rolling(LEFT_EXT_N).min().shift(1).to_numpy()
            a2 = np.where(np.isnan(ext), False, l < ext)
        a4 = (body_pct <= BODY_MAX) & (wick_pct >= WICK_MIN)
        out[d] = {"A1": a1, "A2": a2, "A3": a3_mask, "A4": a4}
    return out


def compute_net_per_anchor(sub: pd.DataFrame) -> tuple:
    if len(sub) == 0:
        return 0.0, 0.0
    dist_abs = np.abs(sub["distance_signed_pct"].to_numpy())
    mask = dist_abs < PROXIMITY_PCT
    if not mask.any():
        return 0.0, 0.0
    df = sub.iloc[mask]
    tf_w = df["tf"].map(TF_WEIGHT).fillna(0).to_numpy()
    age_h = df["age_ms"].to_numpy() / 3_600_000.0
    age_f = 1 + (np.maximum(age_h, 0) / 24.0) ** 0.4
    cls = df["element"].map(CLASS_MAP).fillna("block")
    cls_w = cls.map(CLASS_W).fillna(3).to_numpy()
    dist = np.abs(df["distance_signed_pct"].to_numpy())
    prox = np.maximum(PROXIMITY_FLOOR, 1 - dist / PROXIMITY_PCT)
    mit_mod = df["element"].map(ELEMENT_MIT_MODEL).fillna("wick_fill")
    mit_w = mit_mod.map(MIT_MODEL_W).fillna(0.7).to_numpy()
    strength = tf_w * age_f * cls_w * prox * mit_w
    direction = df["direction"].to_numpy()
    buyer = float(strength[direction == "long"].sum())
    seller = float(strength[direction == "short"].sum())
    return buyer, seller


def load_s7d(symbol: str) -> pd.DataFrame:
    s7d_dir = WAREHOUSE / "data" / "s7d"
    cands = sorted(s7d_dir.glob(f"snapshots_s7d_{symbol}_*.parquet"), key=lambda p: p.stat().st_mtime)
    return pd.read_parquet(cands[-1], columns=["anchor_ts", "zone_id", "element", "tf",
                                                "direction", "distance_signed_pct", "age_ms"])


def run_symbol(symbol: str):
    df_1m = load_1m(symbol)
    df12 = agg_12h(df_1m)
    t12 = df12["ts"].to_numpy()
    h12, l12, c12 = (df12[x].to_numpy() for x in ("high", "low", "close"))
    T = len(df12)

    masks = independent_masks(df12)
    cf_s = np.zeros(T, dtype=bool)
    cf_s[:-2] = (h12[1:-1] < h12[:-2]) & (h12[2:] < h12[:-2])
    cf_l = np.zeros(T, dtype=bool)
    cf_l[:-2] = (l12[1:-1] > l12[:-2]) & (l12[2:] > l12[:-2])
    conf = {"short": cf_s, "long": cf_l}
    confirmable = np.arange(T) <= T - 3
    in_win = (t12 >= START_MS) & (t12 < END_MS)

    s7d = load_s7d(symbol)

    opts = ["A2", "A3", "A4"]
    print(f"\n=== {symbol} ===")
    print(f"{'domain':16s} {'dom_n':>7s} | {'B8C1 n':>7s} {'conf':>6s} {'WR':>7s}")
    for r in range(4):
        for combo in itertools.combinations(opts, r):
            name = "+".join(("A1",) + combo)
            rows = []
            for d in ("short", "long"):
                m = masks[d]["A1"] & in_win
                for cflt in combo:
                    m &= masks[d][cflt]
                idx = np.flatnonzero(m)
                for i in idx:
                    rows.append((int(t12[i]), d, i, bool(conf[d][i]) and bool(confirmable[i]), bool(confirmable[i])))
            if not rows:
                print(f"{name:16s} {0:7,} | {0:7,} {0:6,} {0:6.2f}%")
                continue
            pivots = pd.DataFrame(rows, columns=["ts", "direction", "idx12", "confirmed", "confirmable"])

            ts_i = pivots["ts"].to_numpy() + TF_12H_MS
            ts_im1 = pivots["ts"].to_numpy()
            all_anchors = set(int(x) for x in ts_i) | set(int(x) for x in ts_im1)
            s7d_f = s7d[s7d["anchor_ts"].isin(all_anchors)]
            anchor_net = {}
            for anchor, sub in s7d_f.groupby("anchor_ts"):
                b, s = compute_net_per_anchor(sub)
                anchor_net[int(anchor)] = (b, s, b - s)

            n_hit = 0
            n_hit_conf_able = 0
            n_hit_conf_yes = 0
            for _, row in pivots.iterrows():
                ts_p = int(row["ts"])
                ts_c = ts_p + TF_12H_MS
                direction = row["direction"]
                _, _, net = anchor_net.get(ts_c, (0.0, 0.0, 0.0))
                _, _, net_prev = anchor_net.get(ts_p, (0.0, 0.0, 0.0))
                net_w2 = net + net_prev
                c9a = (direction == "long") and (net <= THR_NET_FL)
                c9b = (direction == "short") and (net >= THR_NET_FH)
                c9c = (direction == "long") and (net_w2 <= THR_NET_W2)
                b8c1 = c9a or c9b or c9c
                if b8c1:
                    n_hit += 1
                    if row["confirmable"]:
                        n_hit_conf_able += 1
                        if row["confirmed"]:
                            n_hit_conf_yes += 1
            wr = 100.0 * n_hit_conf_yes / n_hit_conf_able if n_hit_conf_able else 0.0
            print(f"{name:16s} {len(pivots):7,} | {n_hit:7,} {n_hit_conf_yes:6,} {wr:6.2f}%")


for sym in SYMBOLS:
    run_symbol(sym)
