"""B8 Power Zone — sub-basket B8C1 (Reverse Force Divergence, ∪3), ASVK-portable.

Портировано из ~/smc-warehouse/scripts/фрактал-12h/b8_power_zone.py (WSL, read-only
источник). Никакого нового детектора не требуется — источник данных (`snapshots_s7d`)
уже считается в ASVK каждый цикл (`s7d.py`), схема колонок совпадает буква в букву
с тем, что нужно здесь.

Формула (Phase 4 force framework, canon):
  strength(zone) = TF_weight × age_factor × class_weight × proximity × mit_model_w

  TF_WEIGHT: 1h=1, 2h=2, 4h=4, 6h=6, 12h=12, 1d=24, 2d=48, 3d=72 (canon включает
    ещё 8h=8, которого у нас нет в e12d; ASVK e12d считает больше TF, чем canon —
    15m/30m/1w не входят в TF_WEIGHT, formula сама даёт им вес 0 через fillna(0),
    отдельная фильтрация не нужна)
  age_factor = 1 + (age_hours / 24)^0.4
  CLASS_W: block(ob,ob_vc,block_orders,breaker_block,mitigation_block,rb)=3,
           inefficiency(fvg,i_fvg,rdrb,i_rdrb,marubozu)=2, liquidity(fractal,ob_liq)=1
  proximity = max(0.3, 1 - |distance_pct| / 3.0)   (±3% band, ниже — zone отброшена)
  mit_model_w: sweep(fractal,marubozu)=0.5, first_touch(rb,ob_liq)=1.0, wick_fill(проч.)=0.7

Per 12h bar close:
  buyer_total(i)  = Σ strength(z) для LONG zones (support)
  seller_total(i) = Σ strength(z) для SHORT zones (resistance)
  net(i)          = buyer_total - seller_total
  net_w2(i)       = net(i) + net(i-1)

B8C1 threshold rules:
  c9a: direction=long  AND net ≤ -1000     (FL seller exhaustion)
  c9b: direction=short AND net ≥ +500      (FH buyer exhaustion)
  c9c: direction=long  AND net_w2 ≤ -2000  (FL 2-bar seller dominance)
  B8C1 = c9a ∨ c9b ∨ c9c

Домен: A1+A2 (a1_pre_w & a2_indep) — единственный блок во всей серии (B1/B2/B3/B4/
B5/B9), где A-фильтр реально несёт информацию, а не просто режет выборку: проверка
на всех 11 активах (BTC+ETH+SOL+ADA+AVAX+BNB+DOGE+DOT+LINK+LTC+XRP) показала
устойчивый рост WR при добавлении A2 (pooled 70.47%→82.81%, 10 из 11 активов), а не
шум/переобучение под один актив. A3/A4 почти не влияют (в отличие от A2) — по
решению пользователя берём именно A1+A2, без A4 (не a124_pool, как у B3/B4/B5).
WSL-исходник сидел на a4_body_wick — та же ошибка, что была у B3/B4/B5 до фикса.

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/s7d/snapshots_s7d_{SYM}_2018-01-01_{end}.parquet (последний по дате, читаем
    только 7 нужных колонок — 38.6М строк на BTC читаются за <1с)
Writes:
  data/fractal12h/b8_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import DATA_OUT, BASE, TF_12H_MS


# ── Canonical force framework constants (verbatim из WSL) ──
TF_WEIGHT = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12,
             "1d": 24, "2d": 48, "3d": 72}   # canon имеет ещё 8h=8, у нас нет

CLASS_MAP = {
    "ob": "block", "ob_vc": "block", "block_orders": "block",
    "breaker_block": "block", "mitigation_block": "block", "rb": "block",
    "fvg": "inefficiency", "i_fvg": "inefficiency",
    "rdrb": "inefficiency", "i_rdrb": "inefficiency", "marubozu": "inefficiency",
    "fractal": "liquidity", "ob_liq": "liquidity",
}
CLASS_W = {"block": 3, "inefficiency": 2, "liquidity": 1}

MIT_MODEL_W = {"sweep": 0.5, "first_touch": 1.0, "wick_fill": 0.7}
ELEMENT_MIT_MODEL = {
    "fractal": "sweep", "marubozu": "sweep",
    "rb": "first_touch", "ob_liq": "first_touch",
}

PROXIMITY_PCT = 3.0
PROXIMITY_FLOOR = 0.3

THR_NET_FL = -1000    # c9a: FL if net ≤ this
THR_NET_FH = +500     # c9b: FH if net ≥ this
THR_NET_W2 = -2000    # c9c: FL 2-bar if net_w2 ≤ this


def load_s7d(symbol: str) -> pd.DataFrame:
    """Последний (по mtime) snapshots_s7d_{symbol}_*.parquet, только нужные колонки."""
    s7d_dir = BASE / "data" / "s7d"
    candidates = sorted(s7d_dir.glob(f"snapshots_s7d_{symbol}_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no snapshots_s7d_{symbol}_*.parquet in {s7d_dir}")
    return pd.read_parquet(
        candidates[-1],
        columns=["anchor_ts", "zone_id", "element", "tf", "direction",
                 "distance_signed_pct", "age_ms"],
    )


def compute_net_per_anchor(s7d_at_anchor: pd.DataFrame) -> tuple[float, float]:
    """(buyer_total, seller_total) для одного anchor snapshot.

    Canon hard-cutoff: |distance_pct| < PROXIMITY_PCT — зоны вне ±3% band
    полностью отбрасываются перед суммированием, внутри — soft weighting."""
    if len(s7d_at_anchor) == 0:
        return 0.0, 0.0
    dist_abs = np.abs(s7d_at_anchor["distance_signed_pct"].to_numpy())
    mask = dist_abs < PROXIMITY_PCT
    if not mask.any():
        return 0.0, 0.0
    df = s7d_at_anchor[mask]

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


def compute_b8(a_cand: pd.DataFrame, s7d: pd.DataFrame) -> pd.DataFrame:
    pool = a_cand[a_cand["a1_pre_w"] & a_cand["a2_indep"]].copy()   # A1+A2 — см. докстринг

    ts_i_close = pool["pivot_open_ts_ms"].to_numpy() + TF_12H_MS   # bar i close
    ts_im1_close = pool["pivot_open_ts_ms"].to_numpy()             # bar i-1 close = bar i open
    all_anchors = set(int(t) for t in ts_i_close) | set(int(t) for t in ts_im1_close)
    print(f"  unique anchors needed: {len(all_anchors):,}", file=sys.stderr, flush=True)

    s7d = s7d[s7d["anchor_ts"].isin(all_anchors)]
    print(f"  s7d rows after filter: {len(s7d):,}", file=sys.stderr, flush=True)

    print(f"  computing per-anchor buyer/seller...", file=sys.stderr, flush=True)
    t0 = time.time()
    anchor_net: dict[int, tuple[float, float, float]] = {}
    for anchor, sub in s7d.groupby("anchor_ts"):
        b, s = compute_net_per_anchor(sub)
        anchor_net[int(anchor)] = (b, s, b - s)
    print(f"    {len(anchor_net):,} anchors in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    rows = []
    for _, row in pool.iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        ts_c = ts_pivot + TF_12H_MS
        ts_p = ts_pivot

        buyer, seller, net = anchor_net.get(ts_c, (0.0, 0.0, 0.0))
        _, _, net_prev = anchor_net.get(ts_p, (0.0, 0.0, 0.0))
        net_w2 = net + net_prev

        c9a = (direction == "long") and (net <= THR_NET_FL)
        c9b = (direction == "short") and (net >= THR_NET_FH)
        c9c = (direction == "long") and (net_w2 <= THR_NET_W2)
        b8c1 = c9a or c9b or c9c

        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "b8c1":  b8c1,
            "b8_hit": b8c1,   # пока B8 = B8C1 (единственный sub, как в WSL-каноне)
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b8c1", "b8_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        marker = "  ← B8" if col == "b8_hit" else ""
        print(f"  {col:8s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{marker}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"b8_power_zone (B8C1 only, A1+A2 pool): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    n_domain = int((a_cand["a1_pre_w"] & a_cand["a2_indep"]).sum())
    print(f"  A1+A2 candidates domain: {n_domain:,}", file=sys.stderr, flush=True)

    s7d = load_s7d(args.symbol)
    print(f"  s7d loaded: {len(s7d):,} rows", file=sys.stderr, flush=True)

    hits = compute_b8(a_cand, s7d)
    print_stats(hits)

    out = DATA_OUT / f"b8_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
