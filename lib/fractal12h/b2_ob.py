"""b2_ob — B2 Order Block sub-basket (ASVK-portable, B2C1 + B2C2).

Портировано из ~/smc-warehouse/scripts/фрактал-12h/b2_ob.py (WSL, read-only источник).
B2C1 (block_orders) и B2C2 (ob_liq) — два разных элемента (см. структура_BTC.png:
B2C1 "OB · multi-TF", B2C2 "ob_liq · multi-TF"), считаются оба, но НЕ объединяются:
b2_hit (production/basket signal) = чистый B2C1, как и был — B2C1 уже разобран и
подтверждён, его не трогаем. b2c2 — отдельная исследовательская колонка, не влияет
на b2_hit и на итоговую корзину B1∪...∪B9.

B2C1: FIRST 50%-sweep block_orders zones, multi-TF {12h,1d,2d,3d,1w}, block length
L=last_bar_idx-preceding_idx+1 ≤ 8 (canon-фильтр из meta).

B2C2: FIRST 50%-sweep ob_liq zones, multi-TF {12h,1d,2d,3d,1w} — hit если fs50
совпадает либо на LIQ-marker зоне (primary, узкая), либо на entry_zone (drop/rally
area, парсится из meta) — обе зоны зоны каждого ob_liq события проверяются
независимо. Никакого нового детектора не нужно — block_orders и ob_liq уже
считаются в e12d.py на всех нужных TF.

Домен (оба sub-условия): A1 pre-w pool (НЕ полный A1..A4 каскад) — как B1/B9. Канон
явно документирует A2/A3/A4 как "informational only" (WSL
скрипт_структура_фрактал_12h.py: "A1-anchor pool ... A2/A3/A4 — informational only"),
и эмпирически на BTC подтвердилось для B2C1: WR держится ~78-80% на всех 8
комбинациях A1..A4, а сигнал теряется (169→94, -44%) без роста WR. Проверено на 11
активах (BTC+ETH+SOL+ADA+AVAX+BNB+DOGE+DOT+LINK+LTC+XRP): pooled n=1,969 WR=76.54%,
ни один актив не проваливается к базовому уровню (~41-48%).

Semantic (из pred12h_basket_c1c2c3.py):
  mid = (zone_lo + zone_hi) / 2
  SHORT: high[k] >= mid AND close[k] < zone_lo    (проекция 50% + rejection вниз)
  LONG:  low[k]  <= mid AND close[k] > zone_hi    (проекция 50% + rejection вверх)
  Первое такое k = fs50 для этой зоны. Одна зона fires максимум один раз.

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/events/events_e12d_{SYM}_*.parquet (последний по mtime, через latest_events_path)
  data/{SYM}USDT_1m.csv
Writes:
  data/fractal12h/b2_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import ast
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_1m, agg_12h, DATA_OUT, latest_events_path


OB_TFS = ("12h", "1d", "2d", "3d", "1w")


# ── Loaders ───────────────────────────────────────────────

def load_block_orders_zones(symbol: str) -> pd.DataFrame:
    """block_orders born-events из последнего events_e12d_{symbol}_*.parquet (ASVK e12d),
    только canonical TFs + block length ≤ 8 (canon-фильтр по meta)."""
    p = latest_events_path(symbol)
    df = pd.read_parquet(p)
    df = df[(df["element"] == "block_orders") & (df["kind"] == "born") & df["tf"].isin(OB_TFS)]
    grouped = (df.groupby("zone_id", as_index=False).first()
                 [["zone_id", "ts", "tf", "direction", "zone_lo", "zone_hi", "meta"]]
                 .rename(columns={"ts": "born_ts"}))

    def _block_L(m):
        try:
            d = ast.literal_eval(m) if isinstance(m, str) else m
            return d["last_bar_idx"] - d["preceding_idx"] + 1
        except Exception:
            return 999

    grouped["_L"] = grouped["meta"].map(_block_L)
    return grouped[grouped["_L"] <= 8].drop(columns=["meta", "_L"]).reset_index(drop=True)


def load_ob_liq_zones(symbol: str) -> pd.DataFrame:
    """ob_liq born-events из последнего events_e12d_{symbol}_*.parquet (ASVK e12d).

    Возвращает zone_id, born_ts, tf, direction, liq_lo/liq_hi (LIQ-marker, узкая
    primary-зона) и entry_lo/entry_hi (drop/rally area из meta, если есть)."""
    p = latest_events_path(symbol)
    df = pd.read_parquet(p)
    df = df[(df["element"] == "ob_liq") & (df["kind"] == "born") & df["tf"].isin(OB_TFS)]
    grouped = (df.groupby("zone_id", as_index=False).first()
                 [["zone_id", "ts", "tf", "direction", "zone_lo", "zone_hi", "meta"]]
                 .rename(columns={"ts": "born_ts", "zone_lo": "liq_lo", "zone_hi": "liq_hi"}))
    entry_lo, entry_hi = [], []
    for m in grouped["meta"]:
        try:
            d = ast.literal_eval(m) if isinstance(m, str) else m
            ez = d.get("entry_zone")
            if ez is None or len(ez) != 2:
                entry_lo.append(np.nan); entry_hi.append(np.nan)
            else:
                entry_lo.append(float(ez[0])); entry_hi.append(float(ez[1]))
        except Exception:
            entry_lo.append(np.nan); entry_hi.append(np.nan)
    grouped["entry_lo"] = entry_lo
    grouped["entry_hi"] = entry_hi
    return grouped.drop(columns=["meta"]).reset_index(drop=True)


# ── First 50%-sweep (per zone) ─────────────────────────────

def first_sweep50_idx(zone_lo: float, zone_hi: float, direction: str, born_ts: int,
                       t12: np.ndarray, h12: np.ndarray, l12: np.ndarray, c12: np.ndarray) -> int | None:
    """Первый bar с 50%-sweep + close outside. None если не случилось."""
    if not np.isfinite(zone_lo) or not np.isfinite(zone_hi):
        return None
    if zone_hi - zone_lo <= 0:
        return None
    n12 = len(t12)
    sp = int(np.searchsorted(t12, int(born_ts), side="left"))
    if sp >= n12:
        return None
    mid = (zone_lo + zone_hi) / 2.0
    hh, ll, cc = h12[sp:], l12[sp:], c12[sp:]
    if direction == "short":
        mask = (hh >= mid) & (cc < zone_lo)
    else:
        mask = (ll <= mid) & (cc > zone_hi)
    if not mask.any():
        return None
    return int(sp + int(np.argmax(mask)))


def eval_b2c1(zones: pd.DataFrame,
              t12: np.ndarray, h12: np.ndarray, l12: np.ndarray, c12: np.ndarray) -> set[tuple]:
    """B2C1: FIRST 50%-sweep block_orders. Fires set of (bar_idx, direction)."""
    fires = set()
    for z in zones.to_dict(orient="records"):
        fs = first_sweep50_idx(z["zone_lo"], z["zone_hi"], z["direction"], z["born_ts"],
                                t12, h12, l12, c12)
        if fs is not None:
            fires.add((fs, z["direction"]))
    return fires


def eval_b2c2(zones: pd.DataFrame,
              t12: np.ndarray, h12: np.ndarray, l12: np.ndarray, c12: np.ndarray) -> set[tuple]:
    """B2C2: FIRST 50%-sweep ob_liq — hit если fs50 совпадает либо на LIQ, либо на
    entry_zone. Canon: обе зоны каждого события проверяются независимо."""
    fires = set()
    for z in zones.to_dict(orient="records"):
        fs_liq = first_sweep50_idx(z["liq_lo"], z["liq_hi"], z["direction"], z["born_ts"],
                                    t12, h12, l12, c12)
        if fs_liq is not None:
            fires.add((fs_liq, z["direction"]))
        fs_ob = first_sweep50_idx(z["entry_lo"], z["entry_hi"], z["direction"], z["born_ts"],
                                   t12, h12, l12, c12)
        if fs_ob is not None:
            fires.add((fs_ob, z["direction"]))
    return fires


# ── Assembly ──────────────────────────────────────────────

def compute_b2(a_cand: pd.DataFrame, bo_zones: pd.DataFrame, ol_zones: pd.DataFrame,
               df_12h: pd.DataFrame) -> pd.DataFrame:
    t12 = df_12h["ts"].to_numpy()
    h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy()
    c12 = df_12h["close"].to_numpy()

    print(f"  eval B2C1 (block_orders) on {len(bo_zones):,} zones...", file=sys.stderr, flush=True)
    t0 = time.time()
    B2C1 = eval_b2c1(bo_zones, t12, h12, l12, c12)
    print(f"    {len(B2C1):,} fires in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    print(f"  eval B2C2 (ob_liq) on {len(ol_zones):,} zones...", file=sys.stderr, flush=True)
    t0 = time.time()
    B2C2 = eval_b2c2(ol_zones, t12, h12, l12, c12)
    print(f"    {len(B2C2):,} fires in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    ts_to_idx = {int(t): k for k, t in enumerate(t12)}
    a1 = a_cand[a_cand["a1_pre_w"]].copy()   # A1 pool — см. докстринг: A2/A3/A4 informational only
    rows = []
    for _, row in a1.iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        idx = ts_to_idx.get(ts_pivot)
        if idx is None:
            continue
        key = (idx, direction)
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "b2c1":   key in B2C1,
            "b2c2":   key in B2C2,   # отдельная секция — НЕ входит в b2_hit/basket, см. докстринг
            "b2_hit": key in B2C1,   # production/basket signal — остаётся чистым B2C1, не трогаем
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b2c1", "b2c2", "b2_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        marker = "  ← B2 (production)" if col == "b2_hit" else ("  (не в basket)" if col == "b2c2" else "")
        print(f"  {col:8s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{marker}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"b2_ob (B2C1+B2C2, A1 pool): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    n_a1 = int(a_cand["a1_pre_w"].sum())
    print(f"  A1 candidates domain: {n_a1:,}", file=sys.stderr, flush=True)

    bo_zones = load_block_orders_zones(args.symbol)
    ol_zones = load_ob_liq_zones(args.symbol)
    print(f"  block_orders zones (TFs={OB_TFS}): {len(bo_zones):,}", file=sys.stderr, flush=True)
    print(f"  ob_liq zones (TFs={OB_TFS}):       {len(ol_zones):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_b2(a_cand, bo_zones, ol_zones, df_12h)
    print_stats(hits)

    out = DATA_OUT / f"b2_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
