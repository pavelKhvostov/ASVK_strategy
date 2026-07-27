#!/usr/bin/env python3
"""Этап 2 — FVG фильтры (canonical AlexxxFlow, 2026-07-12).

HTF: 12h/1D/2D/3D/1W.

FILTER 1 — IDM (first-touch wickfill свежей HTF FVG):
  Активная HTF FVG (born < cur_open, retire NaN OR retire > cur_close) И
  cur bar range пересекает FVG range (any direction FVG) И
  cur bar — самый первый 4h бар тронувший FVG (first_wick_touch_ts == cur_open).
  → REJECT (наш OB = wickfill свежей HTF FVG).

FILTER 2 — proximity magnet (0.5×ATR, только свежая FVG):
  Свежая FVG (first_wick_touch_ts > cur_open) в направлении SL:
    LONG:  zone_hi ≤ cur.low  AND (cur.low - zone_hi) ≤ 0.5×ATR
    SHORT: zone_lo ≥ cur.high AND (zone_lo - cur.high) ≤ 0.5×ATR
  → REJECT (свежая FVG рядом = магнит на SL).

BTC 8.5 лет: 576 → 486 → 334 valid (~39/год).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ASVK = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ASVK / "data"
EVENTS_DIR = DATA_DIR / "events"
STAGE_DIR = DATA_DIR / "liq_ob4h_vc"

TFS = ["12h", "1d", "2d", "3d", "1w"]  # HTF only, 4h/6h не считаются HTF (унифицировано с 1h POI 2026-07-14)
ATR_MULTIPLIER = 0.5
ATR_PERIOD = 14
TF_MS_4H = 4*3600_000


def find_latest_events(symbol: str) -> Path:
    files = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No events_e12d_{symbol}_*.parquet found in {EVENTS_DIR}")
    return files[-1]


def find_latest_stage1(symbol: str) -> Path:
    files = sorted(STAGE_DIR.glob(f"ob_canonical_4h_v1_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    files = [p for p in files if "_config" not in p.stem]
    if not files:
        raise FileNotFoundError(f"No ob_canonical_4h_v1_{symbol}_*.parquet found in {STAGE_DIR}")
    return files[-1]


def compute_first_wick_touch(fvgs: pd.DataFrame, bars_4h: pd.DataFrame) -> np.ndarray:
    """Для каждой FVG вернуть ts_ms первого 4h бара range которого пересёкся с FVG после born.
    Если ни один не тронул — 2**62."""
    if len(fvgs) == 0:
        return np.array([], dtype=np.int64)
    bar_ts = bars_4h["ts_ms"].values.astype(np.int64)
    bar_l = bars_4h["low"].values.astype(np.float64)
    bar_h = bars_4h["high"].values.astype(np.float64)
    born = fvgs["born_ts"].values.astype(np.int64)
    lo = fvgs["zone_lo"].values.astype(np.float64)
    hi = fvgs["zone_hi"].values.astype(np.float64)
    result = np.full(len(fvgs), 2**62, dtype=np.int64)
    for i in range(len(fvgs)):
        start_idx = int(np.searchsorted(bar_ts, born[i], side="right"))
        if start_idx >= len(bar_ts): continue
        # bar overlap FVG: bar.high >= FVG.zone_lo AND bar.low <= FVG.zone_hi
        mask = (bar_h[start_idx:] >= lo[i]) & (bar_l[start_idx:] <= hi[i])
        idx = np.argmax(mask)
        if mask[idx]:
            result[i] = bar_ts[start_idx + idx]
    return result


def compute_active_bounds_at(zone_lo, zone_hi, direction, born_ts, target_ts,
                             bars_ts, bars_h, bars_l):
    """Compute active portion of FVG at target_ts (see stage2 Liq_OB1h_VC для семантики)."""
    i0 = int(np.searchsorted(bars_ts, born_ts, side="right"))
    i1 = int(np.searchsorted(bars_ts, target_ts, side="left"))
    if i0 >= i1:
        return zone_lo, zone_hi, False
    hs = bars_h[i0:i1]
    ls = bars_l[i0:i1]
    touched = (hs >= zone_lo) & (ls <= zone_hi)
    if not touched.any():
        return zone_lo, zone_hi, False
    if direction == "short":
        vals = np.minimum(hs[touched], zone_hi)
        mit = float(vals.max())
        return mit, zone_hi, mit >= zone_hi
    else:
        vals = np.maximum(ls[touched], zone_lo)
        mit = float(vals.min())
        return zone_lo, mit, mit <= zone_lo


def aggregate_4h_bars(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["open_time","high","low","close"])
    df["ts_ms"] = pd.to_datetime(df["open_time"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
    df["bucket"] = (df["ts_ms"] // TF_MS_4H) * TF_MS_4H
    agg = df.groupby("bucket").agg(high=("high","max"), low=("low","min")).reset_index()
    agg = agg.rename(columns={"bucket":"ts_ms"}).sort_values("ts_ms").reset_index(drop=True)
    return agg


def load_atr_4h(csv_path: Path) -> pd.DataFrame:
    """Compute ATR(14) on 4h aggregated from 1m CSV. Return DataFrame with ts_ms, atr14."""
    df = pd.read_csv(csv_path, usecols=["open_time","high","low","close"])
    df["ts_ms"] = pd.to_datetime(df["open_time"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
    df = df.sort_values("ts_ms").drop_duplicates("ts_ms").reset_index(drop=True)
    df["bucket"] = (df["ts_ms"] // TF_MS_4H) * TF_MS_4H
    agg = df.groupby("bucket").agg(h=("high","max"), l=("low","min"), c=("close","last")).reset_index()
    agg = agg.rename(columns={"bucket":"ts_ms"}).sort_values("ts_ms").reset_index(drop=True)
    prev_c = np.concatenate([[agg["c"].iloc[0]], agg["c"].values[:-1]])
    tr = np.maximum.reduce([
        agg["h"].values - agg["l"].values,
        np.abs(agg["h"].values - prev_c),
        np.abs(agg["l"].values - prev_c),
    ])
    agg["atr14"] = pd.Series(tr).rolling(ATR_PERIOD).mean().values
    return agg[["ts_ms", "atr14"]]


def check_filters(row: pd.Series, fvg_by_tf: dict, atr_lookup: pd.DataFrame,
                  bars_ts=None, bars_h=None, bars_l=None) -> dict:
    """IDM wickfill противоположной FVG + PROX magnet same/any FVG."""
    ts = int(row["cur_open"])            # cur bar open
    cur_close_ts = ts + TF_MS_4H          # cur bar close
    direction = row["direction"]
    cur_o = float(row["cur_o"])
    cur_l = float(row["cur_l"])
    cur_h = float(row["cur_h"])
    cur_c = float(row["cur_c"])

    idx = np.searchsorted(atr_lookup["ts_ms"].values, ts, side="right") - 1
    atr = float(atr_lookup["atr14"].iloc[idx]) if idx >= ATR_PERIOD else np.nan

    idm_hits = 0
    proximity_hits = 0

    for tf, fvgs in fvg_by_tf.items():
        # Active BEFORE cur formation: born < cur_open AND (retire NaN OR retire > cur_close)
        # Живая после cur close = не destroy'ена нашим OB.
        active_all = fvgs[(fvgs["born_ts"] < ts) &
                          (fvgs["retire_ts"].isna() | (fvgs["retire_ts"] > cur_close_ts))]
        if len(active_all) == 0: continue

        # FILTER 1 (IDM, canon 2026-07-14): overlap проверяется против active_lo/hi
        # (не изначальной zone). Любой wick-fill активной части = IDM. Только полное mit исключает.
        candidates = active_all[
            (active_all["direction"] == direction) &
            (active_all["zone_hi"] >= cur_l) &
            (active_all["zone_lo"] <= cur_h)
        ]
        if len(candidates) > 0 and bars_ts is not None:
            for _, fv in candidates.iterrows():
                a_lo, a_hi, full_mit = compute_active_bounds_at(
                    float(fv.zone_lo), float(fv.zone_hi), fv.direction,
                    int(fv.born_ts), ts, bars_ts, bars_h, bars_l,
                )
                if full_mit: continue
                k_up = (cur_c > a_hi) and (cur_o < a_hi)
                k_dn = (cur_c < a_lo) and (cur_o > a_lo)
                if k_up or k_dn: continue
                if cur_h >= a_lo and cur_l <= a_hi:
                    idm_hits += 1

        # FILTER 2 (proximity magnet): свежая FVG (никогда не тронутая до cur_open)
        # в N×ATR. Тронутая fvg уже не магнит.
        fresh = active_all[active_all["first_wick_touch_ts"] > ts]
        if not np.isnan(atr) and len(fresh) > 0:
            if direction == "long":
                near = fresh[(fresh["zone_hi"] <= cur_l) &
                             ((cur_l - fresh["zone_hi"]) <= ATR_MULTIPLIER * atr)]
            else:
                near = fresh[(fresh["zone_lo"] >= cur_h) &
                             ((fresh["zone_lo"] - cur_h) <= ATR_MULTIPLIER * atr)]
            proximity_hits += len(near)

    return {
        "atr14": atr,
        "idm_hits": idm_hits,
        "proximity_hits": proximity_hits,
        "stage2_pass_idm_only": (idm_hits == 0),
        "stage2_pass_prox_only": (proximity_hits == 0),
        "stage2_passed": (idm_hits == 0) and (proximity_hits == 0),
    }


def build_fvg_intervals(events: pd.DataFrame, tf: str) -> pd.DataFrame:
    sub = events[(events["tf"]==tf) & (events["element"]=="fvg")]
    born = sub[sub["kind"]=="born"].copy()
    # sort retire events by ts, keep FIRST по ts (не по строке parquet)
    ret = sub[sub["kind"].isin(["retire","fill_partial"])].sort_values("ts").drop_duplicates("zone_id", keep="first")
    retire_map = dict(zip(ret["zone_id"].values, ret["ts"].values))
    born["retire_ts"] = born["zone_id"].map(retire_map)
    born["born_ts"] = born["ts"]
    return born[["born_ts","retire_ts","zone_lo","zone_hi","direction"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", choices=["BTC", "ETH", "SOL"])
    ap.add_argument("--end", default=None, help="Period end YYYY-MM-DD (informational, propagates from stage 1)")
    args = ap.parse_args()

    symbol = args.symbol
    CSV_1M = DATA_DIR / f"{symbol}USDT_1m.csv"
    EVENTS_PATH = find_latest_events(symbol)
    STAGE1_PARQUET = find_latest_stage1(symbol)
    OUT_CANONICAL = STAGE_DIR / f"ob_stage2_canonical_{symbol}.parquet"
    OUT_SWEEP = STAGE_DIR / f"ob_stage2_sweep_only_{symbol}.parquet"

    print(f"[cfg] symbol={symbol}")
    print(f"[cfg] STAGE1_PARQUET = {STAGE1_PARQUET.name}")
    print(f"[cfg] EVENTS = {EVENTS_PATH.name}")

    print("[load] events...")
    events = pd.read_parquet(EVENTS_PATH)
    print(f"  {len(events):,} events")

    print("[load] 4h bars for first-touch...")
    bars_4h = aggregate_4h_bars(CSV_1M)
    print(f"  {len(bars_4h):,} 4h bars")

    print("[load] stage1 canonical...")
    v2 = pd.read_parquet(STAGE1_PARQUET)
    canonical = v2[v2["passed"]].copy()
    sweep_only = v2[v2["sweep_any"] & ~v2["engulf"]].copy()
    print(f"  canonical: {len(canonical):,}, sweep_only: {len(sweep_only):,}")

    print(f"[load] ATR14 4h from CSV...")
    atr = load_atr_4h(CSV_1M)
    print(f"  {len(atr):,} 4h bars")

    print(f"[load] FVG intervals per TF + first-touch precompute...")
    fvg_by_tf = {}
    for tf in TFS:
        fdf = build_fvg_intervals(events, tf)
        fdf["first_wick_touch_ts"] = compute_first_wick_touch(fdf, bars_4h)
        fvg_by_tf[tf] = fdf
        n_touch = int((fdf["first_wick_touch_ts"] < 2**62).sum())
        print(f"  FVG {tf}: {len(fdf):,}  first-touched={n_touch:,}")

    for name, cohort in [("canonical", canonical), ("sweep-only", sweep_only)]:
        print(f"\n[check] {name}: n={len(cohort):,}")
        if len(cohort) == 0:
            out = OUT_CANONICAL if name == "canonical" else OUT_SWEEP
            cohort_out = cohort.copy()
            for col in ["atr14", "idm_hits", "proximity_hits", "stage2_pass_idm_only", "stage2_pass_prox_only", "stage2_passed"]:
                cohort_out[col] = pd.Series(dtype="float64" if col == "atr14" else "int64" if "hits" in col else "bool")
            cohort_out.to_parquet(out, index=False)
            print(f"  (empty) saved: {out.name}")
            continue
        _bts = bars_4h["ts_ms"].values.astype(np.int64)
        _bh = bars_4h["high"].values.astype(np.float64)
        _bl = bars_4h["low"].values.astype(np.float64)
        results = cohort.apply(lambda r: check_filters(r, fvg_by_tf, atr, _bts, _bh, _bl), axis=1, result_type="expand")
        cohort_out = pd.concat([cohort.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
        n_idm = (cohort_out["idm_hits"] > 0).sum()
        n_prox = (cohort_out["proximity_hits"] > 0).sum()
        n_idm_only = int(cohort_out["stage2_pass_idm_only"].sum())
        n_prox_only = int(cohort_out["stage2_pass_prox_only"].sum())
        n_both = int(cohort_out["stage2_passed"].sum())
        n_total = len(cohort_out)
        print(f"  IDM hit rejected:  {n_idm:>4} → pass_idm_only:  {n_idm_only:>4} / {n_total}")
        print(f"  Prox hit rejected: {n_prox:>4} → pass_prox_only: {n_prox_only:>4} / {n_total}")
        print(f"  BOTH filters:                       passed:         {n_both:>4} / {n_total}")

        out = OUT_CANONICAL if name == "canonical" else OUT_SWEEP
        cohort_out.to_parquet(out, index=False)
        print(f"  saved: {out.name}")


if __name__ == "__main__":
    main()
