"""FVG_OB canonical 4h — IDM trigger (заменяет Liq_OB stage1+stage2).

Trigger (методичка AlexxxFlow, IDM = inducement):
  cur bar на 4h wick'ом взаимодействует с АКТИВНОЙ HTF FVG.
  → OB образован в момент wick-fill живой FVG (первый, второй, N-ный касание OK,
     пока FVG не полностью замитигирована).

Condition:
  Существует HTF FVG (12h/1d/2d/3d/1w) для которой:
    - active: born_ts < cur_open AND (retire_ts NaN OR retire_ts > cur_close)
      (retire = FULL fill, close сквозь; wick-touch НЕ убивает FVG)
    - zone overlaps cur range: zone_hi >= cur.low AND zone_lo <= cur.high
    → wick-fill активной FVG (не важно который по счёту touch)

PROX filter (аналог Liq stage2):
  Свежая FVG (first_wick_touch_ts > cur_open) в направлении SL:
    LONG:  zone_hi <= cur.low  AND (cur.low - zone_hi) <= 0.5×ATR
    SHORT: zone_lo >= cur.high AND (zone_lo - cur.high) <= 0.5×ATR
  → REJECT (SL magnet)

Output:
  ob_stage2_sweep_only_{sym}.parquet — passed IDM (без учёта PROX)
  ob_stage2_canonical_{sym}.parquet — passed IDM AND clean PROX
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ASVK = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ASVK / "data"
EVENTS_DIR = DATA_DIR / "events"
OUT_DIR = DATA_DIR / "fvg_ob4h_vc"
CSV_DIR = DATA_DIR

HTF_FVG_TFS = ["12h", "1d", "2d", "3d", "1w"]
TF_MS_4H = 4 * 3600 * 1000
TF_MS_1H = 3600 * 1000
TF_MS_HTF = {"4h": 4*TF_MS_1H, "6h": 6*TF_MS_1H, "12h": 12*TF_MS_1H,
             "1d": 24*TF_MS_1H, "2d": 48*TF_MS_1H, "3d": 72*TF_MS_1H, "1w": 168*TF_MS_1H}
ATR_MULTIPLIER = 0.5
ATR_PERIOD = 14


def build_config(symbol: str, end_date: str) -> dict:
    return {
        "symbol":         symbol,
        "target_tf":      "4h",
        "poi_element":    "ob",
        "trigger":        "idm_htf_fvg_first_wick_touch",
        "htf_fvg_tfs":    HTF_FVG_TFS,
        "prox_filter":    True,
        "prox_multiplier": ATR_MULTIPLIER,
        "atr_period":     ATR_PERIOD,
        "period_start":   "2018-01-01",
        "period_end":     end_date,
        "version":        1,
    }


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def find_latest_events(symbol: str) -> Path:
    files = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No events_e12d_{symbol}_*.parquet in {EVENTS_DIR}")
    return files[-1]


def aggregate_4h(csv_path: Path, period_end: str) -> pd.DataFrame:
    print(f"[agg 4h] {csv_path.name}...")
    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=["open_time","open","high","low","close"])
    df["ts_ms"] = pd.to_datetime(df["open_time"], utc=True, format="ISO8601").values.astype("datetime64[ms]").astype("int64")
    tf = TF_MS_4H
    df["bucket"] = (df["ts_ms"] // tf) * tf
    period_end_ms = int(pd.Timestamp(period_end, tz="UTC").value // 1_000_000)
    df = df[df["bucket"] < period_end_ms]
    agg = df.groupby("bucket").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last")
    ).reset_index()
    agg = agg.rename(columns={"bucket":"ts_ms"}).sort_values("ts_ms").reset_index(drop=True)
    print(f"[agg] 4h bars: {len(agg):,} ({time.time()-t0:.1f}s)")
    return agg


def load_atr_4h(bars_4h: pd.DataFrame) -> pd.DataFrame:
    df = bars_4h.copy()
    prev_c = np.concatenate([[df["close"].iloc[0]], df["close"].values[:-1]])
    tr = np.maximum.reduce([
        df["high"].values - df["low"].values,
        np.abs(df["high"].values - prev_c),
        np.abs(df["low"].values - prev_c),
    ])
    df["atr14"] = pd.Series(tr).rolling(ATR_PERIOD).mean().values
    return df[["ts_ms","atr14"]]


def build_fvg_intervals(events: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Возвращает FVG born events + retire_ts для указанного TF."""
    sub = events[(events["tf"] == tf) & (events["element"] == "fvg")]
    born = sub[sub["kind"] == "born"].copy()
    ret = sub[sub["kind"].isin(["retire","fill_partial"])].sort_values("ts").drop_duplicates("zone_id", keep="first")
    retire_map = dict(zip(ret["zone_id"].values, ret["ts"].values))
    born["retire_ts"] = born["zone_id"].map(retire_map)
    born["born_ts"] = born["ts"]
    return born[["born_ts","retire_ts","zone_lo","zone_hi","direction"]].reset_index(drop=True)


def compute_first_wick_touch(fvgs: pd.DataFrame, bars_4h: pd.DataFrame) -> np.ndarray:
    """Для каждой FVG найти первый 4h бар, тронувший её range wick'ом.
    Возвращает ts_ms первого касающегося бара (или 2^62 если не тронута)."""
    n = len(fvgs)
    result = np.full(n, 2**62, dtype=np.int64)
    bar_ts = bars_4h["ts_ms"].values.astype(np.int64)
    bar_l = bars_4h["low"].values.astype(np.float64)
    bar_h = bars_4h["high"].values.astype(np.float64)
    born_arr = fvgs["born_ts"].values.astype(np.int64)
    lo_arr = fvgs["zone_lo"].values.astype(np.float64)
    hi_arr = fvgs["zone_hi"].values.astype(np.float64)

    for i in range(n):
        born = born_arr[i]
        zone_lo = lo_arr[i]
        zone_hi = hi_arr[i]
        # ищем первый бар с ts > born, у которого wick пересёк [zone_lo, zone_hi]
        start_idx = int(np.searchsorted(bar_ts, born, side="right"))
        if start_idx >= len(bar_ts): continue
        sub_l = bar_l[start_idx:]
        sub_h = bar_h[start_idx:]
        touch = (sub_l <= zone_hi) & (sub_h >= zone_lo)
        idx = np.argmax(touch)
        if touch[idx]:
            result[i] = int(bar_ts[start_idx + idx])
    return result


def compute_active_bounds_at(zone_lo, zone_hi, direction, born_ts, target_ts,
                             bars_ts, bars_h, bars_l):
    """Compute active portion at target_ts (see stage2 semantics)."""
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


def check_idm_prox(cur_open: int, cur_o: float, cur_h: float, cur_l: float, cur_c: float,
                   direction: str, atr: float, fvg_by_tf: dict,
                   bars_ts=None, bars_h=None, bars_l=None) -> dict:
    """IDM (canon 2026-07-14): любой wick-fill АКТИВНОЙ части HTF FVG.
    Полная mit исключает. Overlap против active_lo/hi."""
    cur_close_ts = cur_open + TF_MS_4H
    idm_hits_by_tf = {}
    prox_hits_by_tf = {}

    for tf, fvgs in fvg_by_tf.items():
        active = fvgs[(fvgs["born_ts"] < cur_open) &
                      (fvgs["retire_ts"].isna() | (fvgs["retire_ts"] > cur_close_ts))]
        if len(active) == 0: continue

        candidates = active[
            (active["direction"] == direction) &
            (active["zone_hi"] >= cur_l) &
            (active["zone_lo"] <= cur_h)
        ]
        if len(candidates) > 0 and bars_ts is not None:
            n_idm = 0
            for _, fv in candidates.iterrows():
                a_lo, a_hi, full_mit = compute_active_bounds_at(
                    float(fv.zone_lo), float(fv.zone_hi), fv.direction,
                    int(fv.born_ts), cur_open, bars_ts, bars_h, bars_l,
                )
                if full_mit: continue
                k_up = (cur_c > a_hi) and (cur_o < a_hi)
                k_dn = (cur_c < a_lo) and (cur_o > a_lo)
                if k_up or k_dn: continue
                if cur_h >= a_lo and cur_l <= a_hi:
                    n_idm += 1
            if n_idm > 0:
                idm_hits_by_tf[tf] = n_idm

        # PROX magnet
        fresh = active[active["first_wick_touch_ts"] > cur_open]
        if not np.isnan(atr) and len(fresh) > 0:
            if direction == "long":
                near = fresh[(fresh["zone_hi"] <= cur_l) &
                             ((cur_l - fresh["zone_hi"]) <= ATR_MULTIPLIER * atr)]
            else:
                near = fresh[(fresh["zone_lo"] >= cur_h) &
                             ((fresh["zone_lo"] - cur_h) <= ATR_MULTIPLIER * atr)]
            if len(near) > 0:
                prox_hits_by_tf[tf] = len(near)

    idm_total = sum(idm_hits_by_tf.values())
    prox_total = sum(prox_hits_by_tf.values())
    return {
        "idm_hits": idm_total,
        "proximity_hits": prox_total,
        "idm_by_tf": idm_hits_by_tf,
        "prox_by_tf": prox_hits_by_tf,
        "stage2_pass_idm_only": (idm_total > 0),
        "stage2_pass_prox_only": (prox_total == 0),
        "stage2_passed": (idm_total > 0) and (prox_total == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", choices=["BTC", "ETH", "SOL"])
    ap.add_argument("--end", default=None, help="Period end YYYY-MM-DD (default: tomorrow UTC)")
    args = ap.parse_args()

    symbol = args.symbol
    end_date = args.end or (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    CONFIG = build_config(symbol, end_date)
    CSV_1M = CSV_DIR / f"{symbol}USDT_1m.csv"
    EVENTS_PATH = find_latest_events(symbol)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FVG_OB canonical 4h  symbol={symbol}  end={end_date}  cfg_hash={cfg_hash(CONFIG)}")
    print(f"  events: {EVENTS_PATH.name}")
    print(f"  csv:    {CSV_1M.name}")
    print("─" * 60)

    events = pd.read_parquet(EVENTS_PATH)
    print(f"[load] events: {len(events):,}")

    ob_born = events[
        (events["tf"] == "4h") &
        (events["element"] == "ob") &
        (events["kind"] == "born")
    ].sort_values("ts").reset_index(drop=True)
    print(f"[filter] 4h OB born: {len(ob_born):,}")

    bars = aggregate_4h(CSV_1M, end_date)
    ts_to_idx = dict(zip(bars["ts_ms"].values, bars.index.values))
    _bars_ts = bars["ts_ms"].values.astype(np.int64)
    _bars_h = bars["high"].values.astype(np.float64)
    _bars_l = bars["low"].values.astype(np.float64)

    atr_lookup = load_atr_4h(bars)
    print(f"[atr] 4h ATR14: {len(atr_lookup):,}")

    # Build FVG intervals per HTF and compute first_wick_touch
    print("[fvg] intervals + first_wick_touch per HTF ...")
    fvg_by_tf = {}
    for tf in HTF_FVG_TFS:
        fdf = build_fvg_intervals(events, tf)
        fdf["first_wick_touch_ts"] = compute_first_wick_touch(fdf, bars)
        n_touched = int((fdf["first_wick_touch_ts"] < 2**62).sum())
        fvg_by_tf[tf] = fdf
        print(f"   {tf}: n={len(fdf):>5,}  touched={n_touched:>5,}")

    print(f"[audit] scanning {len(ob_born):,} 4h OBs for IDM trigger ...")
    t0 = time.time()
    rows_idm_only = []
    rows_canonical = []
    n_no_bar = 0
    n_gap = 0
    n_idm = 0
    n_canonical = 0

    tf_ms = TF_MS_4H
    for _, r in ob_born.iterrows():
        ob_ts = int(r["ts"])
        direction = r["direction"]
        cur_open = ob_ts - tf_ms
        prev_open = cur_open - tf_ms
        ci = ts_to_idx.get(cur_open)
        pi = ts_to_idx.get(prev_open)
        if ci is None or pi is None or ci < 1:
            n_no_bar += 1
            continue
        if pi != ci - 1:
            n_gap += 1
            continue
        cur = bars.iloc[ci]
        prev = bars.iloc[pi]

        # Engulf (diagnostic)
        if direction == "long":
            engulf = bool(
                (cur["close"] > cur["open"]) and (prev["close"] < prev["open"])
                and (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])
            )
        else:
            engulf = bool(
                (cur["close"] < cur["open"]) and (prev["close"] > prev["open"])
                and (cur["close"] <= prev["open"]) and (cur["open"] >= prev["close"])
            )

        # ATR at cur_open
        idx_atr = int(np.searchsorted(atr_lookup["ts_ms"].values, cur_open, side="right")) - 1
        atr = float(atr_lookup["atr14"].iloc[idx_atr]) if idx_atr >= ATR_PERIOD else np.nan

        result = check_idm_prox(cur_open, float(cur["open"]), float(cur["high"]),
                                 float(cur["low"]), float(cur["close"]),
                                 direction, atr, fvg_by_tf,
                                 _bars_ts, _bars_h, _bars_l)

        if not result["stage2_pass_idm_only"]:
            continue  # no IDM trigger

        n_idm += 1
        row = {
            "ob_ts":     ob_ts,
            "cur_open":  cur_open,
            "direction": direction,
            "cur_o":     float(cur["open"]),
            "cur_h":     float(cur["high"]),
            "cur_l":     float(cur["low"]),
            "cur_c":     float(cur["close"]),
            "prev_o":    float(prev["open"]),
            "prev_h":    float(prev["high"]),
            "prev_l":    float(prev["low"]),
            "prev_c":    float(prev["close"]),
            "engulf":    engulf,
            # sweep_XX = 0 always (для совместимости downstream stage3/4 columns)
            "sweep_1h":  0, "sweep_2h": 0, "sweep_4h": 0, "sweep_6h": 0, "sweep_12h": 0,
            "sweep_any": True,  # meaning: passed IDM trigger
            "passed":    True,
            # IDM specific
            "atr14":                   result["atr14"] if "atr14" in result else atr,
            "idm_hits":                result["idm_hits"],
            "proximity_hits":          result["proximity_hits"],
            "stage2_pass_idm_only":    result["stage2_pass_idm_only"],
            "stage2_pass_prox_only":   result["stage2_pass_prox_only"],
            "stage2_passed":           result["stage2_passed"],
        }
        # атрибуты по HTF (сколько IDM/PROX на каждом TF)
        for tf in HTF_FVG_TFS:
            row[f"idm_hits_{tf}"] = result["idm_by_tf"].get(tf, 0)
            row[f"prox_hits_{tf}"] = result["prox_by_tf"].get(tf, 0)

        rows_idm_only.append(row)
        if result["stage2_passed"]:
            n_canonical += 1
            rows_canonical.append(row)

    df_idm_only = pd.DataFrame(rows_idm_only)
    df_canonical = pd.DataFrame(rows_canonical)
    dt = time.time() - t0
    print(f"[audit] done in {dt:.1f}s (no-bar: {n_no_bar}, gap: {n_gap})")
    print()

    n_all = len(ob_born) - n_no_bar - n_gap
    print("═" * 70)
    print(f"FVG_OB canonical 4h — IDM trigger ({symbol} {CONFIG['period_start']} → {CONFIG['period_end']})")
    print("═" * 70)
    print(f"total 4h OB born (with bars):       {n_all:>6,}")
    print(f"passed IDM trigger (idm_hits > 0):  {n_idm:>6,}  ({n_idm/n_all*100:5.1f}%)")
    print(f"passed IDM + PROX clean:            {n_canonical:>6,}  ({n_canonical/n_all*100:5.1f}%)")
    print()

    if len(df_canonical) > 0:
        print("By direction (canonical):")
        for d in ("long", "short"):
            n_d = int((df_canonical["direction"] == d).sum())
            n_total_d = int((ob_born["direction"] == d).sum())
            print(f"  {d:5s}: {n_d:>4,} / {n_total_d:>4,}  ({n_d/n_total_d*100:5.1f}%)")

        print("\nIDM hits by HTF (canonical):")
        for tf in HTF_FVG_TFS:
            col = f"idm_hits_{tf}"
            n = int((df_canonical[col] > 0).sum())
            print(f"  {tf}: {n:>4,}  ({n/len(df_canonical)*100:5.1f}%)")

    out_sw = OUT_DIR / f"ob_stage2_sweep_only_{symbol}.parquet"
    out_can = OUT_DIR / f"ob_stage2_canonical_{symbol}.parquet"
    df_idm_only.to_parquet(out_sw, index=False)
    df_canonical.to_parquet(out_can, index=False)
    print(f"\nsaved: {out_sw.name} ({len(df_idm_only)} rows)")
    print(f"saved: {out_can.name} ({len(df_canonical)} rows)")

    (OUT_DIR / f"fvg_ob_stage1_config_{symbol}.json").write_text(
        json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
