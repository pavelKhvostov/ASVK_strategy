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

Оптимизировано 2026-07-27: s7d pre-filter (было ~21s → <5s per symbol).
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
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ASVK = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ASVK / "data"
EVENTS_DIR = DATA_DIR / "events"
S7D_DIR = DATA_DIR / "s7d"
OUT_DIR = DATA_DIR / "fvg_ob4h_vc"
CSV_DIR = DATA_DIR

HTF_FVG_TFS = ["12h", "1d", "2d", "3d", "1w"]
TF_MS_4H = 4 * 3600 * 1000
ATR_MULTIPLIER = 0.5
ATR_PERIOD = 14


def build_config(symbol: str, end_date: str) -> dict:
    return {
        "symbol":          symbol,
        "target_tf":       "4h",
        "poi_element":     "ob",
        "trigger":         "idm_htf_fvg_first_wick_touch",
        "htf_fvg_tfs":    HTF_FVG_TFS,
        "prox_filter":     True,
        "prox_multiplier": ATR_MULTIPLIER,
        "atr_period":      ATR_PERIOD,
        "period_start":    "2018-01-01",
        "period_end":      end_date,
        "version":         1,
    }


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def find_latest_events(symbol: str) -> Path:
    files = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No events_e12d_{symbol}_*.parquet in {EVENTS_DIR}")
    return files[-1]


def find_latest_s7d(symbol: str) -> Path:
    files = sorted(S7D_DIR.glob(f"snapshots_s7d_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No snapshots_s7d_{symbol}_*.parquet in {S7D_DIR}")
    return files[-1]


def load_s7d_htf_fvg(symbol: str) -> pd.DataFrame:
    path = find_latest_s7d(symbol)
    table = pq.read_table(
        path,
        columns=["anchor_ts", "tf", "direction", "active_lo", "active_hi",
                 "mit_count_at", "zone_lo", "zone_hi", "born_ts", "zone_id"],
        filters=[("element", "=", "fvg"), ("tf", "in", HTF_FVG_TFS)],
    )
    return table.to_pandas()


def aggregate_4h(csv_path: Path, period_end_ms: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["open_time", "open", "high", "low", "close"])
    df["ts_ms"] = (pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
                   .values.astype("datetime64[ms]").astype("int64"))
    df["bucket"] = (df["ts_ms"] // TF_MS_4H) * TF_MS_4H
    df = df[df["bucket"] < period_end_ms]
    agg = (df.groupby("bucket")
           .agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"))
           .reset_index()
           .rename(columns={"bucket": "ts_ms"})
           .sort_values("ts_ms")
           .reset_index(drop=True))
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
    return df[["ts_ms", "atr14"]]


def compute_active_bounds_at(zone_lo, zone_hi, direction, born_ts, target_ts,
                              bars_ts, bars_h, bars_l):
    i0 = int(np.searchsorted(bars_ts, born_ts, side="right"))
    i1 = int(np.searchsorted(bars_ts, target_ts, side="left"))
    if i0 >= i1:
        return zone_lo, zone_hi, False
    hs = bars_h[i0:i1]; ls = bars_l[i0:i1]
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


def build_ob_df(ob_born: pd.DataFrame, bars_4h: pd.DataFrame) -> pd.DataFrame:
    """Векторизованное построение ob_df из born events + 4h баров."""
    ts_arr = bars_4h["ts_ms"].values.astype(np.int64)
    ts_to_idx = {int(t): i for i, t in enumerate(ts_arr)}

    ob_born = ob_born.copy()
    ob_born["cur_open"] = ob_born["ts"].values.astype(np.int64) - TF_MS_4H
    ob_born["ci"] = ob_born["cur_open"].map(ts_to_idx)
    ob_born["pi"] = (ob_born["cur_open"] - TF_MS_4H).map(ts_to_idx)

    valid = ob_born.dropna(subset=["ci", "pi"]).copy()
    valid["ci"] = valid["ci"].astype(int)
    valid["pi"] = valid["pi"].astype(int)
    valid = valid[valid["pi"] == valid["ci"] - 1].reset_index(drop=True)

    if len(valid) == 0:
        return pd.DataFrame()

    ci = valid["ci"].values
    pi = valid["pi"].values

    cur = bars_4h.iloc[ci].reset_index(drop=True)
    prev = bars_4h.iloc[pi].reset_index(drop=True)

    # Engulf (vectorized)
    is_long = valid["direction"] == "long"
    engulf_long = (
        (cur["close"].values > cur["open"].values) &
        (prev["close"].values < prev["open"].values) &
        (cur["close"].values >= prev["open"].values) &
        (cur["open"].values <= prev["close"].values)
    )
    engulf_short = (
        (cur["close"].values < cur["open"].values) &
        (prev["close"].values > prev["open"].values) &
        (cur["close"].values <= prev["open"].values) &
        (cur["open"].values >= prev["close"].values)
    )
    engulf = np.where(is_long.values, engulf_long, engulf_short)

    ob_df = pd.DataFrame({
        "ob_ts":     valid["ts"].values.astype(np.int64),
        "cur_open":  valid["cur_open"].values.astype(np.int64),
        "direction": valid["direction"].values,
        "cur_o":     cur["open"].values.astype(np.float64),
        "cur_h":     cur["high"].values.astype(np.float64),
        "cur_l":     cur["low"].values.astype(np.float64),
        "cur_c":     cur["close"].values.astype(np.float64),
        "prev_o":    prev["open"].values.astype(np.float64),
        "prev_h":    prev["high"].values.astype(np.float64),
        "prev_l":    prev["low"].values.astype(np.float64),
        "prev_c":    prev["close"].values.astype(np.float64),
        "engulf":    engulf,
        "sweep_1h":  0, "sweep_2h": 0, "sweep_4h": 0, "sweep_6h": 0, "sweep_12h": 0,
        "sweep_any": True,
        "passed":    True,
    })
    return ob_df


def check_filters_fvgob(ob_df: pd.DataFrame, fvg_s7d: pd.DataFrame,
                         fvg_retire_map: dict, bars_4h: pd.DataFrame,
                         atr_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized IDM (accept) + PROX (reject) через s7d pre-filter.

    FVG_OB семантика (обратная Liq_OB):
      stage2_pass_idm_only = (idm_hits > 0)  — IDM IS trigger, not rejection
      stage2_pass_prox_only = (proximity_hits == 0)
      stage2_passed = (idm_hits > 0) AND (proximity_hits == 0)
    """
    ob_df = ob_df.copy()

    # ATR lookup
    atr_ts = atr_df["ts_ms"].values.astype(np.int64)
    atr_vals = atr_df["atr14"].values
    def _atr(t):
        idx = int(np.searchsorted(atr_ts, int(t), side="right")) - 1
        return float(atr_vals[idx]) if idx >= ATR_PERIOD else float("nan")
    ob_df["atr14"] = ob_df["cur_open"].apply(_atr)

    _bars_ts = bars_4h["ts_ms"].values.astype(np.int64)
    _bars_h  = bars_4h["high"].values.astype(np.float64)
    _bars_l  = bars_4h["low"].values.astype(np.float64)

    # ── IDM (accept trigger) ───────────────────────────────────────────────
    # Use ALL s7d HTF FVGs (not just mit_count_at==0) — any touch of active FVG qualifies
    fvg_all_dir = fvg_s7d[["anchor_ts", "tf", "direction",
                             "zone_lo", "zone_hi", "born_ts", "zone_id"]].drop_duplicates()

    idm_cands = ob_df[["ob_ts", "cur_open", "direction", "cur_h", "cur_l", "cur_o", "cur_c"]].merge(
        fvg_all_dir,
        left_on=["cur_open", "direction"],
        right_on=["anchor_ts", "direction"],
        how="inner",
    )
    idm_cands = idm_cands[idm_cands["born_ts"] < idm_cands["cur_open"]]
    idm_cands["first_retire_ts"] = idm_cands["zone_id"].map(fvg_retire_map)
    # retire check: zone must be alive through cur_close = ob_ts
    active_mask = idm_cands["first_retire_ts"].isna() | (idm_cands["first_retire_ts"] > idm_cands["ob_ts"])
    idm_cands = idm_cands[active_mask]
    pre_overlap = (idm_cands["zone_hi"] >= idm_cands["cur_l"]) & (idm_cands["zone_lo"] <= idm_cands["cur_h"])
    idm_cands = idm_cands[pre_overlap].reset_index(drop=True)

    # Track per-TF hits
    idm_hit_map: dict[int, dict] = {}  # ob_ts → {tf: count}
    for row in idm_cands.itertuples(index=False):
        a_lo, a_hi, full_mit = compute_active_bounds_at(
            float(row.zone_lo), float(row.zone_hi), row.direction,
            int(row.born_ts), int(row.cur_open), _bars_ts, _bars_h, _bars_l,
        )
        if full_mit:
            continue
        k_up = (row.cur_c > a_hi) and (row.cur_o < a_hi)
        k_dn = (row.cur_c < a_lo) and (row.cur_o > a_lo)
        if k_up or k_dn:
            continue
        if row.cur_h >= a_lo and row.cur_l <= a_hi:
            d = idm_hit_map.setdefault(int(row.ob_ts), {})
            d[row.tf] = d.get(row.tf, 0) + 1

    ob_df["idm_hits"] = ob_df["ob_ts"].map(
        lambda t: sum(idm_hit_map.get(int(t), {}).values())
    ).fillna(0).astype(int)
    for tf in HTF_FVG_TFS:
        ob_df[f"idm_hits_{tf}"] = ob_df["ob_ts"].map(
            lambda t, _tf=tf: idm_hit_map.get(int(t), {}).get(_tf, 0)
        ).astype(int)

    # ── PROX (reject) ─────────────────────────────────────────────────────
    fvg_for_prox = fvg_s7d[["anchor_ts", "tf", "zone_lo", "zone_hi",
                              "born_ts", "zone_id"]].drop_duplicates()

    prox_join = ob_df[["ob_ts", "cur_open", "direction", "cur_h", "cur_l", "atr14"]].merge(
        fvg_for_prox, left_on="cur_open", right_on="anchor_ts", how="inner",
    )
    prox_join = prox_join[prox_join["born_ts"] < prox_join["cur_open"]]
    prox_join["first_retire_ts"] = prox_join["zone_id"].map(fvg_retire_map)
    prox_active = prox_join["first_retire_ts"].isna() | (prox_join["first_retire_ts"] > prox_join["ob_ts"])
    prox_join = prox_join[prox_active]

    valid_atr = ~prox_join["atr14"].isna()
    prox_long  = (valid_atr & (prox_join["direction"] == "long") &
                  (prox_join["zone_hi"] <= prox_join["cur_l"]) &
                  ((prox_join["cur_l"] - prox_join["zone_hi"]) <= ATR_MULTIPLIER * prox_join["atr14"]))
    prox_short = (valid_atr & (prox_join["direction"] == "short") &
                  (prox_join["zone_lo"] >= prox_join["cur_h"]) &
                  ((prox_join["zone_lo"] - prox_join["cur_h"]) <= ATR_MULTIPLIER * prox_join["atr14"]))
    prox_sl = prox_join[prox_long | prox_short].reset_index(drop=True)

    prox_hit_map: dict[int, dict] = {}  # ob_ts → {tf: count}
    for row in prox_sl.itertuples(index=False):
        i0 = int(np.searchsorted(_bars_ts, int(row.born_ts), side="right"))
        i1 = int(np.searchsorted(_bars_ts, int(row.cur_open), side="right"))
        if i0 < i1:
            touched = (_bars_h[i0:i1] >= row.zone_lo) & (_bars_l[i0:i1] <= row.zone_hi)
            if touched.any():
                continue
        d = prox_hit_map.setdefault(int(row.ob_ts), {})
        d[row.tf] = d.get(row.tf, 0) + 1

    ob_df["proximity_hits"] = ob_df["ob_ts"].map(
        lambda t: sum(prox_hit_map.get(int(t), {}).values())
    ).fillna(0).astype(int)
    for tf in HTF_FVG_TFS:
        ob_df[f"prox_hits_{tf}"] = ob_df["ob_ts"].map(
            lambda t, _tf=tf: prox_hit_map.get(int(t), {}).get(_tf, 0)
        ).astype(int)

    # FVG_OB semantics: IDM is accept trigger
    ob_df["stage2_pass_idm_only"]  = ob_df["idm_hits"] > 0
    ob_df["stage2_pass_prox_only"] = ob_df["proximity_hits"] == 0
    ob_df["stage2_passed"]         = ob_df["stage2_pass_idm_only"] & ob_df["stage2_pass_prox_only"]

    return ob_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", choices=["BTC", "ETH", "SOL"])
    ap.add_argument("--end", default=None, help="Period end YYYY-MM-DD (default: tomorrow UTC)")
    args = ap.parse_args()

    symbol = args.symbol
    end_date = args.end or (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    period_end_ms = int(pd.Timestamp(end_date, tz="UTC").value // 1_000_000)

    CONFIG = build_config(symbol, end_date)
    CSV_1M = CSV_DIR / f"{symbol}USDT_1m.csv"
    EVENTS_PATH = find_latest_events(symbol)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FVG_OB canonical 4h  symbol={symbol}  end={end_date}  cfg_hash={cfg_hash(CONFIG)}")
    print(f"  events: {EVENTS_PATH.name}")
    print(f"  csv:    {CSV_1M.name}")
    print("─" * 60)

    t0 = time.time()

    print("[load] events...")
    events = pd.read_parquet(EVENTS_PATH)
    print(f"  {len(events):,} events")

    print("[load] fvg_retire_map from events...")
    fvg_ret_ev = events[(events["element"] == "fvg") &
                        (events["kind"].isin(["retire", "fill_partial"]))]
    fvg_retire_map: dict = dict(
        fvg_ret_ev.sort_values("ts")
                  .drop_duplicates("zone_id", keep="first")
                  .set_index("zone_id")["ts"]
                  .astype("int64")
    )
    print(f"  retire_map: {len(fvg_retire_map):,} zones")

    print("[load] 4h OB born from events...")
    ob_born = events[
        (events["tf"] == "4h") &
        (events["element"] == "ob") &
        (events["kind"] == "born")
    ].sort_values("ts").reset_index(drop=True)
    print(f"  4h OB born: {len(ob_born):,}")

    del events

    print("[load] 4h bars...")
    bars_4h = aggregate_4h(CSV_1M, period_end_ms)
    print(f"  {len(bars_4h):,} 4h bars ({time.time()-t0:.1f}s)")

    print("[load] ATR14 4h...")
    atr_df = load_atr_4h(bars_4h)

    print("[load] s7d HTF FVG...")
    fvg_s7d = load_s7d_htf_fvg(symbol)
    print(f"  {len(fvg_s7d):,} rows (HTF FVG)")

    print("[build] ob_df from events + bars (vectorized)...")
    ob_df_all = build_ob_df(ob_born, bars_4h)
    n_no_bar = len(ob_born) - len(ob_df_all)
    print(f"  {len(ob_df_all):,} valid OBs (skipped {n_no_bar} no-bar/gap)")

    print("[check] IDM + PROX filters...")
    ob_df_all = check_filters_fvgob(ob_df_all, fvg_s7d, fvg_retire_map, bars_4h, atr_df)

    dt = time.time() - t0
    print(f"[done] {dt:.1f}s total")
    print()

    n_all = len(ob_df_all)
    df_idm_only  = ob_df_all[ob_df_all["stage2_pass_idm_only"]].reset_index(drop=True)
    df_canonical = ob_df_all[ob_df_all["stage2_passed"]].reset_index(drop=True)

    n_idm = len(df_idm_only)
    n_canonical = len(df_canonical)
    print("═" * 70)
    print(f"FVG_OB canonical 4h — IDM trigger ({symbol} {CONFIG['period_start']} → {CONFIG['period_end']})")
    print("═" * 70)
    print(f"total 4h OB born (with bars):       {n_all:>6,}")
    print(f"passed IDM trigger (idm_hits > 0):  {n_idm:>6,}  ({n_idm/n_all*100:5.1f}% of all)" if n_all else "  (none)")
    print(f"passed IDM + PROX clean:            {n_canonical:>6,}  ({n_canonical/n_all*100:5.1f}% of all)" if n_all else "  (none)")
    print()

    if len(df_canonical) > 0:
        print("By direction (canonical):")
        for d in ("long", "short"):
            n_d = int((df_canonical["direction"] == d).sum())
            n_total_d = int((ob_df_all["direction"] == d).sum())
            print(f"  {d:5s}: {n_d:>4,} / {n_total_d:>4,}  ({n_d/n_total_d*100:5.1f}% of all {d})" if n_total_d else f"  {d}: 0")

        print("\nIDM hits by HTF (canonical):")
        for tf in HTF_FVG_TFS:
            col = f"idm_hits_{tf}"
            n = int((df_canonical[col] > 0).sum())
            print(f"  {tf}: {n:>4,}  ({n/len(df_canonical)*100:5.1f}%)")

    out_sw  = OUT_DIR / f"ob_stage2_sweep_only_{symbol}.parquet"
    out_can = OUT_DIR / f"ob_stage2_canonical_{symbol}.parquet"
    df_idm_only.to_parquet(out_sw, index=False)
    df_canonical.to_parquet(out_can, index=False)
    print(f"\nsaved: {out_sw}")
    print(f"saved: {out_can}")

    (OUT_DIR / f"fvg_ob_stage1_config_{symbol}.json").write_text(
        json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
