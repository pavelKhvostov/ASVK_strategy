"""Этап 4 — race между VC и body break после retest OB.

Правильная логика (по user'у):
  1. OB 4h сформирован — режим слежения включается.
  2. Ждём retest OB (цена wick'ом коснулась body).
  3. С момента retest — гонка между:
     - VC-элемент образуется на 1h (правильное направление, zone overlaps body OB)
     - Цена пробивает body насквозь (invalidation)
  4. Если VC.born_ts < body_break_ts → entry candidate.
  5. Если body_break_ts ≤ VC.born_ts → POI мёртв, пропуск.

TF VC: 1h (X-1 per canon).
VC.direction: LONG OB → bullish (dir='long'); SHORT OB → bearish (dir='short')
VC.zone: overlaps с [body_bot, body_top]

Retest: 1m wick touches body:
  LONG:  1m.low ≤ body_top   (wick заходит сверху в body)
  SHORT: 1m.high ≥ body_bot  (wick заходит снизу в body)

Body break: 1m wick прошёл насквозь:
  LONG:  1m.low < body_bot
  SHORT: 1m.high > body_top

Максимальное окно наблюдения (safety cap): 168h.
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
STAGE_DIR = DATA_DIR / "fvg_ob4h_vc"

WATCH_CAP_H = 168     # safety cap для retest
TF_MS_4H = 4 * 3600 * 1000
TF_MS_RETEST = 3600 * 1000   # retest/break TF = 1h (VC tf = X-1 for 4h POI)


def aggregate_to_tf(m1: pd.DataFrame, tf_ms: int) -> pd.DataFrame:
    """Агрегировать 1m OHLC в бары указанного TF."""
    ts = m1["ts_ms"].values.astype(np.int64)
    bucket = (ts // tf_ms) * tf_ms
    df = pd.DataFrame({"ts_ms": bucket, "high": m1["high"].values, "low": m1["low"].values})
    agg = df.groupby("ts_ms").agg(high=("high","max"), low=("low","min")).reset_index()
    return agg.sort_values("ts_ms").reset_index(drop=True)


def find_latest_events(symbol: str) -> Path:
    files = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No events_e12d_{symbol}_*.parquet found in {EVENTS_DIR}")
    return files[-1]


def load_1m(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["open_time","high","low","close"])
    dt = pd.to_datetime(df["open_time"], format="ISO8601", utc=True)
    df["ts_ms"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    df["low"] = df["low"].astype(float)
    df["high"] = df["high"].astype(float)
    df["close"] = df["close"].astype(float)
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    gaps = np.diff(df["ts_ms"].values) > 60_000
    if gaps.any():
        n_gaps = int(gaps.sum())
        print(f"[1m] WARN: {n_gaps} gaps > 1min detected")
    return df[["ts_ms", "low", "high", "close"]]


def build_element_born(events: pd.DataFrame, element: str, tf: str) -> pd.DataFrame:
    sub = events[
        (events["tf"] == tf)
        & (events["element"] == element)
        & (events["kind"] == "born")
    ][["ts", "zone_lo", "zone_hi", "direction"]]
    return sub.rename(columns={"ts": "born_ts"}).sort_values("born_ts").reset_index(drop=True)


def ob_zone(r) -> tuple[float, float]:
    """OB zone по canonical ob.py:
       LONG:  [min(prev.low, cur.low), prev.open]
       SHORT: [prev.open, max(prev.high, cur.high)]
    """
    if r.direction == "long":
        return float(min(r.prev_l, r.cur_l)), float(r.prev_o)
    else:
        return float(r.prev_o), float(max(r.prev_h, r.cur_h))


def find_retest_ts_batch(ob_df: pd.DataFrame, bars: pd.DataFrame, cap_ms: int) -> np.ndarray:
    """Первое время (wick на 1h TF) когда цена вошла в OB zone."""
    result = np.full(len(ob_df), 2**62, dtype=np.int64)
    m1_ts = bars["ts_ms"].values.astype(np.int64)
    m1_low = bars["low"].values.astype(np.float64)
    m1_high = bars["high"].values.astype(np.float64)

    for i, r in enumerate(ob_df.itertuples()):
        cur_close = int(r.cur_open) + TF_MS_4H
        end = cur_close + cap_ms
        zone_lo, zone_hi = ob_zone(r)
        i0 = int(np.searchsorted(m1_ts, cur_close, side="left"))
        i1 = int(np.searchsorted(m1_ts, end, side="left"))
        if i0 >= i1:
            continue
        if r.direction == "long":
            mask = m1_low[i0:i1] <= zone_hi
        else:
            mask = m1_high[i0:i1] >= zone_lo
        idx = np.where(mask)[0]
        if len(idx) > 0:
            result[i] = int(m1_ts[i0 + idx[0]])
    return result


def find_invalidation_ts_batch(ob_df: pd.DataFrame, bars: pd.DataFrame, cap_ms: int) -> np.ndarray:
    """Invalidation = первый wick на 1h TF прошёл за OB zone.
       LONG:  bar.low < zone_lo
       SHORT: bar.high > zone_hi
    """
    result = np.full(len(ob_df), 2**62, dtype=np.int64)
    m1_ts = bars["ts_ms"].values.astype(np.int64)
    m1_low = bars["low"].values.astype(np.float64)
    m1_high = bars["high"].values.astype(np.float64)

    for i, r in enumerate(ob_df.itertuples()):
        cur_close = int(r.cur_open) + TF_MS_4H
        end = cur_close + cap_ms
        zone_lo, zone_hi = ob_zone(r)
        i0 = int(np.searchsorted(m1_ts, cur_close, side="left"))
        i1 = int(np.searchsorted(m1_ts, end, side="left"))
        if i0 >= i1:
            continue
        if r.direction == "long":
            mask = m1_low[i0:i1] < zone_lo
        else:
            mask = m1_high[i0:i1] > zone_hi
        idx = np.where(mask)[0]
        if len(idx) > 0:
            result[i] = int(m1_ts[i0 + idx[0]])
    return result


def check_valid_vc(ob_df: pd.DataFrame, retest_ts: np.ndarray, brk_ts: np.ndarray,
                   vc_events: pd.DataFrame, cap_ms: int) -> np.ndarray:
    """VC born_ts:
      - в окне [retest_ts, window_end) где window_end = min(body_break_ts, cap_end)
        cap_end = cur_open + TF_MS_4H + cap_ms (safety cap если тела не пробило)
      - direction соответствует OB
      - VC.zone overlaps OB zone [ob_lo, ob_hi]
    """
    result = np.zeros(len(ob_df), dtype=bool)
    vc_born = vc_events["born_ts"].values.astype(np.int64)
    vc_lo = vc_events["zone_lo"].values.astype(np.float64)
    vc_hi = vc_events["zone_hi"].values.astype(np.float64)
    vc_dir = vc_events["direction"].values

    for i, r in enumerate(ob_df.itertuples()):
        rt = int(retest_ts[i])
        bt = int(brk_ts[i])
        if rt >= 2**60:
            continue
        cur_open = int(r.cur_open)
        cap_end = cur_open + TF_MS_4H + cap_ms
        window_end = min(bt, cap_end) if bt < 2**60 else cap_end
        need_dir = r.direction
        ob_lo, ob_hi = ob_zone(r)

        mask = (
            (vc_born >= rt)
            & (vc_born < window_end)
            & (vc_dir == need_dir)
            & (vc_hi >= ob_lo)
            & (vc_lo <= ob_hi)
        )
        if mask.any():
            result[i] = True
    return result


def get_vc_born_ts(ob_df: pd.DataFrame, retest_ts: np.ndarray, brk_ts: np.ndarray,
                   vc_events: pd.DataFrame, cap_ms: int) -> np.ndarray:
    """Возвращает min(born_ts) первого валидного VC события, 0 если не найдено."""
    result = np.zeros(len(ob_df), dtype=np.int64)
    vc_born = vc_events["born_ts"].values.astype(np.int64)
    vc_lo = vc_events["zone_lo"].values.astype(np.float64)
    vc_hi = vc_events["zone_hi"].values.astype(np.float64)
    vc_dir = vc_events["direction"].values

    for i, r in enumerate(ob_df.itertuples()):
        rt = int(retest_ts[i])
        bt = int(brk_ts[i])
        if rt >= 2**60:
            continue
        cur_open = int(r.cur_open)
        cap_end = cur_open + TF_MS_4H + cap_ms
        window_end = min(bt, cap_end) if bt < 2**60 else cap_end
        need_dir = r.direction
        ob_lo, ob_hi = ob_zone(r)

        mask = (
            (vc_born >= rt)
            & (vc_born < window_end)
            & (vc_dir == need_dir)
            & (vc_hi >= ob_lo)
            & (vc_lo <= ob_hi)
        )
        hits = vc_born[mask]
        if len(hits) > 0:
            result[i] = int(hits.min())
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", choices=["BTC", "ETH", "SOL"])
    ap.add_argument("--end", default=None, help="Period end YYYY-MM-DD (informational)")
    args = ap.parse_args()

    symbol = args.symbol
    CSV_1M = DATA_DIR / f"{symbol}USDT_1m.csv"
    EVENTS_PATH = find_latest_events(symbol)
    STAGE3_CANONICAL = STAGE_DIR / f"ob_stage3_canonical_{symbol}.parquet"
    STAGE3_SWEEP = STAGE_DIR / f"ob_stage3_sweep_only_{symbol}.parquet"

    print(f"[cfg] symbol={symbol}")
    print(f"[cfg] STAGE3_CANONICAL = {STAGE3_CANONICAL.name}")
    print(f"[cfg] STAGE3_SWEEP = {STAGE3_SWEEP.name}")

    print("Loading v2 + events + 1m ...")
    v2_can = pd.read_parquet(STAGE3_CANONICAL)
    v2_sw = pd.read_parquet(STAGE3_SWEEP)
    events = pd.read_parquet(EVENTS_PATH)
    m1 = load_1m(CSV_1M)
    print(f"  OB canonical: {len(v2_can):,}   sweep_only: {len(v2_sw):,}   events: {len(events):,}   1m: {len(m1):,}")

    sweep_only = v2_sw[v2_sw["stage3_passed"]].copy().reset_index(drop=True)
    canonical = v2_can[v2_can["stage3_passed"]].copy().reset_index(drop=True)
    print(f"  sweep-only: {len(sweep_only):,}   canonical: {len(canonical):,}")

    # VC pools: только 1h (X-1 по канону для 4h POI)
    print("Building VC pools (1h only) ...")
    pools = {}
    pools["rb_1h"] = build_element_born(events, "rb", "1h")
    pools["fvg_1h"] = build_element_born(events, "fvg", "1h")
    pools["snr_1h"] = build_element_born(events, "breaker_block", "1h")
    print(f"  1h:  RB={len(pools['rb_1h']):,}  FVG={len(pools['fvg_1h']):,}  SNR={len(pools['snr_1h']):,}")

    cap_ms = WATCH_CAP_H * 3600 * 1000

    # Aggregate 1m → TF_MS_RETEST (1h) для retest/invalidation
    bars_retest = aggregate_to_tf(m1, TF_MS_RETEST)
    print(f"[retest TF] 1m → {TF_MS_RETEST // 60000}m: {len(bars_retest):,} баров")

    for name, cohort in [("sweep-only", sweep_only), ("canonical", canonical)]:
        print(f"\n[{name}] compute retest_ts + invalidation (wick на {TF_MS_RETEST//60000}m) ...")
        cohort["retest_ts"] = find_retest_ts_batch(cohort, bars_retest, cap_ms)
        cohort["body_break_ts"] = find_invalidation_ts_batch(cohort, bars_retest, cap_ms)
        n_retest = int((cohort["retest_ts"] < 2**60).sum())
        n_brk = int((cohort["body_break_ts"] < 2**60).sum())
        print(f"    retest in {WATCH_CAP_H}h: {n_retest:,} / {len(cohort):,}")
        print(f"    invalidation (за OB zone) in {WATCH_CAP_H}h: {n_brk:,} / {len(cohort):,}")

        print(f"[{name}] check valid VC (race window [retest, invalidation]) ...")
        rt = cohort["retest_ts"].values.astype(np.int64)
        bt = cohort["body_break_ts"].values.astype(np.int64)

        cohort["vc_rb"] = check_valid_vc(cohort, rt, bt, pools["rb_1h"], cap_ms)
        cohort["vc_fvg"] = check_valid_vc(cohort, rt, bt, pools["fvg_1h"], cap_ms)
        cohort["vc_snr"] = check_valid_vc(cohort, rt, bt, pools["snr_1h"], cap_ms)
        cohort["vc_double"] = cohort["vc_rb"] & cohort["vc_fvg"]
        cohort["vc_triple"] = cohort["vc_rb"] & cohort["vc_fvg"] & cohort["vc_snr"]
        cohort["vc_any"] = cohort["vc_rb"] | cohort["vc_fvg"] | cohort["vc_snr"]
        cohort["vc_rb_ts"]  = get_vc_born_ts(cohort, rt, bt, pools["rb_1h"],  cap_ms)
        cohort["vc_fvg_ts"] = get_vc_born_ts(cohort, rt, bt, pools["fvg_1h"], cap_ms)
        cohort["vc_snr_ts"] = get_vc_born_ts(cohort, rt, bt, pools["snr_1h"], cap_ms)

    print()
    print("═" * 105)
    print(f"Этап 4 — RACE (retest → VC/break) на 1h. VC zone overlaps body OB.")
    print("═" * 105)
    print(f"{'Метрика':50s}  {'sweep-only':>20s}  {'canonical':>20s}")
    print("─" * 105)

    def row(label, mask_so, mask_can):
        n_so = int(mask_so.sum()); n_can = int(mask_can.sum())
        so_pct = (n_so / len(sweep_only) * 100) if len(sweep_only) else 0.0
        can_pct = (n_can / len(canonical) * 100) if len(canonical) else 0.0
        print(f"{label:50s}  {n_so:>4,}  ({so_pct:5.1f}%)  {n_can:>4,}  ({can_pct:5.1f}%)")

    row("Total", pd.Series([True]*len(sweep_only)), pd.Series([True]*len(canonical)))
    print()
    row("valid RB 1h",   sweep_only["vc_rb"],   canonical["vc_rb"])
    row("valid FVG 1h",  sweep_only["vc_fvg"],  canonical["vc_fvg"])
    row("valid SNR 1h",  sweep_only["vc_snr"],  canonical["vc_snr"])
    print()
    row("valid RB+FVG (double)",  sweep_only["vc_double"],   canonical["vc_double"])
    row("valid triple",           sweep_only["vc_triple"],   canonical["vc_triple"])
    row("valid ANY (RB|FVG|SNR)", sweep_only["vc_any"],      canonical["vc_any"])
    print()
    row("NO valid VC (отсеиваем)", ~sweep_only["vc_any"],    ~canonical["vc_any"])

    print()
    print("По направлениям (canonical):")
    for d in ("long", "short"):
        can_d = canonical[canonical["direction"] == d]
        n = len(can_d)
        if n == 0:
            continue
        rb = int(can_d["vc_rb"].sum())
        fvg = int(can_d["vc_fvg"].sum())
        snr = int(can_d["vc_snr"].sum())
        dbl = int(can_d["vc_double"].sum())
        tri = int(can_d["vc_triple"].sum())
        any_ = int(can_d["vc_any"].sum())
        print(f"  {d:5s} (n={n:>4,}):  RB={rb:>4,}  FVG={fvg:>4,}  SNR={snr:>4,}  double={dbl:>4,}  triple={tri:>4,}  any={any_:>4,} ({any_/n*100:.1f}%)")

    canonical["stage4_passed"] = canonical["vc_any"] & (canonical["retest_ts"] < 2**60)
    sweep_only["stage4_passed"] = sweep_only["vc_any"] & (sweep_only["retest_ts"] < 2**60)
    n_can_final = int(canonical["stage4_passed"].sum())
    n_sw_final = int(sweep_only["stage4_passed"].sum())
    print(f"\n[stage4 final] canonical: {n_can_final}/{len(canonical)}   sweep-only: {n_sw_final}/{len(sweep_only)}")

    out_sw = STAGE_DIR / f"ob_stage4_race_sweep_only_{symbol}.parquet"
    out_can = STAGE_DIR / f"ob_stage4_race_canonical_{symbol}.parquet"
    sweep_only.to_parquet(out_sw, index=False)
    canonical.to_parquet(out_can, index=False)
    print(f"\nsaved: {out_sw.name}")
    print(f"saved: {out_can.name}")


if __name__ == "__main__":
    main()
