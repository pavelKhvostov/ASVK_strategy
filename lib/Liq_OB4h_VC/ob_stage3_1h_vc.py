"""Этап 3 — VC (Volume Confirmation) на 1h после 4h OB.

По методичке (стр 34-35):
  VC на X-1 (для нас 1h) = появление RB / FVG / SNR правильного направления
  после теста POI (X = 4h OB).

Здесь берём RB и FVG (SNR приблизим через ob_vc позже, если нужно).

Правила проверки:
  для LONG OB   : ищем bullish RB на 1h ИЛИ bullish FVG на 1h
                  born в окне [cur_close, cur_close + 48h]
                  zone пересекается с диапазоном 4h OB [cur.low, cur.high]
  для SHORT OB  : зеркально bearish

Отдельно считаем:
  - RB only
  - FVG only
  - RB и FVG вместе (сильная комбинация по методичке)
  - Любой VC (RB OR FVG)

Ведём параллельно 541 sweep-only и 377 canonical.
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

SEARCH_WINDOW_H = 48   # окно поиска VC от cur close
TF_MS_4H = 4 * 3600 * 1000


def find_latest_events(symbol: str) -> Path:
    files = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No events_e12d_{symbol}_*.parquet found in {EVENTS_DIR}")
    return files[-1]


def build_element_born(events: pd.DataFrame, element: str, tf: str) -> pd.DataFrame:
    sub = events[
        (events["tf"] == tf)
        & (events["element"] == element)
        & (events["kind"] == "born")
    ][["ts", "zone_id", "zone_lo", "zone_hi", "direction"]]
    return sub.rename(columns={"ts": "born_ts"}).sort_values("born_ts").reset_index(drop=True)


def ob_zone(r) -> tuple[float, float]:
    """OB zone canonical:
       LONG:  [min(prev.low, cur.low), prev.open]
       SHORT: [prev.open, max(prev.high, cur.high)]"""
    if r["direction"] == "long":
        return float(min(r["prev_l"], r["cur_l"])), float(r["prev_o"])
    else:
        return float(r["prev_o"]), float(max(r["prev_h"], r["cur_h"]))


def check_vc_batch(ob_df: pd.DataFrame, vc_events: pd.DataFrame, need_dir: dict, window_ms: int) -> np.ndarray:
    """
    Для каждого OB row — True если существует VC event такой что:
      born_ts in [cur_close, cur_close + window]
      direction соответствует OB (need_dir[ob.direction])
      zone overlaps [OB.zone_lo, OB.zone_hi]
    """
    result = np.zeros(len(ob_df), dtype=bool)

    vc_born = vc_events["born_ts"].values.astype(np.int64)
    vc_lo = vc_events["zone_lo"].values.astype(np.float64)
    vc_hi = vc_events["zone_hi"].values.astype(np.float64)
    vc_dir = vc_events["direction"].values

    for i, r in enumerate(ob_df.itertuples()):
        cur_close = int(r.cur_open) + TF_MS_4H
        end = cur_close + window_ms
        ob_dir = r.direction
        want = need_dir[ob_dir]  # какой direction VC ищем
        if ob_dir == "long":
            ob_lo = float(min(r.prev_l, r.cur_l))
            ob_hi = float(r.prev_o)
        else:
            ob_lo = float(r.prev_o)
            ob_hi = float(max(r.prev_h, r.cur_h))

        mask = (
            (vc_born >= cur_close)
            & (vc_born <= end)
            & (vc_dir == want)
            & (vc_hi >= ob_lo)     # overlap с [ob_lo, ob_hi]
            & (vc_lo <= ob_hi)
        )
        if mask.any():
            result[i] = True
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", choices=["BTC", "ETH", "SOL"])
    ap.add_argument("--end", default=None, help="Period end YYYY-MM-DD (informational)")
    args = ap.parse_args()

    symbol = args.symbol
    EVENTS_PATH = find_latest_events(symbol)
    STAGE2_CANONICAL = STAGE_DIR / f"ob_stage2_canonical_{symbol}.parquet"
    STAGE2_SWEEP = STAGE_DIR / f"ob_stage2_sweep_only_{symbol}.parquet"

    print(f"[cfg] symbol={symbol}")
    print(f"[cfg] STAGE2_CANONICAL = {STAGE2_CANONICAL.name}")
    print(f"[cfg] STAGE2_SWEEP = {STAGE2_SWEEP.name}")
    print("Loading stage 2 outputs + events...")
    events = pd.read_parquet(EVENTS_PATH)

    canonical = pd.read_parquet(STAGE2_CANONICAL)
    canonical = canonical[canonical["stage2_passed"]].copy() if "stage2_passed" in canonical.columns else canonical.copy()
    sweep_only = pd.read_parquet(STAGE2_SWEEP)
    sweep_only = sweep_only[sweep_only["stage2_passed"]].copy() if "stage2_passed" in sweep_only.columns else sweep_only.copy()
    print(f"  sweep-only: {len(sweep_only):,}   canonical: {len(canonical):,}")

    # 1h events: RB, FVG, breaker_block (=SNR по AlexxxFlow: RBS/SBR)
    rb_1h = build_element_born(events, "rb", "1h")
    fvg_1h = build_element_born(events, "fvg", "1h")
    snr_1h = build_element_born(events, "breaker_block", "1h")
    print(f"  RB 1h: {len(rb_1h):,}   FVG 1h: {len(fvg_1h):,}   SNR (breaker) 1h: {len(snr_1h):,}")
    print()

    # Направление VC: для LONG OB нужен bullish → 'long'; для SHORT → 'short'
    # (в наших events direction означает направление зоны: long=support/bullish, short=resistance/bearish)
    need_dir = {"long": "long", "short": "short"}
    window_ms = SEARCH_WINDOW_H * 3600 * 1000

    for name, cohort in [("sweep-only", sweep_only), ("canonical", canonical)]:
        print(f"[check] {name}...")
        cohort["vc_rb"] = check_vc_batch(cohort, rb_1h, need_dir, window_ms)
        cohort["vc_fvg"] = check_vc_batch(cohort, fvg_1h, need_dir, window_ms)
        cohort["vc_snr"] = check_vc_batch(cohort, snr_1h, need_dir, window_ms)
        cohort["vc_rb_and_fvg"] = cohort["vc_rb"] & cohort["vc_fvg"]
        cohort["vc_triple"] = cohort["vc_rb"] & cohort["vc_fvg"] & cohort["vc_snr"]  # mega по методичке
        cohort["vc_any"] = cohort["vc_rb"] | cohort["vc_fvg"] | cohort["vc_snr"]

    print()
    print("═" * 100)
    print(f"Этап 3 — VC на 1h после 4h OB в окне {SEARCH_WINDOW_H}h, zone overlaps [cur.low, cur.high]")
    print("═" * 100)
    print(f"{'Метрика':50s}  {'sweep-only (stage2)':>22s}  {'canonical (stage2)':>22s}")
    print("─" * 100)

    def row(label, mask_so, mask_can):
        n_so = int(mask_so.sum())
        n_can = int(mask_can.sum())
        so_pct = (n_so / len(sweep_only) * 100) if len(sweep_only) else 0.0
        can_pct = (n_can / len(canonical) * 100) if len(canonical) else 0.0
        print(f"{label:50s}  {n_so:>4,}  ({so_pct:5.1f}%)  {n_can:>4,}  ({can_pct:5.1f}%)")

    row("Total", pd.Series([True]*len(sweep_only)), pd.Series([True]*len(canonical)))
    print()
    row("VC RB 1h",                     sweep_only["vc_rb"],           canonical["vc_rb"])
    row("VC FVG 1h",                    sweep_only["vc_fvg"],          canonical["vc_fvg"])
    row("VC SNR (breaker_block) 1h",    sweep_only["vc_snr"],          canonical["vc_snr"])
    print()
    row("VC RB + FVG (двойной)",        sweep_only["vc_rb_and_fvg"],   canonical["vc_rb_and_fvg"])
    row("VC RB + SNR + FVG (тройной)",  sweep_only["vc_triple"],       canonical["vc_triple"])
    row("VC any (RB or FVG or SNR)",    sweep_only["vc_any"],          canonical["vc_any"])
    print()
    row("NO VC",                        ~sweep_only["vc_any"],         ~canonical["vc_any"])

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
        combo = int(can_d["vc_rb_and_fvg"].sum())
        triple = int(can_d["vc_triple"].sum())
        any_ = int(can_d["vc_any"].sum())
        print(f"  {d:5s} (n={n:>4,}):  RB={rb:>4,}  FVG={fvg:>4,}  SNR={snr:>4,}  RB+FVG={combo:>4,}  triple={triple:>4,}  any={any_:>4,} ({any_/n*100:.1f}%)")

    # Пропускаем только те, у кого есть VC хотя бы одного типа в 48h окне
    sweep_only["stage3_passed"] = sweep_only["vc_any"]
    canonical["stage3_passed"] = canonical["vc_any"]
    n_can_pass = int(canonical["stage3_passed"].sum())
    n_sw_pass = int(sweep_only["stage3_passed"].sum())
    print(f"\n[stage3 filter] canonical: {n_can_pass}/{len(canonical)}   sweep-only: {n_sw_pass}/{len(sweep_only)}")

    out_sw = STAGE_DIR / f"ob_stage3_sweep_only_{symbol}.parquet"
    out_can = STAGE_DIR / f"ob_stage3_canonical_{symbol}.parquet"
    sweep_only.to_parquet(out_sw, index=False)
    canonical.to_parquet(out_can, index=False)
    print(f"\nsaved: {out_sw.name}")
    print(f"saved: {out_can.name}")


if __name__ == "__main__":
    main()
