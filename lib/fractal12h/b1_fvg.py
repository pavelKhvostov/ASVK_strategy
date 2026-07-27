"""B1 FVG — canon-aligned rewrite (first-touch semantic).

Портировано из ~/smc-warehouse/scripts/фрактал-12h/b1_fvg.py (WSL, read-only источник).

Ключевые отличия от предыдущей версии:
  * first-touch semantic: одна зона fires МАКСИМУМ один раз (первое событие)
  * ATR14 = simple SMA(TR, 14) — как в каноне (не Wilder EWM)
  * reuse ASVK e12d events (не пересчитываем FVG-детектор — G:\\ASVK\\lib\\e12d.py уже
    гоняет fvg-детектор для BTC/ETH/SOL каждые 15 мин)
  * векторизация per-zone через numpy (не Python-loop 12h-bars)
  * fires = set of (bar_idx, direction) tuples; match с A1 pivots на выходе

Sub-conditions (B1_v5, causal-only):
  B1C1  strict sweep  S100 / WIDE                   full puncture + wide zones
  B1C2  strict sweep  S50  / AGE-WIDE               aged + wide, 50%-sweep
  B1C3  strict sweep  S70  / AGE-WIDE               aged + wide, 70%-sweep
  B1C4  strict sweep  S50  / HTF-WIDE               HTF wide zones

Direction: SHORT pivot ↔ SHORT FVG (resistance above). LONG pivot ↔ LONG FVG (support below).
TFs: {12h, 1d, 2d, 3d} — canon использует также W (weekly); у e12d ASVK нет 1w в SIMPLE_DETECTORS
для fvg на 1w, поэтому небольшое расхождение с WSL-каноном ожидается.

Reads:  G:\\ASVK\\data\\fractal12h\\a_candidates_{SYM}_{start}_{end}.parquet
        G:\\ASVK\\data\\events\\events_e12d_{SYM}_*.parquet (последний по дате)
        G:\\ASVK\\data\\{SYM}USDT_1m.csv
Writes: G:\\ASVK\\data\\fractal12h\\b1_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — для maxv (level-1)
from common import load_1m, agg_12h, DATA_OUT, TF_12H_MS, latest_events_path
from maxv import latest_maxv_path


# ── Каноничные параметры (заморожены per B1_fvg.md v4) ──
ATR_N       = 14
WIDE_MULT   = 0.7
AGE_N       = 50
HTF_TFS     = ("1d", "2d", "3d", "1w")   # canon HTF = D, 2D, 3D, W
FVG_TFS     = ("12h",) + HTF_TFS


def load_atr14(symbol: str, t12: np.ndarray) -> np.ndarray:
    """ATR14(12h) — level-1 shared indicator, см. G:\\ASVK\\lib\\maxv.py (считается один
    раз в pipeline Block 1, здесь только читается и матчится по ts)."""
    mv = pd.read_parquet(latest_maxv_path(symbol), columns=["ts", "atr14"])
    s = pd.Series(mv["atr14"].to_numpy(), index=mv["ts"].to_numpy())
    return s.reindex(t12).to_numpy()


# ── Loaders ───────────────────────────────────────────────

def load_fvg_zones(symbol: str) -> pd.DataFrame:
    """FVG born-events из последнего events_e12d_{symbol}_*.parquet (ASVK e12d), только canonical TFs."""
    p = latest_events_path(symbol)
    df = pd.read_parquet(p)
    df = df[(df["element"] == "fvg") & (df["kind"] == "born") & df["tf"].isin(FVG_TFS)]
    return (df.groupby("zone_id", as_index=False).first()
              [["zone_id", "ts", "tf", "direction", "zone_lo", "zone_hi"]]
              .rename(columns={"ts": "born_ts"}))


# ── Indicators ────────────────────────────────────────────
# ATR14(12h) — level-1 shared indicator, читается через load_atr14() выше
# (см. G:\ASVK\lib\maxv.py). Здесь остаётся только то, что специфично для B1.

# ── Zone-events builder (векторизованно per zone) ────────

def build_zone_events(zones: pd.DataFrame,
                      t12: np.ndarray, h12: np.ndarray,
                      l12: np.ndarray, c12: np.ndarray) -> list[dict]:
    """Per-zone: список 12h-баров где wick касался + метрики (pen, ci, co_far).

    Один проход per zone через numpy = сотни зон/сек вместо десятков.
    """
    n12 = len(t12)
    out = []
    for z in zones.to_dict(orient="records"):
        zlo, zhi = float(z["zone_lo"]), float(z["zone_hi"])
        w = zhi - zlo
        if w <= 0:
            continue
        sp = int(np.searchsorted(t12, int(z["born_ts"]), side="left"))
        if sp >= n12:
            continue
        hh, ll, cc = h12[sp:], l12[sp:], c12[sp:]
        direction = z["direction"]
        if direction == "short":
            touch = hh >= zlo
            pen = (np.minimum(hh, zhi) - zlo) / w * 100.0
            co_far = cc < zlo
        else:
            touch = ll <= zhi
            pen = (zhi - np.maximum(ll, zlo)) / w * 100.0
            co_far = cc > zhi
        ci = (cc >= zlo) & (cc <= zhi)
        idx_local = np.flatnonzero(touch)
        if idx_local.size == 0:
            continue
        # events: chronological list of (k_global, pen, ci, co_far)
        k_global = (sp + idx_local).astype(np.int64)
        events = list(zip(k_global.tolist(),
                          pen[idx_local].tolist(),
                          ci[idx_local].tolist(),
                          co_far[idx_local].tolist()))
        out.append({
            "tf": z["tf"], "direction": direction,
            "zlo": zlo, "zhi": zhi, "born_ts": int(z["born_ts"]),
            "events": events,
        })
    return out


# ── Filter ────────────────────────────────────────────────

def zone_passes(z: dict, k: int, ftype: str,
                t12: np.ndarray, atr14: np.ndarray) -> bool:
    """Canonical фильтр зоны на bar k."""
    if ftype == "ANY":
        return True
    width = z["zhi"] - z["zlo"]
    atr = atr14[k] if atr14[k] > 0 else 1.0
    wide = width / atr >= WIDE_MULT
    if ftype == "WIDE":
        return wide
    age = (t12[k] - z["born_ts"]) // TF_12H_MS
    aged = age >= AGE_N
    if ftype == "AGE50":
        return aged
    if ftype == "AGE50_WIDE":
        return aged and wide
    if ftype == "HTF_WIDE":
        return (z["tf"] in HTF_TFS) and wide
    return False


# ── Evaluators (first-touch semantic) ────────────────────

def eval_classic(zone_events: list[dict], pen_min: float, ftype: str,
                 t12: np.ndarray, atr14: np.ndarray) -> set[tuple]:
    """Strict sweep classic: одна зона fires ОДИН РАЗ на первое qualifying event."""
    fires = set()
    for z in zone_events:
        for (k, pen, ci, co_far) in z["events"]:
            if pen < pen_min:
                continue
            if not co_far:
                continue
            if not zone_passes(z, k, ftype, t12, atr14):
                continue
            fires.add((k, z["direction"]))
            break
    return fires





# ── Assembly ──────────────────────────────────────────────

def compute_b1(a_cand: pd.DataFrame, zones: pd.DataFrame,
               df_12h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    t12 = df_12h["ts"].to_numpy()
    h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy()
    c12 = df_12h["close"].to_numpy()
    atr14 = load_atr14(symbol, t12)

    print(f"  building zone events...", file=sys.stderr, flush=True)
    t0 = time.time()
    zone_events = build_zone_events(zones, t12, h12, l12, c12)
    total_ev = sum(len(z["events"]) for z in zone_events)
    print(f"  {len(zone_events):,} zones × avg {total_ev/max(len(zone_events),1):.1f} events "
          f"= {total_ev:,} total events in {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)

    print(f"  evaluating B1C1..B1C4...", file=sys.stderr, flush=True)
    t0 = time.time()
    B1C1 = eval_classic(zone_events, 100, "WIDE",       t12, atr14)
    B1C2 = eval_classic(zone_events,  50, "AGE50_WIDE", t12, atr14)
    B1C3 = eval_classic(zone_events,  70, "AGE50_WIDE", t12, atr14)
    B1C4 = eval_classic(zone_events,  50, "HTF_WIDE",   t12, atr14)
    B1_union = B1C1 | B1C2 | B1C3 | B1C4
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    # Match с A1 pool (все Williams-confirmable pivots, БЕЗ A2/A3/A4 фильтров)
    ts_to_idx = {int(t): k for k, t in enumerate(t12)}
    a1 = a_cand.copy()  # a_candidates уже все A1 pivots
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
            "direction": direction,
            "confirmable": bool(row["confirmable"]),
            "confirmed":   bool(row["confirmed"]),
            "b1c1": key in B1C1, "b1c2": key in B1C2, "b1c3": key in B1C3,
            "b1c4": key in B1C4,
            "b1_hit": key in B1_union,
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b1c1", "b1c2", "b1c3", "b1c4", "b1_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        marker = "  ← B1" if col == "b1_hit" else ""
        print(f"  {col:8s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{marker}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-08")
    args = ap.parse_args()

    print(f"b1_fvg (ASVK-portable): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {len(a_cand):,}", file=sys.stderr, flush=True)

    zones = load_fvg_zones(args.symbol)
    print(f"  FVG zones (TFs={FVG_TFS}): {len(zones):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_b1(a_cand, zones, df_12h, args.symbol)
    print_stats(hits)

    out = DATA_OUT / f"b1_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
