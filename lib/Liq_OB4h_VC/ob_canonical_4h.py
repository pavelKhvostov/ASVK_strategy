"""OB canonical 4h v2 — first-touch sweep semantics.

Отличие от v1:
  v1: FL живой если retire_ts из events ещё не наступил (formal retire).
      Bug: retire для 1D fractal происходит только при close 1D bar → FL
      формально жив ещё сутки после реального пробития wick'ом.
  v2: FL живой ТОЛЬКО пока НИ ОДИН bar не пробил его low.
      Sweep valid ⇔ prev или cur — ПЕРВЫЙ bar, пробивший FL wick'ом.
      Требование: max(low всех баров от born до prev_open) ≥ FL.level
                  AND min(prev.low, cur.low) < FL.level

Semantics OB canonical (методичка стр 12):
  1) Свечное поглощение 1 свечой (bull/bear engulf).
  2) Снятие ликвидности FL/FH — first touch prev/cur своим range.

Sweep TFs: 4h, 12h, 1D. Проверяем first-touch по 4h bars
(это самый жёсткий базис — если 1D FL пробит на 4h wick'ом, значит снят).
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
OUT_DIR = DATA_DIR / "liq_ob4h_vc"

TF_MS = {"4h": 4*3600*1000, "6h": 6*3600*1000, "12h": 12*3600*1000, "1d": 24*3600*1000, "2d": 2*24*3600*1000, "3d": 3*24*3600*1000, "1w": 7*24*3600*1000}


def build_config(symbol: str, end_date: str) -> dict:
    return {
        "symbol":            symbol,
        "target_tf":         "4h",
        "poi_element":       "ob",
        "engulf":            "diagnostic_only",
        "sweep_tfs":          ["4h", "6h", "12h", "1d", "2d", "3d", "1w"],
        "sweep_semantics":   "first_touch_by_4h_bars",
        "sweep_bar_scope":   "prev_or_cur",
        "period_start":      "2018-01-01",
        "period_end":        end_date,
        "version":           4,
    }


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def find_latest_events(symbol: str) -> Path:
    files = sorted(EVENTS_DIR.glob(f"events_e12d_{symbol}_*.parquet"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No events_e12d_{symbol}_*.parquet found in {EVENTS_DIR}")
    return files[-1]


def aggregate_4h(csv_path: Path, period_end: str) -> pd.DataFrame:
    print(f"[agg] loading {csv_path.name} ...")
    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=["open_time","open","high","low","close"])
    dt = pd.to_datetime(df["open_time"], format="ISO8601", utc=True)
    df["ts_ms"] = dt.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
    tf = TF_MS["4h"]
    df["bucket"] = (df["ts_ms"] // tf) * tf
    agg = df.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index().rename(columns={"bucket": "ts_ms"})
    period_end_ms = int(pd.Timestamp(period_end, tz="UTC").value // 1_000_000)
    last_full = (period_end_ms // tf) * tf
    agg = agg[agg["ts_ms"] < last_full].reset_index(drop=True)
    print(f"[agg] 4h bars: {len(agg):,} ({time.time() - t0:.1f}s)")
    return agg


def build_fractal_intervals(events: pd.DataFrame, tf: str, direction: str) -> pd.DataFrame:
    sub = events[
        (events["tf"] == tf)
        & (events["element"] == "fractal")
        & (events["direction"] == direction)
    ]
    born = sub[sub["kind"] == "born"][["ts", "zone_id", "zone_lo", "zone_hi"]].rename(
        columns={"ts": "born_ts"}
    )
    ret = sub[sub["kind"] == "retire"][["ts", "zone_id"]].rename(columns={"ts": "retire_ts"})
    m = born.merge(ret, on="zone_id", how="left")
    m["level"] = m["zone_lo"] if direction == "long" else m["zone_hi"]
    return m.sort_values("born_ts").reset_index(drop=True)


def compute_first_breach_ts(
    fractals: pd.DataFrame,
    bars: pd.DataFrame,
    direction: str,
) -> np.ndarray:
    """
    Для каждого фрактала возвращает ts_ms 4h-бара, который первым пробил его wick'ом.
    Если не пробит до конца данных — возвращает 2**62.

    LONG (FL):  ищем первый bar с low  < FL.level  ПОСЛЕ born_ts
    SHORT (FH): ищем первый bar с high > FH.level  ПОСЛЕ born_ts

    Возвращает shape (len(fractals),) int64.
    """
    if len(fractals) == 0:
        return np.array([], dtype=np.int64)

    bar_ts = bars["ts_ms"].values.astype(np.int64)
    bar_low = bars["low"].values.astype(np.float64)
    bar_high = bars["high"].values.astype(np.float64)

    born = fractals["born_ts"].values.astype(np.int64)
    level = fractals["level"].values.astype(np.float64)

    result = np.full(len(fractals), 2**62, dtype=np.int64)
    for i in range(len(fractals)):
        # side='left' — включаем бар born_ts. Если fractal breach'нут В момент рождения
        # (высокая внутри born_ts бара уже за уровнем), first_breach = born_ts, а не следующий бар.
        start_idx = int(np.searchsorted(bar_ts, born[i], side="left"))
        if start_idx >= len(bar_ts):
            continue
        if direction == "long":
            mask = bar_low[start_idx:] < level[i]
        else:
            mask = bar_high[start_idx:] > level[i]
        idx = np.argmax(mask)
        if mask[idx]:
            result[i] = bar_ts[start_idx + idx]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", choices=["BTC", "ETH", "SOL"])
    ap.add_argument("--end", default=None, help="Period end YYYY-MM-DD (default: tomorrow UTC)")
    args = ap.parse_args()

    symbol = args.symbol
    end_date = args.end or (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    CONFIG = build_config(symbol, end_date)
    CSV_1M = DATA_DIR / f"{symbol}USDT_1m.csv"
    EVENTS_PATH = find_latest_events(symbol)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"OB canonical 4h v2  symbol={symbol}  end={end_date}  cfg_hash={cfg_hash(CONFIG)}")
    print(f"  events: {EVENTS_PATH.name}")
    print(f"  csv:    {CSV_1M.name}")
    print("─" * 60)

    events = pd.read_parquet(EVENTS_PATH)
    print(f"[load] events: {len(events):,}")

    ob_born = events[
        (events["tf"] == "4h")
        & (events["element"] == "ob")
        & (events["kind"] == "born")
    ].sort_values("ts").reset_index(drop=True)
    print(f"[filter] 4h OB born: {len(ob_born):,}")

    bars = aggregate_4h(CSV_1M, end_date)
    ts_to_idx = dict(zip(bars["ts_ms"].values, bars.index.values))

    print("[frac] intervals + first-breach ...")
    fr = {}
    for tf in CONFIG["sweep_tfs"]:
        fr[tf] = {}
        for d in ("long", "short"):
            fdf = build_fractal_intervals(events, tf, d)
            fdf["first_breach_ts"] = compute_first_breach_ts(fdf, bars, d)
            fr[tf][d] = fdf
            n_breach = int((fdf["first_breach_ts"] < 2**62).sum())
            print(f"   {tf}  {d}:  n={len(fdf):>6,}  breached={n_breach:>6,}")

    tf_ms = TF_MS["4h"]

    print(f"[audit] scanning {len(ob_born):,} OBs...")
    t0 = time.time()
    rows = []
    n_no_bar = 0
    for _, r in ob_born.iterrows():
        ob_ts = int(r["ts"])
        direction = r["direction"]
        cur_open = ob_ts - tf_ms
        prev_open = cur_open - tf_ms
        ci = ts_to_idx.get(cur_open)
        if ci is None or ci < 1:
            n_no_bar += 1
            continue
        cur = bars.iloc[ci]
        prev = bars.iloc[ci - 1]

        if direction == "long":
            engulf = bool(
                (cur["close"] > cur["open"])
                and (prev["close"] < prev["open"])
                and (cur["close"] >= prev["open"])
                and (cur["open"] <= prev["close"])
            )
        else:
            engulf = bool(
                (cur["close"] < cur["open"])
                and (prev["close"] > prev["open"])
                and (cur["close"] <= prev["open"])
                and (cur["open"] >= prev["close"])
            )

        sweep_hits = {}
        if direction == "long":
            # LONG: sweep wick идёт ВНИЗ через FL, ниже body prev bar.
            #   min_low = внешний край свипа (нижний wick)
            #   body_bot = prev.close (тело bearish prev — нижняя граница body)
            #   valid sweep zone для FL: (min_low, body_bot) — фрактал между wick_bot и body
            min_low = float(min(prev["low"], cur["low"]))
            body_bot = float(prev["close"])
        else:
            # SHORT: sweep wick идёт ВВЕРХ через FH, выше body prev bar.
            #   max_high = внешний край свипа (верхний wick)
            #   body_top = prev.close (тело bullish prev — верхняя граница body)
            #   valid sweep zone для FH: (body_top, max_high) — фрактал между body и wick_top
            max_high = float(max(prev["high"], cur["high"]))
            body_top = float(prev["close"])

        for tf in CONFIG["sweep_tfs"]:
            fdf = fr[tf][direction]
            born_ok = fdf["born_ts"].values < cur_open
            first_breach = fdf["first_breach_ts"].values
            breach_at_prev_or_cur = (first_breach == prev_open) | (first_breach == cur_open)
            level = fdf["level"].values
            if direction == "long":
                # Fractal FL должен быть в sweep zone: (min_low, body_bot)
                breach_price = (level > min_low) & (level < body_bot)
            else:
                # Fractal FH должен быть в sweep zone: (body_top, max_high)
                breach_price = (level < max_high) & (level > body_top)
            mask = born_ok & breach_at_prev_or_cur & breach_price
            n_swept = int(mask.sum())
            if n_swept > 0:
                sweep_hits[tf] = n_swept

        sweep_any = len(sweep_hits) > 0
        # engulf filter removed 2026-07-12: crypto cur.open=prev.close, so
        # canonical ob.py (cur.close > prev.open) already implies full body engulf.
        # engulf column kept as diagnostic only.
        passed = sweep_any

        rows.append({
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
            "sweep_4h":  sweep_hits.get("4h", 0),
            "sweep_6h":  sweep_hits.get("6h", 0),
            "sweep_12h": sweep_hits.get("12h", 0),
            "sweep_1d":  sweep_hits.get("1d", 0),
            "sweep_2d":  sweep_hits.get("2d", 0),
            "sweep_3d":  sweep_hits.get("3d", 0),
            "sweep_1w":  sweep_hits.get("1w", 0),
            "sweep_any": sweep_any,
            "passed":    passed,
        })

    df = pd.DataFrame(rows)
    dt = time.time() - t0
    print(f"[audit] done in {dt:.1f}s (skipped no-bar: {n_no_bar})")
    print()

    n_all = len(df)
    n_engulf = int(df["engulf"].sum())
    n_sweep = int(df["sweep_any"].sum())
    n_pass = int(df["passed"].sum())
    print("═" * 60)
    print(f"OB canonical 4h v2 — first-touch sweep ({symbol} {CONFIG['period_start']} → {CONFIG['period_end']})")
    print("═" * 60)
    print(f"total OB born (with bars): {n_all:>6,}")
    if n_all > 0:
        print(f"  engulf (diagnostic):     {n_engulf:>6,}  ({n_engulf/n_all*100:5.1f}%)")
        print(f"  passed sweep (1st touch):{n_sweep:>6,}  ({n_sweep/n_all*100:5.1f}%)")
        print(f"  PASSED both (canonical): {n_pass:>6,}  ({n_pass/n_all*100:5.1f}%)")
    print()
    print("By direction (canonical):")
    for d in ["long", "short"]:
        n = int(((df["direction"] == d) & df["passed"]).sum())
        n_tot = int((df["direction"] == d).sum())
        if n_tot > 0:
            print(f"  {d:5s}: {n:>4,} / {n_tot:>4,}  ({n/n_tot*100:.1f}%)")
    print()
    print("Sweep TF distribution among canonical:")
    can = df[df["passed"]]
    if len(can) > 0 and n_pass > 0:
        for tf in CONFIG['sweep_tfs']:
            n = int((can[f"sweep_{tf}"] > 0).sum())
            print(f"  {tf}:  {n:>4,}  ({n/n_pass*100:5.1f}%)")

    slug = "ob_canonical_4h_v1"
    h = cfg_hash(CONFIG)
    out_p = OUT_DIR / f"{slug}_{symbol}_{h}.parquet"
    out_cfg = OUT_DIR / f"{slug}_{symbol}_{h}_config.json"
    df.to_parquet(out_p, index=False)
    out_cfg.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out_p}")
    print(f"saved: {out_cfg}")


if __name__ == "__main__":
    main()
