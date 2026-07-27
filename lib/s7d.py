"""s7d — per-anchor active-zone snapshots (residual-at-T). Reads e12d output.

Depends only on:
    - data/events/events_e12d_{SYMBOL}_{start}_{end}.parquet   (from e12d)
    - график/{SYMBOL}USDT_1m.csv                                (autonomous)
Writes:
    - data/s7d/snapshots_s7d_{SYMBOL}_{start}_{end}.parquet
    - data/s7d/snapshot_s7d_{SYMBOL}_shards/*.parquet (intermediate)

Contract:
    Anchor cadence: --anchor-tf (default 1h). Anchor ts = bar close time.
    Per-row scope: --scope-pct % от текущей цены (default ±10%).
    Zone considered "active" at T if born_ts ≤ T AND (retire_ts > T OR NaN).
    Residual-at-T = last fill_partial ≤ T (per-zone bisect).

Per-row schema:
    anchor_ts, current_price,
    zone_id, element, tf, direction, role,
    zone_lo, zone_hi          — INITIAL born bounds
    active_lo, active_hi      — RESIDUAL bounds at T
    mit_count_at              — how many fill_partial ≤ T
    level, distance_signed_pct, price_in_zone, dist_to_edge_pct,
    age_ms, mit_pct, symbol

Speedups over s7c:
    - pandas CSV load (60s → ~4s)
    - Prefilter alive zones by INITIAL bounds vs price band before residual bisect
      (10× fewer bisects for far-from-price zones)
    - Column arrays pickled to workers via buffer protocol (not DataFrame)
    - Default n-workers=30 (PC1)
    - CLI: --anchor-tf, --scope-pct
    - Autonomous — no traid-bot refs
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

WAREHOUSE = pathlib.Path(__file__).resolve().parent.parent  # ASVK-standalone

TF_MS: dict[str, int] = {
    "15m": 900_000,     "30m": 1_800_000,
    "1h": 3_600_000,    "2h": 7_200_000,
    "4h": 14_400_000,   "6h": 21_600_000,
    "12h": 43_200_000,  "1d": 86_400_000,
    "2d": 172_800_000,  "3d": 259_200_000,
    "1w": 604_800_000,
}

# Anchor offsets (ms) — sync с e12d TF_ANCHORS.
# TV BINANCE:BTCUSDT verified 2026-07-10:
#   1w = Monday-anchored (2017-01-02 UTC)
#   2d = +1 day shift (Fri-aligned)
TF_ANCHORS: dict[str, int] = {
    "1w": 1_483_315_200_000,
    "2d": 86_400_000,
}


# ─── Load events + build zone repo ─────────────────────────

def load_events(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = ["ts", "kind", "zone_id", "element", "tf", "direction",
                 "zone_lo", "zone_hi", "active_lo", "active_hi", "role"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"e12d parquet missing columns: {missing}")
    return df


def build_zone_repo(events_df: pd.DataFrame) -> dict:
    """Build zones + fill timelines as flat numpy arrays.

    Returns dict with:
        zone_id, element_codes, element_categories,
        tf_codes, tf_categories,
        direction_codes, direction_categories,
        role_codes, role_categories,
        zone_lo, zone_hi   (float64)
        born_ts (int64), retire_ts (float64, NaN allowed)
        fill_ts, fill_lo, fill_hi (flat float64/int64)
        fill_offset (int64, len n_zones+1)
    """
    events_df = events_df.sort_values("ts").reset_index(drop=True)

    born = events_df[events_df["kind"] == "born"].set_index("zone_id")[
        ["ts", "element", "tf", "direction", "role", "zone_lo", "zone_hi"]
    ].rename(columns={"ts": "born_ts"})

    retire = (
        events_df[events_df["kind"] == "retire"]
        .groupby("zone_id")["ts"].first().rename("retire_ts")
    )

    zones = born.join(retire, how="left").reset_index()
    zones = zones.sort_values("born_ts").reset_index(drop=True)
    n_zones = len(zones)

    # Categoricals — cheaper to pickle as int8 codes + tuple of labels
    elem_cat = pd.Categorical(zones["element"])
    tf_cat = pd.Categorical(zones["tf"])
    dir_cat = pd.Categorical(zones["direction"])
    role_cat = pd.Categorical(zones["role"])

    zone_id_arr = zones["zone_id"].values.astype("int64")
    zone_lo_arr = zones["zone_lo"].values.astype("float64")
    zone_hi_arr = zones["zone_hi"].values.astype("float64")
    born_ts_arr = zones["born_ts"].values.astype("int64")
    retire_ts_arr = zones["retire_ts"].values.astype("float64")  # NaN allowed

    # Fill timelines: for each zone, sorted list of (ts, active_lo, active_hi)
    fills = events_df[events_df["kind"] == "fill_partial"][
        ["zone_id", "ts", "active_lo", "active_hi"]
    ].copy()

    # Sanity: exclude fills before born (defensive, e.g. ob_vc artifact)
    born_ts_map = born["born_ts"]
    fills["born_ts_zone"] = fills["zone_id"].map(born_ts_map)
    fills = fills[fills["ts"] >= fills["born_ts_zone"]]

    zone_id_to_idx = {int(zid): i for i, zid in enumerate(zone_id_arr)}
    fills["zone_idx"] = fills["zone_id"].map(zone_id_to_idx)
    fills = fills.dropna(subset=["zone_idx"]).astype({"zone_idx": "int64"})
    fills = fills.sort_values(["zone_idx", "ts"]).reset_index(drop=True)

    # Sanity: residual bounds valid (lo ≤ hi)
    invalid = fills["active_lo"].values > fills["active_hi"].values
    if invalid.any():
        n_bad = int(invalid.sum())
        raise AssertionError(
            f"Sanity FAIL: {n_bad:,} fill_partial с invalid active bounds (lo > hi). "
            f"e12d parquet повреждён — fix detector."
        )

    counts = np.zeros(n_zones, dtype="int64")
    np.add.at(counts, fills["zone_idx"].values, 1)
    fill_offset = np.zeros(n_zones + 1, dtype="int64")
    np.cumsum(counts, out=fill_offset[1:])

    return {
        "zone_id": zone_id_arr,
        "element_codes": elem_cat.codes.astype("int32"),
        "element_categories": tuple(elem_cat.categories),
        "tf_codes": tf_cat.codes.astype("int8"),
        "tf_categories": tuple(tf_cat.categories),
        "direction_codes": dir_cat.codes.astype("int8"),
        "direction_categories": tuple(dir_cat.categories),
        "role_codes": role_cat.codes.astype("int32"),
        "role_categories": tuple(role_cat.categories),
        "zone_lo": zone_lo_arr,
        "zone_hi": zone_hi_arr,
        "born_ts": born_ts_arr,
        "retire_ts": retire_ts_arr,
        "fill_ts": fills["ts"].values.astype("int64"),
        "fill_lo": fills["active_lo"].values.astype("float64"),
        "fill_hi": fills["active_hi"].values.astype("float64"),
        "fill_offset": fill_offset,
    }


# ─── Load 1m CSV + aggregate to anchor TF ───────────────────

def load_anchors(csv_path: pathlib.Path, tf_ms: int,
                  start_ts: int, end_ts: int, anchor_ms: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Read 1m CSV, sort/dedupe, aggregate to anchor_tf.
    anchor_ms=0 → epoch-floor. anchor_ms>0 → shift-anchored (1w Monday / 2d Friday).
    Anchor_ts = bar close time (bucket_open + tf_ms). Returns (anchor_ts, close)."""
    df = pd.read_csv(csv_path, usecols=["open_time", "close"])
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    ts = dt.astype("datetime64[ms, UTC]").astype("int64").values
    close = df["close"].values.astype("float64")

    order = np.argsort(ts, kind="stable")
    ts = ts[order]; close = close[order]
    _, unique_idx = np.unique(ts, return_index=True)
    ts = ts[unique_idx]; close = close[unique_idx]

    if anchor_ms == 0:
        buckets = ts // tf_ms * tf_ms
    else:
        buckets = ((ts - anchor_ms) // tf_ms) * tf_ms + anchor_ms
    g = pd.DataFrame({"bucket": buckets, "close": close}).groupby("bucket", sort=True)["close"].last()
    anchor_open_ts = g.index.values.astype("int64")
    anchor_close = g.values.astype("float64")
    anchor_ts = anchor_open_ts + tf_ms

    mask = (anchor_ts >= start_ts) & (anchor_ts <= end_ts)
    return anchor_ts[mask], anchor_close[mask]


# ─── Per-chunk worker (numpy-native) ────────────────────────

def process_chunk(chunk_idx: int, anchor_ts_chunk: np.ndarray, price_chunk: np.ndarray,
                   repo: dict, shard_dir: pathlib.Path, scope_pct: float) -> tuple:
    born_ts = repo["born_ts"]
    retire_ts = repo["retire_ts"]
    zone_lo = repo["zone_lo"]
    zone_hi = repo["zone_hi"]
    zone_id = repo["zone_id"]
    elem_codes = repo["element_codes"]
    elem_cat = repo["element_categories"]
    tf_codes = repo["tf_codes"]
    tf_cat = repo["tf_categories"]
    dir_codes = repo["direction_codes"]
    dir_cat = repo["direction_categories"]
    role_codes = repo["role_codes"]
    role_cat = repo["role_categories"]
    fill_ts = repo["fill_ts"]
    fill_lo = repo["fill_lo"]
    fill_hi = repo["fill_hi"]
    fill_offset = repo["fill_offset"]

    parts: list[pd.DataFrame] = []

    for a_ts_v, price_v in zip(anchor_ts_chunk, price_chunk):
        a_ts = int(a_ts_v)
        price = float(price_v)
        band_lo = price * (1 - scope_pct / 100)
        band_hi = price * (1 + scope_pct / 100)

        # Step 1: alive
        born_end = int(np.searchsorted(born_ts, a_ts, side="right"))
        if born_end == 0:
            continue
        retire_slice = retire_ts[:born_end]
        alive_mask = np.isnan(retire_slice) | (retire_slice > a_ts)
        alive_idx = np.where(alive_mask)[0]
        if len(alive_idx) == 0:
            continue

        # Step 2: prefilter by INITIAL bounds (conservative — after shrinkage only narrower)
        init_lo = zone_lo[alive_idx]
        init_hi = zone_hi[alive_idx]
        pre = (init_hi >= band_lo) & (init_lo <= band_hi)
        cand_idx = alive_idx[pre]
        if len(cand_idx) == 0:
            continue

        # Step 3: residual bounds at T (per-zone bisect)
        n = len(cand_idx)
        lo_at = zone_lo[cand_idx].astype("float64").copy()
        hi_at = zone_hi[cand_idx].astype("float64").copy()
        mit_count = np.zeros(n, dtype="int32")
        for j in range(n):
            k = int(cand_idx[j])
            s = int(fill_offset[k]); e = int(fill_offset[k + 1])
            if e <= s:
                continue
            pos = int(np.searchsorted(fill_ts[s:e], a_ts, side="right"))
            if pos > 0:
                mit_count[j] = pos
                lo_at[j] = fill_lo[s + pos - 1]
                hi_at[j] = fill_hi[s + pos - 1]

        # Step 4: final filter by residual bounds
        final = (hi_at >= band_lo) & (lo_at <= band_hi)
        if not final.any():
            continue
        sel = cand_idx[final]

        # Assemble
        n_sel = len(sel)
        row = {
            "anchor_ts": np.full(n_sel, a_ts, dtype="int64"),
            "current_price": np.full(n_sel, price, dtype="float64"),
            "zone_id": zone_id[sel],
            "element": np.array([elem_cat[c] for c in elem_codes[sel]], dtype=object),
            "tf": np.array([tf_cat[c] for c in tf_codes[sel]], dtype=object),
            "direction": np.array([dir_cat[c] for c in dir_codes[sel]], dtype=object),
            "role": np.array([role_cat[c] for c in role_codes[sel]], dtype=object),
            "zone_lo": zone_lo[sel],
            "zone_hi": zone_hi[sel],
            "active_lo": lo_at[final],
            "active_hi": hi_at[final],
            "mit_count_at": mit_count[final],
            "born_ts": born_ts[sel],
            "retire_ts": retire_ts[sel],
        }
        # Derived
        level = (row["active_lo"] + row["active_hi"]) / 2.0
        row["level"] = level
        row["distance_signed_pct"] = (price - level) / price * 100.0
        row["price_in_zone"] = (row["active_lo"] <= price) & (price <= row["active_hi"])
        dist_edge = np.maximum(0.0, np.maximum(
            row["active_lo"] - price, price - row["active_hi"]
        ))
        row["dist_to_edge_pct"] = dist_edge / price * 100.0
        row["age_ms"] = np.full(n_sel, a_ts, dtype="int64") - row["born_ts"]
        w_init = row["zone_hi"] - row["zone_lo"]
        w_now = row["active_hi"] - row["active_lo"]
        with np.errstate(divide="ignore", invalid="ignore"):
            mit_pct = 1.0 - np.where(w_init > 0, w_now / w_init, 0.0)
        mit_pct = np.clip(mit_pct, 0.0, 1.0)
        row["mit_pct"] = mit_pct

        parts.append(pd.DataFrame(row))

    if not parts:
        return None, 0
    df = pd.concat(parts, ignore_index=True)
    path = shard_dir / f"shard_{chunk_idx:05d}.parquet"
    df.to_parquet(path, compression="zstd", compression_level=9)
    return path, len(df)


# ─── Main ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", choices=["BTC", "ETH", "SOL"])
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-06-23")
    ap.add_argument("--anchor-tf", default="1h", choices=list(TF_MS))
    ap.add_argument("--scope-pct", type=float, default=10.0)
    ap.add_argument("--n-workers", type=int, default=30)
    ap.add_argument("--chunk-size", type=int, default=500)
    args = ap.parse_args()

    events_path = WAREHOUSE / f"data/events/events_e12d_{args.symbol}_{args.start}_{args.end}.parquet"
    csv_path = WAREHOUSE / f"data/{args.symbol}USDT_1m.csv"
    out_path = WAREHOUSE / f"data/s7d/snapshots_s7d_{args.symbol}_{args.start}_{args.end}.parquet"
    shard_dir = WAREHOUSE / f"data/s7d/snapshot_s7d_{args.symbol}_shards"

    if not events_path.exists():
        raise FileNotFoundError(f"e12d output not found: {events_path}\nRun e12d first.")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    start_ts = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000)
    anchor_tf_ms = TF_MS[args.anchor_tf]

    print(f"s7d: {args.symbol} {args.start}..{args.end}", file=sys.stderr, flush=True)
    print(f"  events: {events_path.name}", file=sys.stderr, flush=True)
    print(f"  anchor_tf: {args.anchor_tf}  scope: ±{args.scope_pct}%", file=sys.stderr, flush=True)

    t = time.time()
    print("Loading events...", file=sys.stderr, flush=True)
    events_df = load_events(events_path)
    print(f"  {len(events_df):,} events in {time.time()-t:.1f}s", file=sys.stderr, flush=True)

    t = time.time()
    print("Building zone repo (zones + fill timelines)...", file=sys.stderr, flush=True)
    repo = build_zone_repo(events_df)
    n_zones = len(repo["zone_id"])
    n_fills = len(repo["fill_ts"])
    print(f"  {n_zones:,} zones  {n_fills:,} fills in {time.time()-t:.1f}s",
          file=sys.stderr, flush=True)
    del events_df

    t = time.time()
    print(f"Loading 1m + agg to {args.anchor_tf}...", file=sys.stderr, flush=True)
    anchor_ts, anchor_price = load_anchors(csv_path, anchor_tf_ms, start_ts, end_ts,
                                            anchor_ms=TF_ANCHORS.get(args.anchor_tf, 0))
    print(f"  {len(anchor_ts):,} anchors in {time.time()-t:.1f}s", file=sys.stderr, flush=True)

    shard_dir.mkdir(exist_ok=True, parents=True)
    for p in shard_dir.glob("shard_*.parquet"):
        p.unlink()

    chunks = [
        (i // args.chunk_size,
         anchor_ts[i:i + args.chunk_size],
         anchor_price[i:i + args.chunk_size])
        for i in range(0, len(anchor_ts), args.chunk_size)
    ]
    print(f"Processing {len(anchor_ts):,} anchors in {len(chunks)} chunks × {args.chunk_size} on {args.n_workers} workers...",
          file=sys.stderr, flush=True)

    t = time.time()
    results = Parallel(n_jobs=args.n_workers, backend="loky", batch_size=1)(
        delayed(process_chunk)(idx, chunk, prices, repo, shard_dir, args.scope_pct)
        for idx, chunk, prices in chunks
    )
    total_rows = sum(n for _, n in results)
    print(f"  compute done in {time.time()-t:.1f}s, {total_rows:,} rows",
          file=sys.stderr, flush=True)

    t = time.time()
    print("Concat shards...", file=sys.stderr, flush=True)
    dfs = [pd.read_parquet(p) for p, n in results if p is not None]
    if not dfs:
        print("No shards produced.", file=sys.stderr)
        return
    df = pd.concat(dfs, ignore_index=True).sort_values(["anchor_ts", "tf", "element"])
    df["symbol"] = args.symbol
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", compression_level=9, index=False)
    print(f"  {len(df):,} rows → {out_path.name} in {time.time()-t:.1f}s",
          file=sys.stderr, flush=True)

    per_anchor = df.groupby("anchor_ts").size()
    shrunk_pct = (df["mit_count_at"] >= 1).mean() * 100
    print(f"\nStats: anchors={per_anchor.index.nunique():,}  "
          f"rows/anchor min={per_anchor.min()} median={int(per_anchor.median())} "
          f"max={per_anchor.max()}", file=sys.stderr)
    print(f"Zones shrunk (mit_count_at≥1): {shrunk_pct:.1f}% of rows",
          file=sys.stderr)


if __name__ == "__main__":
    main()


# ============================================================
# ПУБЛИЧНЫЙ API ДЛЯ ЧТЕНИЯ s7d СНИМКОВ
# ============================================================

def load_snapshot(path, anchor_ts, fresh_only=None):
    """Загружает s7d снимок на конкретный anchor_ts.

    ВНИМАНИЕ: активная зона ≠ свежая зона!
    s7d retire происходит только при полной митигации (бар закрылся сквозь зону).
    После wick-touch зона остаётся активной, но уже тронута (mit_count_at > 0).

    Параметр fresh_only ОБЯЗАТЕЛЕН — нельзя получить данные без явного выбора:

        fresh_only=True  → только нетронутые зоны (mit_count_at == 0)
                           используй для TP-целей, IDM-фильтров, сигналов
        fresh_only=False → все активные зоны (включая тронутые)
                           ОБЯЗАТЕЛЬНО включи mit_count_at как ML-признак!
                           используй active_lo/active_hi как реальные границы

    Args:
        path:       путь к snapshots_s7d_*.parquet
        anchor_ts:  timestamp (ms) — берётся ближайший anchor <= anchor_ts
        fresh_only: True/False — ОБЯЗАТЕЛЬНЫЙ параметр

    Returns:
        DataFrame снимка

    Raises:
        ValueError если fresh_only не передан
    """
    if fresh_only is None:
        raise ValueError(
            "\n\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  СТОП! load_snapshot() требует fresh_only=True или False        ║\n"
            "║                                                                  ║\n"
            "║  АКТИВНАЯ ЗОНА ≠ СВЕЖАЯ ЗОНА                                    ║\n"
            "║  s7d не отзывает зону при wick-touch — только при полной        ║\n"
            "║  митигации. mit_count_at > 0 = зона уже тронута виком.          ║\n"
            "║                                                                  ║\n"
            "║  fresh_only=True  → mit_count_at == 0 (для TP/сигналов)        ║\n"
            "║  fresh_only=False → все зоны (ВКЛЮЧИ mit_count_at в ML-фичи!)  ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n"
        )

    import numpy as np
    df = pd.read_parquet(path)
    anchors = np.sort(df["anchor_ts"].unique())
    idx = int(np.searchsorted(anchors, anchor_ts, side="right")) - 1
    if idx < 0:
        raise ValueError(f"Нет anchor <= {anchor_ts} в {path}")
    snap = df[df["anchor_ts"] == anchors[idx]].copy()

    if fresh_only:
        snap = snap[snap["mit_count_at"] == 0].reset_index(drop=True)

    return snap
