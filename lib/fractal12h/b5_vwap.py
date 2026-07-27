"""B5 VWAP — sub-basket B5C1 (≥1 decisive-swept W-aligned VWAP, FULL_DISP), ASVK-portable.

Портировано из ~/smc-warehouse/scripts/фрактал-12h/b5_vwap.py (WSL, read-only источник),
изначально с механикой SWEEP (просто touch+close-beyond) + confluence≥2. Разносторонний
research (G:\\Claude\\research) показал тот же паттерн, что и на B2/B4: SWEEP был слабым
(WR 65-73% на 11 активах), а confluence≥2 — костылём, компенсировавшим слабую механику,
а не содержательным требованием. С FULL_DISP (решительное закрытие за VWAP на
0.5×ATR14, не просто по нужную сторону) хватает и ОДНОГО decisive-swept VWAP: WR
растёт на 8.8-10.8pp на BTC/ETH/SOL БЕЗ потери объёма (n тоже растёт или держится).
Механика заменена 2026-07-25.

Semantic:
  Anchored VWAP от D-фрактала (Williams n=2), W-aligned (совпал с W-фракталом того же
  side, допуск уровня 0.01), готов через ready_ms = D.ts + 3 дня после подтверждения.
  Якоря — level-1 shared indicator, см. G:\\ASVK\\lib\\vwap_anchors.py (детекция D/W
  фракталов + W-alignment один раз на символ за цикл, не зависит от конкретного
  пивота — сам VWAP по найденному якорю дёшев и считается здесь inline через cumsum).

  На 12h pivot bar (пул A1, как у B1/B2/B9 — НЕ a124_pool: сравнение показало, что
  A2/A4 почти не меняют WR (72.97%→74.60%, +1.63pp), но режут n почти вдвое
  (185→126, -32%), поэтому по решению пользователя B5 остаётся на чистом A1):
    side = "FH" if direction=="short" else "FL"
    margin = 0.5 × ATR14(12h)[i]
    Для каждого relevant D-якоря (тот же side, aligned_W, ready_ms ≤ pivot_open):
      VWAP v = Σ(close×vol с anchor.ts до pivot_close) / Σvol
      SHORT: high_pivot > v AND close_pivot < v − margin
      LONG:  low_pivot  < v AND close_pivot > v + margin
    B5C1 fires если количество decisive-swept W-aligned VWAP ≥ 1

Кросс-проверка (n / WR, SWEEP≥2 → FULL_DISP≥1):
    BTC n=185 WR=72.97% → n=204 WR=81.86%
    ETH n=386 WR=64.94% → n=293 WR=75.77%
    SOL n=133 WR=66.92% → n=177 WR=75.71%

Reads:
  data/fractal12h/a_candidates_{SYM}_{start}_{end}.parquet
  data/vwap_anchors/vwap_anchors_{SYM}_*.parquet (последний по дате)
  data/{SYM}USDT_1m.csv (через df_1m, переданный run_fractal12h.py)
Writes:
  data/fractal12h/b5_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — для vwap_anchors (level-1)
from common import load_1m, agg_12h, DATA_OUT, TF_12H_MS
from vwap_anchors import latest_vwap_anchors_path


MIN_W_SWEPT = 1


def load_vwap_anchors(symbol: str) -> list[dict]:
    """D-фракталы, W-aligned=True, из level-1 vwap_anchors.py."""
    df = pd.read_parquet(latest_vwap_anchors_path(symbol))
    df = df[df["aligned_W"] == True]
    return df[["ts", "side", "level", "ready_ms"]].to_dict(orient="records")


def atr14_sma(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """ATR14 = SMA(TR, 14) — canon-совместимо (не Wilder EWM), см. lib/maxv.py."""
    n = len(h)
    c_prev = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - c_prev), np.abs(l - c_prev)])
    tr[0] = 0.0
    csum = np.cumsum(tr)
    atr = np.full(n, np.nan)
    atr[14:] = (csum[14:] - csum[:n - 14]) / 14
    return atr


def compute_b5(a_cand: pd.DataFrame, df_1m: pd.DataFrame, df_12h: pd.DataFrame,
               anchors: list[dict]) -> pd.DataFrame:
    t12 = df_12h["ts"].to_numpy()
    h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy()
    c12 = df_12h["close"].to_numpy()
    atr12 = atr14_sma(h12, l12, c12)

    # Cumsum на 1m для быстрых anchored VWAP query
    ts_1m = df_1m["ts"].to_numpy()
    cls_1m = df_1m["close"].to_numpy()
    vol_1m = df_1m["volume"].to_numpy()
    pv_cum = np.concatenate([[0.0], np.cumsum(cls_1m * vol_1m)])
    vol_cum = np.concatenate([[0.0], np.cumsum(vol_1m)])

    def vwap_at(anchor_ts: int, end_ts: int) -> float | None:
        i_a = int(np.searchsorted(ts_1m, anchor_ts, side="left"))
        i_e = int(np.searchsorted(ts_1m, end_ts, side="right")) - 1
        if i_a > i_e or i_e < 0:
            return None
        pv = pv_cum[i_e + 1] - pv_cum[i_a]
        v = vol_cum[i_e + 1] - vol_cum[i_a]
        return pv / v if v > 0 else None

    ts_to_idx12 = {int(t): k for k, t in enumerate(t12)}
    pool = a_cand[a_cand["a1_pre_w"]].copy()   # чистый A1 — см. докстринг

    print(f"  computing B5C1 per pivot ({len(pool):,}), {len(anchors):,} W-aligned anchors...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    rows = []
    for _, row in pool.iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        pi = ts_to_idx12.get(ts_pivot)
        if pi is None:
            continue
        pivot_close_ts = ts_pivot + TF_12H_MS
        bh, bl, bc_p = h12[pi], l12[pi], c12[pi]
        side = "FH" if direction == "short" else "FL"
        margin = 0.5 * atr12[pi] if not np.isnan(atr12[pi]) else 0.0

        n_swept = 0
        for anc in anchors:
            if anc["side"] != side or anc["ready_ms"] > ts_pivot:
                continue
            v = vwap_at(anc["ts"], pivot_close_ts)
            if v is None:
                continue
            if side == "FH":
                if bh > v and bc_p < v - margin:
                    n_swept += 1
            else:
                if bl < v and bc_p > v + margin:
                    n_swept += 1
            if n_swept >= MIN_W_SWEPT:
                break   # раннее прекращение — уже прошёл порог

        b5c1 = n_swept >= MIN_W_SWEPT
        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "b5c1":  b5c1,
            "b5_hit": b5c1,   # пока B5 = B5C1 (единственный sub, как в WSL-каноне)
        })
    print(f"    done in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b5c1", "b5_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        marker = "  ← B5" if col == "b5_hit" else ""
        print(f"  {col:8s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{marker}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    print(f"b5_vwap (B5C1 only, A1 pool): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {int(a_cand['a1_pre_w'].sum()):,}",
          file=sys.stderr, flush=True)

    anchors = load_vwap_anchors(args.symbol)
    print(f"  W-aligned VWAP anchors: {len(anchors):,}", file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_b5(a_cand, df_1m, df_12h, anchors)
    print_stats(hits)

    out = DATA_OUT / f"b5_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
