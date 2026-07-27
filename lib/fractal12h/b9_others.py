"""B9 Others — sub-basket B9C1 (P11+overlay) + B9C2 (maxV, ex-B1C7) + B9C3 (momentum) + B9C4 (climax).

Портировано из ~/smc-warehouse/scripts/фрактал-12h/b9_others.py (WSL, read-only источник).

B9C1 semantic:
  P11_count 4-window OR-basket + overlay:
    window_15m внутри последних N × 15m 12h бара
    FH (short): cnt = count(close < open в 15m)
    FL (long):  cnt = count(close > open в 15m)
    P11_N = cnt / len(window_15m)
    p11_or = (P11_8 >= 0.65) ∨ (P11_12 >= 0.75) ∨ (P11_16 >= 0.65) ∨ (P11_24 >= 0.65)
  Overlay: close_match (c[i] в направлении разворота) AND range/ATR14 ≥ 1.2

B9C2 semantic (ex-B1C7):
  Для pivot bar i с maxV = close of 1m bar с абс. max объёмом bull/bear группы в i-1:
    SHORT: h[i] > maxV(i-1) AND c[i] < maxV(i-1) AND (h[i]-maxV)/ATR14 >= 0.7
    LONG:  l[i] < maxV(i-1) AND c[i] > maxV(i-1) AND (maxV-l[i])/ATR14 >= 0.7

B9C3 semantic (momentum reversal bar):
  close_match AND body/range >= 0.7 (Marubozu-like)

B9C4 semantic (climax bar):
  close_match AND body/range >= 0.5 AND close_pos >= 0.75 AND range/ATR14 >= 1.5

Causality: ✅ всё внутри 12h бара i.
Pool: A1 (все pivots из a_candidates).

Reads:
  G:\\ASVK\\data\\fractal12h\\a_candidates_{SYM}_{start}_{end}.parquet
  G:\\ASVK\\data\\{SYM}USDT_1m.csv
Writes:
  G:\\ASVK\\data\\fractal12h\\b9_hits_{SYM}_{start}_{end}.parquet
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
from common import load_1m, agg_12h, agg_15m, DATA_OUT, TF_12H_MS, TF_15M_MS
from maxv import latest_maxv_path


WINDOWS = [
    (8,  0.65),   # 2h  окно
    (12, 0.75),   # 3h  окно
    (16, 0.65),   # 4h  окно
    (24, 0.65),   # 6h  окно
]
B9C1_RANGE_ATR_MIN = 1.2
B9C2_DIST_ATR_MIN  = 0.7   # canonical (ex-B1C7)
B9C3_BODY_MIN      = 0.7
B9C4_BODY_MIN      = 0.5
B9C4_CLOSE_POS_MIN = 0.75
B9C4_RANGE_ATR_MIN = 1.5


def compute_b9(a_cand: pd.DataFrame,
                df_12h: pd.DataFrame, df_15m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    ts_15m = df_15m["ts"].to_numpy()
    o_15m  = df_15m["open"].to_numpy()
    c_15m  = df_15m["close"].to_numpy()

    ts_12h = df_12h["ts"].to_numpy()
    o_12h  = df_12h["open"].to_numpy()
    h_12h  = df_12h["high"].to_numpy()
    l_12h  = df_12h["low"].to_numpy()
    c_12h  = df_12h["close"].to_numpy()
    ts_to_idx = {int(t): i for i, t in enumerate(ts_12h)}

    # ATR14 + maxV — level-1 shared indicators (G:\ASVK\lib\maxv.py), считаются один раз
    # в pipeline Block 1, здесь только читаются и матчатся по ts.
    mv_df = pd.read_parquet(latest_maxv_path(symbol))
    mv_by_ts = pd.Series(mv_df["atr14"].to_numpy(), index=mv_df["ts"].to_numpy())
    maxv_by_ts = pd.Series(mv_df["maxv"].to_numpy(), index=mv_df["ts"].to_numpy())
    atr14 = mv_by_ts.reindex(ts_12h).to_numpy()
    maxv = maxv_by_ts.reindex(ts_12h).to_numpy()

    a1 = a_cand[a_cand["a1_pre_w"]].copy()

    rows = []
    for _, row in a1.iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        pt_end = ts_pivot + TF_12H_MS

        idx = ts_to_idx.get(ts_pivot)
        if idx is None:
            continue
        o_i = o_12h[idx]; h_i = h_12h[idx]
        l_i = l_12h[idx]; c_i = c_12h[idx]
        rng = h_i - l_i
        atr_i = atr14[idx] if atr14[idx] > 0 else 1.0
        body_ratio = abs(c_i - o_i) / rng if rng > 0 else 0.0
        close_match = (c_i < o_i) if direction == "short" else (c_i > o_i)
        close_pos = ((h_i - c_i) if direction == "short" else (c_i - l_i)) / rng if rng > 0 else 0.0
        range_atr = rng / atr_i

        # ── B9C1: P11 + close_match + range_atr ≥ 1.2 ──
        i_hi = int(np.searchsorted(ts_15m, pt_end, side="left"))
        p11_or = False
        for N, thr in WINDOWS:
            cut_ts = pt_end - N * TF_15M_MS
            i_lo = int(np.searchsorted(ts_15m, cut_ts, side="left"))
            if i_hi <= i_lo:
                continue
            o_win = o_15m[i_lo:i_hi]
            c_win = c_15m[i_lo:i_hi]
            if direction == "short":
                cnt = int(np.sum(c_win < o_win))
            else:
                cnt = int(np.sum(c_win > o_win))
            n_win = len(o_win)
            ratio = cnt / n_win if n_win > 0 else 0.0
            if ratio >= thr:
                p11_or = True
                break
        b9c1 = p11_or and close_match and (range_atr >= B9C1_RANGE_ATR_MIN)

        # ── B9C2: maxV sweep + dist ≥ 0.7 × ATR ──
        b9c2 = False
        if idx >= 1:
            mv = maxv[idx-1]
            if not np.isnan(mv) and atr_i > 0:
                if direction == "short" and h_i > mv and c_i < mv:
                    dist = (h_i - mv) / atr_i
                    if dist >= B9C2_DIST_ATR_MIN:
                        b9c2 = True
                elif direction == "long" and l_i < mv and c_i > mv:
                    dist = (mv - l_i) / atr_i
                    if dist >= B9C2_DIST_ATR_MIN:
                        b9c2 = True

        # ── B9C3: momentum bar ──
        b9c3 = close_match and (body_ratio >= B9C3_BODY_MIN)

        # ── B9C4: climax bar ──
        b9c4 = (close_match and (body_ratio >= B9C4_BODY_MIN)
                and (close_pos >= B9C4_CLOSE_POS_MIN)
                and (range_atr >= B9C4_RANGE_ATR_MIN))

        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction":        direction,
            "confirmable":      bool(row["confirmable"]),
            "confirmed":        bool(row["confirmed"]),
            "b9c1":  b9c1,
            "b9c2":  b9c2,
            "b9c3":  b9c3,
            "b9c4":  b9c4,
            "b9_hit": b9c1 or b9c2 or b9c3 or b9c4,
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b9c1", "b9c2", "b9c3", "b9c4", "b9_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        marker = "  ← B9" if col == "b9_hit" else ""
        print(f"  {col:8s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{marker}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-08")
    args = ap.parse_args()

    print(f"b9_others (ASVK-portable, C1 P11 + C2 maxV + C3 momentum + C4 climax): "
          f"{args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1 candidates domain: {int(a_cand['a1_pre_w'].sum()):,}",
          file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)
    df_15m = agg_15m(df_1m)
    print(f"  15m bars: {len(df_15m):,}   12h bars: {len(df_12h):,}",
          file=sys.stderr, flush=True)

    hits = compute_b9(a_cand, df_12h, df_15m, args.symbol)
    print_stats(hits)

    out = DATA_OUT / f"b9_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
