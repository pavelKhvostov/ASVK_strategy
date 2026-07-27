"""Falling wedge (клин) — разметка 3H+2L, канон (ASVK-portable).

Портировано из ~/smc-warehouse/scripts/Bulkowski/Клин/scan_wedges_3H_2L.py (WSL,
read-only источник, 2026-07-19 — самая свежая ревизия среди scan_wedges.py /
scan_rising_wedges.py). Геометрия и все пороги — verbatim, без изменений.

Отличие от WSL-оригинала: born-детекция там считала HMA(78)/WMA(50) локально внутри
скрипта (свой _wma/_hma); здесь читает те же ряды из level-1 shared indicators
(lib/trendline.py --tf 1h --length 78, lib/wma.py) — единая формула на весь проект,
без дублирующей реализации HMA.

Falling wedge = бычий разворотный паттерн (два падающих sloping trendline, сходятся,
born = момент разворота момента вверх после сжатия) → direction = "long".

Разметка: H1 → L1 → H2 → L2 → H3 (3 верхних касания, 2 нижних).
  Upper trendline: H1↔H3 endpoints, H2 — промежуточная точка (tolerance-check)
  Lower trendline: L1↔L2 endpoints

Canon-фильтры (verbatim из WSL):
  1. Prominent pivots (2× median h-l)
  2. Оба slope < 0 (falling)
  3. |slope| ≥ 0.005 %/бар для обоих
  4. Convergence ≥ 2.00×
  5. Middle H2 dev ≤ tol_touch = max(0.5%*price, 0.8*ATR14)
  6. Boundary integrity, allow 2 false breaks + recovery 5 баров
  7. Duration 21..200 баров
  8. Height ≥ 4%
  9. Born: первый бар после H3 (≤48 баров), где HMA(78) < WMA(50) И HMA развернулась
     вверх (HMA[j] > HMA[j-2])

Статус (по аналогии с fractal12h confirmable/confirmed):
  PENDING   — H3 сформирован, но ещё не прошло 48 баров с born-поиском (либо born
              ещё не найден в пределах уже прошедшего окна)
  CONFIRMED — born найден
  FAILED    — прошло 48 баров после H3, born не найден (окно закрыто без сигнала)

Никакого measure rule/target/stop — в WSL-источнике их нет ни в коде, ни в
документации для клина (только геометрия + born), не придумываем.

Reads:
  G:\\ASVK\\data\\{SYM}USDT_1m.csv
  G:\\ASVK\\data\\trendline\\trendline_{SYM}_1h78_*.parquet  (HMA-78/1h, born)
  G:\\ASVK\\data\\wma\\wma_{SYM}_*.parquet                    (WMA-50/1h, born)
Writes:
  G:\\ASVK\\data\\patterns\\wedge_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — level-1
from common import load_1m, agg_1h, DATA_OUT, TF_1H_MS
from trendline import latest_trendline_path
from wma import latest_wma_path


MIN_BARS  = 21
MAX_BARS  = 200
MIN_HEIGHT_PCT = 4.0
CONV_RATIO_MIN = 2.00
TOL_TOUCH_ATR  = 0.8
TOL_TOUCH_PCT  = 0.005
TOL_BREAK_PCT  = 0.0005
MAX_FALSE_BR   = 2
RECOVERY_BARS  = 5
MIN_ABS_SLOPE_PCT = 0.005
BORN_MAX_SEARCH = 48


def _atr14(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return np.concatenate([[np.nan], pd.Series(tr).rolling(14).mean().to_numpy()])


def _check_wedge_falling_3H(h, l, c, atr, H1_i, L1_i, H2_i, L2_i, H3_i):
    H1_p, L1_p, H2_p, L2_p, H3_p = h[H1_i], l[L1_i], h[H2_i], l[L2_i], h[H3_i]
    slope_H = (H3_p - H1_p) / (H3_i - H1_i)
    slope_L = (L2_p - L1_p) / (L2_i - L1_i)
    if slope_L >= 0 or slope_H >= 0:
        return None
    if abs(slope_L) * 100 / L1_p < MIN_ABS_SLOPE_PCT:
        return None
    if abs(slope_H) * 100 / H1_p < MIN_ABS_SLOPE_PCT:
        return None
    conv = abs(slope_H) / abs(slope_L)
    if conv < CONV_RATIO_MIN:
        return None
    line_H_at_H2 = H1_p + slope_H * (H2_i - H1_i)
    dev_H2 = H2_p - line_H_at_H2
    atr_H2 = atr[H2_i] if not np.isnan(atr[H2_i]) else 400
    tol_touch = max(TOL_TOUCH_PCT * H2_p, TOL_TOUCH_ATR * atr_H2)
    if abs(dev_H2) > tol_touch:
        return None

    intcp_L = L1_p - slope_L * L1_i
    intcp_H = H1_p - slope_H * H1_i
    anchor_bars = {H1_i, L1_i, H2_i, L2_i, H3_i}
    start_scan = H1_i + 1
    end_scan   = H3_i - 1
    false_br_count = 0
    j = start_scan
    while j <= end_scan:
        if j in anchor_bars:
            j += 1; continue
        line_low  = intcp_L + slope_L * j
        line_high = intcp_H + slope_H * j
        tb_price = TOL_BREAK_PCT * c[j]
        outside = (c[j] < line_low - tb_price) or (c[j] > line_high + tb_price)
        if outside:
            false_br_count += 1
            if false_br_count > MAX_FALSE_BR:
                return None
            recovered = False
            for k in range(j + 1, min(j + RECOVERY_BARS + 1, end_scan + 1)):
                ll = intcp_L + slope_L * k
                lh = intcp_H + slope_H * k
                tb_k = TOL_BREAK_PCT * c[k]
                if (c[k] >= ll - tb_k) and (c[k] <= lh + tb_k):
                    recovered = True; break
            if not recovered:
                return None
        j += 1

    dur = H3_i - H1_i
    if dur < MIN_BARS or dur > MAX_BARS:
        return None
    peak = max(H1_p, H2_p, H3_p)
    trough = min(L1_p, L2_p)
    height = peak - trough
    BO_price = c[H3_i]
    height_pct = height / BO_price * 100
    if height_pct < MIN_HEIGHT_PCT:
        return None

    return {
        'H1_idx': H1_i, 'L1_idx': L1_i, 'H2_idx': H2_i,
        'L2_idx': L2_i, 'H3_idx': H3_i,
        'H1_p': H1_p, 'L1_p': L1_p, 'H2_p': H2_p, 'L2_p': L2_p, 'H3_p': H3_p,
        'slope_L': slope_L, 'slope_H': slope_H, 'conv': conv,
        'height_pct': height_pct, 'duration': dur,
    }


def _try_H1(H1_i, h, l, c, atr, FH, FL):
    results = []
    for L1_i in FL[(FL > H1_i) & (FL <= H1_i + MAX_BARS)]:
        for H2_i in FH[(FH > L1_i) & (FH <= H1_i + MAX_BARS)]:
            for L2_i in FL[(FL > H2_i) & (FL <= H1_i + MAX_BARS)]:
                for H3_i in FH[(FH > L2_i) & (FH <= H1_i + MAX_BARS)]:
                    r = _check_wedge_falling_3H(h, l, c, atr, H1_i, L1_i, H2_i, L2_i, H3_i)
                    if r is not None:
                        results.append(r)
    return results


def detect_wedge_geometry(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                          ts: np.ndarray) -> list[dict]:
    atr = _atr14(h, l, c)
    prom_price = np.median(h - l) * 2
    FH, _ = find_peaks(h, prominence=prom_price, distance=3)
    FL, _ = find_peaks(-l, prominence=prom_price, distance=3)

    all_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_try_H1)(int(H1_i), h, l, c, atr, FH, FL) for H1_i in FH
    )
    patterns = [r for chunk in all_results for r in chunk]
    if not patterns:
        return []

    df_p = pd.DataFrame(patterns)
    df_p['window_key'] = df_p['H1_idx'].astype(str) + '_' + df_p['H3_idx'].astype(str)
    df_p = df_p.sort_values('conv', ascending=False).drop_duplicates('window_key')
    df_p = df_p.sort_values('H1_idx').reset_index(drop=True)
    return df_p.to_dict('records')


def apply_born(patterns: list[dict], t_arr: np.ndarray, hma78: np.ndarray,
               wma50: np.ndarray, n_bars: int) -> list[dict]:
    out = []
    for p in patterns:
        H3_i = p['H3_idx']
        confirmable = (H3_i + BORN_MAX_SEARCH) < n_bars   # окно поиска уже целиком прошло
        born_i = None
        search_end = min(H3_i + BORN_MAX_SEARCH + 1, n_bars)
        for j in range(H3_i + 1, search_end):
            if np.isnan(wma50[j]) or np.isnan(hma78[j]) or np.isnan(hma78[j - 2]):
                continue
            if hma78[j] < wma50[j] and hma78[j] > hma78[j - 2]:
                born_i = j
                break

        if born_i is not None:
            status = "CONFIRMED"
            signal_ts = int(t_arr[born_i]) + TF_1H_MS
        elif not confirmable:
            status = "PENDING"
            signal_ts = int(t_arr[H3_i]) + TF_1H_MS   # для сортировки "по свежести" в TUI
        else:
            status = "FAILED"
            signal_ts = int(t_arr[H3_i]) + TF_1H_MS

        p2 = dict(p)
        p2['born_idx'] = born_i
        p2['status'] = status
        p2['signal_ts'] = signal_ts
        out.append(p2)
    return out


def compute_wedge(df_1h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    o = df_1h["open"].to_numpy(); h = df_1h["high"].to_numpy()
    l = df_1h["low"].to_numpy();  c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()
    n_bars = len(t_arr)

    geo = detect_wedge_geometry(o, h, l, c, t_arr)

    tl = pd.read_parquet(latest_trendline_path(symbol, variant="1h78"), columns=["ts", "mhull"])
    hma78 = pd.Series(tl["mhull"].to_numpy(), index=tl["ts"].to_numpy()).reindex(t_arr).to_numpy()

    wm = pd.read_parquet(latest_wma_path(symbol), columns=["ts", "wma50"])
    wma50 = pd.Series(wm["wma50"].to_numpy(), index=wm["ts"].to_numpy()).reindex(t_arr).to_numpy()

    final = apply_born(geo, t_arr, hma78, wma50, n_bars)

    rows = []
    for p in final:
        rows.append({
            "signal_ts": p['signal_ts'],
            "direction": "long",
            "pattern": "wedge_falling",
            "ts_H1": int(t_arr[p['H1_idx']]), "ts_H3": int(t_arr[p['H3_idx']]),
            "H1_p": p['H1_p'], "L1_p": p['L1_p'], "H2_p": p['H2_p'],
            "L2_p": p['L2_p'], "H3_p": p['H3_p'],
            "conv": p['conv'], "height_pct": p['height_pct'], "duration": p['duration'],
            "status": p['status'],
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    n = len(hits)
    n_conf = int((hits["status"] == "CONFIRMED").sum()) if n else 0
    n_pend = int((hits["status"] == "PENDING").sum()) if n else 0
    n_fail = int((hits["status"] == "FAILED").sum()) if n else 0
    print(f"  wedge_falling  n={n}  CONFIRMED={n_conf}  PENDING={n_pend}  FAILED={n_fail}",
          file=sys.stderr, flush=True)


def main() -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-23")
    args = ap.parse_args()

    print(f"wedge (falling, 3H+2L): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    df_1m = load_1m(args.symbol)
    df_1h = agg_1h(df_1m)
    print(f"  1h bars: {len(df_1h):,}", file=sys.stderr, flush=True)

    hits = compute_wedge(df_1h, args.symbol)

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms   = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print_stats(hits)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"wedge_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
