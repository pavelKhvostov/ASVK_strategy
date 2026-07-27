"""H&S Bottom (Inverted H&S, LONG) — зеркало hs_top.py (Bulkowski Ch.38).

Геометрия (зеркало hs_top):
  Head (H)  = Williams N=2 FL с НАИМЕНЬШЕЙ ценой (самое глубокое дно)
  LS        = FH до H, l[LS] > head_p (левое плечо мельче головы)
  i_la      = HIGHEST FH между LS и H (левая подмышка = пик)
  RS        = FH после H, l[RS] > head_p, выбирается как LOWEST FL
              в окне (H, BR) → самое глубокое правое плечо (=наиболее заметное)
  i_ra      = HIGHEST FH между H и RS (правая подмышка = пик)
  Neckline  = линия через (i_la, LA_p) и (i_ra, RA_p) (два пика-подмышки)
  BR        = первое закрытие ВЫШЕ neckline после RS
  MAX_SPAN LS→RS = 72 бара, MAX_CONFIRM RS→BR = 7 баров (как hs_top)

Условие 1a: между LS и H (и H и RS) нет FL, которые касаются/прокалывают
линию LS→H (H→RS) снизу.
Условие 5: |neckline slope| ≤ max_nl_slope_pct_per_bar.

TG (зеркало hs_top):
  pattern_height = nl_h - head_p  (neckline над головой)
  down_sloping = slope < 0
  base = c_br if down_sloping else RA_p
  target_full = base + pattern_height
  target_half = base + 0.5 * pattern_height
  target_2x   = base + 2.0 * pattern_height

FVG(2): bullish FVG (direction=long) в (RS, BR].
MA(2a_long/2c_long/2d_long): применяется на баре BR.
2b: traveled/distance < max_path_traveled_pct.

Reads:
  G:\\ASVK\\data\\{SYM}USDT_1m.csv
  G:\\ASVK\\data\\events\\events_e12d_{SYM}_*.parquet   (FVG long)
  G:\\ASVK\\data\\trendline\\trendline_{SYM}_1h78_*.parquet
  G:\\ASVK\\data\\wma\\wma_{SYM}_*.parquet
Writes:
  G:\\ASVK\\data\\patterns\\hs_bottom_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import load_1m, agg_1h, DATA_OUT, TF_1H_MS
from trendline import latest_trendline_path
from wma import latest_wma_path
from block5_common import fractals


@dataclass
class HSBottomParams:
    max_span:                 int   = 72
    max_confirm_bars:         int   = 7
    max_nl_slope_pct_per_bar: float = 0.05
    n_workers:                int   = 30
    max_path_traveled_pct:    float = 0.60


def _empty_stats() -> dict:
    return {
        'n_fh': 0, 'n_fl': 0,
        'rej_invalidated': 0, 'rej_no_breakout': 0,
        'passed_geometry': 0,
        'rej_fvg': 0, 'rej_hma_wma': 0, 'rej_path_2b': 0,
        'passed_canon': 0,
    }


def _line_value_at(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def _atr14_nolookahead(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    out = np.zeros_like(tr)
    cum = 0.0
    for i in range(len(tr)):
        cum += tr[i]
        if i >= 14:
            cum -= tr[i - 14]
            out[i] = cum / 14.0
        else:
            out[i] = cum / (i + 1)
    return out


def _process_head_chunk(head_chunk: list[int], h: np.ndarray, l: np.ndarray,
                         c: np.ndarray, FH: np.ndarray, FL: np.ndarray,
                         fl_set: set[int], n_bars: int,
                         params: HSBottomParams) -> tuple[list[dict], dict]:
    MAX_SPAN = params.max_span
    MAX_CONFIRM = params.max_confirm_bars
    local_stats = {'rej_invalidated': 0, 'rej_no_breakout': 0}
    out: list[dict] = []

    for i_h in head_chunk:
        i_h = int(i_h)
        if i_h < 6 or i_h > n_bars - MAX_CONFIRM - 2:
            continue
        head_p = l[i_h]  # цена головы (наименьшее дно)

        # LS = FL до головы, l[LS] > head_p (мельче головы)
        ls_cands = [i for i in FL if i_h - MAX_SPAN <= i < i_h and l[i] > head_p]
        if not ls_cands:
            continue

        for i_ls in ls_cands:
            LS_p = l[i_ls]
            # левая подмышка = HIGHEST FH между LS и head
            left_fhs = [i for i in FH if i_ls < i < i_h]
            if not left_fhs:
                continue
            i_la = max(left_fhs, key=lambda i: h[i])
            LA_p = h[i_la]

            max_scan = min(n_bars, i_ls + MAX_SPAN + MAX_CONFIRM + 1)
            i_br = i_rs = i_ra = None
            RS_p = RA_p = slope = None
            invalidated = False

            for j in range(i_h + 3, max_scan):
                if c[j] < head_p:
                    invalidated = True
                    break
                # RS candidates = FL после головы, l[RS] > head_p
                rs_cands_j = [i for i in fl_set if i_h < i < j and l[i] > head_p]
                if not rs_cands_j:
                    continue
                # RS = самый НИЗКИЙ FL (наиболее глубокое правое плечо = наиболее заметное)
                cur_i_rs = min(rs_cands_j, key=lambda i: l[i])
                if cur_i_rs - i_ls > MAX_SPAN:
                    continue
                cur_RS_p = l[cur_i_rs]
                # 1d: между RS и BR ≤ MAX_CONFIRM баров
                if j - cur_i_rs > MAX_CONFIRM:
                    continue
                # нет более низкого FL между RS и j (не прерывать плечо)
                if j > cur_i_rs + 1:
                    if float(l[cur_i_rs + 1:j + 1].min()) <= cur_RS_p:
                        continue
                # правая подмышка = HIGHEST FH между головой и RS
                right_fhs = [i for i in FH if i_h < i < cur_i_rs]
                if not right_fhs:
                    continue
                cur_i_ra = max(right_fhs, key=lambda i: h[i])
                cur_RA_p = h[cur_i_ra]
                # 1a: промежуточные FL не касаются линий LS→H и H→RS снизу
                cond_fail = False
                for i_int in FL:
                    i_int = int(i_int)
                    if i_ls < i_int < i_h:
                        if l[i_int] <= _line_value_at(i_ls, LS_p, i_h, head_p, i_int):
                            cond_fail = True
                            break
                    elif i_h < i_int < cur_i_rs:
                        if l[i_int] <= _line_value_at(i_h, head_p, cur_i_rs, cur_RS_p, i_int):
                            cond_fail = True
                            break
                if cond_fail:
                    continue
                # Neckline через две подмышки (FH)
                cur_slope = (cur_RA_p - LA_p) / (cur_i_ra - i_la)
                nl_j = LA_p + cur_slope * (j - i_la)
                if c[j] > nl_j:
                    i_br = j; i_rs = cur_i_rs; i_ra = cur_i_ra
                    RS_p = cur_RS_p; RA_p = cur_RA_p; slope = cur_slope
                    break

            if invalidated:
                local_stats['rej_invalidated'] += 1
                continue
            if i_br is None:
                local_stats['rej_no_breakout'] += 1
                continue

            # 5: |neckline slope %/бар|
            nl_slope_pct_per_bar = slope * 100 / LA_p
            if abs(nl_slope_pct_per_bar) > params.max_nl_slope_pct_per_bar:
                continue

            nl_h = LA_p + slope * (i_h - i_la)
            nl_br = LA_p + slope * (i_br - i_la)
            pattern_height = nl_h - head_p  # положительное: neckline выше головы
            if pattern_height <= 0:
                continue
            down_sloping = slope < 0
            c_br = c[i_br]
            base = c_br if down_sloping else RA_p
            target_full = base + pattern_height
            target_half = base + 0.5 * pattern_height
            target_2x   = base + 2.0 * pattern_height

            out.append({
                'i_ls': i_ls, 'i_la': i_la, 'i_h': i_h, 'i_ra': i_ra, 'i_rs': i_rs,
                'i_breakout': i_br,
                'ls_p': float(LS_p), 'la_p': float(LA_p), 'head_p': float(head_p),
                'ra_p': float(RA_p), 'rs_p': float(RS_p),
                'ts_ls': 0, 'ts_la': 0, 'ts_h': 0, 'ts_ra': 0, 'ts_rs': 0,
                'ts_breakout': 0,
                'pattern_height': float(pattern_height),
                'breakout_close': float(c_br),
                'breakout_base': float(base),
                'target_half': float(target_half),
                'target_full': float(target_full),
                'target_2x': float(target_2x),
                'stop': float(RS_p * 0.995),  # чуть ниже правого плеча
                'nl_at_br': float(nl_br),
                'down_sloping_neckline': bool(down_sloping),
                'neckline_at_head': float(nl_h),
                'width_bars': int(i_rs - i_ls),
            })

    return out, local_stats


def detect_hs_bottom_geometry(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                               ts: np.ndarray,
                               params: Optional[HSBottomParams] = None,
                               ) -> tuple[list[dict], dict]:
    if params is None:
        params = HSBottomParams()

    n_bars = len(c)
    stats = _empty_stats()
    if n_bars < 20:
        return [], stats

    h = h.astype(np.float64); l = l.astype(np.float64); c = c.astype(np.float64)

    FH = fractals(h, 2, 'high')
    FL = fractals(l, 2, 'low')
    stats['n_fh'] = len(FH); stats['n_fl'] = len(FL)
    fl_set = set(int(i) for i in FL)

    # Внешний цикл по FL (голова = самое глубокое дно) — зеркало hs_top (голова = пик FH)
    n_workers = max(1, min(params.n_workers, os.cpu_count() or 4))
    head_chunks = np.array_split(FL, n_workers)
    head_chunks = [chunk.tolist() for chunk in head_chunks if len(chunk) > 0]

    if len(head_chunks) > 1:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_head_chunk)(chunk, h, l, c, FH, FL, fl_set, n_bars, params)
            for chunk in head_chunks
        )
    else:
        chunk_results = [_process_head_chunk(chunk, h, l, c, FH, FL, fl_set, n_bars, params)
                          for chunk in head_chunks]

    patterns: list[dict] = []
    for res, local_stats in chunk_results:
        patterns.extend(res)
        for k, v in local_stats.items():
            stats[k] += v

    for pat in patterns:
        pat['ts_ls'] = int(ts[pat['i_ls']]); pat['ts_la'] = int(ts[pat['i_la']])
        pat['ts_h']  = int(ts[pat['i_h']]);  pat['ts_ra'] = int(ts[pat['i_ra']])
        pat['ts_rs'] = int(ts[pat['i_rs']]); pat['ts_breakout'] = int(ts[pat['i_breakout']])

    seen: set[int] = set(); uniq: list[dict] = []
    for pat in patterns:
        if pat['ts_breakout'] not in seen:
            seen.add(pat['ts_breakout']); uniq.append(pat)
    stats['passed_geometry'] = len(uniq)

    atr14 = _atr14_nolookahead(h, l, c)
    for pat in uniq:
        atr_h = float(atr14[pat['i_h']])
        pat['atr14_at_head'] = atr_h
        pat['height_atr'] = float(pat['pattern_height'] / atr_h) if atr_h > 0 else 0.0

    return uniq, stats


def apply_canon_filters(geo_patterns: list[dict], t_arr: np.ndarray,
                         h_arr: np.ndarray, l_arr: np.ndarray, n_bars: int,
                         hma78_mhull: np.ndarray, hma78_shull: np.ndarray,
                         wma50: np.ndarray, stats: dict,
                         params: HSBottomParams) -> list[dict]:
    out = []
    for pat in geo_patterns:
        i_br = pat['i_breakout']
        # 2) Bullish FVG inline: h[i-1] < l[i+1] в (i_rs, i_br+3]
        fvg_ok = False
        for i_fvg in range(pat['i_rs'] + 1, min(n_bars - 1, i_br + 3)):
            if 0 < i_fvg < n_bars - 1:
                if h_arr[i_fvg - 1] < l_arr[i_fvg + 1]:
                    fvg_ok = True; break
        if not fvg_ok:
            stats['rej_fvg'] += 1
            continue

        hma_br = hma78_mhull[i_br]
        hma_br2 = hma78_shull[i_br]
        wma_br = wma50[i_br]
        close_br = pat['breakout_close']

        if any(np.isnan(x) for x in (hma_br, hma_br2, wma_br)):
            stats['rej_hma_wma'] += 1
            continue

        # 2a_long: close > WMA AND close > HMA (бычий режим)
        # 2c_long: HMA растёт
        # 2d_long: HMA < WMA (pre-golden-cross)
        cond_2a = (wma_br < close_br) and (hma_br < close_br)
        cond_2c = close_br > hma_br2
        cond_2d = hma_br < wma_br
        if not (cond_2a and cond_2c and cond_2d):
            stats['rej_hma_wma'] += 1
            continue

        # 2b: traveled/distance < max_path_traveled_pct
        nl_at_br = pat['nl_at_br']
        target_full = pat['target_full']
        distance_up = target_full - nl_at_br
        traveled_up = close_br - nl_at_br
        path_ok = True
        if distance_up > 0:
            path_ok = (traveled_up / distance_up) < params.max_path_traveled_pct
        if not path_ok:
            stats['rej_path_2b'] += 1
            continue

        stats['passed_canon'] += 1
        out.append(pat)
    return out


def compute_hs_bottom(df_1h: pd.DataFrame, symbol: str,
                       params: Optional[HSBottomParams] = None,
                       ) -> tuple[pd.DataFrame, dict]:
    if params is None:
        params = HSBottomParams()

    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    geo, stats = detect_hs_bottom_geometry(h, l, c, t_arr, params)
    n_bars = len(h)


    tl = pd.read_parquet(latest_trendline_path(symbol, variant="1h78"),
                          columns=["ts", "mhull", "shull"])
    hma_mhull = pd.Series(tl["mhull"].to_numpy(), index=tl["ts"].to_numpy()).reindex(t_arr).to_numpy()
    hma_shull = pd.Series(tl["shull"].to_numpy(), index=tl["ts"].to_numpy()).reindex(t_arr).to_numpy()

    wm = pd.read_parquet(latest_wma_path(symbol), columns=["ts", "wma50"])
    wma50 = pd.Series(wm["wma50"].to_numpy(), index=wm["ts"].to_numpy()).reindex(t_arr).to_numpy()

    final = apply_canon_filters(geo, t_arr, h, l, n_bars, hma_mhull, hma_shull, wma50, stats, params)

    rows = []
    for pat in final:
        rows.append({
            "signal_ts": pat['ts_breakout'] + TF_1H_MS,
            "direction": "long",
            "pattern": "hs_bottom",
            "ts_breakout": pat['ts_breakout'],
            "ts_ls": pat['ts_ls'], "ts_h": pat['ts_h'], "ts_rs": pat['ts_rs'],
            "head_p": pat['head_p'], "ls_p": pat['ls_p'], "rs_p": pat['rs_p'],
            "breakout_close": pat['breakout_close'],
            "target_half": pat['target_half'], "target_full": pat['target_full'],
            "target_2x": pat['target_2x'], "stop": pat['stop'],
            "width_bars": pat['width_bars'], "height_atr": pat['height_atr'],
            "status": "CONFIRMED",
        })
    return pd.DataFrame(rows), stats


def print_stats(stats: dict) -> None:
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"    rej_invalidated          {stats['rej_invalidated']}", file=sys.stderr, flush=True)
    print(f"    rej_no_breakout          {stats['rej_no_breakout']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry   n={stats['passed_geometry']}", file=sys.stderr, flush=True)
    print(f"    rej_fvg (2)              {stats['rej_fvg']}", file=sys.stderr, flush=True)
    print(f"    rej_hma_wma (2a/2c/2d)   {stats['rej_hma_wma']}", file=sys.stderr, flush=True)
    print(f"    rej_path_2b              {stats['rej_path_2b']}", file=sys.stderr, flush=True)
    print(f"  passed_canon (full)   n={stats['passed_canon']}  ← HS_BOTTOM", file=sys.stderr, flush=True)


def main() -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-26")
    args = ap.parse_args()

    print(f"hs_bottom: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    df_1h = agg_1h(load_1m(args.symbol))
    hits, stats = compute_hs_bottom(df_1h, args.symbol)
    print_stats(stats)

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    hits = hits[(hits["signal_ts"] >= start_ms) & (hits["signal_ts"] < end_ms)].reset_index(drop=True)
    print(f"  in window: {len(hits):,} signals", file=sys.stderr, flush=True)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"hs_bottom_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
