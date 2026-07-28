"""OB_VC — HTF OB + LTF FVG composite.

Условия #1-9 (canon). Condition #9 (FVG not consumed) проверяется через check_tf (default 15m).

Direction = HTF OB direction. Born ts = max(HTF cur close, primary FVG c3 close).
Mit start = первый HTF idx с open_time >= born_ts.
"""
from __future__ import annotations
from typing import Mapping

import numpy as np
from numba import njit

from mit import wick_fill_events

HTF_TO_LTF: dict[str, tuple[str, ...]] = {
    "1h":  ("15m", "30m"), "2h":  ("15m", "30m"),
    "4h":  ("1h", "2h"),    "6h":  ("1h", "2h"),
    "12h": ("4h", "6h"),    "1d":  ("4h", "6h"),
    "2d":  ("12h",),         "3d":  ("12h",),
}


def _fvg_list(o, h, l, c, ts):
    """Все FVG на массиве: list of (c3_idx, direction, zone_lo, zone_hi, c1_open_time, c3_open_time)."""
    n = len(o)
    if n < 3:
        return []
    long_c3 = np.where(h[:-2] < l[2:])[0] + 2
    short_c3 = np.where(l[:-2] > h[2:])[0] + 2
    out = []
    for i in long_c3:
        out.append((int(i), "long", float(h[i-2]), float(l[i]), int(ts[i-2]), int(ts[i])))
    for i in short_c3:
        out.append((int(i), "short", float(h[i]), float(l[i-2]), int(ts[i-2]), int(ts[i])))
    out.sort(key=lambda r: r[0])
    return out


@njit(cache=True)
def _first_fh_above_nb(h_arr, start_idx, threshold, N):
    """Первый Williams N=n FH с center.high > threshold.
    Returns (center_idx, high). Если не найдено — (-1, 0.0)."""
    n = len(h_arr)
    lo = start_idx if start_idx > N else N
    for i in range(lo, n - N):
        if h_arr[i] <= threshold:
            continue
        ok = True
        for k in range(1, N+1):
            if h_arr[i] <= h_arr[i-k] or h_arr[i] <= h_arr[i+k]:
                ok = False
                break
        if ok:
            return (i, h_arr[i])
    return (-1, 0.0)


@njit(cache=True)
def _first_fl_below_nb(l_arr, start_idx, threshold, N):
    """Первый Williams N=n FL с center.low < threshold.
    Returns (center_idx, low). Если не найдено — (-1, 0.0)."""
    n = len(l_arr)
    lo = start_idx if start_idx > N else N
    for i in range(lo, n - N):
        if l_arr[i] >= threshold:
            continue
        ok = True
        for k in range(1, N+1):
            if l_arr[i] >= l_arr[i-k] or l_arr[i] >= l_arr[i+k]:
                ok = False
                break
        if ok:
            return (i, l_arr[i])
    return (-1, 0.0)


def _first_fh_above(h_arr, l_arr, start_idx, threshold, N=2):
    """Wrapper — сохраняет старую сигнатуру. Возвращает (idx, high) или None."""
    idx, val = _first_fh_above_nb(h_arr, int(start_idx), float(threshold), int(N))
    if idx == -1:
        return None
    return (int(idx), float(val))


def _first_fl_below(h_arr, l_arr, start_idx, threshold, N=2):
    idx, val = _first_fl_below_nb(l_arr, int(start_idx), float(threshold), int(N))
    if idx == -1:
        return None
    return (int(idx), float(val))


def _htf_idx_ge_ts(ts_arr, target):
    return int(np.searchsorted(ts_arr, target, side="left"))


def detect(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, ts: np.ndarray,
    tf_ms: int, htf: str,
    ltf_arrays: Mapping[str, tuple],       # {ltf_name: (o,h,l,c,ts)}
    ltf_ms_map: Mapping[str, int],
    n_fractal: int = 2,
    check_tf: str = "15m",
) -> list[dict]:
    allowed_ltfs = HTF_TO_LTF.get(htf)
    if not allowed_ltfs:
        return []

    n = len(o)
    if n < 2:
        return []

    is_bull = c > o
    is_bear = c < o
    ob_mask = np.zeros(n, dtype=bool)
    ob_dir_long = is_bear[:-1] & is_bull[1:] & (c[1:] > o[:-1])
    ob_dir_short = is_bull[:-1] & is_bear[1:] & (c[1:] < o[:-1])
    ob_mask[1:] = ob_dir_long | ob_dir_short

    # Precompute LTF FVGs
    ltf_fvgs = {}
    for ltf in allowed_ltfs:
        if ltf not in ltf_arrays:
            continue
        arrs = ltf_arrays[ltf]
        ltf_fvgs[ltf] = _fvg_list(*arrs)

    # Check_tf arrays + ms
    check_arrs = ltf_arrays.get(check_tf)
    check_tf_ms = ltf_ms_map.get(check_tf)
    if check_arrs is not None and check_tf_ms is not None:
        _, ch_h, ch_l, _, ch_ts = check_arrs
    else:
        ch_h = ch_l = ch_ts = None

    events: list[dict] = []
    zone_id = 0

    for i in range(1, n):
        if not ob_mask[i]:
            continue
        prev_o = float(o[i-1]); prev_l = float(l[i-1]); prev_h = float(h[i-1])
        cur_l = float(l[i]); cur_h = float(h[i])
        if ob_dir_long[i-1]:
            direction = "long"
            drop_area = (min(prev_l, cur_l), prev_o)
            role = "support"
        else:
            direction = "short"
            drop_area = (prev_o, max(prev_h, cur_h))
            role = "resistance"

        ob_prev_open_ts = int(ts[i-1])
        cur_open_ts = int(ts[i])

        # First opposite fractal + fh_confirm_ts на каждом LTF
        first_boundary: dict[str, float] = {}
        first_confirm_ts: dict[str, int] = {}
        for ltf in allowed_ltfs:
            arrs = ltf_arrays.get(ltf)
            if not arrs:
                continue
            _, lh, ll, _, lts = arrs
            ltf_ms = ltf_ms_map.get(ltf)
            if ltf_ms is None:
                continue
            start_ltf = _htf_idx_ge_ts(lts, cur_open_ts)
            if direction == "long":
                res = _first_fh_above(lh, ll, start_ltf, drop_area[1], N=n_fractal)
                if res is None: continue
                fh_idx, boundary = res
                first_boundary[ltf] = boundary
                first_confirm_ts[ltf] = int(lts[fh_idx]) + (n_fractal + 1) * ltf_ms
            else:
                res = _first_fl_below(lh, ll, start_ltf, drop_area[0], N=n_fractal)
                if res is None: continue
                fl_idx, boundary = res
                first_boundary[ltf] = boundary
                first_confirm_ts[ltf] = int(lts[fl_idx]) + (n_fractal + 1) * ltf_ms

        if not first_boundary:
            continue

        # Собираем candidates. Каждый tracks LTF и fh_confirm_ts своего LTF.
        candidates = []  # (fvg_close_ts, fvg_zone, ltf, fh_confirm_ts)
        for ltf, boundary in first_boundary.items():
            fvgs = ltf_fvgs.get(ltf, [])
            ltf_ms = ltf_ms_map[ltf]
            fh_confirm_ts = first_confirm_ts[ltf]
            if direction == "long":
                allowed_range = (drop_area[0], boundary)
            else:
                allowed_range = (boundary, drop_area[1])
            for c3_idx, fvg_dir, fvg_lo, fvg_hi, c1_ts, c3_ts in fvgs:
                if fvg_dir != direction:
                    continue
                fvg_close_ts = c3_ts + ltf_ms
                if fvg_close_ts > fh_confirm_ts:
                    break  # sorted by c3_idx; дальше только позже
                if c1_ts < ob_prev_open_ts:
                    continue
                if fvg_lo >= drop_area[1] or fvg_hi <= drop_area[0]:
                    continue
                if not (fvg_lo >= allowed_range[0] and fvg_hi <= allowed_range[1]):
                    continue

                # Condition #9: check_tf min/max в окне [fvg_close, fh_confirm].
                # Включаем только check-бары, ЗАКРЫВШИЕСЯ к fh_confirm_ts:
                #   bar.open + check_tf_ms <= fh_confirm_ts  ↔  bar.open <= fh_confirm_ts - check_tf_ms
                if ch_ts is not None and check_tf_ms is not None:
                    lo_idx = int(np.searchsorted(ch_ts, fvg_close_ts, side="left"))
                    hi_idx = int(np.searchsorted(ch_ts, fh_confirm_ts - check_tf_ms, side="right"))
                    if hi_idx > lo_idx:
                        if direction == "long":
                            wmin = float(ch_l[lo_idx:hi_idx].min())
                            if wmin <= fvg_lo:
                                continue
                        else:
                            wmax = float(ch_h[lo_idx:hi_idx].max())
                            if wmax >= fvg_hi:
                                continue

                candidates.append((fvg_close_ts, (fvg_lo, fvg_hi), ltf, fh_confirm_ts))

        if not candidates:
            continue

        # Каждая валидирующая LTF FVG = самостоятельная ob_vc-зона со своим
        # zone_id и lifecycle.
        candidates.sort(key=lambda x: x[0])
        htf_cur_close_ts = int(ts[i]) + tf_ms

        for fvg_close_ts, fvg_zone, fvg_ltf, fvg_fh_confirm_ts in candidates:
            zone_id += 1
            # No lookahead: композит observable когда HTF cur закрылся, эта FVG закрылась,
            # И opposite fractal ЭТОЙ LTF подтвердился.
            born_ts = max(htf_cur_close_ts, fvg_close_ts, fvg_fh_confirm_ts)
            mit_start_idx = _htf_idx_ge_ts(ts, born_ts)

            traj = wick_fill_events(l, h, ts, tf_ms, born_ts, fvg_zone, direction, mit_start_idx)
            for ev in traj:
                ev["role"] = role
                ev["zone_id"] = zone_id
                ev["meta"] = {
                    "ob_bar_idx": int(i),
                    "htf": htf,
                    "entry_zone": drop_area,
                    "ltf": fvg_ltf,
                    "fvg_close_ts": int(fvg_close_ts),
                    "fh_confirm_ts": int(fvg_fh_confirm_ts),
                    "n_fvg_at_ob": len(candidates),
                }
                events.append(ev)

    return events
