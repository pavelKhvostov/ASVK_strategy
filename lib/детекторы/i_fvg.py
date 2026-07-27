"""i-FVG (Inverse FVG). numpy.

Canon v2: FVG-A + FVG-B противоположного направления. between bars шринкают A.zone.
Если A уцелела, hover B касается shrunk_A, overlap непусто → ZoI = overlap.

Direction = B.direction. Born ts = b_c3.open_time + tf_ms.
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events


def _fvg_masks(o, h, l, c):
    """Все FVG (c3_idx, direction, zone) в один проход numpy."""
    n = len(o)
    if n < 3:
        return np.array([], dtype=np.int64), np.array([], dtype=object), np.array([], dtype=object)
    long_mask = h[:-2] < l[2:]
    short_mask = l[:-2] > h[2:]
    long_c3 = np.where(long_mask)[0] + 2
    short_c3 = np.where(short_mask)[0] + 2

    fvg_records = []   # (c3_idx, direction, zone_lo, zone_hi)
    for i in long_c3:
        fvg_records.append((int(i), "long", float(h[i-2]), float(l[i])))
    for i in short_c3:
        fvg_records.append((int(i), "short", float(h[i]), float(l[i-2])))
    fvg_records.sort(key=lambda r: r[0])
    return fvg_records


def _wick_shrink_zone(lows, highs, zone_lo, zone_hi, direction, i0, i1):
    """Прошинкать zone через bars [i0, i1). Возвращает (lo, hi) или None если consumed."""
    lo, hi = zone_lo, zone_hi
    if direction == "long":
        for k in range(i0, i1):
            v = lows[k]
            if v > hi:
                continue
            if v <= lo:
                return None
            hi = float(v)
    else:
        for k in range(i0, i1):
            v = highs[k]
            if v < lo:
                continue
            if v >= hi:
                return None
            lo = float(v)
    return (lo, hi)


# TF-specific max_between:
#   LTF (≤2h): near-term inversion window 200 баров.
#   HTF (>2h): unlimited — на дневных/недельных парах A-B бывают далеко, но композит валиден.
LTF_MAX_MS = 2 * 60 * 60_000   # 2h


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int, max_between: int | None = None) -> list[dict]:
    if max_between is None:
        max_between = 200 if tf_ms <= LTF_MAX_MS else 10**9   # HTF unlimited
    fvgs = _fvg_masks(o, h, l, c)   # list of tuples
    events: list[dict] = []
    zone_id = 0

    for a_pos, (a_c3_idx, a_dir, a_lo, a_hi) in enumerate(fvgs):
        for b_pos in range(a_pos + 1, len(fvgs)):
            b_c3_idx, b_dir, b_lo, b_hi = fvgs[b_pos]
            b_c1_idx = b_c3_idx - 2
            # Early exit: fvgs отсортированы по c3_idx → как только окно превышено, дальше только хуже
            if b_c1_idx - a_c3_idx - 1 > max_between:
                break
            if b_dir == a_dir:
                continue
            if b_c1_idx <= a_c3_idx:
                continue

            shrunk = _wick_shrink_zone(l, h, a_lo, a_hi, a_dir, a_c3_idx + 1, b_c1_idx)
            if shrunk is None:
                # Если A потеряна на между, то и все дальше по b_pos будут падать
                break
            s_lo, s_hi = shrunk

            # Хотя бы одна из B's 3 свечей касается shrunk_A (wick overlap)
            b_touch = False
            for k in (b_c1_idx, b_c1_idx + 1, b_c3_idx):
                if l[k] < s_hi and h[k] > s_lo:
                    b_touch = True
                    break
            if not b_touch:
                continue

            # Overlap shrunk_A и B.zone
            lo = max(s_lo, b_lo); hi = min(s_hi, b_hi)
            if lo >= hi:
                continue
            overlap = (float(lo), float(hi))

            direction = b_dir
            role = "support" if direction == "long" else "resistance"
            born_ts = int(ts[b_c3_idx]) + tf_ms
            zone_id += 1
            traj = wick_fill_events(l, h, ts, tf_ms, born_ts, overlap, direction, b_c3_idx + 1)
            for ev in traj:
                ev["role"] = role; ev["zone_id"] = zone_id
                ev["meta"] = {
                    "a_c3_idx": int(a_c3_idx), "b_c3_idx": int(b_c3_idx),
                    "a_zone": (a_lo, a_hi), "shrunk_a": (s_lo, s_hi),
                    "b_zone": (b_lo, b_hi),
                }
                events.append(ev)

    return events
