"""RB (Rejection Block) — bar с доминирующим фитилём.

body>0; upper ≥ 2×lower AND upper ≥ 3×body → TOP RB (short/resistance)
        lower ≥ 2×upper AND lower ≥ 3×body → BOTTOM RB (long/support)

Zone:
    LONG:  (low, body_bottom)
    SHORT: (body_top, high)
Mit: first-touch fraction=0.5 (mid wick = entry level).
"""
from __future__ import annotations
import numpy as np

from mit import first_touch_events

K1 = 2.0   # dominant ≥ K1 × other
K2 = 3.0   # dominant ≥ K2 × body


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int) -> list[dict]:
    n = len(o)
    if n < 1:
        return []

    body = np.abs(o - c)
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    upper = h - body_top
    lower = body_bot - l

    valid = (body > 0) & (upper > 0) & (lower > 0)
    top_mask = valid & (upper >= K1 * lower) & (upper >= K2 * body)
    bottom_mask = valid & (lower >= K1 * upper) & (lower >= K2 * body)
    top_idx = np.where(top_mask)[0]
    bottom_idx = np.where(bottom_mask)[0]

    events: list[dict] = []
    zone_id = 0

    for i in top_idx:
        zone = (float(body_top[i]), float(h[i]))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = first_touch_events(l, h, ts, tf_ms, born_ts, zone, "short", i+1, fraction=0.5)
        for ev in traj:
            ev["role"] = "resistance"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    for i in bottom_idx:
        zone = (float(l[i]), float(body_bot[i]))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = first_touch_events(l, h, ts, tf_ms, born_ts, zone, "long", i+1, fraction=0.5)
        for ev in traj:
            ev["role"] = "support"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    return events
