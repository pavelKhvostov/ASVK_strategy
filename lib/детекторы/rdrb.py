"""RDRB — 3-bar rally-drop-rally / drop-rally-drop.

SHORT (bear middle):
    c2.close < c1.low
    (c3.body_top..c3.high) ∩ (c1.low..c1.body_bottom) непусто
    c1.body_bottom > c3.body_top
    POI = (block_bottom, c1.body_bottom)   resistance

LONG (bull middle) — зеркально:
    c2.close > c1.high
    (c3.low..c3.body_bottom) ∩ (c1.body_top..c1.high) непусто
    c3.body_bottom > c1.body_top
    POI = (c1.body_top, block_top)   support

Mit: wick-fill. Born ts = c3.open_time + tf_ms.
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int) -> list[dict]:
    n = len(o)
    if n < 3:
        return []

    is_bull = c > o
    is_bear = c < o
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)

    c1 = slice(None, n-2); c2 = slice(1, n-1); c3 = slice(2, n)

    # SHORT: c2 bear, c2.close < c1.low, overlap((c3.body_top, c3.high), (c1.low, c1.body_bot)),
    #        c1.body_bot > c3.body_top
    short_cond = (
        is_bear[c2]
        & (c[c2] < l[c1])
        & (np.maximum(body_top[c3], l[c1]) < np.minimum(h[c3], body_bot[c1]))
        & (body_bot[c1] > body_top[c3])
    )
    # LONG: c2 bull, c2.close > c1.high, overlap((c3.low, c3.body_bot), (c1.body_top, c1.high)),
    #       c3.body_bot > c1.body_top
    long_cond = (
        is_bull[c2]
        & (c[c2] > h[c1])
        & (np.maximum(l[c3], body_top[c1]) < np.minimum(body_bot[c3], h[c1]))
        & (body_bot[c3] > body_top[c1])
    )

    short_idx = np.where(short_cond)[0] + 2   # c3 idx
    long_idx = np.where(long_cond)[0] + 2

    events: list[dict] = []
    zone_id = 0

    for i in short_idx:
        block_bottom = max(float(l[i-2]), float(body_top[i]))
        poi = (block_bottom, float(body_bot[i-2]))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, poi, "short", i+1)
        for ev in traj:
            ev["role"] = "resistance"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    for i in long_idx:
        block_top = min(float(h[i-2]), float(body_bot[i]))
        poi = (float(body_top[i-2]), block_top)
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, poi, "long", i+1)
        for ev in traj:
            ev["role"] = "support"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    return events
