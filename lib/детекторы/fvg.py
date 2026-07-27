"""FVG — 3-bar gap. numpy-native.

LONG:  c1.high < c3.low  → zone=(c1.high, c3.low)   support
SHORT: c1.low  > c3.high → zone=(c3.high, c1.low)  resistance
Born ts = c3.open_time + tf_ms. Mit: wick-fill от c3+1.
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int) -> list[dict]:
    n = len(o)
    if n < 3:
        return []

    # c1 = i-2, c3 = i, condition on i от 2 до n-1
    long_mask = h[:-2] < l[2:]
    short_mask = l[:-2] > h[2:]
    long_idx = np.where(long_mask)[0] + 2   # c3 idx
    short_idx = np.where(short_mask)[0] + 2

    events: list[dict] = []
    zone_id = 0

    for i in long_idx:
        zone = (float(h[i-2]), float(l[i]))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, zone, "long", i+1)
        for ev in traj:
            ev["role"] = "support"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    for i in short_idx:
        zone = (float(h[i]), float(l[i-2]))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, zone, "short", i+1)
        for ev in traj:
            ev["role"] = "resistance"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    return events
