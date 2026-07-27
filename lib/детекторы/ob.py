"""Order Block (OB) — numpy-native детектор + mit-трекинг.

Canon:
    LONG:  prev bear, cur bull, cur.close > prev.open.
           zone = [min(prev.low, cur.low), prev.open]   role = support
    SHORT: prev bull, cur bear, cur.close < prev.open.
           zone = [prev.open, max(prev.high, cur.high)] role = resistance

Born ts = cur.open_time + tf_ms.
Mit: wick-fill от cur+1.
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int) -> list[dict]:
    n = len(o)
    if n < 2:
        return []

    is_bull = c > o
    is_bear = c < o

    # LONG OB на i: is_bear[i-1] & is_bull[i] & (c[i] > o[i-1])
    long_mask = is_bear[:-1] & is_bull[1:] & (c[1:] > o[:-1])
    short_mask = is_bull[:-1] & is_bear[1:] & (c[1:] < o[:-1])
    long_idx = np.where(long_mask)[0] + 1   # cur idx
    short_idx = np.where(short_mask)[0] + 1

    # Заготовки границ
    events: list[dict] = []
    zone_id = 0

    for i in long_idx:
        zone_lo = float(min(l[i-1], l[i]))
        zone_hi = float(o[i-1])
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, (zone_lo, zone_hi), "long", i+1)
        for ev in traj:
            ev["role"] = "support"
            ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    for i in short_idx:
        zone_lo = float(o[i-1])
        zone_hi = float(max(h[i-1], h[i]))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, (zone_lo, zone_hi), "short", i+1)
        for ev in traj:
            ev["role"] = "resistance"
            ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    return events
