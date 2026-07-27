"""OB Liquidity (ob_liq).

LONG: prev bear, cur bull, cur.close > prev.open,
      prev.lower_wick > 3×cur.lower_wick, prev.lower_wick > prev.body,
      cur.low > prev.low.
      ZoI = (prev.low, cur.low)  narrow LIQ marker, support.

SHORT — зеркально: cur.high < prev.high; ZoI = (cur.high, prev.high) resistance.

Mit: first-touch fraction=1.0 (любой wick касается ZoI → CONSUMED).
"""
from __future__ import annotations
import numpy as np

from mit import first_touch_events


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int) -> list[dict]:
    n = len(o)
    if n < 2:
        return []

    is_bull = c > o
    is_bear = c < o
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    lower_wick = body_bot - l
    upper_wick = h - body_top
    body = np.abs(o - c)

    # LONG:  prev bear, cur bull, cur.close > prev.open,
    #        prev.lower > 3×cur.lower, prev.lower > prev.body, cur.low > prev.low
    long_cond = (
        is_bear[:-1] & is_bull[1:] & (c[1:] > o[:-1])
        & (lower_wick[:-1] > 3 * lower_wick[1:])
        & (lower_wick[:-1] > body[:-1])
        & (l[1:] > l[:-1])
    )
    long_idx = np.where(long_cond)[0] + 1

    # SHORT: prev bull, cur bear, cur.close < prev.open,
    #        prev.upper > 3×cur.upper, prev.upper > prev.body, cur.high < prev.high
    short_cond = (
        is_bull[:-1] & is_bear[1:] & (c[1:] < o[:-1])
        & (upper_wick[:-1] > 3 * upper_wick[1:])
        & (upper_wick[:-1] > body[:-1])
        & (h[1:] < h[:-1])
    )
    short_idx = np.where(short_cond)[0] + 1

    events: list[dict] = []
    zone_id = 0

    for i in long_idx:
        zone = (float(l[i-1]), float(l[i]))
        entry_zone = (float(min(l[i-1], l[i])), float(o[i-1]))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = first_touch_events(l, h, ts, tf_ms, born_ts, zone, "long", i+1, fraction=1.0)
        for ev in traj:
            ev["role"] = "support"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i), "entry_zone": entry_zone}
            events.append(ev)

    for i in short_idx:
        zone = (float(h[i]), float(h[i-1]))
        entry_zone = (float(o[i-1]), float(max(h[i-1], h[i])))
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = first_touch_events(l, h, ts, tf_ms, born_ts, zone, "short", i+1, fraction=1.0)
        for ev in traj:
            ev["role"] = "resistance"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i), "entry_zone": entry_zone}
            events.append(ev)

    return events
