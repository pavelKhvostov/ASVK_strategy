"""Marubozu — свеча без фитиля со стороны open. Точка (level).

    LONG (bull):  open == low  AND close > open
        level = open, direction="long", sweep_low.
    SHORT (bear): open == high AND close < open
        level = open, direction="short", sweep_high.

Mit: sweep. Consumed при wick касании уровня:
    LONG:  bar.low  < level → CONSUMED
    SHORT: bar.high > level → CONSUMED

Born ts = c.open_time + tf_ms.
"""
from __future__ import annotations
import numpy as np

from mit import sweep_events


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int) -> list[dict]:
    n = len(o)
    if n < 1:
        return []

    long_mask = (o == l) & (c > o)
    short_mask = (o == h) & (c < o)
    long_idx = np.where(long_mask)[0]
    short_idx = np.where(short_mask)[0]

    events: list[dict] = []
    zone_id = 0

    for i in long_idx:
        level = float(o[i])
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = sweep_events(l, h, ts, tf_ms, born_ts, level, "long", i + 1)
        for ev in traj:
            ev["role"] = "sweep_low"
            ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    for i in short_idx:
        level = float(o[i])
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = sweep_events(l, h, ts, tf_ms, born_ts, level, "short", i + 1)
        for ev in traj:
            ev["role"] = "sweep_high"
            ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    return events
