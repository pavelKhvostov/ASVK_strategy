"""Mitigation Block — полностью пробитый OB + Правило 1. numpy.

LONG OB пробит вниз → MB short/resistance (zone = drop area above activator).
SHORT OB пробит вверх → MB long/support (zone = rally area below activator).
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events

MAX_BARS_TO_BREAKOUT = 30


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int,
           max_bars_to_breakout: int = MAX_BARS_TO_BREAKOUT) -> list[dict]:
    n = len(o)
    if n < 5:
        return []

    is_bull = c > o
    is_bear = c < o

    long_ob_mask = is_bear[:-1] & is_bull[1:] & (c[1:] > o[:-1])
    short_ob_mask = is_bull[:-1] & is_bear[1:] & (c[1:] < o[:-1])
    long_ob_idx = np.where(long_ob_mask)[0] + 1
    short_ob_idx = np.where(short_ob_mask)[0] + 1

    events: list[dict] = []
    zone_id = 0

    # LONG OB → SHORT MB (пробой вниз)
    for i in long_ob_idx:
        drop_low = float(min(l[i-1], l[i]))
        prev_open = float(o[i-1])
        broken_level = drop_low
        ob_zone = (drop_low, prev_open)

        scan_end = min(max_bars_to_breakout, n - i - 1)
        armed_at = -1
        for k in range(scan_end):
            bar_idx = i + 1 + k
            if c[bar_idx] >= broken_level:
                continue
            # пробойная свеча найдена; нужны 3 confirming
            confirm_start = bar_idx + 1
            confirm_end = confirm_start + 3
            if confirm_end > n:
                break
            ok = True
            for j in range(confirm_start, confirm_end):
                if not (o[j] < broken_level and c[j] < broken_level):
                    ok = False; break
            if ok:
                armed_at = confirm_end - 1
            break

        if armed_at < 0:
            continue

        born_ts = int(ts[armed_at]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, ob_zone, "short", armed_at + 1)
        for ev in traj:
            ev["role"] = "resistance"; ev["zone_id"] = zone_id
            ev["meta"] = {"ob_bar_idx": int(i), "ob_direction": "long",
                          "armed_at_idx": int(armed_at), "broken_level": float(broken_level)}
            events.append(ev)

    # SHORT OB → LONG MB (пробой вверх)
    for i in short_ob_idx:
        rally_high = float(max(h[i-1], h[i]))
        prev_open = float(o[i-1])
        broken_level = rally_high
        ob_zone = (prev_open, rally_high)

        scan_end = min(max_bars_to_breakout, n - i - 1)
        armed_at = -1
        for k in range(scan_end):
            bar_idx = i + 1 + k
            if c[bar_idx] <= broken_level:
                continue
            confirm_start = bar_idx + 1
            confirm_end = confirm_start + 3
            if confirm_end > n:
                break
            ok = True
            for j in range(confirm_start, confirm_end):
                if not (o[j] > broken_level and c[j] > broken_level):
                    ok = False; break
            if ok:
                armed_at = confirm_end - 1
            break

        if armed_at < 0:
            continue

        born_ts = int(ts[armed_at]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, ob_zone, "long", armed_at + 1)
        for ev in traj:
            ev["role"] = "support"; ev["zone_id"] = zone_id
            ev["meta"] = {"ob_bar_idx": int(i), "ob_direction": "short",
                          "armed_at_idx": int(armed_at), "broken_level": float(broken_level)}
            events.append(ev)

    return events
