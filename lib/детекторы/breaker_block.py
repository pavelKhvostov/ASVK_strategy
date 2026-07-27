"""Breaker Block.

    - Activation window post[0..3] (bar 3-6).
    - Bullish flip (LONG OB, close > prev.high): zone [prev.open, prev.high] BELOW activator → SUPPORT (long).
    - Bearish flip (SHORT OB, close < prev.low):  zone [prev.low, prev.open]  ABOVE activator → RESISTANCE (short).

Mit: wick-fill от бара после activator.
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events

ACTIVATION_WINDOW = 4   # bar 3-6 = post[0..3]


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int,
           activation_window: int = ACTIVATION_WINDOW) -> list[dict]:
    n = len(o)
    if n < 3:
        return []

    is_bull = c > o
    is_bear = c < o

    # OB pairs (cur idx i)
    long_ob_mask = is_bear[:-1] & is_bull[1:] & (c[1:] > o[:-1])
    short_ob_mask = is_bull[:-1] & is_bear[1:] & (c[1:] < o[:-1])
    long_ob_idx = np.where(long_ob_mask)[0] + 1
    short_ob_idx = np.where(short_ob_mask)[0] + 1

    events: list[dict] = []
    zone_id = 0

    # Bullish flip: LONG OB, threshold = prev.high
    for i in long_ob_idx:
        prev_open = float(o[i-1]); prev_high = float(h[i-1])
        if prev_high <= prev_open:
            continue  # дегенерат
        threshold = prev_high
        # Activation scan: post[0..3] = candles[i+1..i+4]
        n_win = min(activation_window, n - i - 1)
        activated_at = -1
        for k in range(n_win):
            if c[i + 1 + k] > threshold:
                activated_at = i + 1 + k
                break
        if activated_at < 0:
            continue

        zone = (prev_open, prev_high)   # zone below activator → SUPPORT
        born_ts = int(ts[activated_at]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, zone, "long", activated_at + 1)
        for ev in traj:
            ev["role"] = "support"; ev["zone_id"] = zone_id
            ev["meta"] = {"ob_bar_idx": int(i), "activator_idx": int(activated_at),
                          "ob_direction": "long"}
            events.append(ev)

    # Bearish flip: SHORT OB, threshold = prev.low
    for i in short_ob_idx:
        prev_low = float(l[i-1]); prev_open = float(o[i-1])
        if prev_open <= prev_low:
            continue
        threshold = prev_low
        n_win = min(activation_window, n - i - 1)
        activated_at = -1
        for k in range(n_win):
            if c[i + 1 + k] < threshold:
                activated_at = i + 1 + k
                break
        if activated_at < 0:
            continue

        zone = (prev_low, prev_open)   # zone above activator → RESISTANCE
        born_ts = int(ts[activated_at]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, zone, "short", activated_at + 1)
        for ev in traj:
            ev["role"] = "resistance"; ev["zone_id"] = zone_id
            ev["meta"] = {"ob_bar_idx": int(i), "activator_idx": int(activated_at),
                          "ob_direction": "short"}
            events.append(ev)

    return events
