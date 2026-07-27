"""Numpy-array mit-функции. Возвращают события (born + fill_partial + retire).

Contract: numpy arrays as inputs, early-exit Python loops для короткоживущих зон.
Средняя длина зоны 20-100 баров → tight loop быстрее векторизации на всей серии
(не аллоцирует слайсы по 200k+ элементов).

Три модели:
    wick_fill    — постепенное сжатие фитилём
    first_touch  — одноразовое consumption при касании
    sweep        — точечный level, wick касается → CONSUMED

Canon: только low/high (no close-check). direction: long | short.
LONG zone тестируется bar.low сверху, SHORT — bar.high снизу.
"""
from __future__ import annotations
from typing import Literal

import numpy as np

Direction = Literal["long", "short"]
Interval = tuple[float, float]


def wick_fill_events(
    lows: np.ndarray, highs: np.ndarray, ts: np.ndarray, tf_ms: int,
    born_ts: int, initial_zone: Interval, direction: Direction, start_idx: int,
) -> list[dict]:
    """Wick-fill mit через tight Python loop с early exit.

    LONG:  bar.low > active_hi → skip; bar.low ≤ zone_lo → retire; else fill (active_hi=low)
    SHORT: bar.high < active_lo → skip; bar.high ≥ zone_hi → retire; else fill (active_lo=high)
    """
    zone_lo, zone_hi = float(initial_zone[0]), float(initial_zone[1])
    events: list[dict] = [{
        "ts": int(born_ts), "kind": "born", "direction": direction,
        "zone_lo": zone_lo, "zone_hi": zone_hi,
        "active_lo": zone_lo, "active_hi": zone_hi,
    }]

    n = len(lows)
    if start_idx >= n:
        return events

    if direction == "long":
        active_hi = zone_hi
        for i in range(start_idx, n):
            v = lows[i]
            if v >= active_hi:
                continue   # strictly less than active_hi для нового shrink
            if v <= zone_lo:
                events.append({
                    "ts": int(ts[i]) + tf_ms, "kind": "retire", "direction": "long",
                    "zone_lo": zone_lo, "zone_hi": zone_hi,
                    "active_lo": zone_lo, "active_hi": zone_lo,
                })
                return events
            active_hi = float(v)
            events.append({
                "ts": int(ts[i]) + tf_ms, "kind": "fill_partial", "direction": "long",
                "zone_lo": zone_lo, "zone_hi": zone_hi,
                "active_lo": zone_lo, "active_hi": active_hi,
            })
    else:  # short
        active_lo = zone_lo
        for i in range(start_idx, n):
            v = highs[i]
            if v <= active_lo:
                continue   # strictly greater than active_lo для нового shrink
            if v >= zone_hi:
                events.append({
                    "ts": int(ts[i]) + tf_ms, "kind": "retire", "direction": "short",
                    "zone_lo": zone_lo, "zone_hi": zone_hi,
                    "active_lo": zone_hi, "active_hi": zone_hi,
                })
                return events
            active_lo = float(v)
            events.append({
                "ts": int(ts[i]) + tf_ms, "kind": "fill_partial", "direction": "short",
                "zone_lo": zone_lo, "zone_hi": zone_hi,
                "active_lo": active_lo, "active_hi": zone_hi,
            })

    return events


def first_touch_events(
    lows: np.ndarray, highs: np.ndarray, ts: np.ndarray, tf_ms: int,
    born_ts: int, initial_zone: Interval, direction: Direction, start_idx: int,
    fraction: float = 1.0,
) -> list[dict]:
    """First-touch: consumed при касании wick'ом consume_level."""
    zone_lo, zone_hi = float(initial_zone[0]), float(initial_zone[1])
    consume_level = (zone_lo + (zone_hi - zone_lo) * fraction) if direction == "long" \
                    else (zone_hi - (zone_hi - zone_lo) * fraction)

    events: list[dict] = [{
        "ts": int(born_ts), "kind": "born", "direction": direction,
        "zone_lo": zone_lo, "zone_hi": zone_hi,
        "active_lo": zone_lo, "active_hi": zone_hi,
    }]

    n = len(lows)
    if start_idx >= n:
        return events

    arr = lows if direction == "long" else highs
    # numpy vectorized: consume — first idx с попаданием
    slice_ = arr[start_idx:]
    hit_mask = (slice_ <= consume_level) if direction == "long" else (slice_ >= consume_level)
    if hit_mask.any():
        retire_local = int(hit_mask.argmax())
        events.append({
            "ts": int(ts[start_idx + retire_local]) + tf_ms, "kind": "retire", "direction": direction,
            "zone_lo": zone_lo, "zone_hi": zone_hi,
            "active_lo": float(consume_level), "active_hi": float(consume_level),
        })
    return events


def sweep_events(
    lows: np.ndarray, highs: np.ndarray, ts: np.ndarray, tf_ms: int,
    born_ts: int, level: float, direction: Direction, start_idx: int,
) -> list[dict]:
    """Sweep: точечный level, wick касается → CONSUMED.

    direction="short" (FH): consume при bar.high > level.
    direction="long"  (FL): consume при bar.low  < level.
    """
    level = float(level)
    events: list[dict] = [{
        "ts": int(born_ts), "kind": "born", "direction": direction,
        "zone_lo": level, "zone_hi": level,
        "active_lo": level, "active_hi": level,
    }]

    n = len(lows)
    if start_idx >= n:
        return events

    arr = highs if direction == "short" else lows
    slice_ = arr[start_idx:]
    hit_mask = (slice_ > level) if direction == "short" else (slice_ < level)
    if hit_mask.any():
        retire_local = int(hit_mask.argmax())
        events.append({
            "ts": int(ts[start_idx + retire_local]) + tf_ms, "kind": "retire", "direction": direction,
            "zone_lo": level, "zone_hi": level,
            "active_lo": level, "active_hi": level,
        })
    return events
