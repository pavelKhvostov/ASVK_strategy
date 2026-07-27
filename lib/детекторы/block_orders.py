"""Block Orders — variable N+M composite. numpy assist для detection.

Слайс: candles[i] = preceding, candles[i+1] = initial#1, ...
(N1, N2) != (1, 1) — иначе canon-OB.

Zone (canon 2026-06-15):
    LONG:  (block.low, block.open)   support
    SHORT: (block.open, block.high)  resistance

Логика вариативная (run length), поэтому детекция через Python loop с numpy бустом.
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events

MAX_WINDOW = 20


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int, max_window: int = MAX_WINDOW) -> list[dict]:
    n = len(o)
    if n < 3:
        return []

    is_bull = c > o
    is_bear = c < o
    is_doji = c == o

    events: list[dict] = []
    zone_id = 0

    for i in range(n - 2):
        if is_doji[i]:
            continue
        first = i + 1
        if is_doji[first]:
            continue

        initial_bear = is_bear[first]
        # preceding противоположна initial
        if initial_bear and not is_bull[i]:
            continue
        if (not initial_bear) and not is_bear[i]:
            continue

        # Initial run length
        j = 1
        while j < max_window and (i + j) < n:
            k = i + j
            if initial_bear and is_bear[k]:
                j += 1
            elif (not initial_bear) and is_bull[k]:
                j += 1
            else:
                break
        n_initial = j - 1
        if n_initial < 1 or (i + j) >= n:
            continue

        block_open = float(o[first])

        # Counter run
        counter_bull = initial_bear
        m = 0
        crossed = False
        while (j + m) < max_window and (i + j + m) < n:
            k = i + j + m
            ok = (counter_bull and is_bull[k]) or ((not counter_bull) and is_bear[k])
            if not ok:
                break
            m += 1
            if (initial_bear and c[k] > block_open) or ((not initial_bear) and c[k] < block_open):
                crossed = True
                break
        if not crossed:
            continue
        n_counter = m
        if n_initial == 1 and n_counter == 1:
            continue

        last_idx = i + n_initial + n_counter
        block_slice = slice(i + 1, i + 1 + n_initial + n_counter)
        hi = float(h[block_slice].max())
        lo = float(l[block_slice].min())

        if initial_bear:
            direction = "long"; role = "support"; zone = (lo, block_open)
        else:
            direction = "short"; role = "resistance"; zone = (block_open, hi)

        born_ts = int(ts[last_idx]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, zone, direction, last_idx + 1)
        for ev in traj:
            ev["role"] = role; ev["zone_id"] = zone_id
            ev["meta"] = {"preceding_idx": int(i), "last_bar_idx": int(last_idx)}
            events.append(ev)

    return events
