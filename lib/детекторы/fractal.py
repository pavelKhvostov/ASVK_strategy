"""Fractal (Williams N=2 default) — точечный pivot-level.

FH: center.high строго > всех 2N соседей → short (sweep_high)
FL: center.low  строго < всех 2N соседей → long  (sweep_low)

Born ts = (center + N).open_time + tf_ms (момент подтверждения pivot).
Mit: sweep от confirm+1.
"""
from __future__ import annotations
import numpy as np

from mit import sweep_events

N_DEFAULT = 2


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int, n: int = N_DEFAULT) -> list[dict]:
    N = len(o)
    if N < 2 * n + 1:
        return []

    # Sliding window: для каждого i от n до N-n-1, сравнить с 2n соседями
    # Строгий (>) для FH, (<) для FL
    # Векторизация через broadcast сравнений
    center_h = h[n:N-n]  # shape (M,) где M = N-2n
    center_l = l[n:N-n]

    # Собираем соседей (2n массивов)
    neighbors_h = np.stack([h[i:i+len(center_h)] for i in range(2*n+1) if i != n], axis=0)
    neighbors_l = np.stack([l[i:i+len(center_l)] for i in range(2*n+1) if i != n], axis=0)

    is_fh = np.all(center_h > neighbors_h, axis=0)
    is_fl = np.all(center_l < neighbors_l, axis=0)

    # Исключаем случай "оба" (defensive)
    fh_mask = is_fh & ~is_fl
    fl_mask = is_fl & ~is_fh

    fh_center_idx = np.where(fh_mask)[0] + n   # абсолютный center idx
    fl_center_idx = np.where(fl_mask)[0] + n

    events: list[dict] = []
    zone_id = 0

    for i in fh_center_idx:
        confirm_idx = int(i + n)
        level = float(h[i])
        born_ts = int(ts[confirm_idx]) + tf_ms
        zone_id += 1
        traj = sweep_events(l, h, ts, tf_ms, born_ts, level, "short", confirm_idx+1)
        for ev in traj:
            ev["role"] = "sweep_high"; ev["zone_id"] = zone_id
            ev["meta"] = {"center_bar_idx": int(i), "confirm_bar_idx": confirm_idx, "n": n}
            events.append(ev)

    for i in fl_center_idx:
        confirm_idx = int(i + n)
        level = float(l[i])
        born_ts = int(ts[confirm_idx]) + tf_ms
        zone_id += 1
        traj = sweep_events(l, h, ts, tf_ms, born_ts, level, "long", confirm_idx+1)
        for ev in traj:
            ev["role"] = "sweep_low"; ev["zone_id"] = zone_id
            ev["meta"] = {"center_bar_idx": int(i), "confirm_bar_idx": confirm_idx, "n": n}
            events.append(ev)

    return events
