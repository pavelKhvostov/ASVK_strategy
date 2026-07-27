"""i-RDRB — 4-bar inverse RDRB. numpy.

SHORT RDRB → LONG i-RDRB: c4 bull AND c4.close > block_top
LONG  RDRB → SHORT i-RDRB: c4 bear AND c4.close < block_bottom

POI:
    LONG i-RDRB:  liq=(c3.body_top, c1.low) if c3.body_top<c1.low else None
                  POI = (c3.body_top, block_top)
    SHORT i-RDRB: liq=(c1.high, c3.body_bot) if c1.high<c3.body_bot else None
                  POI = (block_bottom, c3.body_bot)
"""
from __future__ import annotations
import numpy as np

from mit import wick_fill_events


def detect(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
           ts: np.ndarray, tf_ms: int) -> list[dict]:
    n = len(o)
    if n < 4:
        return []

    is_bull = c > o
    is_bear = c < o
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)

    # Индексы: c1=i-3, c2=i-2, c3=i-1, c4=i (i от 3 до n-1)
    c1s = slice(None, n-3); c2s = slice(1, n-2); c3s = slice(2, n-1); c4s = slice(3, n)

    # Underlying RDRB: SHORT (c2 bear) или LONG (c2 bull)
    short_rdrb = (
        is_bear[c2s] & (c[c2s] < l[c1s])
        & (np.maximum(body_top[c3s], l[c1s]) < np.minimum(h[c3s], body_bot[c1s]))
        & (body_bot[c1s] > body_top[c3s])
    )
    long_rdrb = (
        is_bull[c2s] & (c[c2s] > h[c1s])
        & (np.maximum(l[c3s], body_top[c1s]) < np.minimum(body_bot[c3s], h[c1s]))
        & (body_bot[c3s] > body_top[c1s])
    )

    # break-level каждого underlying rdrb для проверки c4 close
    short_bt = np.minimum(body_bot[c1s], h[c3s])   # SHORT rdrb block_top
    long_bb = np.maximum(body_top[c1s], l[c3s])    # LONG rdrb block_bottom

    # i-RDRB LONG (после SHORT rdrb): c4 bull, c4.close > block_top
    long_irdrb = short_rdrb & is_bull[c4s] & (c[c4s] > short_bt)
    # i-RDRB SHORT (после LONG rdrb): c4 bear, c4.close < block_bottom
    short_irdrb = long_rdrb & is_bear[c4s] & (c[c4s] < long_bb)

    long_idx = np.where(long_irdrb)[0] + 3   # i (c4) absolute
    short_idx = np.where(short_irdrb)[0] + 3

    events: list[dict] = []
    zone_id = 0

    for i in long_idx:
        c1_idx = i - 3; c3_idx = i - 1
        # SHORT rdrb block_top = min(c1.body_bottom, c3.high)
        bt = float(min(body_bot[c1_idx], h[c3_idx]))
        c3_top = float(body_top[c3_idx]); c1_lo = float(l[c1_idx])
        if c3_top < c1_lo:
            poi = (c3_top, bt)
        else:
            bb = float(max(l[c1_idx], body_top[c3_idx]))
            poi = (bb, bt)
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, poi, "long", i+1)
        for ev in traj:
            ev["role"] = "support"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    for i in short_idx:
        c1_idx = i - 3; c3_idx = i - 1
        bb = float(max(body_top[c1_idx], l[c3_idx]))
        c1_hi = float(h[c1_idx]); c3_bot = float(body_bot[c3_idx])
        if c1_hi < c3_bot:
            poi = (bb, c3_bot)
        else:
            bt = float(min(h[c1_idx], body_bot[c3_idx]))
            poi = (bb, bt)
        born_ts = int(ts[i]) + tf_ms
        zone_id += 1
        traj = wick_fill_events(l, h, ts, tf_ms, born_ts, poi, "short", i+1)
        for ev in traj:
            ev["role"] = "resistance"; ev["zone_id"] = zone_id
            ev["meta"] = {"born_bar_idx": int(i)}
            events.append(ev)

    return events
