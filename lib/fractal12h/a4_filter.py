"""a4_filter — A4 body+wick: независимый тест "не marubozu" по своему направлению.

НЕЗАВИСИМЫЙ per-bar тест: тело свечи ≤ 80% диапазона, а wick в сторону, релевантную
направлению пивота (верхний для short/FH, нижний для long/FL), ≥ 3% диапазона.
Считается на каждом баре, без AND с A1/A2/A3. См. докстринг a_cascade.py.

    body_pct  = |close-open| / (high-low)
    wick_pct  = (верхний или нижний wick, по направлению) / (high-low)
    a4[i] = body_pct[i] <= 0.80 AND wick_pct[i] >= 0.03
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BODY_MAX = 0.80
WICK_MIN = 0.03


def compute_a4(df_12h: pd.DataFrame) -> dict:
    o = df_12h["open"].to_numpy()
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    c = df_12h["close"].to_numpy()

    hl = h - l
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    body_pct = np.divide(body, hl, out=np.zeros_like(body), where=(hl > 0))
    up_wick_pct = np.divide(upper_wick, hl, out=np.zeros_like(upper_wick), where=(hl > 0))
    lo_wick_pct = np.divide(lower_wick, hl, out=np.zeros_like(lower_wick), where=(hl > 0))

    a4_short = (body_pct <= BODY_MAX) & (up_wick_pct >= WICK_MIN)
    a4_long = (body_pct <= BODY_MAX) & (lo_wick_pct >= WICK_MIN)

    return {
        "short": a4_short, "long": a4_long,
        "body_pct": body_pct, "up_wick_pct": up_wick_pct, "lo_wick_pct": lo_wick_pct,
    }
