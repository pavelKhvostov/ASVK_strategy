"""a2_filter — A2 ext_5: независимый тест "шире локального экстремума за 5 баров слева".

НЕЗАВИСИМЫЙ per-bar тест — считается на КАЖДОМ баре, не требует, чтобы бар уже
прошёл A1. Раньше (до 2026-07-23) в a_cascade.py считался кумулятивно как
`A1 AND ext5`, из-за чего домены, построенные поверх более поздних кумулятивных
стадий (см. a4_filter.py), незаметно тащили в себя и A1, и A2, и A3 разом — не давая
способа взять, например, "A1+A2+A4, но без A3". См. докстринг a_cascade.py.

    short (FH): h[i] > max(h[i-5:i])
    long  (FL): l[i] < min(l[i-5:i])
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LEFT_EXT_N = 5


def compute_a2(df_12h: pd.DataFrame) -> dict:
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()

    left5_max = pd.Series(h).rolling(LEFT_EXT_N).max().shift(1).to_numpy()
    a2_short = np.where(np.isnan(left5_max), False, h > left5_max)

    left5_min = pd.Series(l).rolling(LEFT_EXT_N).min().shift(1).to_numpy()
    a2_long = np.where(np.isnan(left5_min), False, l < left5_min)

    return {"short": a2_short, "long": a2_long}
