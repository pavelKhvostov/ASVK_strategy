"""a1_filter — A1 Pre-W: 3-свечный локальный экстремум (ASVK-portable, независимый).

Базовая стадия A-каскада — определяет, что вообще считается кандидатом-пивотом
(Williams-style 3-bar local extremum). В отличие от A2/A3/A4 (см. a2_filter.py,
a3_filter.py, a4_filter.py), у A1 нет более ранней стадии, от которой можно быть
"независимым" — это сам вход в домен.

Также здесь же Williams n=2 right-confirmation (свойство самого пивота — подтвердился
ли он двумя следующими барами, а не ещё один A-фильтр) — неотделимо от A1-события,
поэтому осталось в этом же файле, а не размазано отдельным скриптом.

    A1 short (FH, fractal high): h[i] > h[i-1] AND h[i] > h[i-2]
    A1 long  (FL, fractal low):  l[i] < l[i-1] AND l[i] < l[i-2]

    confirmed (Williams n=2 right):
    short: h[i+1] < h[i] AND h[i+2] < h[i]
    long:  l[i+1] > l[i] AND l[i+2] > l[i]
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_a1(df_12h: pd.DataFrame) -> dict:
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    T = len(df_12h)

    a1_short = np.zeros(T, dtype=bool)
    a1_short[2:] = (h[2:] > h[1:-1]) & (h[2:] > h[:-2])

    a1_long = np.zeros(T, dtype=bool)
    a1_long[2:] = (l[2:] < l[1:-1]) & (l[2:] < l[:-2])

    return {"short": a1_short, "long": a1_long}


def compute_confirmation(df_12h: pd.DataFrame) -> dict:
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    T = len(df_12h)

    conf_short = np.zeros(T, dtype=bool)
    conf_short[:-2] = (h[1:-1] < h[:-2]) & (h[2:] < h[:-2])

    conf_long = np.zeros(T, dtype=bool)
    conf_long[:-2] = (l[1:-1] > l[:-2]) & (l[2:] > l[:-2])

    confirmable = np.arange(T) <= T - 3
    return {"short": conf_short, "long": conf_long, "confirmable": confirmable}
