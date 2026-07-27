"""a3_filter — A3 color: независимый тест смены цвета свечи / 3 однонаправленных подряд.

НЕЗАВИСИМЫЙ per-bar тест, одинаковый для short/long (чисто свечная геометрия, не
зависит от направления пивота). Как и A2/A4 — считается на каждом баре без AND с
предыдущими стадиями. См. докстринг a_cascade.py про причину такого разделения.

По явной инструкции пользователя (проверено по транскрипту прошлой сессии: "A3
исключили пока из расчетов", "Фильтры a1 a2 a4 остаются") A3 сейчас НЕ входит в
рабочий домен ни одного блока (см. a_cascade.py: a124_pool = a1 & a2 & a4, без a3) —
файл остаётся для справки/будущего использования, не для того чтобы его прямо сейчас
на что-то накладывали.

    opp_colors[i] = color[i] != color[i-1], оба non-doji
    three_same[i] = color[i] == color[i-1] == color[i-2], все non-doji
    a3[i] = opp_colors[i] OR three_same[i]
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_a3(df_12h: pd.DataFrame) -> dict:
    o = df_12h["open"].to_numpy()
    c = df_12h["close"].to_numpy()
    T = len(df_12h)

    color = np.where(c > o, 1, np.where(c < o, -1, 0))
    non_doji = color != 0

    opp_colors = np.zeros(T, dtype=bool)
    opp_colors[1:] = (color[1:] != color[:-1]) & non_doji[1:] & non_doji[:-1]

    three_same = np.zeros(T, dtype=bool)
    three_same[2:] = (
        (color[2:] == color[1:-1]) & (color[2:] == color[:-2])
        & non_doji[2:] & non_doji[1:-1] & non_doji[:-2]
    )
    a3_mask = opp_colors | three_same

    # одна и та же маска для обоих направлений — цветовой паттерн не зависит от
    # short/long, но возвращаем dict для единообразия с a1/a2/a4
    return {"short": a3_mask, "long": a3_mask}
