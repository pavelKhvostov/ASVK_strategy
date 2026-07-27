"""Визуальная схема проекта фрактал-12h — структура_BTC.png (ASVK-portable).

Адаптация G:\\Claude\\reference\\фрактал-12h\\скрипт_структура_фрактал_12h.py (WSL
канон) под ASVK-portable данные (G:\\ASVK\\data\\fractal12h\\*_hits_*.parquet, актуальные
на 2026-07-25) — числа берутся ИЗ ДАННЫХ, не из тела скрипта, чтобы схема не расходилась
с кодом (та же дисциплина, что в WSL-оригинале, см. комментарий у S=... ниже).

Правка 2026-07-25 (запрошено пользователем: "подправь структуру png ... сверху до
уровня B5"), затрагивает только слои Header/A-cascade/B1/B2/B3/B4/B5:
  1. Header subtitle: было "A2/A3/A4 — informational only" для ВСЕХ блоков — это
     неверно для B3/B4 (они работают на a124_pool = A1+A2+A4, реальный фильтр,
     не informational). Разделено по факту.
  2. B2 sub-basket: B2C2 (mitigation... нет, ob_liq) визуально помечен как
     research/не в basket (пунктирная рамка + красная пометка) — с 2026-07-24
     b2_hit = ЧИСТЫЙ b2c1, b2c2 больше не входит в union (раньше диаграмма могла
     считать их объединением).
  3. B4 sub-basket: было 2 под-условия (B4C1 HMA-78 SWEEP, B4C2 HMA-200 SWEEP) —
     стало 6 (B4C1..B4C6), все на механике FULL_DISP (не SWEEP — механика
     заменена после MA-family research, см. lib/fractal12h/b4_hma.py докстринг).
     Геометрия строки уже была рассчитана на 6 слотов (как у B2/B3), просто не
     использовалась целиком.

B6/B7/B8/B9/Basket-result — рендерятся как раньше (не менялись в этой сессии).

Layers: Header -> A-cascade -> B1 -> B2 -> B3 -> B4 -> B5 -> B6(planned) ->
        B7(planned) -> B8 -> B9 -> Basket result.

Usage:
    python gen_structure.py --symbol BTC --start 2020-01-01 --end 2026-07-25
Output:
    G:\\ASVK\\Export_12h\\структура_{SYMBOL}.png
"""
import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(pathlib.Path("/mnt/g/ASVK/lib/fractal12h")))
from common import DATA_OUT  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--symbol", default="BTC")
_ap.add_argument("--start", default="2020-01-01")
_ap.add_argument("--end", default="2026-07-25")
_ap.add_argument("--out-suffix", default="")
_args = _ap.parse_args()
SYMBOL, START, END = _args.symbol, _args.start, _args.end


def _stats(mask, df):
    n = int(mask.sum())
    cm = mask & df["confirmable"]
    nc = int(cm.sum())
    k = int(df.loc[cm, "confirmed"].sum())
    return n, (100.0 * k / nc if nc else 0.0), k, nc


def _load(block):
    p = DATA_OUT / f"{block}_hits_{SYMBOL}_{START}_{END}.parquet"
    if not p.exists():
        raise SystemExit(f"нет файла: {p}\nсначала прогони {block} для {SYMBOL}")
    return pd.read_parquet(p)


_a = pd.read_parquet(DATA_OUT / f"a_candidates_{SYMBOL}_{START}_{END}.parquet")
S = {}
for _stage, _key in [("a1_pre_w", "A1"), ("a2_indep", "A2"),
                     ("a3_indep", "A3"), ("a4_indep", "A4")]:
    S[_key] = _stats(_a[_stage], _a)
S["BASE"] = S["A1"]

for _b, _subs in [("b1", 4), ("b2", 2), ("b3", 1), ("b4", 6),
                  ("b5", 1), ("b8", 1), ("b9", 4)]:
    _df = _load(_b)
    S[_b.upper()] = _stats(_df[f"{_b}_hit"], _df)
    for _i in range(1, _subs + 1):
        S[f"{_b.upper()}C{_i}"] = _stats(_df[f"{_b}c{_i}"], _df)

_bk = _load("basket")
S["BASKET"] = _stats(_bk["basket_hit"], _bk)
_END_LABEL = (pd.Timestamp(END.replace("T", " "), tz="UTC")
              .tz_convert("Europe/Moscow").strftime("%Y-%m-%d %H:%M MSK"))


def N(key):
    return f"{S[key][0]:,}".replace(",", " ")


def WR(key):
    return f"{S[key][1]:.2f}"


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

fig = plt.figure(figsize=(18, 28), dpi=130)
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(-14, 160)
ax.set_aspect("equal")
ax.axis("off")

C_DATA = "#2C3E50"
C_CASC = "#7F8C8D"
C_BASE = "#16A085"
C_PREC = "#27AE60"
C_RECL = "#E67E22"
C_HEAD = "#34495E"
C_HIGH = "#E74C3C"
C_BG = "#ECF0F1"
C_ARROW = "#7F8C8D"
C_NEW = "#FFF5E6"


def box(x, y, w, h, text, color, text_color="white", fontsize=11, weight="normal", title=None, title_color="white"):
    bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.5",
                         linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.95)
    ax.add_patch(bb)
    if title:
        ax.text(x + w/2, y + h - 0.9, title, ha="center", va="top",
                fontsize=fontsize+1, fontweight="bold", color=title_color)
        ax.text(x + w/2, y + h/2 - 0.7, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight)


def boxw(x, y, w, h, text, color="white", border=C_HEAD, text_color="#2C3E50", fontsize=10,
         weight="normal", linestyle="solid"):
    bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.4",
                         linewidth=1.2, edgecolor=border, facecolor=color, alpha=0.95,
                         linestyle=linestyle)
    ax.add_patch(bb)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=weight)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


def label(x, y, text, fontsize=10, color="#2C3E50", weight="normal", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color, fontweight=weight)


def sub_block(dx, w, row_y, row_h, color, sname, badge_txt, params, n_val, wr):
    """Один D-блок (BxCy): рамка + имя + badge-плашка + params (со списком A-фильтров) + n/WR."""
    boxw(dx, row_y, w, row_h, "", color="white", border=color, fontsize=9)
    ax.text(dx + w/2, row_y + row_h - 1.3, sname,
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
    bw = w * 0.85
    bh = 1.1
    bx = dx + (w - bw) / 2
    by = row_y + row_h - 3.3
    bb = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.05,rounding_size=0.2",
                        linewidth=0, facecolor=color, alpha=0.9)
    ax.add_patch(bb)
    ax.text(dx + w/2, by + bh/2, badge_txt,
            ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    ax.text(dx + w/2, row_y + row_h - 4.8, params,
            ha="center", va="center", fontsize=7, color="#34495E",
            fontweight="bold", family="monospace")
    ax.text(dx + w/2, row_y + 2.7, f"n = {n_val}",
            ha="center", va="center", fontsize=8.5, color="#34495E", fontweight="bold")
    ax.text(dx + w/2, row_y + 1.2, f"P(W) {wr}%",
            ha="center", va="center", fontsize=9.5, color=color, fontweight="bold")


# ─── TITLE ─────────────────────────────────────────────────────
ax.text(50, 158, f"Прогноз формирования 12h фрактала · {SYMBOL}",
        ha="center", va="center", fontsize=22, fontweight="bold", color=C_HEAD)
ax.text(50, 155,
        f"{START} → {_END_LABEL}   ·   A1-anchor pool: {N('A1')}",
        ha="center", va="center", fontsize=10, color=C_CASC)
ax.text(50, 153,
        "B1/B2/B5/B9 — чистый A1 (A2/A3/A4 informational only)   ·   "
        "B3/B4 — A1+A2+A4 без A3 (a124_pool, реальный фильтр)   ·   "
        "B8 — A1+A2 (a12, уникальный домен)",
        ha="center", va="center", fontsize=8.5, color=C_CASC, style="italic")

# ─── LAYER 2: Cascade A1-A4 ──────────────────────────────────
y = 136
container = FancyBboxPatch((6, y), 88, 16, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=1.5, edgecolor=C_CASC, facecolor="#F8F9FA", alpha=0.5)
ax.add_patch(container)
ax.text(50, y + 14.6, "A · Cascade  A1 → A2 → A3 → A4   ·   BASELINE = A1",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color=C_CASC)

cascade_data = [
    (8,  y + 2, 14.5, 10.5, "A1 · Pre-W",    "3-свечный\nлокальный\nэкстремум",      f"n = {N('A1')}\nWR {WR('A1')}%"),
    (24, y + 2, 14.5, 10.5, "A2 · ext_5",    "5 свечей левее\nменьший\nэкстремум",    f"n = {N('A2')}\nWR {WR('A2')}%"),
    (40, y + 2, 14.5, 10.5, "A3 · color",    "смена цвета i-1, i\nили\n3 подряд однонапр.\n(без доджей)", f"n = {N('A3')}\nWR {WR('A3')}%"),
    (56, y + 2, 14.5, 10.5, "A4 · body+wick", "убирает\nпризнаки\nмарубозу",            f"n = {N('A4')}\nWR {WR('A4')}%"),
    (72, y + 2, 20,   10.5, "BASELINE",      "",                                       f"n = {N('BASE')}\nWR {WR('BASE')}%"),
]

for i, (x, yy, w, h, *_rest) in enumerate(cascade_data):
    is_baseline = (i == 4)
    boxw(x, yy, w, h, "", color=C_BASE if is_baseline else "white",
         border=C_BASE if is_baseline else C_CASC, fontsize=9)

for i, (x, yy, w, h, title, body, count) in enumerate(cascade_data):
    is_baseline = (i == 4)
    cx = x + w / 2
    if is_baseline:
        ax.text(cx, yy + h*0.72, title, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
        ax.text(cx, yy + h*0.28, count, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", linespacing=1.4)
    else:
        ax.text(cx, yy + h - 1.0, title, ha="center", va="top",
                fontsize=12, fontweight="bold", color="#2C3E50")
        if body:
            n_lines = body.count("\n") + 1
            body_size = 8.5 if n_lines <= 3 else 7.8
            ax.text(cx, yy + h*0.55, body, ha="center", va="center",
                    fontsize=body_size, color="#34495E", linespacing=1.25, style="italic")
        ax.text(cx, yy + 1.8, count, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_BASE, linespacing=1.4)

for i in range(4):
    sx = cascade_data[i][0] + cascade_data[i][2]
    arrow(sx, y + 7.25, cascade_data[i+1][0], y + 7.25, color=C_CASC, lw=1.5)

# ─── LAYER: B1 (FVG) ─────────────────────────────────────────
y = 120
container = FancyBboxPatch((3, y), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_HIGH, facecolor=C_NEW, alpha=0.6)
ax.add_patch(container)
row_y = y + 0.75
row_h = 11

c4_w = 9; c4_x = 5
boxw(c4_x, row_y, c4_w, row_h, "", color=C_HIGH, border=C_HIGH, fontsize=9)
ax.text(c4_x + c4_w/2, row_y + row_h - 2.5, "B1", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(c4_x + c4_w/2, row_y + 4, f"n = {N('B1')}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(c4_x + c4_w/2, row_y + 2, f"P(W) {WR('B1')}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

fvg_w = 7; fvg_x = c4_x + c4_w + 1
boxw(fvg_x, row_y, fvg_w, row_h, "", color="#FFF9F0", border=C_RECL, fontsize=9)
ax.text(fvg_x + fvg_w/2, row_y + row_h/2 + 2.5, "Fair", ha="center", va="center", fontsize=10, fontweight="bold", color=C_RECL)
ax.text(fvg_x + fvg_w/2, row_y + row_h/2 + 0.7, "Value", ha="center", va="center", fontsize=10, fontweight="bold", color=C_RECL)
ax.text(fvg_x + fvg_w/2, row_y + row_h/2 - 1.1, "Gap", ha="center", va="center", fontsize=10, fontweight="bold", color=C_RECL)
ax.text(fvg_x + fvg_w/2, row_y + row_h/2 - 3.2, "inefficiency class", ha="center", va="center", fontsize=6, color=C_RECL, style="italic")

d_blocks = [
    ("B1C1", "strict sweep", "S100/WIDE\nA1", S["B1C1"][0], round(S["B1C1"][1], 2)),
    ("B1C2", "strict sweep", "S50/AGE-W\nA1", S["B1C2"][0], round(S["B1C2"][1], 2)),
    ("B1C3", "strict sweep", "S70/AGE-W\nA1", S["B1C3"][0], round(S["B1C3"][1], 2)),
    ("B1C4", "strict sweep", "S50/HTF-W\nA1", S["B1C4"][0], round(S["B1C4"][1], 2)),
]
d_start = fvg_x + fvg_w + 1.5
d_w = (96 - d_start - 5 * 0.4) / 6
for i, (sname, badge_txt, params, n_val, wr) in enumerate(d_blocks):
    dx = d_start + i * (d_w + 0.4)
    sub_block(dx, d_w, row_y, row_h, C_PREC, sname, badge_txt, params, n_val, wr)

# ─── LAYER: B2 (Order Block) — только B2C1 (B2C2 убран из схемы 2026-07-25) ──
y_b2 = 105
container = FancyBboxPatch((3, y_b2), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_HIGH, facecolor=C_NEW, alpha=0.6)
ax.add_patch(container)
row_y = y_b2 + 0.75
row_h = 11

b2_w = 9; b2_x = 5
boxw(b2_x, row_y, b2_w, row_h, "", color=C_HIGH, border=C_HIGH, fontsize=9)
ax.text(b2_x + b2_w/2, row_y + row_h - 2.5, "B2", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b2_x + b2_w/2, row_y + 4, f"n = {N('B2')}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b2_x + b2_w/2, row_y + 2, f"P(W) {WR('B2')}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

C_PURPLE = "#8E44AD"
ob_w = 7; ob_x = b2_x + b2_w + 1
boxw(ob_x, row_y, ob_w, row_h, "", color="#F5EEF8", border=C_PURPLE, fontsize=9)
ax.text(ob_x + ob_w/2, row_y + row_h/2 + 2, "Order", ha="center", va="center", fontsize=11, fontweight="bold", color=C_PURPLE)
ax.text(ob_x + ob_w/2, row_y + row_h/2 - 0.3, "Block", ha="center", va="center", fontsize=11, fontweight="bold", color=C_PURPLE)
ax.text(ob_x + ob_w/2, row_y + row_h/2 - 3, "block class", ha="center", va="center", fontsize=6.5, color=C_PURPLE, style="italic")

d_start_b2 = ob_x + ob_w + 1.5
d_w_b2 = (96 - d_start_b2 - 5 * 0.4) / 6
color_active = C_PREC

sub_block(d_start_b2, d_w_b2, row_y, row_h, color_active, "B2C1", "FIRST 50%-sweep",
          "OB \u00b7 multi-TF\nA1", S["B2C1"][0], round(S["B2C1"][1], 2))

# ─── LAYER: B3 (Fractal Liquidity) ────────────────────────────
y_b3 = 90
container = FancyBboxPatch((3, y_b3), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_HIGH, facecolor=C_NEW, alpha=0.6)
ax.add_patch(container)
row_y = y_b3 + 0.75
row_h = 11

b3_w = 9; b3_x = 5
boxw(b3_x, row_y, b3_w, row_h, "", color=C_HIGH, border=C_HIGH, fontsize=9)
ax.text(b3_x + b3_w/2, row_y + row_h - 2.5, "B3", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b3_x + b3_w/2, row_y + 4, f"n = {N('B3')}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b3_x + b3_w/2, row_y + 2, f"P(W) {WR('B3')}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

C_TEAL = "#1ABC9C"
fl_w = 7; fl_x = b3_x + b3_w + 1
boxw(fl_x, row_y, fl_w, row_h, "", color="#E8F8F5", border=C_TEAL, fontsize=9)
ax.text(fl_x + fl_w/2, row_y + row_h/2 + 2, "Fractal", ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEAL)
ax.text(fl_x + fl_w/2, row_y + row_h/2 - 0.3, "Liquidity", ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEAL)
ax.text(fl_x + fl_w/2, row_y + row_h/2 - 3, "liquidity class", ha="center", va="center", fontsize=6.5, color=C_TEAL, style="italic")

d_start_b3 = fl_x + fl_w + 1.5
d_w_b3 = (96 - d_start_b3 - 5 * 0.4) / 6
sub_block(d_start_b3, d_w_b3, row_y, row_h, color_active, "B3C1", "maxV sweep", "maxV(i-1)\nA124",
          N("B3C1"), WR("B3C1"))

# ─── LAYER: B4 (MA-family) — B4C1..B4C6, все FULL_DISP ────────
y_b4 = 75
container = FancyBboxPatch((3, y_b4), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_HIGH, facecolor=C_NEW, alpha=0.6)
ax.add_patch(container)
row_y = y_b4 + 0.75
row_h = 11

b4_w = 9; b4_x = 5
boxw(b4_x, row_y, b4_w, row_h, "", color=C_HIGH, border=C_HIGH, fontsize=9)
ax.text(b4_x + b4_w/2, row_y + row_h - 2.5, "B4", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b4_x + b4_w/2, row_y + 4, f"n = {N('B4')}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b4_x + b4_w/2, row_y + 2, f"P(W) {WR('B4')}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

C_NAVY = "#2874A6"
hma_w = 7; hma_x = b4_x + b4_w + 1
boxw(hma_x, row_y, hma_w, row_h, "", color="#EBF5FB", border=C_NAVY, fontsize=9)
ax.text(hma_x + hma_w/2, row_y + row_h/2 + 2, "Sliding", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NAVY)
ax.text(hma_x + hma_w/2, row_y + row_h/2 - 0.3, "MA", ha="center", va="center", fontsize=11, fontweight="bold", color=C_NAVY)
ax.text(hma_x + hma_w/2, row_y + row_h/2 - 3, "trend-line class", ha="center", va="center", fontsize=6.5, color=C_NAVY, style="italic")

d_start_b4 = hma_x + hma_w + 1.5
d_w_b4 = (96 - d_start_b4 - 5 * 0.4) / 6

b4_subs = [
    ("B4C1", "HMA78\n12h\u222aD\u00b7A124"),
    ("B4C2", "HMA200\nD\u00b7A124"),
    ("B4C3", "THMA9\n12h\u00b7A124"),
    ("B4C4", "WMA50\nD·A124"),
    ("B4C5", "THMA9\nD·A124"),
    ("B4C6", "EHMA20\nD·A124"),
]
for i, (sname, params) in enumerate(b4_subs):
    dx = d_start_b4 + i * (d_w_b4 + 0.4)
    key = sname
    sub_block(dx, d_w_b4, row_y, row_h, color_active, sname, "FULL_DISP", params,
              S[key][0], round(S[key][1], 2))

# ─── LAYER: B5 (VWAP) ─────────────────────────────────────────
y_b5 = 60
container = FancyBboxPatch((3, y_b5), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_HIGH, facecolor=C_NEW, alpha=0.6)
ax.add_patch(container)
row_y = y_b5 + 0.75
row_h = 11

b5_w = 9; b5_x = 5
boxw(b5_x, row_y, b5_w, row_h, "", color=C_HIGH, border=C_HIGH, fontsize=9)
ax.text(b5_x + b5_w/2, row_y + row_h - 2.5, "B5", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b5_x + b5_w/2, row_y + 4, f"n = {N('B5')}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b5_x + b5_w/2, row_y + 2, f"P(W) {WR('B5')}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

C_AMBER = "#B7950B"
vwap_w = 7; vwap_x = b5_x + b5_w + 1
boxw(vwap_x, row_y, vwap_w, row_h, "", color="#FEF5E7", border=C_AMBER, fontsize=9)
ax.text(vwap_x + vwap_w/2, row_y + row_h/2 + 1.5, "VWAP", ha="center", va="center", fontsize=13, fontweight="bold", color=C_AMBER)
ax.text(vwap_x + vwap_w/2, row_y + row_h/2 - 1.5, "anchored", ha="center", va="center", fontsize=8, color=C_AMBER, style="italic")
ax.text(vwap_x + vwap_w/2, row_y + row_h/2 - 3.2, "volume MA", ha="center", va="center", fontsize=6.5, color=C_AMBER, style="italic")

d_start_b5 = vwap_x + vwap_w + 1.5
d_w_b5 = (96 - d_start_b5 - 5 * 0.4) / 6
sub_block(d_start_b5, d_w_b5, row_y, row_h, color_active, "B5C1", "FULL_DISP",
          "\u22651 W-aligned\nVWAP \u00b7 A1", N("B5C1"), WR("B5C1"))

# ─── LAYER: B6 RSI (planned) ───────────────────────────────────
y_b6 = 45
C_PLAN = "#A9A9A9"
container = FancyBboxPatch((3, y_b6), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_PLAN, facecolor="#FAFAFA", alpha=0.5, linestyle="dashed")
ax.add_patch(container)
row_y = y_b6 + 0.75
row_h = 11
b6_w = 9; b6_x = 5
boxw(b6_x, row_y, b6_w, row_h, "", color=C_PLAN, border=C_PLAN, fontsize=9)
ax.text(b6_x + b6_w/2, row_y + row_h - 2.5, "B6", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b6_x + b6_w/2, row_y + 4, "n = \u2014", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b6_x + b6_w/2, row_y + 2, "P(W) \u2014", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
C_CRIMSON = "#C0392B"
rsi_x = b6_x + b6_w + 1; rsi_w = 7
bb = FancyBboxPatch((rsi_x, row_y), rsi_w, row_h, boxstyle="round,pad=0.1,rounding_size=0.4",
                    linewidth=1.5, edgecolor=C_CRIMSON, facecolor="#FADBD8", alpha=0.85, linestyle="dashed")
ax.add_patch(bb)
ax.text(rsi_x + rsi_w/2, row_y + row_h/2 + 1.5, "RSI", ha="center", va="center", fontsize=14, fontweight="bold", color=C_CRIMSON)
ax.text(rsi_x + rsi_w/2, row_y + row_h/2 - 1.5, "Relative", ha="center", va="center", fontsize=7, color=C_CRIMSON, style="italic")
ax.text(rsi_x + rsi_w/2, row_y + row_h/2 - 3, "Strength Idx", ha="center", va="center", fontsize=6.5, color=C_CRIMSON, style="italic")
right_x = rsi_x + rsi_w + 1.5
right_w = 96 - right_x
ax.text(right_x + right_w/2, row_y + row_h/2 + 1.5, "no B6Cx implemented yet",
        ha="center", va="center", fontsize=11, color="#7F8C8D", style="italic", fontweight="bold")
ax.text(right_x + right_w/2, row_y + row_h/2 - 1, "candidates: overbought/oversold · divergence · multi-TF · StochRSI · cross/breakout",
        ha="center", va="center", fontsize=7.5, color="#95A5A6", style="italic")
ax.text(right_x + right_w/2, row_y + row_h/2 - 3.3, "B6 RSI — planned",
        ha="center", va="center", fontsize=6.5, color="#BDC3C7", family="monospace")

# ─── LAYER: B7 MoneyHands (planned) ────────────────────────────
y_b7 = 30
container = FancyBboxPatch((3, y_b7), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_PLAN, facecolor="#FAFAFA", alpha=0.5, linestyle="dashed")
ax.add_patch(container)
row_y = y_b7 + 0.75
b7_w = 9; b7_x = 5
boxw(b7_x, row_y, b7_w, row_h, "", color=C_PLAN, border=C_PLAN, fontsize=9)
ax.text(b7_x + b7_w/2, row_y + row_h - 2.5, "B7", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b7_x + b7_w/2, row_y + 4, "n = \u2014", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b7_x + b7_w/2, row_y + 2, "P(W) \u2014", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
C_OLIVE = "#7D6608"
mh_x = b7_x + b7_w + 1; mh_w = 7
bb = FancyBboxPatch((mh_x, row_y), mh_w, row_h, boxstyle="round,pad=0.1,rounding_size=0.4",
                    linewidth=1.5, edgecolor=C_OLIVE, facecolor="#FCF3CF", alpha=0.85, linestyle="dashed")
ax.add_patch(bb)
ax.text(mh_x + mh_w/2, row_y + row_h/2 + 1.5, "Money", ha="center", va="center", fontsize=11, fontweight="bold", color=C_OLIVE)
ax.text(mh_x + mh_w/2, row_y + row_h/2 - 1.2, "Hands", ha="center", va="center", fontsize=11, fontweight="bold", color=C_OLIVE)
ax.text(mh_x + mh_w/2, row_y + row_h/2 - 3.3, "smart money", ha="center", va="center", fontsize=6, color=C_OLIVE, style="italic")
right_x = mh_x + mh_w + 1.5
right_w = 96 - right_x
ax.text(right_x + right_w/2, row_y + row_h/2 + 1.5, "no B7Cx implemented yet",
        ha="center", va="center", fontsize=11, color="#7F8C8D", style="italic", fontweight="bold")
ax.text(right_x + right_w/2, row_y + row_h/2 - 1, "candidate B7C1: pivot-money-hands LONG-cascade rule (bear + cascade \u2264 1h \u2192 62.9%)",
        ha="center", va="center", fontsize=7.5, color="#95A5A6", style="italic")
ax.text(right_x + right_w/2, row_y + row_h/2 - 3.3, "B7 Money Hands — planned",
        ha="center", va="center", fontsize=6.5, color="#BDC3C7", family="monospace")

# ─── LAYER: B8 Power Zone ───────────────────────────────────────
y_b8 = 15
container = FancyBboxPatch((3, y_b8), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_HIGH, facecolor=C_NEW, alpha=0.6)
ax.add_patch(container)
row_y = y_b8 + 0.75
row_h = 11
b8_w = 9; b8_x = 5
boxw(b8_x, row_y, b8_w, row_h, "", color=C_HIGH, border=C_HIGH, fontsize=9)
ax.text(b8_x + b8_w/2, row_y + row_h - 2.5, "B8", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b8_x + b8_w/2, row_y + 4, f"n = {N('B8')}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b8_x + b8_w/2, row_y + 2, f"P(W) {WR('B8')}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
C_POWER = "#922B21"
pz_w = 7; pz_x = b8_x + b8_w + 1
boxw(pz_x, row_y, pz_w, row_h, "", color="#F2D7D5", border=C_POWER, fontsize=9)
ax.text(pz_x + pz_w/2, row_y + row_h/2 + 1.5, "Power", ha="center", va="center", fontsize=12, fontweight="bold", color=C_POWER)
ax.text(pz_x + pz_w/2, row_y + row_h/2 - 1.5, "Zone", ha="center", va="center", fontsize=12, fontweight="bold", color=C_POWER)
ax.text(pz_x + pz_w/2, row_y + row_h/2 - 3.5, "force extreme", ha="center", va="center", fontsize=6, color=C_POWER, style="italic")
d_start_b8 = pz_x + pz_w + 1.5
d_w_b8 = (96 - d_start_b8 - 5 * 0.4) / 6
sub_block(d_start_b8, d_w_b8, row_y, row_h, color_active, "B8C1", "reverse force",
          "divergence (\u222a3)\nA12", N("B8C1"), WR("B8C1"))

# ─── LAYER: B9 Others ───────────────────────────────────────────
y_b9 = 0
container = FancyBboxPatch((3, y_b9), 94, 12.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                             linewidth=2, edgecolor=C_HIGH, facecolor=C_NEW, alpha=0.6)
ax.add_patch(container)
row_y = y_b9 + 0.75
b9_w = 9; b9_x = 5
boxw(b9_x, row_y, b9_w, row_h, "", color=C_HIGH, border=C_HIGH, fontsize=9)
ax.text(b9_x + b9_w/2, row_y + row_h - 2.5, "B9", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
ax.text(b9_x + b9_w/2, row_y + 4, f"n = {N('B9')}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.text(b9_x + b9_w/2, row_y + 2, f"P(W) {WR('B9')}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
C_SLATE = "#515A5A"
ot_w = 7; ot_x = b9_x + b9_w + 1
boxw(ot_x, row_y, ot_w, row_h, "", color="#EAEDED", border=C_SLATE, fontsize=9)
ax.text(ot_x + ot_w/2, row_y + row_h/2 + 1.5, "Others", ha="center", va="center", fontsize=13, fontweight="bold", color=C_SLATE)
ax.text(ot_x + ot_w/2, row_y + row_h/2 - 1.2, "catch-all", ha="center", va="center", fontsize=8, color=C_SLATE, style="italic")
ax.text(ot_x + ot_w/2, row_y + row_h/2 - 3.3, "misc primitives", ha="center", va="center", fontsize=6, color=C_SLATE, style="italic")
d_start_b9 = ot_x + ot_w + 1.5
d_w_b9 = (96 - d_start_b9 - 5 * 0.4) / 6
b9_subs = [
    ("B9C1", "P11+overlay", "close_m+rng\u22651.2\nA1"),
    ("B9C2", "maxV depth", "dist\u22650.7\u00d7ATR\nA1"),
    ("B9C3", "momentum bar", "body/rng\u22650.7\nA1"),
    ("B9C4", "climax bar", "b\u22650.5+pos+rng\nA1"),
]
for i, (sname, badge_txt, params) in enumerate(b9_subs):
    dx = d_start_b9 + i * (d_w_b9 + 0.4)
    sub_block(dx, d_w_b9, row_y, row_h, color_active, sname, badge_txt, params,
              S[sname][0], round(S[sname][1], 2))

# ─── FINAL RESULT: Basket union ────────────────────────────────
y = -13
result_container = FancyBboxPatch((6, y), 88, 11.0, boxstyle="round,pad=0.3,rounding_size=0.5",
                                    linewidth=2.5, edgecolor=C_HIGH, facecolor=C_HIGH, alpha=0.95)
ax.add_patch(result_container)
ax.text(50, y + 9.3, "ИТОГ — Basket B1 ∪ B2 ∪ … ∪ B9",
        ha="center", va="center", fontsize=12, fontweight="bold", color="white")
ax.text(50, y + 6.8,
        f"n = {N('BASKET')}    ·    conf = {S['BASKET'][2]}    ·    P(W) = {WR('BASKET')}%    ·    Δ = {S['BASKET'][1]-S['BASE'][1]:+.2f} pp",
        ha="center", va="center", fontsize=13, fontweight="bold", color="white")
ax.text(50, y + 4.3,
        f"Всего подтверждённых 12h-фракталов (A1 baseline): {S['BASE'][2]:,}".replace(",", " ") +
        f"  из {N('BASE')} A1-пивотов",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ffd9d3")
ax.text(50, y + 1.8,
        f"selectivity {S['BASKET'][0]}/{S['BASE'][3]} ≈ {100*S['BASKET'][0]/S['BASE'][3]:.0f}%   ·   "
        f"B6/B7 planned, не входят в union",
        ha="center", va="center", fontsize=8, color="white", style="italic")

out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
out = out_dir / f"структура_{SYMBOL}{_args.out_suffix}.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved: {out}")
plt.close()
