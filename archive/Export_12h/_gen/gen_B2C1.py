"""B2C1 rule — presentation-quality schematic with canon notes.

Правило: FIRST 50%-sweep zone block_orders (составной ордер-блок переменной длины,
см. lib/детекторы/block_orders.py), multi-TF {12h,1d,2d,3d,1w}, длина блока
L = last_bar_idx - preceding_idx + 1  <=  8 (canon-фильтр из meta). БЕЗ WIDE/AGE —
в отличие от семьи B1, здесь ровно один порог (mid = 50%), без вариаций pen.

Source of truth (G:\\ASVK\\lib\\fractal12h\\b2_ob.py):
    mid = (zone_lo + zone_hi) / 2
    SHORT: high[k] >= mid  AND  close[k] < zone_lo   (первое такое k = fs50)
    LONG:  low[k]  <= mid  AND  close[k] > zone_hi
    Домен: a_cand[a1_pre_w]  — чистый A1-пул, как у B1/B9 (A2/A3/A4 informational only,
    B2C1 == b2_hit, B2C2 (ob_liq) не перенесён).

Zone block_orders (canon 2026-06-15, lib/детекторы/block_orders.py):
    preceding (противоположный цвет) + initial run (N same-color) + counter run (M,
    пересекающий block_open) → LONG zone=(block.low, block.open) support
                                 SHORT zone=(block.open, block.high) resistance

Стиль 1-в-1 повторяет B1C1..B1C4.png (см. gen_B1C4.py как эталон вёрстки).

Refs:
  G:\\ASVK\\lib\\fractal12h\\b2_ob.py             — реализация (eval_b2c1, first_sweep50_idx)
  G:\\ASVK\\lib\\детекторы\\block_orders.py        — источник зон (variable N+M composite)
  scripts/фрактал-12h/b2_ob.py (WSL канон)

Output:
  G:\\ASVK\\Export_12h\\B2C1.png
"""
from __future__ import annotations

import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#333",
    "axes.linewidth": 0.8,
})

RED_FILL = "#ffd6d6"
RED_EDGE = "#b30000"
GRN_FILL = "#d4ecd4"
GRN_EDGE = "#006400"
FIRE_C   = "#111"
ACCENT   = "#0057b8"
WARN     = "#c0392b"
MID_C    = "#7a2ba0"
LEN_C    = "#b8860b"


def draw_candle(ax, x, o, h, l, c, w=0.55, edge="#111",
                fill_up="white", fill_dn="#222", lw=0.9, alpha=1.0):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=2, alpha=alpha)
    color = fill_up if c >= o else fill_dn
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.03),
                            facecolor=color, edgecolor=edge, lw=lw,
                            zorder=3, alpha=alpha))


def _draw_ellipsis(ax, x_start, x_end, y):
    for xf in [0.30, 0.50, 0.70]:
        ax.plot(x_start + (x_end - x_start) * xf, y,
                marker="o", markersize=3, color="#888", zorder=1)


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.55, 4, 2.2],
                           hspace=0.28, wspace=0.14,
                           left=0.045, right=0.985, top=0.965, bottom=0.045)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#1c1c2e", edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.62, "B2C1 — first 50%-sweep · Block Orders · L \u2264 8",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.20,
            "Правило поиска Block-Orders-sweep для 12h-фрактала  ·  "
            "первое qualifying event  ·  lifecycle L0 (зона живая вечно)",
            ha="center", va="center", fontsize=11.5, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_block(ax, x0, color_fill, color_edge):
    """Составной block_orders: preceding (1) + initial run (N=2) + counter run (M=2)."""
    # preceding — противоположный цвет (bull, если блок формирует SHORT resistance)
    draw_candle(ax, x0, 99.6, 100.9, 99.3, 100.6, w=0.5, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    # initial run (2 bear) — формируют resistance
    draw_candle(ax, x0 + 0.75, 100.4, 100.9, 99.0, 99.3, w=0.5, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    draw_candle(ax, x0 + 1.5, 99.9, 100.3, 98.2, 98.4, w=0.5, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    # counter run (2 bull, второй пересекает block_open)
    draw_candle(ax, x0 + 2.25, 98.5, 99.4, 98.3, 99.2, w=0.5, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    draw_candle(ax, x0 + 3.0, 99.2, 100.75, 99.1, 100.7, w=0.5, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup  ·  A1-pivot LOW \u2192 50%-sweep Block Order (resistance)",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 100.6, 105.5   # (block.open, block.high)
    mid = (zone_lo + zone_hi) / 2.0
    x_block0 = 1.0
    x_fire = 9.0

    ax.add_patch(Rectangle((x_block0, zone_lo), (x_fire + 0.2) - x_block0,
                           zone_hi - zone_lo,
                           facecolor=RED_FILL, edgecolor=RED_EDGE,
                           linewidth=1.4, alpha=0.85, zorder=1))
    ax.text(5.2, zone_hi - 0.5, "Block Order zone  ·  (block.open, block.high)  ·  resistance",
            fontsize=9.2, color="#800000", fontweight="bold", ha="center")
    ax.text(5.2, zone_lo + 0.35, "(TF \u2208 12h \u00b7 1d \u00b7 2d \u00b7 3d \u00b7 1w)",
            fontsize=8, color="#800000", ha="center", style="italic")

    ax.plot([x_block0, x_fire + 0.2], [mid, mid],
            ls=(0, (5, 2, 1, 2)), color=MID_C, lw=1.3, alpha=0.9)
    ax.text(x_fire + 0.35, mid, "mid = 50 %\n(zone_lo+zone_hi)/2", fontsize=8,
            color=MID_C, va="center", fontweight="bold")

    ax.text(11.15, zone_hi, "zone_hi", fontsize=8.5, color=RED_EDGE, va="center", fontweight="bold")
    ax.text(11.15, zone_lo, "zone_lo", fontsize=8.5, color=RED_EDGE, va="center", fontweight="bold")

    _draw_block(ax, x_block0, RED_FILL, RED_EDGE)
    ax.annotate("", xy=(x_block0 + 3.0, 97.4), xytext=(x_block0, 97.4),
                arrowprops=dict(arrowstyle="<->", color=LEN_C, lw=1.6))
    ax.text(x_block0 + 1.5, 96.7, "L = last_bar_idx \u2212 preceding_idx + 1  \u2264  8",
            fontsize=8.7, color=LEN_C, ha="center", fontweight="bold")

    _draw_ellipsis(ax, x_block0 + 3.6, 6.0, zone_lo + 1.5)

    ax.plot(x_block0 + 0.2, 94.5, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_block0 + 0.2, 93.6, "A1-pivot LOW\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    draw_candle(ax, x_fire, o=101.0, h=103.4, l=98.8, c=99.3,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 104.0, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 104.8, "FIRE", fontsize=9.5, fontweight="bold", color="#8a5a00", ha="center")

    ax.annotate("high доходит МИНИМУМ до mid\n(high \u2265 mid)",
                xy=(x_fire + 0.05, 103.3), xytext=(6.3, 107.2),
                fontsize=9.3, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4),
                ha="left", va="center")
    ax.annotate("close < zone_lo\n(полный rejection ниже block.open)",
                xy=(x_fire + 0.10, 99.3), xytext=(2.2, 95.2),
                fontsize=9.3, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4),
                ha="left", va="center")

    ax.set_xlim(-0.4, 12.0)
    ax.set_ylim(91.0, 109.5)
    ax.set_xlabel("12h bars \u2192", fontsize=9.5)
    ax.set_ylabel("price", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel(ax) -> None:
    ax.set_title("LONG setup  ·  A1-pivot HIGH \u2192 50%-sweep Block Order (support)",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 94.5, 99.4   # (block.low, block.open)
    mid = (zone_lo + zone_hi) / 2.0
    x_block0 = 1.0
    x_fire = 9.0

    ax.add_patch(Rectangle((x_block0, zone_lo), (x_fire + 0.2) - x_block0,
                           zone_hi - zone_lo,
                           facecolor=GRN_FILL, edgecolor=GRN_EDGE,
                           linewidth=1.4, alpha=0.85, zorder=1))
    ax.text(5.2, zone_lo + 0.4, "Block Order zone  ·  (block.low, block.open)  ·  support",
            fontsize=9.2, color="#004400", fontweight="bold", ha="center")
    ax.text(5.2, zone_hi - 0.4, "(TF \u2208 12h \u00b7 1d \u00b7 2d \u00b7 3d \u00b7 1w)",
            fontsize=8, color="#004400", ha="center", style="italic")

    ax.plot([x_block0, x_fire + 0.2], [mid, mid],
            ls=(0, (5, 2, 1, 2)), color=MID_C, lw=1.3, alpha=0.9)
    ax.text(x_fire + 0.35, mid, "mid = 50 %\n(zone_lo+zone_hi)/2", fontsize=8,
            color=MID_C, va="center", fontweight="bold")

    ax.text(11.15, zone_hi, "zone_hi", fontsize=8.5, color=GRN_EDGE, va="center", fontweight="bold")
    ax.text(11.15, zone_lo, "zone_lo", fontsize=8.5, color=GRN_EDGE, va="center", fontweight="bold")

    _draw_block(ax, x_block0, GRN_FILL, GRN_EDGE)
    ax.annotate("", xy=(x_block0 + 3.0, 101.6), xytext=(x_block0, 101.6),
                arrowprops=dict(arrowstyle="<->", color=LEN_C, lw=1.6))
    ax.text(x_block0 + 1.5, 102.3, "L = last_bar_idx \u2212 preceding_idx + 1  \u2264  8",
            fontsize=8.7, color=LEN_C, ha="center", fontweight="bold")

    ax.plot(x_block0 + 0.2, 106.5, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_block0 + 0.2, 107.4, "A1-pivot HIGH\n(12h fractal)", fontsize=8,
            color=WARN, ha="center", fontweight="bold")

    draw_candle(ax, x_fire, o=98.9, h=100.9, l=96.4, c=100.6,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 95.4, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 94.6, "FIRE", fontsize=9.5, fontweight="bold", color="#8a5a00", ha="center")

    ax.annotate("low доходит МИНИМУМ до mid\n(low \u2264 mid)",
                xy=(x_fire + 0.05, 96.5), xytext=(6.3, 91.6),
                fontsize=9.3, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4),
                ha="left", va="center")
    ax.annotate("close > zone_hi\n(полный rejection выше block.open)",
                xy=(x_fire + 0.10, 100.6), xytext=(2.2, 104.4),
                fontsize=9.3, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4),
                ha="left", va="center")

    ax.set_xlim(-0.4, 12.0)
    ax.set_ylim(90.0, 109.0)
    ax.set_xlabel("12h bars \u2192", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Формальное правило (все 3 условия, first-touch)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "(1) 50%-sweep — wick доходит ХОТЯ БЫ до середины зоны:\n"
        "         mid = (zone_lo + zone_hi) / 2\n"
        "         SHORT:  high[k] >= mid          LONG:  low[k] <= mid\n\n"
        "(2) rejection — close закрывается ЗА пределами зоны целиком:\n"
        "         SHORT:  close[k] < zone_lo      LONG:  close[k] > zone_hi\n\n"
        "(3) L <= 8 — длина составного блока (canon-фильтр по meta):\n"
        "         L = last_bar_idx - preceding_idx + 1  <=  8\n\n"
        "Zone: LONG=(block.low, block.open) support\n"
        "      SHORT=(block.open, block.high) resistance\n\n"
        "Fire = (bar_idx, direction), матчится с A1-pivot 12h. Первое qualifying k = fs50,\n"
        "одна зона fires МАКС один раз. БЕЗ WIDE/AGE фильтров (в отличие от B1)."
    )
    ax.text(0.04, 0.78, rule_text, fontsize=9.3, color="#222",
            ha="left", va="top", transform=ax.transAxes,
            family="DejaVu Sans Mono")


def _draw_notes_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#fff5f0", edgecolor="#c0392b",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Canon-заметки  \u00b7  lifecycle L0",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "\u2022  Источник зон \u2014 block_orders (lib/\u0434\u0435\u0442\u0435\u043a\u0442\u043e\u0440\u044b/block_orders.py), НЕ каноничный\n"
        "   2-свечной OB: preceding (противоположный цвет) + initial run (N same-color)\n"
        "   + counter run (M, пересекающий block_open) \u2014 переменная длина блока.\n\n"
        "\u2022  Домен пивота \u2014 ЧИСТЫЙ A1 (a_cand[a1_pre_w]), как у B1/B9. A2/A3/A4\n"
        "   informational only \u2014 не режут домен (см. b2_ob.py докстринг).\n\n"
        "\u2022  Единственный порог \u2014 mid = 50 % (в отличие от B1, где pen варьируется\n"
        "   S50/S70/S100). Нет WIDE, нет AGE \u2014 только L <= 8 и multi-TF фильтр.\n\n"
        "\u2022  B2C1 == b2_hit \u2014 B2C2 (ob_liq, зеркальный по liquidity) в ASVK\n"
        "   не перенесён из WSL-канона.\n\n"
        "\u2022  Прошлые wick fills НЕ уменьшают зону (canon lifecycle L0), как во всей серии.\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b2_ob.py (eval_b2c1, first_sweep50_idx)  \u00b7\n"
        "       G:\\ASVK\\lib\\\u0434\u0435\u0442\u0435\u043a\u0442\u043e\u0440\u044b\\block_orders.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=9.1, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B2C1.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
