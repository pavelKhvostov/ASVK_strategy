"""B1C4 rule — presentation-quality schematic with canon notes.

Правило: strict sweep FVG, S50 (половинный пробой) + HTF (зона рождена НЕ на 12h,
а на 1d/2d/3d/1w) + WIDE (width/ATR14 >= 0.7). БЕЗ требования AGE — HTF-зоны
изначально "медленные" (родились на старшем ТФ), отдельный age-фильтр избыточен.

Source of truth (G:\\ASVK\\lib\\fractal12h\\b1_fvg.py):
    B1C4 = eval_classic(zone_events, pen_min=50, ftype="HTF_WIDE", ...)
    zone_passes(..., "HTF_WIDE") = (zone.tf in HTF_TFS) and wide
    HTF_TFS = ("1d", "2d", "3d", "1w")   WIDE_MULT = 0.7

Стиль 1-в-1 повторяет B1C1.png/B1C2.png/B1C3.png (Desktop/export_12h).

Refs:
  G:\\ASVK\\lib\\fractal12h\\b1_fvg.py           — реализация (eval_classic S50/HTF_WIDE)
  scripts/фрактал-12h/эталон/B1C4_strict_S50_htf_wide.py (WSL канон)

Output:
  Desktop/export_12h/B1C4.png
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
HTF_C    = "#b8860b"
S50_C    = "#7a2ba0"


def draw_candle(ax, x, o, h, l, c, w=0.55, edge="#111",
                fill_up="white", fill_dn="#222", lw=0.9, alpha=1.0):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=2, alpha=alpha)
    color = fill_up if c >= o else fill_dn
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.03),
                            facecolor=color, edgecolor=edge, lw=lw,
                            zorder=3, alpha=alpha))


def _draw_born_marker(ax, x, y_bot, y_top, color):
    ax.plot([x, x], [y_bot, y_top], ls=(0, (4, 3)), color=color, lw=1.2, alpha=0.75)
    ax.text(x, y_top + 0.3, "FVG born\n(t = 0)", fontsize=8, color=color,
            ha="center", fontweight="bold")


def _draw_htf_badge(ax, x, y, color):
    ax.add_patch(FancyBboxPatch((x - 0.75, y - 0.55), 2.5, 1.1,
                                 boxstyle="round,pad=0.08",
                                 facecolor="#fff6df", edgecolor=color, linewidth=1.3))
    ax.text(x + 0.5, y, "born TF ∈ {1d, 2d, 3d, 1w}\n(НЕ 12h)", fontsize=8.2,
            color=color, ha="center", va="center", fontweight="bold")


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
    ax.text(0.5, 0.62, "B1C4 — strict sweep · S50 · HTF · WIDE",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.20,
            "Правило поиска FVG-sweep для 12h-фрактала  ·  "
            "первое qualifying event  ·  lifecycle L0  ·  зона рождена на старшем ТФ",
            ha="center", va="center", fontsize=11.5, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup  ·  A1-pivot LOW → half-sweep FVG-SHORT (HTF resistance)",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 100.0, 106.0
    zone_50 = zone_lo + 0.5 * (zone_hi - zone_lo)
    x_born = 0.9
    x_fire = 9.0

    ax.add_patch(Rectangle((x_born + 0.3, zone_lo), (x_fire + 0.2) - (x_born + 0.3),
                           zone_hi - zone_lo,
                           facecolor=RED_FILL, edgecolor=RED_EDGE,
                           linewidth=1.4, alpha=0.85, zorder=1))
    ax.text(5.0, zone_hi - 0.55, "FVG-SHORT zone  ·  original (drop area)",
            fontsize=9.5, color="#800000", fontweight="bold", ha="center")
    ax.text(5.0, zone_lo + 0.35, "WIDE ✓ (width/ATR14(12h) ≥ 0.7)",
            fontsize=8.5, color="#800000", ha="center", style="italic")

    ax.plot([x_born + 0.3, x_fire + 0.2], [zone_50, zone_50],
            ls=(0, (5, 2, 1, 2)), color=S50_C, lw=1.2, alpha=0.85)
    ax.text(x_fire + 0.35, zone_50, "50 %\n(S50 line)", fontsize=8,
            color=S50_C, va="center", fontweight="bold")

    ax.text(11.15, zone_hi, "zone_hi", fontsize=8.5, color=RED_EDGE,
            va="center", fontweight="bold")
    ax.text(11.15, zone_lo, "zone_lo", fontsize=8.5, color=RED_EDGE,
            va="center", fontweight="bold")

    _draw_born_marker(ax, x_born, zone_lo - 0.5, zone_hi + 0.2, RED_EDGE)
    _draw_htf_badge(ax, 2.6, 92.6, HTF_C)

    _draw_ellipsis(ax, x_born + 0.6, 4.2, zone_lo - 1.0)

    ax.plot(x_born + 0.4, 96.0, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_born + 0.4, 95.1, "A1-pivot LOW\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    for x, lo, hi, o, c in [
        (4.6, 98.7, 102.4, 99.5, 100.8),
        (5.3, 98.9, 101.5, 99.9, 100.5),
        (6.0, 99.2, 100.6, 100.0, 100.3),
        (6.7, 99.0, 100.4, 99.7, 100.1),
        (7.4, 99.0, 101.6, 99.9, 100.6),
        (8.1, 99.3, 100.5, 100.2, 100.2),
    ]:
        draw_candle(ax, x, o, hi, lo, c, w=0.42, edge="#666",
                    fill_up="#e7e7ea", fill_dn="#a9a9b0", lw=0.8, alpha=0.85)

    for i, (x, top) in enumerate([(4.6, 102.6), (7.4, 101.8)]):
        ax.plot(x, top + 0.15, marker="o", markersize=13, color="#f4e4a8",
                markeredgecolor="#a17d00", markeredgewidth=1.0, zorder=6)
        ax.text(x, top + 0.15, str(i + 1), fontsize=8, color="#7a5c00",
                fontweight="bold", ha="center", va="center", zorder=7)

    draw_candle(ax, x_fire, o=100.4, h=103.7, l=98.4, c=98.9,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 104.3, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 105.1, "FIRE", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate("wick пересекает линию 50 %\n(pen ≥ 50 %, S50) — половинный пробой",
                xy=(x_fire + 0.05, 103.6), xytext=(2.4, 108.5),
                fontsize=9.5, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4),
                ha="left", va="center")
    ax.annotate("close возвращается НИЖЕ original zone_lo\n→ co_far ✓  (rejection)",
                xy=(x_fire + 0.10, 98.9), xytext=(2.4, 95.0),
                fontsize=9.5, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4),
                ha="left", va="center")

    ax.set_xlim(-0.4, 12.0)
    ax.set_ylim(90.0, 111.0)
    ax.set_xlabel("12h bars →", fontsize=9.5)
    ax.set_ylabel("price", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel(ax) -> None:
    ax.set_title("LONG setup  ·  A1-pivot HIGH → half-sweep FVG-LONG (HTF support)",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 94.0, 100.0
    zone_50 = zone_hi - 0.5 * (zone_hi - zone_lo)
    x_born = 0.9
    x_fire = 9.0

    ax.add_patch(Rectangle((x_born + 0.3, zone_lo), (x_fire + 0.2) - (x_born + 0.3),
                           zone_hi - zone_lo,
                           facecolor=GRN_FILL, edgecolor=GRN_EDGE,
                           linewidth=1.4, alpha=0.85, zorder=1))
    ax.text(5.0, zone_lo + 0.35, "FVG-LONG zone  ·  original (drop area)",
            fontsize=9.5, color="#004400", fontweight="bold", ha="center")
    ax.text(5.0, zone_hi - 0.55, "WIDE ✓ (width/ATR14(12h) ≥ 0.7)",
            fontsize=8.5, color="#004400", ha="center", style="italic")

    ax.plot([x_born + 0.3, x_fire + 0.2], [zone_50, zone_50],
            ls=(0, (5, 2, 1, 2)), color=S50_C, lw=1.2, alpha=0.85)
    ax.text(x_fire + 0.35, zone_50, "50 %\n(S50 line)", fontsize=8,
            color=S50_C, va="center", fontweight="bold")

    ax.text(11.15, zone_hi, "zone_hi", fontsize=8.5, color=GRN_EDGE,
            va="center", fontweight="bold")
    ax.text(11.15, zone_lo, "zone_lo", fontsize=8.5, color=GRN_EDGE,
            va="center", fontweight="bold")

    _draw_born_marker(ax, x_born, zone_lo - 0.2, zone_hi + 0.5, GRN_EDGE)
    _draw_htf_badge(ax, 2.6, 111.0, HTF_C)

    _draw_ellipsis(ax, x_born + 0.6, 4.2, zone_hi + 1.2)

    ax.plot(x_born + 0.4, 108.5, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_born + 0.4, 109.4, "A1-pivot HIGH\n(12h fractal)", fontsize=8,
            color=WARN, ha="center", fontweight="bold")

    for x, lo, hi, o, c in [
        (4.6, 97.6, 101.3, 100.5, 99.2),
        (5.3, 98.5, 101.1, 100.1, 99.5),
        (6.0, 99.4, 100.8, 99.9, 99.7),
        (6.7, 99.6, 101.0, 99.9, 100.3),
        (7.4, 98.4, 101.0, 99.9, 99.4),
        (8.1, 99.5, 100.7, 99.8, 99.8),
    ]:
        draw_candle(ax, x, o, hi, lo, c, w=0.42, edge="#666",
                    fill_up="#e7e7ea", fill_dn="#a9a9b0", lw=0.8, alpha=0.85)

    for i, (x, bot) in enumerate([(4.6, 97.4), (7.4, 98.2)]):
        ax.plot(x, bot - 0.15, marker="o", markersize=13, color="#f4e4a8",
                markeredgecolor="#a17d00", markeredgewidth=1.0, zorder=6)
        ax.text(x, bot - 0.15, str(i + 1), fontsize=8, color="#7a5c00",
                fontweight="bold", ha="center", va="center", zorder=7)

    draw_candle(ax, x_fire, o=99.6, h=101.6, l=96.3, c=105.1 if False else 101.1,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 95.7, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 94.9, "FIRE", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate("wick пересекает линию 50 %\n(pen ≥ 50 %, S50) — половинный пробой",
                xy=(x_fire + 0.05, 96.4), xytext=(2.4, 91.5),
                fontsize=9.5, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4),
                ha="left", va="center")
    ax.annotate("close возвращается ВЫШЕ original zone_hi\n→ co_far ✓  (rejection)",
                xy=(x_fire + 0.10, 101.1), xytext=(2.4, 104.6),
                fontsize=9.5, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4),
                ha="left", va="center")

    ax.set_xlim(-0.4, 12.0)
    ax.set_ylim(89.0, 113.5)
    ax.set_xlabel("12h bars →", fontsize=9.5)
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
        "(1) S50   — wick доходит ХОТЯ БЫ до середины оригинальной зоны:\n"
        "         pen = (min(wick_extreme, zone_far) - zone_near) / "
        "(zone_hi - zone_lo) * 100  >=  50 %\n\n"
        "(2) co_far — close возвращается за original near-side (rejection drop area):\n"
        "         SHORT:  cc < zone_lo        LONG:  cc > zone_hi\n\n"
        "(3) HTF   — зона рождена НЕ на 12h, а на старшем ТФ:\n"
        "         zone.tf ∈ {1d, 2d, 3d, 1w}\n\n"
        "(4) WIDE  — зона не «нитка» (ATR14 всегда считается на 12h-серии):\n"
        "         (zone_hi - zone_lo) / ATR14(12h)  >=  0.7\n\n"
        "Fire = (bar_idx, direction), матчится с A1-pivot 12h. Одна зона fires МАКС один раз.\n"
        "AGE отдельно НЕ проверяется — HTF-происхождение уже подразумевает выдержанность."
    )
    ax.text(0.04, 0.78, rule_text, fontsize=9.5, color="#222",
            ha="left", va="top", transform=ax.transAxes,
            family="DejaVu Sans Mono")


def _draw_notes_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#fff5f0", edgecolor="#c0392b",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Canon-заметки  ·  lifecycle L0",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "•  Место в семье B1: единственное условие, ортогональное по TF-происхождению зоны,\n"
        "   а не по глубине pen. B1C1=S100+WIDE, B1C2=S50+AGE+WIDE, B1C3=S70+AGE+WIDE —\n"
        "   все три на любом TF. B1C4 = S50+WIDE, но ТОЛЬКО зоны с D/2D/3D/W.\n\n"
        "•  Зона на 12h НЕ проходит, даже если widе и половинный пробой — критерий tf ∈ HTF_TFS\n"
        "   отсекает её независимо от остальных условий.\n\n"
        "•  ATR14 считается ВСЕГДА на 12h-серии (atr14_sma(h12,l12,c12)), а не на родном\n"
        "   таймфрейме зоны — так WIDE остаётся сравнимым между B1C1..B1C4.\n\n"
        "•  Загружаются ВСЕ FVG born из e12d (element='fvg' & kind='born'), TF-фильтр\n"
        "   применяется на этапе zone_passes, а не при загрузке зон.\n\n"
        "•  Прошлые wick fills НЕ уменьшают зону, НЕ retire её (canon lifecycle L0,\n"
        "   как и B1C1..B1C3).\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b1_fvg.py (eval_classic S50/HTF_WIDE)  ·  \n"
        "       эталон/B1C4_strict_S50_htf_wide.py"
    )
    ax.text(0.04, 0.78, notes_text, fontsize=9.3, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/c/Users/Вадим/Desktop/export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B1C4.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
