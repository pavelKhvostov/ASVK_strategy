"""B9C4 rule — presentation-quality schematic + canon notes.

Правило: climax bar — экстремально широкий разворотный бар с close у самого края.
Мягче по body (>=0.5 против 0.7 у B9C3 momentum), но требует close В КВАРТИЛИ
дальнего от direction экстремума (close_pos >= 0.75) И огромный range
относительно волатильности (range/ATR14 >= 1.5) — то, чего momentum НЕ требует.

Source of truth (G:\\ASVK\\lib\\fractal12h\\b9_others.py):
    close_match = (c12[i] < o12[i]) if SHORT else (c12[i] > o12[i])
    close_pos   = (h12[i]-c12[i])/rng   if SHORT else (c12[i]-l12[i])/rng
    body_ratio  = |c12[i]-o12[i]| / rng
    range_atr   = rng / atr14[i]
    B9C4 = close_match AND body_ratio>=0.5 AND close_pos>=0.75 AND range_atr>=1.5

Стиль 1-в-1 повторяет B9C3.png (momentum-bar family) + доп. close-position bracket
и "ghost"-свеча среднего размера (ATR14) для наглядного сравнения масштаба.

Refs:
  G:\\ASVK\\lib\\fractal12h\\b9_others.py   — реализация условия (b9c4)
  G:\\ASVK\\lib\\maxv.py                   — ATR14(12h), level-1 shared indicator

Output:
  Desktop/export_12h/B9C4.png
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

RED_EDGE = "#b30000"
GRN_EDGE = "#006400"
FIRE_C   = "#111"
ACCENT   = "#0057b8"
WARN     = "#c0392b"
BODY_C   = "#c0392b"
WICK_C   = "#8e44ad"
POS_C    = "#0e7c61"
GHOST_C  = "#999"


def draw_candle(ax, x, o, h, l, c, w=0.55, edge="#111",
                fill_up="white", fill_dn="#222", lw=0.9, alpha=1.0):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=2, alpha=alpha)
    color = fill_up if c >= o else fill_dn
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.03),
                            facecolor=color, edgecolor=edge, lw=lw,
                            zorder=3, alpha=alpha))


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.65, 4, 2.4],
                           hspace=0.3, wspace=0.14,
                           left=0.045, right=0.985, top=0.965, bottom=0.045)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#1c1c2e", edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.70, "B9C4 — Climax bar · body ≥ 0.5 · close_pos ≥ 0.75 · range ≥ 1.5×ATR14",
            ha="center", va="center", fontsize=19.5, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.42, "A1 anchor  (pool: все Williams-confirmable pivots)  ·  "
                       "экстремальная версия B9C3 momentum",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#ffc857", transform=ax.transAxes)
    ax.text(0.5, 0.15,
            "мягче по телу, но требует close у самого края + аномальный размер бара  ·  "
            "causal (all data ≤ close bar i)",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup  ·  A1-pivot HIGH → close у LOW края + аномальный range",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    for x, lo, hi, o, c in [
        (1.4, 97.5, 101.0, 98.3, 100.2),
        (2.6, 98.8, 102.2, 100.8, 101.6),
    ]:
        draw_candle(ax, x, o, hi, lo, c, w=0.4, edge="#666",
                    fill_up="#e7e7ea", fill_dn="#a9a9b0", lw=0.8, alpha=0.75)
    ax.text(1.4, 96.3, "bar i-2", fontsize=8, color="#555", ha="center", style="italic")
    ax.text(2.6, 103.4, "bar i-1", fontsize=8, color="#555", ha="center", style="italic")

    x_fire = 5.6
    po, ph, pl, pc = 105.5, 110.0, 98.5, 99.0
    rng = ph - pl
    body = po - pc
    body_ratio = body / rng
    close_pos = (ph - pc) / rng
    atr14 = 6.5
    range_atr = rng / atr14

    # ghost "типичный ATR14 бар" рядом, для масштаба
    x_ghost = x_fire - 1.3
    ghost_mid = (ph + pl) / 2
    ax.add_patch(Rectangle((x_ghost - 0.24, ghost_mid - atr14 / 2), 0.48, atr14,
                           facecolor="none", edgecolor=GHOST_C, lw=1.3,
                           linestyle=(0, (3, 2)), zorder=2))
    ax.text(x_ghost, ghost_mid - atr14 / 2 - 0.9, "типичный\nбар (ATR14)\n"
                                                    f"= {atr14:.1f}",
            fontsize=7.6, color=GHOST_C, ha="center", fontweight="bold")

    ax.add_patch(Rectangle((x_fire - 0.34, pc), 0.68, body,
                            facecolor="#ffdcdc", edgecolor="none", zorder=1))
    draw_candle(ax, x_fire, o=po, h=ph, l=pl, c=pc,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, ph + 0.7, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, ph + 1.5, "FIRE\n(bar i)", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    # range bracket
    x_r = x_fire + 0.9
    ax.annotate("", xy=(x_r, ph), xytext=(x_r, pl),
                arrowprops=dict(arrowstyle="<->", color=WICK_C, lw=1.8))
    ax.text(x_r + 0.15, (ph + pl) / 2 + 1.2,
            f"range = h-l = {rng:.1f}\n/ATR14={atr14:.1f} = {range_atr:.2f} (≥1.5 ✓)",
            fontsize=8.6, color=WICK_C, fontweight="bold", va="center")

    # close-position bracket: нижний квартиль range, отмеченный чертой на 25%
    q_y = pl + 0.25 * rng
    ax.plot([x_fire - 0.5, x_fire + 0.5], [q_y, q_y], color=POS_C, lw=1.6,
            ls=(0, (4, 2)), zorder=4)
    ax.text(x_fire - 0.62, q_y, "25%\nline", fontsize=7.3, color=POS_C,
            ha="right", va="center", fontweight="bold")
    ax.annotate("close_pos = (h-c)/range\n"
                f"= {close_pos:.2f}  (≥0.75 ✓)\nclose в нижнем квартиле",
                xy=(x_fire + 0.25, pc), xytext=(7.4, 95.2),
                fontsize=9.0, color=POS_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=POS_C, lw=1.3),
                ha="left", va="center")

    ax.text(x_fire, pl - 2.0, f"body / range = {body_ratio:.2f}  (≥ 0.5 ✓)",
            fontsize=10, color=BODY_C, fontweight="bold", ha="center")

    ax.plot(x_fire, ph + 3.2, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire + 0.9, ph + 3.2, "A1-pivot HIGH\n(bar i > i-1, i-2)",
            fontsize=8, color=ACCENT, va="center", fontweight="bold")

    ax.set_xlim(0.3, 12.5)
    ax.set_ylim(91.0, 116.0)
    ax.set_xlabel("12h bars →", fontsize=9.5)
    ax.set_ylabel("price", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel(ax) -> None:
    ax.set_title("LONG setup  ·  A1-pivot LOW → close у HIGH края + аномальный range",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    for x, lo, hi, o, c in [
        (1.4, 100.0, 103.5, 102.7, 100.8),
        (2.6, 98.8, 102.2, 101.4, 99.6),
    ]:
        draw_candle(ax, x, o, hi, lo, c, w=0.4, edge="#666",
                    fill_up="#e7e7ea", fill_dn="#a9a9b0", lw=0.8, alpha=0.75)
    ax.text(1.4, 104.7, "bar i-2", fontsize=8, color="#555", ha="center", style="italic")
    ax.text(2.6, 97.6, "bar i-1", fontsize=8, color="#555", ha="center", style="italic")

    x_fire = 5.6
    pl, ph, po, pc = 90.0, 101.5, 94.5, 101.0
    rng = ph - pl
    body = pc - po
    body_ratio = body / rng
    close_pos = (pc - pl) / rng
    atr14 = 6.5
    range_atr = rng / atr14

    x_ghost = x_fire - 1.3
    ghost_mid = (ph + pl) / 2
    ax.add_patch(Rectangle((x_ghost - 0.24, ghost_mid - atr14 / 2), 0.48, atr14,
                           facecolor="none", edgecolor=GHOST_C, lw=1.3,
                           linestyle=(0, (3, 2)), zorder=2))
    ax.text(x_ghost, ghost_mid + atr14 / 2 + 0.5, "типичный\nбар (ATR14)\n"
                                                    f"= {atr14:.1f}",
            fontsize=7.6, color=GHOST_C, ha="center", fontweight="bold")

    ax.add_patch(Rectangle((x_fire - 0.34, po), 0.68, body,
                            facecolor="#d4ecd4", edgecolor="none", zorder=1))
    draw_candle(ax, x_fire, o=po, h=ph, l=pl, c=pc,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, pl - 0.7, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, pl - 1.5, "FIRE\n(bar i)", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    x_r = x_fire + 0.9
    ax.annotate("", xy=(x_r, ph), xytext=(x_r, pl),
                arrowprops=dict(arrowstyle="<->", color=WICK_C, lw=1.8))
    ax.text(x_r + 0.15, (ph + pl) / 2 - 1.2,
            f"range = h-l = {rng:.1f}\n/ATR14={atr14:.1f} = {range_atr:.2f} (≥1.5 ✓)",
            fontsize=8.6, color=WICK_C, fontweight="bold", va="center")

    q_y = ph - 0.25 * rng
    ax.plot([x_fire - 0.5, x_fire + 0.5], [q_y, q_y], color=POS_C, lw=1.6,
            ls=(0, (4, 2)), zorder=4)
    ax.text(x_fire - 0.62, q_y, "25%\nline", fontsize=7.3, color=POS_C,
            ha="right", va="center", fontweight="bold")
    ax.annotate("close_pos = (c-l)/range\n"
                f"= {close_pos:.2f}  (≥0.75 ✓)\nclose в верхнем квартиле",
                xy=(x_fire + 0.25, pc), xytext=(7.4, 105.5),
                fontsize=9.0, color=POS_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=POS_C, lw=1.3),
                ha="left", va="center")

    ax.text(x_fire, ph + 1.6, f"body / range = {body_ratio:.2f}  (≥ 0.5 ✓)",
            fontsize=10, color="#006400", fontweight="bold", ha="center")

    ax.plot(x_fire, pl - 3.2, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire + 0.9, pl - 3.2, "A1-pivot LOW\n(bar i < i-1, i-2)",
            fontsize=8, color=WARN, va="center", fontweight="bold")

    ax.set_xlim(0.3, 12.5)
    ax.set_ylim(85.0, 110.0)
    ax.set_xlabel("12h bars →", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Формальное правило (4 условия, causal)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "(1) close direction match pivot direction:\n"
        "    SHORT (FH):  c12[i] < o12[i]      LONG (FL):  c12[i] > o12[i]\n\n"
        "(2) body / range >= 0.5   (мягче, чем 0.7 у B9C3 momentum):\n"
        "    |c12[i] - o12[i]| / (h12[i] - l12[i])   >=   0.5\n\n"
        "(3) close_pos >= 0.75   (close у самого дальнего от pivot края):\n"
        "    SHORT:  (h12[i] - c12[i]) / range   >=   0.75\n"
        "    LONG:   (c12[i] - l12[i]) / range   >=   0.75\n\n"
        "(4) range / ATR14(i) >= 1.5   (аномальный по размеру бар):\n"
        "    (h12[i] - l12[i]) / atr14[i]   >=   1.5\n\n"
        "Fire = (bar_idx, direction), матчится с A1-pivot. ATR14 читается\n"
        "из data/maxv/ (level-1, единый источник с B1 WIDE и B9C2)."
    )
    ax.text(0.04, 0.78, rule_text, fontsize=9.2, color="#222",
            ha="left", va="top", transform=ax.transAxes,
            family="DejaVu Sans Mono")


def _draw_notes_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#fff5f0", edgecolor="#c0392b",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Canon-заметки  ·  экстремальный complement к B9C3",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "•  B9C4 vs B9C3: у momentum (B9C3) требование строже по body (≥0.7), но нет\n"
        "   ограничения на абсолютный размер бара — маленький бар с телом 0.7 тоже проходит.\n"
        "   B9C4 разрешает тело от 0.5, но взамен требует close у самого края (≥0.75 квартиль)\n"
        "   И огромный range относительно ATR14 (≥1.5×) — ловит именно ВЗРЫВНЫЕ бары,\n"
        "   а не просто «чистые» по форме.\n\n"
        "•  Пересечение с B9C3 есть, но не полное: momentum-бар с body=0.72 и range=1.1×ATR14\n"
        "   попадёт в B9C3, но НЕ в B9C4 (range < 1.5). И наоборот — бар с body=0.55,\n"
        "   близко к краю, огромный range попадёт в B9C4, но НЕ в B9C3 (body < 0.7).\n\n"
        "•  «Climax» — термин из classic TA (climax bar/exhaustion bar): аномальный объём\n"
        "   и диапазон в момент разворота, часто последний импульс перед сменой тренда.\n\n"
        "•  Причинность: только open, close, high, low бара i + ATR14(i) (past-only SMA-14).\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b9_others.py (b9c4)  ·  G:\\ASVK\\lib\\maxv.py (atr14)"
    )
    ax.text(0.04, 0.78, notes_text, fontsize=9.15, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/c/Users/Вадим/Desktop/export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B9C4.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
