"""B9C3 rule — presentation-quality schematic + canon notes.

Правило: momentum reversal bar на pivot bar i.
Close закрыт в направлении разворота + body/range >= 0.7 (Marubozu-like).
НЕ FVG-механика, НЕ maxV. Pure candlestick signature.

Перенесено 1-в-1 из WSL ~/smc-warehouse/scripts/фрактал-12h/B9C3.py
(этот файл уже соответствует актуальному канону G:\\ASVK\\lib\\fractal12h\\b9_others.py —
только путь вывода изменён под Windows Desktop).

Refs:
  scripts/фрактал-12h/эталон/B9C3_momentum_bar.py     — canonical
  G:\\ASVK\\lib\\fractal12h\\b9_others.py               — ASVK-portable порт (b9c3)

Output:
  Desktop/export_12h/B9C3.png
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


def draw_candle(ax, x, o, h, l, c, w=0.55, edge="#111",
                fill_up="white", fill_dn="#222", lw=0.9, alpha=1.0):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=2, alpha=alpha)
    color = fill_up if c >= o else fill_dn
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.03),
                            facecolor=color, edgecolor=edge, lw=lw,
                            zorder=3, alpha=alpha))


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
    ax.text(0.5, 0.68, "B9C3 — Momentum reversal bar · body/range ≥ 0.7",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.42, "A1 anchor  (pool: все Williams-confirmable pivots)  ·  "
                       "ортогональный к FVG и maxV",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#ffc857", transform=ax.transAxes)
    ax.text(0.5, 0.15,
            "pure candlestick signature (Marubozu-like)  ·  "
            "causal (all data ≤ close bar i)",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup  ·  A1-pivot HIGH → close RED + body ≥ 0.7 × range",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    for x, lo, hi, o, c in [
        (1.6, 96.0, 100.5, 97.0, 99.5),
        (3.0, 97.5, 101.5, 100.0, 100.8),
    ]:
        draw_candle(ax, x, o, hi, lo, c, w=0.42, edge="#666",
                    fill_up="#e7e7ea", fill_dn="#a9a9b0", lw=0.8, alpha=0.75)
    ax.text(1.6, 94.8, "bar i-2", fontsize=8, color="#555", ha="center", style="italic")
    ax.text(3.0, 96.3, "bar i-1", fontsize=8, color="#555", ha="center", style="italic")

    x_fire = 5.5
    ph = 108.0
    po = 106.5
    pc = 101.0
    pl = 100.5
    rng = ph - pl
    body = po - pc
    body_ratio = body / rng

    ax.add_patch(Rectangle((x_fire - 0.34, pc), 0.68, body,
                            facecolor="#ffdcdc", edgecolor="none", zorder=1))
    draw_candle(ax, x_fire, o=po, h=ph, l=pl, c=pc,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, ph + 0.7, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, ph + 1.5, "FIRE\n(bar i)", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    x_r = x_fire + 0.85
    ax.annotate("", xy=(x_r, ph), xytext=(x_r, pl),
                arrowprops=dict(arrowstyle="<->", color=WICK_C, lw=1.8))
    ax.text(x_r + 0.15, (ph + pl) / 2,
            f"range\n= h - l\n= {rng:.1f}",
            fontsize=9, color=WICK_C, fontweight="bold", va="center")

    x_b = x_fire + 2.4
    ax.annotate("", xy=(x_b, po), xytext=(x_b, pc),
                arrowprops=dict(arrowstyle="<->", color=BODY_C, lw=2.2))
    ax.text(x_b + 0.15, (po + pc) / 2,
            f"body\n= |c - o|\n= {body:.1f}",
            fontsize=9, color=BODY_C, fontweight="bold", va="center")

    ax.text(x_fire, pl - 1.4, f"body / range = {body_ratio:.2f}  (≥ 0.7 ✓)",
            fontsize=10.5, color=BODY_C, fontweight="bold", ha="center")

    ax.annotate("① close < open (RED bar)\n   pivot dir = SHORT",
                xy=(x_fire + 0.25, (po + pc) / 2), xytext=(1.5, 91.0),
                fontsize=9.5, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3),
                ha="left", va="center")
    ax.annotate("② upper wick tiny\n   (h ≈ open)",
                xy=(x_fire + 0.25, ph), xytext=(1.5, 112.0),
                fontsize=9.5, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3),
                ha="left", va="center")

    ax.plot(x_fire, ph + 3.4, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire + 0.5, ph + 3.4, "A1-pivot HIGH\n(bar i > i-1, i-2)",
            fontsize=8, color=ACCENT, va="center", fontweight="bold")

    ax.set_xlim(0.4, 11.5)
    ax.set_ylim(89.0, 115.0)
    ax.set_xlabel("12h bars →", fontsize=9.5)
    ax.set_ylabel("price", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel(ax) -> None:
    ax.set_title("LONG setup  ·  A1-pivot LOW → close GREEN + body ≥ 0.7 × range",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    for x, lo, hi, o, c in [
        (1.6, 99.5, 104.0, 103.0, 100.5),
        (3.0, 98.5, 102.5, 100.0, 99.2),
    ]:
        draw_candle(ax, x, o, hi, lo, c, w=0.42, edge="#666",
                    fill_up="#e7e7ea", fill_dn="#a9a9b0", lw=0.8, alpha=0.75)
    ax.text(1.6, 105.3, "bar i-2", fontsize=8, color="#555", ha="center", style="italic")
    ax.text(3.0, 103.8, "bar i-1", fontsize=8, color="#555", ha="center", style="italic")

    x_fire = 5.5
    pl = 92.0
    po = 93.5
    pc = 99.0
    ph = 99.5
    rng = ph - pl
    body = pc - po
    body_ratio = body / rng

    ax.add_patch(Rectangle((x_fire - 0.34, po), 0.68, body,
                            facecolor="#d4ecd4", edgecolor="none", zorder=1))
    draw_candle(ax, x_fire, o=po, h=ph, l=pl, c=pc,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, pl - 0.7, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, pl - 1.5, "FIRE\n(bar i)", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    x_r = x_fire + 0.85
    ax.annotate("", xy=(x_r, ph), xytext=(x_r, pl),
                arrowprops=dict(arrowstyle="<->", color=WICK_C, lw=1.8))
    ax.text(x_r + 0.15, (ph + pl) / 2,
            f"range\n= h - l\n= {rng:.1f}",
            fontsize=9, color=WICK_C, fontweight="bold", va="center")

    x_b = x_fire + 2.4
    ax.annotate("", xy=(x_b, pc), xytext=(x_b, po),
                arrowprops=dict(arrowstyle="<->", color="#006400", lw=2.2))
    ax.text(x_b + 0.15, (pc + po) / 2,
            f"body\n= |c - o|\n= {body:.1f}",
            fontsize=9, color="#006400", fontweight="bold", va="center")

    ax.text(x_fire, ph + 1.4, f"body / range = {body_ratio:.2f}  (≥ 0.7 ✓)",
            fontsize=10.5, color="#006400", fontweight="bold", ha="center")

    ax.annotate("① close > open (GREEN bar)\n   pivot dir = LONG",
                xy=(x_fire + 0.25, (po + pc) / 2), xytext=(1.5, 108.5),
                fontsize=9.5, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3),
                ha="left", va="center")
    ax.annotate("② lower wick tiny\n   (l ≈ open)",
                xy=(x_fire + 0.25, pl), xytext=(1.5, 87.5),
                fontsize=9.5, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3),
                ha="left", va="center")

    ax.plot(x_fire, pl - 3.4, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire + 0.5, pl - 3.4, "A1-pivot LOW\n(bar i < i-1, i-2)",
            fontsize=8, color=WARN, va="center", fontweight="bold")

    ax.set_xlim(0.4, 11.5)
    ax.set_ylim(85.0, 111.0)
    ax.set_xlabel("12h bars →", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Формальное правило (2 условия, causal)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "(1) close direction match pivot direction:\n"
        "    SHORT (FH):  c12[i] < o12[i]     (bar closed RED)\n"
        "    LONG  (FL):  c12[i] > o12[i]     (bar closed GREEN)\n\n"
        "(2) body / range ≥ 0.7:\n"
        "    |c12[i] - o12[i]|  /  (h12[i] - l12[i])   ≥   0.7\n\n"
        "Marubozu-like signature:\n"
        "    маленькие оба wick, большое тело в направлении разворота.\n"
        "    Разворот УЖЕ произошёл внутри 12h бара.\n\n"
        "Fire = (bar_idx, direction), матчится с A1-pivot.\n"
        "НЕ требует FVG-зон и НЕ требует maxV level."
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
    ax.text(0.5, 0.90, "Canon-заметки  ·  универсальное правило (2026-07-21)",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "•  B9C3 — third orthogonal channel:  B1 (FVG) → B9C2 (maxV) → B9C3 (candle).\n"
        "   Найден при поиске complements для missed FVG-triggered фракталов.\n\n"
        "•  Логика: pivot bar сам показал разворот внутри своего диапазона.\n"
        "   SHORT HIGH: wick top в первой половине бара, close значительно ниже.\n"
        "   LONG LOW:   wick bottom в первой половине бара, close значительно выше.\n\n"
        "•  Universality (mature 2022+, все 11 активов):  11/11 ≥ 75 %\n"
        "   Mean WR = 87.4 %,  Min = 77.1 % (LINK).\n"
        "   Total n ≈ 300 за 6 лет  (≈ 4-5 сигналов/актив/год).\n\n"
        "•  Причинность: только open, close, high, low бара i.\n"
        "   A1 pre-w known на close i (past-only: bar i > bar i-1 AND bar i > bar i-2).\n"
        "   Prediction target = confirmed (i+1, i+2 не пробьют экстремум i).\n\n"
        "•  Orthogonal к FVG basket (B1): большинство B9C3-only сигналов не входят в B1C1..C4.\n"
        "   Unique-only mean WR ≈ 83 %.\n\n"
        "Refs:  эталон/B9C3_momentum_bar.py  ·  G:\\ASVK\\lib\\fractal12h\\b9_others.py (b9c3)"
    )
    ax.text(0.04, 0.78, notes_text, fontsize=9.3, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/c/Users/Вадим/Desktop/export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B9C3.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
