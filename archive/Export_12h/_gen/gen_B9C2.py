"""B9C2 rule — presentation-quality schematic + canon notes.

Правило: maxV depth sweep (ex-B1C7, rescued 2026-07-21 в B9).
maxV(i-1) = close 1m-бара с абсолютным max объёмом в bull/bear группе
внутри ПРЕДЫДУЩЕГО 12h бара (i-1). Pivot bar i пронзает этот уровень
и закрывается обратно — глубина пробоя нормирована на ATR14(12h).

Source of truth (G:\\ASVK\\lib\\fractal12h\\b9_others.py):
    SHORT: h12[i] > maxv[i-1] AND c12[i] < maxv[i-1]
           AND (h12[i]-maxv[i-1]) / atr14[i] >= 0.7
    LONG:  l12[i] < maxv[i-1] AND c12[i] > maxv[i-1]
           AND (maxv[i-1]-l12[i]) / atr14[i] >= 0.7
maxV сам по себе — level-1 shared indicator, см. G:\\ASVK\\lib\\maxv.py.

Стиль 1-в-1 повторяет B1C1..B1C4.png (zone/level-sweep family) + B9-family header.

Refs:
  G:\\ASVK\\lib\\maxv.py                        — maxV(k)/ATR14(k), считается один раз (level-1)
  G:\\ASVK\\lib\\fractal12h\\b9_others.py         — реализация условия (b9c2)

Output:
  Desktop/export_12h/B9C2.png
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
MAXV_C   = "#8e44ad"
INSET_C  = "#b8860b"


def draw_candle(ax, x, o, h, l, c, w=0.55, edge="#111",
                fill_up="white", fill_dn="#222", lw=0.9, alpha=1.0):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=2, alpha=alpha)
    color = fill_up if c >= o else fill_dn
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.03),
                            facecolor=color, edgecolor=edge, lw=lw,
                            zorder=3, alpha=alpha))


def draw_tick(ax, x, y, color, big=False):
    ax.plot([x, x], [y - (0.35 if big else 0.12), y + (0.35 if big else 0.12)],
            color=color, lw=(2.4 if big else 0.9), alpha=(1.0 if big else 0.55), zorder=4)


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.6, 4, 2.3],
                           hspace=0.3, wspace=0.14,
                           left=0.045, right=0.985, top=0.965, bottom=0.045)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#1c1c2e", edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.68, "B9C2 — maxV depth sweep · dist/ATR14 ≥ 0.7",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.42, "A1 anchor  (pool: все Williams-confirmable pivots)  ·  "
                       "ex-B1C7, rescued 2026-07-21 в B9",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#ffc857", transform=ax.transAxes)
    ax.text(0.5, 0.15,
            "maxV(i-1) — level-1 shared indicator (G:\\ASVK\\lib\\maxv.py)  ·  "
            "НЕ FVG-зона, единственный price LEVEL",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup  ·  A1-pivot LOW → pierce maxV(i-1) (resistance)",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    maxv = 100.0
    atr14 = 4.9

    # bar i-1 — "maxV anchor" bar
    x_prev = 2.6
    draw_candle(ax, x_prev, o=97.5, h=101.5, l=96.0, c=98.8,
                w=0.55, edge="#555", fill_up="#eee", fill_dn="#bbb", lw=1.0, alpha=0.9)
    ax.text(x_prev, 95.3, "bar i-1", fontsize=8.5, color="#555", ha="center", style="italic")

    # inset callout (top-left corner — свободная зона, не пересекает maxV line/candles) —
    # облако 1m тиков, один — max объём (жирный, подписан)
    bx, by, bw, bh = 0.0, 104.1, 2.15, 4.4
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                 boxstyle="round,pad=0.08",
                                 facecolor="#fff6df", edgecolor=INSET_C, linewidth=1.2, zorder=5))
    ax.text(bx + bw / 2, by + bh - 0.55, "все 1m бары i-1\n(bull/bear группы)",
            fontsize=6.9, color="#8a5a00", ha="center", style="italic", fontweight="bold")
    for dx, dy in [(0.25, 0.4), (0.45, 1.1), (0.65, 0.65), (1.4, 1.7),
                  (1.65, 2.3), (1.85, 0.85)]:
        draw_tick(ax, bx + dx, by + dy, "#999")
    draw_tick(ax, bx + 1.05, by + 1.4, "#8a5a00", big=True)  # max-volume tick
    ax.text(bx + bw / 2, by + 0.5, "max объём\n(1m) → maxV", fontsize=6.9, color="#8a5a00",
            ha="center", fontweight="bold")
    ax.annotate("", xy=(x_prev, 101.5), xytext=(bx + bw / 2, by),
                arrowprops=dict(arrowstyle="<-", color=INSET_C, lw=1.1))

    # maxV level line (dashed, extends across chart)
    x_maxv_start = x_prev + 0.35
    x_fire = 8.0
    ax.plot([x_maxv_start, x_fire + 0.5], [maxv, maxv],
            ls=(0, (6, 3)), color=MAXV_C, lw=1.8, zorder=2)
    ax.text(x_maxv_start - 0.15, maxv + 0.35,
            "maxV(i-1) = close 1m-бара с max объёмом", fontsize=8.5,
            color=MAXV_C, fontweight="bold", ha="left")

    # A1-pivot marker
    ax.plot(1.0, 94.0, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(1.0, 93.1, "A1-pivot LOW\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    # FIRE candle i — pierces above maxV, closes back below
    ho, hh, hl, hc = 99.0, 103.5, 97.0, 99.3
    draw_candle(ax, x_fire, o=ho, h=hh, l=hl, c=hc,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, hh + 0.6, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, hh + 1.4, "FIRE\n(bar i)", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    # depth bracket
    x_d = x_fire + 1.0
    ax.annotate("", xy=(x_d, hh), xytext=(x_d, maxv),
                arrowprops=dict(arrowstyle="<->", color=WARN, lw=2.0))
    dist = hh - maxv
    ax.text(x_d + 0.15, (hh + maxv) / 2,
            f"depth = h[i] - maxV\n= {dist:.1f}\n(/ATR14={atr14:.1f} = {dist/atr14:.2f} ≥ 0.7 ✓)",
            fontsize=8.7, color=WARN, fontweight="bold", va="center")

    ax.annotate("wick пронзает maxV(i-1)\nh[i] > maxV(i-1)",
                xy=(x_fire + 0.05, maxv + 1.5), xytext=(5.4, 108.0),
                fontsize=9.3, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3),
                ha="left", va="center")
    ax.annotate("close возвращается НИЖЕ maxV(i-1)\n→ rejection",
                xy=(x_fire + 0.10, hc), xytext=(5.4, 91.0),
                fontsize=9.3, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3),
                ha="left", va="center")

    ax.set_xlim(-0.2, 12.5)
    ax.set_ylim(89.0, 111.0)
    ax.set_xlabel("12h bars →", fontsize=9.5)
    ax.set_ylabel("price", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel(ax) -> None:
    ax.set_title("LONG setup  ·  A1-pivot HIGH → pierce maxV(i-1) (support)",
                 fontsize=12.5, fontweight="bold", pad=10, color="#333")

    maxv = 95.0
    atr14 = 4.9

    x_prev = 2.6
    draw_candle(ax, x_prev, o=97.0, h=99.0, l=93.5, c=95.8,
                w=0.55, edge="#555", fill_up="#eee", fill_dn="#bbb", lw=1.0, alpha=0.9)
    ax.text(x_prev, 100.9, "bar i-1", fontsize=8.5, color="#555", ha="center", style="italic")

    # inset callout (bottom-left corner — свободная зона в LONG-панели)
    bx, by, bw, bh = 0.0, 85.7, 2.15, 4.4
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                 boxstyle="round,pad=0.08",
                                 facecolor="#fff6df", edgecolor=INSET_C, linewidth=1.2, zorder=5))
    ax.text(bx + bw / 2, by + bh - 0.55, "все 1m бары i-1\n(bull/bear группы)",
            fontsize=6.9, color="#8a5a00", ha="center", style="italic", fontweight="bold")
    for dx, dy in [(0.25, 0.4), (0.45, 1.1), (0.65, 0.65), (1.4, 1.7),
                  (1.65, 2.3), (1.85, 0.85)]:
        draw_tick(ax, bx + dx, by + dy, "#999")
    draw_tick(ax, bx + 1.05, by + 1.4, "#8a5a00", big=True)
    ax.text(bx + bw / 2, by + 0.5, "max объём\n(1m) → maxV", fontsize=6.9, color="#8a5a00",
            ha="center", fontweight="bold")
    ax.annotate("", xy=(x_prev, 93.5), xytext=(bx + bw / 2, by + bh),
                arrowprops=dict(arrowstyle="<-", color=INSET_C, lw=1.1))

    x_maxv_start = x_prev + 0.35
    x_fire = 8.0
    ax.plot([x_maxv_start, x_fire + 0.5], [maxv, maxv],
            ls=(0, (6, 3)), color=MAXV_C, lw=1.8, zorder=2)
    ax.text(x_maxv_start - 0.15, maxv - 0.7,
            "maxV(i-1) = close 1m-бара с max объёмом", fontsize=8.5,
            color=MAXV_C, fontweight="bold", ha="left")

    ax.plot(1.0, 105.5, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(1.0, 106.4, "A1-pivot HIGH\n(12h fractal)", fontsize=8,
            color=WARN, ha="center", fontweight="bold")

    ho, hh, hl, hc = 96.0, 98.0, 91.5, 95.7
    draw_candle(ax, x_fire, o=ho, h=hh, l=hl, c=hc,
                w=0.55, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, hl - 0.6, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, hl - 1.4, "FIRE\n(bar i)", fontsize=9.5, fontweight="bold",
            color="#8a5a00", ha="center")

    x_d = x_fire + 1.0
    ax.annotate("", xy=(x_d, maxv), xytext=(x_d, hl),
                arrowprops=dict(arrowstyle="<->", color=WARN, lw=2.0))
    dist = maxv - hl
    ax.text(x_d + 0.15, (maxv + hl) / 2,
            f"depth = maxV - l[i]\n= {dist:.1f}\n(/ATR14={atr14:.1f} = {dist/atr14:.2f} ≥ 0.7 ✓)",
            fontsize=8.7, color=WARN, fontweight="bold", va="center")

    ax.annotate("wick пронзает maxV(i-1)\nl[i] < maxV(i-1)",
                xy=(x_fire + 0.05, maxv - 1.5), xytext=(5.4, 88.0),
                fontsize=9.3, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3),
                ha="left", va="center")
    ax.annotate("close возвращается ВЫШЕ maxV(i-1)\n→ rejection",
                xy=(x_fire + 0.10, hc), xytext=(5.4, 100.5),
                fontsize=9.3, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3),
                ha="left", va="center")

    ax.set_xlim(-0.2, 12.5)
    ax.set_ylim(85.5, 108.0)
    ax.set_xlabel("12h bars →", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Формальное правило (3 условия, causal)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "maxV(k) = close 1m-бара с абсолютным max объёмом среди bull ИЛИ bear\n"
        "группы внутри 12h бара k (bull, если max_bull_vol >= max_bear_vol, иначе bear).\n\n"
        "(1) SHORT (FH):  h12[i] > maxV(i-1)     — wick пронзает уровень\n"
        "    LONG  (FL):  l12[i] < maxV(i-1)\n\n"
        "(2) SHORT:  c12[i] < maxV(i-1)          — close возвращается за уровень\n"
        "    LONG:   c12[i] > maxV(i-1)\n\n"
        "(3) depth / ATR14(i) >= 0.7:\n"
        "    SHORT:  (h12[i] - maxV(i-1)) / atr14[i]  >=  0.7\n"
        "    LONG:   (maxV(i-1) - l12[i]) / atr14[i]  >=  0.7\n\n"
        "Fire = (bar_idx, direction), матчится с A1-pivot. maxV/ATR14 читаются\n"
        "из data/maxv/ (level-1, считаются один раз per pipeline cycle)."
    )
    ax.text(0.04, 0.78, rule_text, fontsize=9.4, color="#222",
            ha="left", va="top", transform=ax.transAxes,
            family="DejaVu Sans Mono")


def _draw_notes_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#fff5f0", edgecolor="#c0392b",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.90, "Canon-заметки  ·  ex-B1C7, rescued 2026-07-21",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "•  B9C2 — second orthogonal channel:  B1 (FVG-зона) → B9C2 (maxV-уровень) → B9C3 (свеча).\n"
        "   Изначально был B1C7 внутри B1-семьи, но механика НЕ FVG-based — единственный\n"
        "   price LEVEL (не зона с zone_lo/zone_hi), поэтому вынесен в B9 (catch-all, 2026-07-21).\n\n"
        "•  maxV — «где стоял крупный игрок» на предыдущем 12h баре: 1m-бар с максимальным\n"
        "   объёмом в доминирующей (bull/bear) группе. Это НЕ VWAP и НЕ POC всего профиля —\n"
        "   просто close одного конкретного 1m-бара.\n\n"
        "•  maxV(i-1) считается на баре i-1, sweep проверяется на баре i («i-1» в названии —\n"
        "   maxV формируется в прошлом, пробивается в настоящем).\n\n"
        "•  maxV и ATR14(12h) — level-1 shared indicators (G:\\ASVK\\lib\\maxv.py), считаются\n"
        "   один раз в pipeline Block 1 (после e12d/s7d) и пишутся в data/maxv/. Читаются\n"
        "   отсюда же WIDE-фильтром всей B1-семьи — единый источник правды.\n\n"
        "•  Причинность: maxV(i-1) известен на close бара i-1, полностью до открытия бара i.\n\n"
        "Refs:  G:\\ASVK\\lib\\maxv.py  ·  G:\\ASVK\\lib\\fractal12h\\b9_others.py (b9c2)"
    )
    ax.text(0.04, 0.78, notes_text, fontsize=9.2, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/c/Users/Вадим/Desktop/export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B9C2.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
