"""B4C3 rule (production, в basket) — THMA-9(12h) FULL_DISP · Triple Hull MA · TF=12h.

Правило (источник истины G:\\ASVK\\lib\\fractal12h\\b4_hma.py):
    margin = 0.5 × ATR14(12h)[i]
    SHORT: high[i] > THMA_prev   AND  close[i] < THMA_prev - margin
    LONG:  low[i]  < THMA_prev   AND  close[i] > THMA_prev + margin
    Одна TF — THMA-9(12h), тот же ряд, что и pivot (prev = бар i-1).

    Найден research'ем MA-family (G:\\Claude\\research\\ma_family_explore.py — 6 типов
    MA × 7 длин × 5 TF на BTC) как кандидат с наибольшим приростом n при жадном отборе
    поверх уже выбранных HMA-78/HMA-200 (2026-07-25). Короткая длина (9) — самый
    быстрый/шумный из всей серии B4, поэтому и самый большой объём (n=156 на BTC).

    Домен: a_cand[a124_pool] — A1+A2+A4, БЕЗ A3, как весь B3/B4.

Кросс-проверка (n / WR, ОТДЕЛЬНО по каждому активу — не бандл с C4-C6):
    BTC n=156 WR=86.54%   ETH n=119 WR=84.87%   SOL n=126 WR=84.13%
    ADA n=97  WR=81.44%   AVAX n=124 WR=79.03%  BNB n=113 WR=80.53%
    DOGE n=91 WR=87.91%   DOT n=128 WR=74.22%   LINK n=128 WR=80.47%
    LTC n=126 WR=86.51%   XRP n=103 WR=84.47%   pooled n=1,311
    (пересчитано отдельно от bundle-проверки C3-C6, см. G:\\Claude\\research\\
    b4c3c6_cross_asset.py / b4c3c6_per_asset.parquet)

Правая (LONG) панель — РЕАЛЬНЫЙ пример: BTC, THMA-9(12h), сигнал зажёгся
2026-07-17 12:00 UTC, Williams n=2 подтверждение по i+1/i+2 видно на графике.

Refs:
  G:\\ASVK\\lib\\fractal12h\\b4_hma.py       — реализация
  G:\\ASVK\\lib\\trendline.py                — источник THMA (variant 12h9Thma, --mode Thma)
  G:\\Claude\\research\\ma_family_explore.py — источник механики/скана
  G:\\Claude\\research\\b4c3c6_cross_asset.py — per-candidate cross-asset проверка

Output:
  G:\\ASVK\\Export_12h\\B4C3.png
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

FIRE_C   = "#111"
ACCENT   = "#0057b8"
WARN     = "#c0392b"
LEVEL_C  = "#7a2ba0"


def draw_candle(ax, x, o, h, l, c, w=0.55, edge="#111",
                 fill_up="white", fill_dn="#222", lw=0.9, alpha=1.0, zorder=3):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=zorder - 1, alpha=alpha)
    color = fill_up if c >= o else fill_dn
    body = max(abs(c - o), (h - l) * 0.02)
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, body,
                            facecolor=color, edgecolor=edge, lw=lw,
                            zorder=zorder, alpha=alpha))


def _draw_ellipsis(ax, x_start, x_end, y):
    for xf in [0.30, 0.50, 0.70]:
        ax.plot(x_start + (x_end - x_start) * xf, y,
                marker="o", markersize=3, color="#888", zorder=1)


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.66, 4, 2.4],
                            hspace=0.30, wspace=0.14,
                            left=0.045, right=0.985, top=0.965, bottom=0.045)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#1c1c2e", edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.68, "B4C3 — THMA-9(12h) FULL_DISP \u00b7 Triple Hull MA \u00b7 TF=12h",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "PRODUCTION (в basket, развёрнуто 2026-07-25)  \u00b7  домен A1+A2+A4 без A3 (a124_pool)",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")
    ax.text(0.5, 0.10,
            "BTC n=156 WR=86.54%  \u00b7  ETH n=119 WR=84.87%  \u00b7  SOL n=126 WR=84.13%  \u00b7  "
            "+ 8 активов, диапазон 74-88%",
            ha="center", va="center", fontsize=10, color="#9aa0c0",
            transform=ax.transAxes)

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel_real(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup (схема)  \u00b7  A1-pivot HIGH \u2192 h[i]>THMA_prev, c[i]<THMA_prev-margin",
                 fontsize=11.6, fontweight="bold", pad=10, color="#333")

    level = 101.0
    margin = 1.6
    x0, x_fire = 1.0, 8.5

    ax.plot([x0, x_fire + 1.5], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(x_fire + 1.7, level, "THMA_prev", fontsize=9.5,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0, x_fire + 1.5], [level - margin, level - margin],
            ls=(0, (2, 2)), color=WARN, lw=1.4, alpha=0.85)
    ax.text(x_fire + 1.7, level - margin, "THMA_prev \u2212 margin\n(0.5\u00d7ATR14)", fontsize=8,
            color=WARN, va="center", fontweight="bold")

    draw_candle(ax, x0, o=99.5, h=100.4, l=98.7, c=100.0, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    draw_candle(ax, x0 + 1.1, o=100.0, h=100.9, l=99.3, c=100.6, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    ax.text(x0 + 0.5, 97.9, "бар i-1  (формирует THMA_prev)", fontsize=7.6,
            color="#555", ha="center", style="italic")

    _draw_ellipsis(ax, x0 + 2.0, x_fire - 1.3, level - 4.5)

    ax.plot(x0 + 0.3, 96.0, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x0 + 0.3, 95.2, "A1-pivot HIGH\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    draw_candle(ax, x_fire, o=101.0, h=103.4, l=97.5, c=98.4,
                w=0.6, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 104.1, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 104.8, "FIRE", fontsize=9.5, fontweight="bold", color="#8a5a00", ha="center")

    ax.annotate("high > THMA_prev\n(прокол уровня вверх)",
                xy=(x_fire + 0.02, 103.35), xytext=(4.7, 107.2),
                fontsize=9.0, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4), ha="left", va="center")
    ax.annotate("close < THMA_prev \u2212 margin\n(решительный уход, не полу-мера)",
                xy=(x_fire + 0.05, 98.4), xytext=(1.0, 93.6),
                fontsize=9.0, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4), ha="left", va="center")

    ax.set_xlim(-0.4, 12.7)
    ax.set_ylim(92.0, 109.0)
    ax.set_xlabel("12h bars \u2192", fontsize=9.5)
    ax.set_ylabel("price (условно)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel_real(ax) -> None:
    ax.set_title("LONG setup \u2014 РЕАЛЬНЫЙ пример: BTC 2026-07-17 12:00 UTC (confirmed)",
                 fontsize=11.6, fontweight="bold", pad=10, color="#333")

    level = 63278.55
    margin = 556.89
    thr = level + margin

    bars = [
        (-2, "07-16 12h", 64256.52, 64896.00, 63748.74, 63830.20),
        (-1, "07-17 00h", 63830.20, 64067.69, 62666.00, 63298.01),
        (0,  "07-17 12h", 63298.00, 64387.99, 62537.56, 63931.67),
        (1,  "07-18 00h", 63931.67, 64097.22, 63886.65, 64069.89),
        (2,  "07-18 12h", 64069.89, 64865.00, 63963.00, 64834.22),
    ]
    x0 = 3.6
    xs = {k: x0 + (k + 2) * 1.35 for k, *_ in bars}
    x_fire = xs[0]

    ax.plot([x0 - 1.3, xs[2] + 0.9], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(xs[2] + 1.05, level, f"THMA_prev={level:,.0f}", fontsize=8.4,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0 - 1.3, xs[2] + 0.9], [thr, thr],
            ls=(0, (2, 2)), color=ACCENT, lw=1.4, alpha=0.85)
    ax.text(xs[2] + 1.05, thr, f"+margin={thr:,.0f}", fontsize=8.0,
            color=ACCENT, va="center", fontweight="bold")

    ax.text(xs[-2] - 1.1, 65100, "THMA-9(12h) на баре i-1\n(2026-07-17 00h)",
            fontsize=7.8, color=LEVEL_C, ha="left", style="italic")

    for k, lbl, o, h, l, c in bars:
        x = xs[k]
        is_fire = (k == 0)
        draw_candle(ax, x, o, h, l, c, w=0.75,
                    edge=FIRE_C if is_fire else "#333",
                    fill_up="white", fill_dn="#2a2a2a",
                    lw=1.6 if is_fire else 1.0)
        ax.text(x, l - 260, lbl, fontsize=7.6, color="#555", ha="center", rotation=0)

    ax.plot(x_fire, 62200, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire, 61850, "A1-pivot LOW (12h, l[i]<l[i-1] и l[i]<l[i-2])", fontsize=7.6,
            color=WARN, ha="center", fontweight="bold")

    ax.plot(x_fire, 65300, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 65600, "FIRE  (i, 2026-07-17 12:00)", fontsize=9, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate(f"low={62537.56:,.0f} < THMA_prev={level:,.0f}",
                xy=(x_fire - 0.05, 62537.56), xytext=(xs[-2] - 1.9, 62050),
                fontsize=8.2, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3), ha="left", va="center")
    ax.annotate(f"close={63931.67:,.0f} > {thr:,.0f}",
                xy=(x_fire, 63931.67), xytext=(xs[1] - 0.1, 64550),
                fontsize=8.4, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3), ha="left", va="center")

    ax.annotate("Williams n=2 confirm: l[i+1]>l[i] и l[i+2]>l[i]  \u2192  CONFIRMED",
                xy=(xs[2], 63963.00), xytext=(xs[2] + 0.4, 64500),
                fontsize=8.3, color="#006400", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#006400", lw=1.2), ha="left", va="center")

    ax.set_xlim(x0 - 3.4, xs[2] + 3.2)
    ax.set_ylim(61600, 65900)
    ax.set_xlabel("12h bars \u2192  (реальные даты BTC)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.92, "Формальное правило (FULL_DISP, одна TF)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "margin = 0.5 \u00d7 ATR14(12h)[i]\n\n"
        "SHORT:  high[i] > THMA_prev   AND   close[i] < THMA_prev \u2212 margin\n"
        "LONG:   low[i]  < THMA_prev   AND   close[i] > THMA_prev + margin\n\n"
        "b4c3 = fire(THMA-9, 12h)  \u2014  ОДНА TF, тот же ряд, что и pivot\n"
        "  (prev = бар i-1 напрямую, без cross-TF day-lookup, в отличие от C4-C6).\n\n"
        "THMA (Triple Hull MA) = WMA(2\u00d7WMA(src,n/2)\u2212WMA(src,n), \u221an), где внутренние\n"
        "  WMA считаются с длинами n/3, n/2, n \u2014 см. trend_line_asvk.py.\n\n"
        "Домен: a_cand[a124_pool] = A1+A2+A4, БЕЗ A3. Как весь B3/B4."
    )
    ax.text(0.04, 0.80, rule_text, fontsize=9.3, color="#222",
            ha="left", va="top", transform=ax.transAxes,
            family="DejaVu Sans Mono")


def _draw_notes_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#eef6ff", edgecolor="#0057b8",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.92, "Canon-заметки \u00b7 история находки",
            ha="center", fontsize=12, fontweight="bold", color="#0057b8",
            transform=ax.transAxes)
    notes_text = (
        "\u2022  B4C3 отсутствовал в WSL-каноне \u2014 найден жадным отбором поверх\n"
        "   HMA-78/HMA-200 (G:\\Claude\\research\\ma_family_explore.py, 6 типов MA\n"
        "   \u00d7 7 длин \u00d7 5 TF на BTC, кросс-проверка на 10 активах).\n\n"
        "\u2022  Самый большой объём во всей серии B4 (n=156 на BTC) \u2014 короткая\n"
        "   длина (9) реагирует чаще остальных, но FULL_DISP держит WR на\n"
        "   уровне остальных условий (74-88% по 11 активам).\n\n"
        "\u2022  ВАЖНО: C3-C6 отобраны жадным алгоритмом на BTC \u2014 при совместной\n"
        "   проверке всех 4-х разом WR вырос лишь на 5/10 активов (см. докстринг\n"
        "   b4_hma.py). Цифры на этой картинке \u2014 ОТДЕЛЬНАЯ per-candidate\n"
        "   проверка (b4c3c6_cross_asset.py), не бандл \u2014 обе валидны, но не\n"
        "   взаимозаменяемы (маргинальная ценность != собственный WR).\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b4_hma.py  \u00b7  G:\\ASVK\\lib\\trendline.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=8.9, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B4C3.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
