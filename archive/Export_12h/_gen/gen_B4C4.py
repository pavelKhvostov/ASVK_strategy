"""B4C4 rule (production, в basket) — WMA-50(D) FULL_DISP · плоская WMA (без Hull) · TF=D.

Правило (источник истины G:\\ASVK\\lib\\fractal12h\\b4_hma.py):
    margin = 0.5 × ATR14(12h)[i]
    SHORT: high[i] > WMA_prev   AND  close[i] < WMA_prev - margin
    LONG:  low[i]  < WMA_prev   AND  close[i] > WMA_prev + margin
    Одна TF — WMA-50(D), предыдущий уже закрытый D-бар (cross-TF lookup, как B4C2).

    Единственный кандидат серии C3-C6 БЕЗ Hull-обёртки (плоская WMA) — жадный отбор
    показал, что классическая скользящая на длине 50 добавляет независимый объём
    поверх HMA-78/200/THMA-9, не дублируя их сигналы (2026-07-25).

    Домен: a_cand[a124_pool] — A1+A2+A4, БЕЗ A3, как весь B3/B4.

Кросс-проверка (n / WR, ОТДЕЛЬНО по каждому активу — не бандл с C3/C5/C6):
    BTC n=38 WR=97.37%   ETH n=28 WR=78.57%   SOL n=27 WR=70.37%
    ADA n=26 WR=80.77%   AVAX n=28 WR=71.43%  BNB n=36 WR=80.56%
    DOGE n=41 WR=78.05%  DOT n=28 WR=64.29%   LINK n=24 WR=87.50%
    LTC n=41 WR=87.80%   XRP n=40 WR=85.00%   pooled n=357
    (пересчитано отдельно от bundle-проверки C3-C6, см. G:\\Claude\\research\\
    b4c3c6_cross_asset.py / b4c3c6_per_asset.parquet; самый маленький объём и самый
    широкий разброс WR по активам во всей серии B4 — 64-97%)

Правая (LONG) панель — РЕАЛЬНЫЙ пример: BTC, WMA-50(D), сигнал зажёгся
2026-01-08 12:00 UTC, Williams n=2 подтверждение по i+1/i+2 видно на графике.

Refs:
  G:\\ASVK\\lib\\fractal12h\\b4_hma.py       — реализация
  G:\\ASVK\\lib\\trendline.py                — источник WMA (variant D50Wma, --mode Wma)
  G:\\Claude\\research\\ma_family_explore.py — источник механики/скана
  G:\\Claude\\research\\b4c3c6_cross_asset.py — per-candidate cross-asset проверка

Output:
  G:\\ASVK\\Export_12h\\B4C4.png
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
    ax.text(0.5, 0.68, "B4C4 — WMA-50(D) FULL_DISP \u00b7 плоская WMA (без Hull) \u00b7 TF=D",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "PRODUCTION (в basket, развёрнуто 2026-07-25)  \u00b7  домен A1+A2+A4 без A3 (a124_pool)",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")
    ax.text(0.5, 0.10,
            "BTC n=38 WR=97.37%  \u00b7  ETH n=28 WR=78.57%  \u00b7  SOL n=27 WR=70.37%  \u00b7  "
            "+ 8 активов, диапазон 64-97% (самый широкий во всей серии)",
            ha="center", va="center", fontsize=10, color="#9aa0c0",
            transform=ax.transAxes)

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel_real(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup (схема)  \u00b7  A1-pivot HIGH \u2192 h[i]>WMA_prev, c[i]<WMA_prev-margin",
                 fontsize=11.6, fontweight="bold", pad=10, color="#333")

    level = 101.0
    margin = 1.6
    x0, x_fire = 1.0, 8.5

    ax.plot([x0, x_fire + 1.5], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(x_fire + 1.7, level, "WMA_prev", fontsize=9.5,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0, x_fire + 1.5], [level - margin, level - margin],
            ls=(0, (2, 2)), color=WARN, lw=1.4, alpha=0.85)
    ax.text(x_fire + 1.7, level - margin, "WMA_prev \u2212 margin\n(0.5\u00d7ATR14)", fontsize=8,
            color=WARN, va="center", fontweight="bold")

    draw_candle(ax, x0, o=99.5, h=100.4, l=98.7, c=100.0, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    draw_candle(ax, x0 + 1.1, o=100.0, h=100.9, l=99.3, c=100.6, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    ax.text(x0 + 0.5, 97.9, "предыдущий закрытый D-бар", fontsize=7.6,
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

    ax.annotate("high > WMA_prev\n(прокол уровня вверх)",
                xy=(x_fire + 0.02, 103.35), xytext=(4.7, 107.2),
                fontsize=9.0, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4), ha="left", va="center")
    ax.annotate("close < WMA_prev \u2212 margin\n(решительный уход, не полу-мера)",
                xy=(x_fire + 0.05, 98.4), xytext=(1.0, 93.6),
                fontsize=9.0, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4), ha="left", va="center")

    ax.set_xlim(-0.4, 12.7)
    ax.set_ylim(92.0, 109.0)
    ax.set_xlabel("12h bars \u2192", fontsize=9.5)
    ax.set_ylabel("price (условно)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel_real(ax) -> None:
    ax.set_title("LONG setup \u2014 РЕАЛЬНЫЙ пример: BTC 2026-01-08 12:00 UTC (confirmed)",
                 fontsize=11.6, fontweight="bold", pad=10, color="#333")

    level = 89367.66
    margin = 881.30
    thr = level + margin

    bars = [
        (-2, "01-07 12h", 92087.21, 92299.40, 90675.52, 91364.16),
        (-1, "01-08 00h", 91364.16, 91687.99, 89641.84, 90226.77),
        (0,  "01-08 12h", 90226.77, 91493.00, 89311.00, 91099.99),
        (1,  "01-09 00h", 91100.00, 91632.10, 89694.66, 90463.11),
        (2,  "01-09 12h", 90463.12, 92082.55, 89850.00, 90641.28),
    ]
    x0 = 3.6
    xs = {k: x0 + (k + 2) * 1.35 for k, *_ in bars}
    x_fire = xs[0]

    ax.plot([x0 - 1.3, xs[2] + 0.9], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(xs[2] + 1.05, level, f"WMA-50(D)_prev={level:,.0f}", fontsize=8.0,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0 - 1.3, xs[2] + 0.9], [thr, thr],
            ls=(0, (2, 2)), color=ACCENT, lw=1.4, alpha=0.85)
    ax.text(xs[2] + 1.05, thr, f"+margin={thr:,.0f}", fontsize=8.0,
            color=ACCENT, va="center", fontweight="bold")

    ax.text(xs[-2] - 1.1, 92700, "WMA-50(D) на предыдущем\nзакрытом дне (2026-01-07)",
            fontsize=7.8, color=LEVEL_C, ha="left", style="italic")

    for k, lbl, o, h, l, c in bars:
        x = xs[k]
        is_fire = (k == 0)
        draw_candle(ax, x, o, h, l, c, w=0.75,
                    edge=FIRE_C if is_fire else "#333",
                    fill_up="white", fill_dn="#2a2a2a",
                    lw=1.6 if is_fire else 1.0)
        ax.text(x, l - 300, lbl, fontsize=7.6, color="#555", ha="center", rotation=0)

    ax.plot(x_fire, 88700, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire, 88350, "A1-pivot LOW (12h, l[i]<l[i-1] и l[i]<l[i-2])", fontsize=7.6,
            color=WARN, ha="center", fontweight="bold")

    ax.plot(x_fire, 92900, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 93250, "FIRE  (i, 2026-01-08 12:00)", fontsize=9, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate(f"low={89311.00:,.0f} < WMA_prev={level:,.0f}",
                xy=(x_fire - 0.05, 89311.00), xytext=(xs[-2] - 1.9, 88550),
                fontsize=8.2, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3), ha="left", va="center")
    ax.annotate(f"close={91099.99:,.0f} > {thr:,.0f}",
                xy=(x_fire, 91099.99), xytext=(xs[1] - 0.1, 92200),
                fontsize=8.4, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3), ha="left", va="center")

    ax.annotate("Williams n=2 confirm: l[i+1]>l[i] и l[i+2]>l[i]  \u2192  CONFIRMED",
                xy=(xs[2], 89850.00), xytext=(xs[2] + 0.4, 90600),
                fontsize=8.3, color="#006400", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#006400", lw=1.2), ha="left", va="center")

    ax.set_xlim(x0 - 3.4, xs[2] + 3.2)
    ax.set_ylim(88200, 93600)
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
        "SHORT:  high[i] > WMA_prev   AND   close[i] < WMA_prev \u2212 margin\n"
        "LONG:   low[i]  < WMA_prev   AND   close[i] > WMA_prev + margin\n\n"
        "b4c4 = fire(WMA-50, D)  \u2014  ОДНА TF, cross-TF lookup (как b4c2)\n\n"
        "WMA_prev \u2014 значение WMA ПРЕДЫДУЩЕГО уже закрытого D-бара (день ДО дня\n"
        "  pivot-бара \u2014 LIVE-safe, тот же принцип, что у b4c2/D200).\n\n"
        "WMA(src,n) \u2014 линейно-взвешенное среднее (без Hull-обёртки), n=50.\n"
        "  Единственный из C3-C6 БЕЗ Hull-производной \u2014 см. trend_line_asvk.py.\n\n"
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
        "\u2022  Единственный кандидат серии C3-C6 без Hull-обёртки \u2014 плоская\n"
        "   WMA-50(D), найдена жадным отбором как независимая (не дублирующая\n"
        "   HMA-78/200/THMA-9) добавка к объёму (2026-07-25).\n\n"
        "\u2022  Самый маленький объём во всей серии B4 (n=38 на BTC) и самый широкий\n"
        "   разброс WR по 11 активам (64-97%) \u2014 меньше выборка на каждый\n"
        "   актив, чем у C3/C5/C6, оценка менее устойчива статистически.\n\n"
        "\u2022  ВАЖНО: C3-C6 отобраны жадным алгоритмом на BTC \u2014 при совместной\n"
        "   проверке всех 4-х разом WR вырос лишь на 5/10 активов (см. докстринг\n"
        "   b4_hma.py). Цифры на этой картинке \u2014 ОТДЕЛЬНАЯ per-candidate\n"
        "   проверка (b4c3c6_cross_asset.py), не бандл.\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b4_hma.py  \u00b7  G:\\ASVK\\lib\\trendline.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=8.9, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B4C4.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
