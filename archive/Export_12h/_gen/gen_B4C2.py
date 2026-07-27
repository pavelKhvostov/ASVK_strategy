"""B4C2 rule (production, в basket) — HMA-200(D) FULL_DISP · Hull MA · TF=D.

Правило (источник истины G:\\ASVK\\lib\\fractal12h\\b4_hma.py):
    margin = 0.5 × ATR14(12h)[i]
    SHORT: high[i] > HMA_prev   AND  close[i] < HMA_prev - margin
    LONG:  low[i]  < HMA_prev   AND  close[i] > HMA_prev + margin
    Одна TF — HMA-200(D), без multi-TF union (в отличие от B4C1).

    Изначально B4C2 работал на механике SWEEP (просто касание+закрытие по сторону,
    WR 59-70% на BTC/ETH/SOL). Тот же research, что нашёл FULL_DISP для B4C1,
    применён и здесь: механика заменена 2026-07-25, индикатор (HMA-200 D) НЕ
    менялся — тот же trendline.py, тот же variant D200.

    Домен: a_cand[a124_pool] — A1+A2+A4, БЕЗ A3, как весь B3/B4.

Кросс-проверка (n / WR, все 11 активов) — до/после замены механики:
    BTC:  SWEEP n=71 WR=70.42%  \u2192  FULL_DISP n=26 WR=88.46%
    ETH:  SWEEP n=58 WR=65.52%  \u2192  FULL_DISP n=14 WR=85.71%
    SOL:  SWEEP n=61 WR=59.02%  \u2192  FULL_DISP n=18 WR=83.33%
    + 8 активов на FULL_DISP: ADA 90.0% AVAX 70.0% BNB 66.7% DOGE 87.5%
    DOT 76.5% LINK 79.2% LTC 90.0% XRP 76.0%

Правая (LONG) панель — РЕАЛЬНЫЙ пример: BTC, HMA-200(D), сигнал зажёгся
2025-07-25 00:00 UTC, Williams n=2 подтверждение по i+1/i+2 видно на графике.

Refs:
  G:\\ASVK\\lib\\fractal12h\\b4_hma.py       — реализация
  G:\\ASVK\\lib\\trendline.py                — источник HMA (variant D200)
  G:\\Claude\\research\\ma_family_explore.py — источник механики/скана

Output:
  G:\\ASVK\\Export_12h\\B4C2.png
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
    ax.text(0.5, 0.68, "B4C2 — HMA-200(D) FULL_DISP \u00b7 Hull MA \u00b7 TF=D",
            ha="center", va="center", fontsize=21, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "PRODUCTION (в basket)  \u00b7  механика заменена SWEEP\u2192FULL_DISP 2026-07-25  \u00b7  "
            "домен a124_pool",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")
    ax.text(0.5, 0.10,
            "BTC: 70.42%\u219288.46%  \u00b7  ETH: 65.52%\u219285.71%  \u00b7  SOL: 59.02%\u219283.33%  (SWEEP\u2192FULL_DISP)",
            ha="center", va="center", fontsize=10, color="#9aa0c0",
            transform=ax.transAxes)

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel_real(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup (схема)  \u00b7  A1-pivot LOW \u2192 h[i]>HMA_prev, c[i]<HMA_prev-margin",
                 fontsize=11.6, fontweight="bold", pad=10, color="#333")

    level = 101.0
    margin = 1.6
    x0, x_fire = 1.0, 8.5

    ax.plot([x0, x_fire + 1.5], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(x_fire + 1.7, level, "HMA-200(D)_prev", fontsize=9.0,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0, x_fire + 1.5], [level - margin, level - margin],
            ls=(0, (2, 2)), color=WARN, lw=1.4, alpha=0.85)
    ax.text(x_fire + 1.7, level - margin, "\u2212margin\n(0.5\u00d7ATR14)", fontsize=8,
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
    ax.text(x0 + 0.3, 95.2, "A1-pivot LOW\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    draw_candle(ax, x_fire, o=101.0, h=103.4, l=97.5, c=98.4,
                w=0.6, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 104.1, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 104.8, "FIRE", fontsize=9.5, fontweight="bold", color="#8a5a00", ha="center")

    ax.annotate("high > HMA_prev\n(прокол уровня вверх)",
                xy=(x_fire + 0.02, 103.35), xytext=(4.7, 107.2),
                fontsize=9.0, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4), ha="left", va="center")
    ax.annotate("close < HMA_prev \u2212 margin\n(решительный уход, не полу-мера)",
                xy=(x_fire + 0.05, 98.4), xytext=(1.0, 93.6),
                fontsize=9.0, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4), ha="left", va="center")

    ax.set_xlim(-0.4, 12.7)
    ax.set_ylim(92.0, 109.0)
    ax.set_xlabel("12h bars \u2192", fontsize=9.5)
    ax.set_ylabel("price (условно)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel_real(ax) -> None:
    ax.set_title("LONG setup \u2014 РЕАЛЬНЫЙ пример: BTC 2025-07-25 00:00 UTC (confirmed)",
                 fontsize=11.6, fontweight="bold", pad=10, color="#333")

    level = 114957.93
    margin = 1085.20
    thr = level + margin

    bars = [
        (-2, "07-24 00h", 118756.00, 119273.36, 117103.10, 118600.00),
        (-1, "07-24 12h", 118600.00, 119450.00, 117832.32, 118340.99),
        (0,  "07-25 00h", 118340.98, 118451.57, 114723.16, 116543.58),
        (1,  "07-25 12h", 116543.59, 117630.00, 114908.00, 117614.31),
        (2,  "07-26 00h", 117614.31, 118217.45, 117138.38, 117776.11),
    ]
    x0 = 3.6
    xs = {k: x0 + (k + 2) * 1.35 for k, *_ in bars}
    x_fire = xs[0]

    ax.plot([x0 - 1.3, xs[2] + 0.9], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(xs[2] + 1.05, level, f"HMA-200(D)_prev={level:,.0f}", fontsize=8.0,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0 - 1.3, xs[2] + 0.9], [thr, thr],
            ls=(0, (2, 2)), color=ACCENT, lw=1.4, alpha=0.85)
    ax.text(xs[2] + 1.05, thr, f"+margin={thr:,.0f}", fontsize=8.0,
            color=ACCENT, va="center", fontweight="bold")

    ax.text(xs[-2] - 1.1, 119900, "HMA-200(D) на предыдущем\nзакрытом дне (2025-07-24)",
            fontsize=7.8, color=LEVEL_C, ha="left", style="italic")

    for k, lbl, o, h, l, c in bars:
        x = xs[k]
        is_fire = (k == 0)
        draw_candle(ax, x, o, h, l, c, w=0.75,
                    edge=FIRE_C if is_fire else "#333",
                    fill_up="white", fill_dn="#2a2a2a",
                    lw=1.6 if is_fire else 1.0)
        ax.text(x, l - 340, lbl, fontsize=7.6, color="#555", ha="center", rotation=0)

    ax.plot(x_fire, 113300, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire, 112700, "A1-pivot LOW (12h, l[i]<l[i-1] и l[i]<l[i-2])", fontsize=7.6,
            color=WARN, ha="center", fontweight="bold")

    ax.plot(x_fire, 119200, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 119600, "FIRE  (i, 2025-07-25 00:00)", fontsize=9, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate(f"low={114723.16:,.0f} < HMA_prev={level:,.0f}",
                xy=(x_fire - 0.05, 114723.16), xytext=(xs[-2] - 1.9, 113500),
                fontsize=8.2, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3), ha="left", va="center")
    ax.annotate(f"close={116543.58:,.0f} > {thr:,.0f}",
                xy=(x_fire, 116543.58), xytext=(xs[1] - 0.1, 118700),
                fontsize=8.4, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3), ha="left", va="center")

    ax.annotate("Williams n=2 confirm: l[i+1]>l[i] и l[i+2]>l[i]  \u2192  CONFIRMED",
                xy=(xs[2], 117138.38), xytext=(xs[2] + 0.4, 117900),
                fontsize=8.3, color="#006400", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#006400", lw=1.2), ha="left", va="center")

    ax.set_xlim(x0 - 3.4, xs[2] + 3.2)
    ax.set_ylim(112300, 121200)
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
        "SHORT:  high[i] > HMA_prev   AND   close[i] < HMA_prev \u2212 margin\n"
        "LONG:   low[i]  < HMA_prev   AND   close[i] > HMA_prev + margin\n\n"
        "b4c2 = fire(HMA-200, D)  \u2014  ОДНА TF, без multi-TF union\n"
        "  (в отличие от b4c1, который OR-ит 12h и D).\n\n"
        "HMA_prev \u2014 значение mhull ПРЕДЫДУЩЕГО уже закрытого D-бара\n"
        "  (день ДО дня pivot-бара \u2014 Правило LIVE, тот же принцип, что и раньше,\n"
        "  сам indicator не менялся).\n\n"
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
    ax.text(0.5, 0.92, "Canon-заметки \u00b7 история правки",
            ha="center", fontsize=12, fontweight="bold", color="#0057b8",
            transform=ax.transAxes)
    notes_text = (
        "\u2022  Изначально B4C2 работал на SWEEP (простое касание+закрытие) \u2014\n"
        "   WR 59-70% на BTC/ETH/SOL, самый слабый блок в корзине.\n\n"
        "\u2022  Та же находка, что и для B4C1 (research по MA-family): FULL_DISP\n"
        "   вместо SWEEP \u2014 индикатор HMA-200(D) НЕ менялся, поменялось только\n"
        "   условие подтверждения.\n\n"
        "\u2022  Эффект самый сильный из всей серии: +18\u2026+24 п.п. WR на BTC/\n"
        "   ETH/SOL, ценой объёма (n падает на 63-76%, т.к. половинчатые\n"
        "   срабатывания SWEEP отсеиваются).\n\n"
        "\u2022  Правка боевая: b4_hit = b4c1 OR b4c2, оба входят в basket_hit\n"
        "   напрямую. _sweep() удалён из b4_hma.py как мёртвый код.\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b4_hma.py  \u00b7  G:\\ASVK\\lib\\trendline.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=8.9, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B4C2.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
