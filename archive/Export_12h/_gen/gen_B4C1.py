"""B4C1 rule (production, в basket) — HMA-78(12h∪D) FULL_DISP · Hull MA · TF=12h∪D.

Правило (источник истины G:\\ASVK\\lib\\fractal12h\\b4_hma.py):
    margin = 0.5 × ATR14(12h)[i]
    SHORT: high[i] > HMA_prev   AND  close[i] < HMA_prev - margin
    LONG:  low[i]  < HMA_prev   AND  close[i] > HMA_prev + margin
    Multi-TF OR: HMA-78(12h) ∪ HMA-78(D) — каждая TF даёт свой независимый fire,
    объединяются как b4c1 = fire_12h OR fire_d.

    Канонически B4C1 определялся как SWEEP (просто касание+закрытие по сторону),
    но с этой механикой был слишком слаб (WR 57-64% на 11 активах) и никогда не
    переносился в ASVK. Разносторонний research (block-family, затем MA-family scan,
    G:\\Claude\\research\\ma_family_explore.py) показал: слабость была в механике,
    не в индикаторе. С FULL_DISP та же HMA-78 даёт WR 73-84% на всех 11 активах.
    Развёрнуто в ASVK 2026-07-25.

    Домен: a_cand[a124_pool] — A1+A2+A4, БЕЗ A3, как весь B3/B4.

Кросс-проверка (n / WR, все 11 активов):
    BTC n=89 WR=78.65%   ETH n=60 WR=81.67%   SOL n=63 WR=80.95%
    ADA n=58 WR=84.48%   AVAX n=59 WR=72.88%  BNB n=66 WR=81.82%
    DOGE n=64 WR=75.00%  DOT n=62 WR=79.03%   LINK n=68 WR=76.47%
    LTC n=81 WR=81.25%   XRP n=67 WR=82.09%

Правая (LONG) панель — РЕАЛЬНЫЙ пример: BTC, HMA-78(12h), сигнал зажёгся
2026-04-20 00:00 UTC, Williams n=2 подтверждение по i+1/i+2 видно на графике.

Refs:
  G:\\ASVK\\lib\\fractal12h\\b4_hma.py       — реализация
  G:\\ASVK\\lib\\trendline.py                — источник HMA (variant 12h78, D78)
  G:\\Claude\\research\\ma_family_explore.py — источник механики/скана

Output:
  G:\\ASVK\\Export_12h\\B4C1.png
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
GRN_EDGE = "#006400"


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
    ax.text(0.5, 0.68, "B4C1 — HMA-78(12h\u222aD) FULL_DISP \u00b7 Hull MA \u00b7 TF=12h\u222aD",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "PRODUCTION (в basket, развёрнуто 2026-07-25)  \u00b7  домен A1+A2+A4 без A3 (a124_pool)",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")
    ax.text(0.5, 0.10,
            "BTC n=89 WR=78.65%  \u00b7  ETH n=60 WR=81.67%  \u00b7  SOL n=63 WR=80.95%  \u00b7  "
            "+ 8 активов, диапазон 73-84%",
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
    ax.text(x_fire + 1.7, level, "HMA_prev", fontsize=9.5,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0, x_fire + 1.5], [level - margin, level - margin],
            ls=(0, (2, 2)), color=WARN, lw=1.4, alpha=0.85)
    ax.text(x_fire + 1.7, level - margin, "HMA_prev \u2212 margin\n(0.5\u00d7ATR14)", fontsize=8,
            color=WARN, va="center", fontweight="bold")

    draw_candle(ax, x0, o=99.5, h=100.4, l=98.7, c=100.0, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    draw_candle(ax, x0 + 1.1, o=100.0, h=100.9, l=99.3, c=100.6, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    ax.text(x0 + 0.5, 97.9, "бар i-1  (формирует HMA_prev)", fontsize=7.6,
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
    ax.set_title("LONG setup \u2014 РЕАЛЬНЫЙ пример: BTC 2026-04-20 00:00 UTC (confirmed)",
                 fontsize=11.6, fontweight="bold", pad=10, color="#333")

    level = 74086.40
    margin = 936.00
    thr = level + margin

    bars = [
        (-2, "04-19 00h", 75691.76, 75847.42, 74867.72, 75604.11),
        (-1, "04-19 12h", 75604.11, 76240.66, 73762.90, 73801.79),
        (0,  "04-20 00h", 73801.80, 75572.00, 73724.31, 75192.71),
        (1,  "04-20 12h", 75192.72, 76558.62, 74702.00, 75840.97),
        (2,  "04-21 00h", 75840.97, 76927.57, 75474.77, 76452.94),
    ]
    x0 = 3.6
    xs = {k: x0 + (k + 2) * 1.35 for k, *_ in bars}
    x_fire = xs[0]

    ax.plot([x0 - 1.3, xs[2] + 0.9], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(xs[2] + 1.05, level, f"HMA_prev={level:,.0f}", fontsize=8.4,
            color=LEVEL_C, va="center", fontweight="bold")

    ax.plot([x0 - 1.3, xs[2] + 0.9], [thr, thr],
            ls=(0, (2, 2)), color=ACCENT, lw=1.4, alpha=0.85)
    ax.text(xs[2] + 1.05, thr, f"+margin={thr:,.0f}", fontsize=8.0,
            color=ACCENT, va="center", fontweight="bold")

    ax.text(xs[-2] - 1.1, 77400, "HMA-78(12h) на баре i-1\n(2026-04-19 12h)",
            fontsize=7.8, color=LEVEL_C, ha="left", style="italic")

    for k, lbl, o, h, l, c in bars:
        x = xs[k]
        is_fire = (k == 0)
        draw_candle(ax, x, o, h, l, c, w=0.75,
                    edge=FIRE_C if is_fire else "#333",
                    fill_up="white", fill_dn="#2a2a2a",
                    lw=1.6 if is_fire else 1.0)
        ax.text(x, l - 280, lbl, fontsize=7.6, color="#555", ha="center", rotation=0)

    ax.plot(x_fire, 72700, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire, 72300, "A1-pivot LOW (12h, l[i]<l[i-1] и l[i]<l[i-2])", fontsize=7.6,
            color=WARN, ha="center", fontweight="bold")

    ax.plot(x_fire, 77000, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 77400, "FIRE  (i, 2026-04-20 00:00)", fontsize=9, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate(f"low={73724.31:,.0f} < HMA_prev={level:,.0f}",
                xy=(x_fire - 0.05, 73724.31), xytext=(xs[-2] - 1.9, 72900),
                fontsize=8.4, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3), ha="left", va="center")
    ax.annotate(f"close={75192.71:,.0f} > {thr:,.0f}",
                xy=(x_fire, 75192.71), xytext=(xs[1] - 0.1, 76600),
                fontsize=8.6, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3), ha="left", va="center")

    ax.annotate("Williams n=2 confirm: l[i+1]>l[i] и l[i+2]>l[i]  \u2192  CONFIRMED",
                xy=(xs[2], 75474.77), xytext=(xs[2] + 0.4, 76000),
                fontsize=8.3, color="#006400", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#006400", lw=1.2), ha="left", va="center")

    ax.set_xlim(x0 - 3.4, xs[2] + 3.2)
    ax.set_ylim(72000, 78200)
    ax.set_xlabel("12h bars \u2192  (реальные даты BTC)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.92, "Формальное правило (FULL_DISP, multi-TF OR)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "margin = 0.5 \u00d7 ATR14(12h)[i]\n\n"
        "SHORT:  high[i] > HMA_prev   AND   close[i] < HMA_prev \u2212 margin\n"
        "LONG:   low[i]  < HMA_prev   AND   close[i] > HMA_prev + margin\n\n"
        "b4c1 = fire(HMA-78, 12h)  OR  fire(HMA-78, D)  \u2014 каждая TF независимо,\n"
        "  объединяются через OR (multi-TF union, как у B1/B2).\n\n"
        "HMA_prev \u2014 значение на ПРЕДЫДУЩЕМ уже закрытом баре нужной TF (LIVE):\n"
        "  для 12h \u2014 бар i-1 того же ряда; для D \u2014 предыдущий закрытый день.\n\n"
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
        "\u2022  Канонический B4C1 = HMA-78 SWEEP (просто касание) никогда не\n"
        "   переносился в ASVK \u2014 WR 57-64% на 11 активах, слишком слабо.\n\n"
        "\u2022  Разносторонний research (block-family, затем MA-family) нашёл\n"
        "   общий паттерн: FULL_DISP (решительный уход close за уровень на\n"
        "   0.5\u00d7ATR14) стабильно бьёт SWEEP \u2014 не только на HMA, но и на\n"
        "   block_orders/ob_liq (B2), mitigation_block/rb (research).\n\n"
        "\u2022  С FULL_DISP та же HMA-78 даёт WR 73-84% на всех 11 активах \u2014\n"
        "   развёрнуто в b4_hma.py 2026-07-25, без изменения самого\n"
        "   индикатора (trendline.py/trend_line_asvk.py не тронуты).\n\n"
        "\u2022  b4_hit = b4c1 OR b4c2 (оба теперь FULL_DISP, оба в basket).\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b4_hma.py  \u00b7  G:\\ASVK\\lib\\trendline.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=8.9, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B4C1.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
