"""B2C3 (КАНДИДАТ, не в production/basket) — FULL_DISP · Mitigation Block · TF=12h.

Найдено в разностороннем скане block-класса (ob, breaker_block, mitigation_block, rb;
5 механик x 11 TF, см. G:\\Claude\\research\\block_family_explore.py). SWEEP50 (канон
C1/C2) на этих элементах слабее — новая механика FULL_DISP (полное вытеснение цены на
целую ширину зоны за БЛИЖНИЙ край, а не мера-к-mid) даёт стабильно высокий WR именно
на TF=12h (multi-TF union здесь ХУЖЕ, чем чистый 12h — в отличие от C1/C2).

Правило (b2_ob.py пока НЕ содержит эту механику — это research-находка,
G:\\Claude\\research\\block_family_explore.py::mech_full_disp):
    w = zone_hi - zone_lo
    SHORT: high[k] >= zone_hi   AND  close[k] < zone_lo - w
    LONG:  low[k]  <= zone_lo   AND  close[k] > zone_hi + w
    Zone источник: mitigation_block (lib/детекторы/mitigation_block.py) — полностью
    пробитый OB + Правило 1, zone = drop/rally area над/под activator.
    TF: 12h ТОЛЬКО (не multi-TF — проверено, union 12h+1d+2d+3d размывает WR).
    Домен: чистый A1 pre-w pool, как весь B2.

Кросс-проверка (n / WR, FULL_DISP@12h):
    BTC  n=65  WR=87.69%      ETH  n=68  WR=83.82%      SOL  n=50  WR=88.00%
  (для сравнения B2C1=70-81%, B2C2=63-75% на тех же активах)

Правый (LONG) панель — РЕАЛЬНЫЙ пример, не схема: BTC, зона mitigation_block
родилась 2024-09-20 12:00 UTC, сигнал зажёгся 2026-06-05 12:00 UTC (i-бар), Williams
n=2 подтверждение по барам i+1/i+2 — оба уже видны на графике.

Refs:
  G:\\Claude\\research\\block_family_explore.py  — источник механики/скана
  G:\\ASVK\\lib\\fractal12h\\b2_ob.py              — стиль вёрстки (эталон gen_B2C1.py)
  G:\\ASVK\\lib\\детекторы\\mitigation_block.py     — источник зон

Output:
  G:\\ASVK\\Export_12h\\B2C3_candidate_mitigation_FULL_DISP.png
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
FAR_C    = "#7a2ba0"


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
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.62, 4, 2.4],
                            hspace=0.30, wspace=0.14,
                            left=0.045, right=0.985, top=0.965, bottom=0.045)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#1c1c2e", edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.66, "B2C3 (кандидат) — FULL_DISP \u00b7 Mitigation Block \u00b7 TF=12h",
            ha="center", va="center", fontsize=21, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.28,
            "НЕ в production/basket \u2014 research-находка (разносторонний скан block-класса)  \u00b7  "
            "BTC n=65 WR=87.69%  \u00b7  ETH n=68 WR=83.82%  \u00b7  SOL n=50 WR=88.00%",
            ha="center", va="center", fontsize=11.2, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel_real(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup (схема)  \u00b7  A1-pivot LOW \u2192 full displacement (resistance)",
                 fontsize=12.2, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 100.6, 102.5
    w = zone_hi - zone_lo
    x0, x_fire = 1.0, 8.5
    close_thr = zone_lo - w

    ax.add_patch(Rectangle((x0, zone_lo), (x_fire + 0.6) - x0, zone_hi - zone_lo,
                            facecolor=RED_FILL, edgecolor=RED_EDGE, linewidth=1.4,
                            alpha=0.85, zorder=1))
    ax.text(4.5, zone_hi + 0.35, "Mitigation Block zone (resistance)  \u00b7  TF=12h",
            fontsize=9.2, color="#800000", fontweight="bold", ha="center")

    ax.plot([x0, x_fire + 0.6], [close_thr, close_thr],
            ls=(0, (5, 2, 1, 2)), color=FAR_C, lw=1.4, alpha=0.9)
    ax.text(x_fire + 0.75, close_thr, "zone_lo \u2212 w\n(close-порог)", fontsize=8,
            color=FAR_C, va="center", fontweight="bold")

    ax.text(x_fire + 1.55, zone_hi, "zone_hi", fontsize=8.3, color=RED_EDGE, va="center", fontweight="bold")
    ax.text(x_fire + 1.55, zone_lo, "zone_lo", fontsize=8.3, color=RED_EDGE, va="center", fontweight="bold")
    ax.annotate("", xy=(x_fire + 1.35, zone_hi), xytext=(x_fire + 1.35, zone_lo),
                arrowprops=dict(arrowstyle="<->", color=RED_EDGE, lw=1.2))
    ax.text(x_fire + 1.9, (zone_hi + zone_lo) / 2, "w", fontsize=9, color=RED_EDGE,
            va="center", fontweight="bold")

    _draw_ellipsis(ax, x0 - 0.7, x0, (zone_lo + zone_hi) / 2)
    ax.text(x0 - 0.35, zone_hi + 1.1, "born давно\n(история)", fontsize=7.6,
            color="#800000", ha="center", style="italic")

    ax.plot(x0 + 0.3, 96.3, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x0 + 0.3, 95.5, "A1-pivot LOW\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    draw_candle(ax, x_fire, o=101.4, h=103.0, l=97.3, c=98.0,
                w=0.6, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 103.7, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 104.4, "FIRE", fontsize=9.5, fontweight="bold", color="#8a5a00", ha="center")

    ax.annotate("high \u2265 zone_hi\n(пробой ДАЛЬНЕГО края)",
                xy=(x_fire + 0.02, 102.95), xytext=(4.6, 106.5),
                fontsize=9.1, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4), ha="left", va="center")
    ax.annotate("close < zone_lo \u2212 w\n(вынос на целую ширину зоны)",
                xy=(x_fire + 0.05, 98.0), xytext=(1.0, 93.5),
                fontsize=9.1, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4), ha="left", va="center")

    ax.set_xlim(-0.4, 12.0)
    ax.set_ylim(91.5, 108.0)
    ax.set_xlabel("12h bars \u2192", fontsize=9.5)
    ax.set_ylabel("price (условно)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel_real(ax) -> None:
    ax.set_title("LONG setup \u2014 РЕАЛЬНЫЙ пример: BTC 2026-06-05 12:00 UTC (confirmed)",
                 fontsize=12.2, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 59993.02, 60395.80
    w = zone_hi - zone_lo
    close_thr = zone_hi + w

    bars = [
        (-2, "06-04 12h", 62546.00, 64494.92, 62392.00, 63885.99),
        (-1, "06-05 00h", 63885.99, 63978.00, 61126.01, 61964.99),
        (0,  "06-05 12h", 61964.99, 62457.86, 59130.91, 61056.47),
        (1,  "06-06 00h", 61056.47, 61530.05, 59500.00, 60802.91),
        (2,  "06-06 12h", 60802.90, 61185.26, 60393.96, 60884.62),
    ]
    x0 = 4.0
    xs = {k: x0 + (k + 2) * 1.35 for k, *_ in bars}
    x_fire = xs[0]

    ax.add_patch(Rectangle((x0 - 0.9, zone_lo), (xs[2] + 0.9) - (x0 - 0.9), w,
                            facecolor=GRN_FILL, edgecolor=GRN_EDGE, linewidth=1.4,
                            alpha=0.85, zorder=1))
    ax.text(xs[-1], zone_lo - 260, "Mitigation Block zone (support)\nborn 2024-09-20 12:00 UTC",
            fontsize=8.6, color="#004400", fontweight="bold", ha="left")

    ax.plot([x0 - 0.9, xs[2] + 0.9], [close_thr, close_thr],
            ls=(0, (5, 2, 1, 2)), color=FAR_C, lw=1.4, alpha=0.9)
    ax.text(xs[2] + 1.05, close_thr, f"zone_hi+w={close_thr:,.0f}\n(close-порог)", fontsize=7.8,
            color=FAR_C, va="center", fontweight="bold")

    _draw_ellipsis(ax, x0 - 2.2, x0 - 0.9, (zone_lo + zone_hi) / 2)

    for k, lbl, o, h, l, c in bars:
        x = xs[k]
        is_fire = (k == 0)
        draw_candle(ax, x, o, h, l, c, w=0.75,
                    edge=FIRE_C if is_fire else "#333",
                    fill_up="white", fill_dn="#2a2a2a",
                    lw=1.6 if is_fire else 1.0)
        ax.text(x, l - 210, lbl, fontsize=7.6, color="#555", ha="center", rotation=0)

    ax.plot(x_fire, 58650, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire, 58230, "A1-pivot LOW (12h,  l[i]<l[i-1] и l[i]<l[i-2])", fontsize=7.6,
            color=WARN, ha="center", fontweight="bold")

    ax.plot(x_fire, 63400, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 63750, "FIRE  (i, 2026-06-05 12:00)", fontsize=9, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate(f"low={59130.91:,.0f} ≤ zone_lo={zone_lo:,.0f}",
                xy=(x_fire - 0.05, 59200), xytext=(xs[-2] - 1.6, 58650),
                fontsize=8.4, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3), ha="left", va="center")
    ax.annotate(f"close={61056.47:,.0f} > {close_thr:,.0f}",
                xy=(x_fire, 61056.47), xytext=(xs[1] - 0.2, 63000),
                fontsize=8.6, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3), ha="left", va="center")

    ax.annotate("Williams n=2 confirm: l[i+1]>l[i] и l[i+2]>l[i]  →  CONFIRMED",
                xy=(xs[2], 60393.96), xytext=(xs[2] + 0.5, 61900),
                fontsize=8.3, color="#006400", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#006400", lw=1.2), ha="left", va="center")

    ax.set_xlim(x0 - 3.4, xs[2] + 3.0)
    ax.set_ylim(57700, 65200)
    ax.set_xlabel("12h bars \u2192  (реальные даты BTC)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.92, "Формальное правило FULL_DISP (первое qualifying k)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "w = zone_hi \u2212 zone_lo\n\n"
        "SHORT:  high[k] >= zone_hi   AND   close[k] < zone_lo \u2212 w\n"
        "LONG:   low[k]  <= zone_lo   AND   close[k] > zone_hi + w\n\n"
        "\u2014 в отличие от SWEEP50 (канон C1/C2, mid + close за БЛИЖНИМ краем),\n"
        "  здесь нужен пробой ДАЛЬНЕГО края + close на целую ширину зоны ЗА\n"
        "  ближним \u2014 полное вытеснение, не полу-мера.\n\n"
        "TF: 12h ТОЛЬКО. Проверено: union {12h,1d,2d,3d} снижает WR (разбавление\n"
        "  зонами младших TF) \u2014 в отличие от C1/C2, где multi-TF работает лучше.\n\n"
        "Домен: a_cand[a1_pre_w] \u2014 чистый A1, как весь B2 (A2/A3/A4 informational).\n"
        "Fire = (bar_idx, direction), первое k, одна зона fires МАКС один раз."
    )
    ax.text(0.04, 0.80, rule_text, fontsize=9.3, color="#222",
            ha="left", va="top", transform=ax.transAxes,
            family="DejaVu Sans Mono")


def _draw_notes_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#fff5f0", edgecolor="#c0392b",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.92, "Как это нашли \u00b7 статус",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "\u2022  Разносторонний скан: 4 block-элемента (ob, breaker_block,\n"
        "   mitigation_block, rb) \u00d7 11 TF \u00d7 5 механик (SWEEP50, FULL_DISP,\n"
        "   TOUCH, BIRTH, PROXIMITY) на BTC \u2014 220 комбинаций.\n\n"
        "\u2022  BIRTH/TOUCH/PROXIMITY провалились (WR 38\u201358%, на уровне A1-\n"
        "   бейзлайна 41.47% или ниже) \u2014 отрицательный результат, не просто\n"
        "   пропущенный вариант.\n\n"
        "\u2022  FULL_DISP@12h выделился на mitigation_block И на rb (n=63-85,\n"
        "   WR 81-90% на BTC/ETH/SOL) \u2014 rb-версия (\"Rejection Block\") ждёт\n"
        "   отдельного B2C4-кандидата.\n\n"
        "\u2022  СТАТУС: research-находка в G:\\Claude\\research\\, НЕ портирована\n"
        "   в ASVK b2_ob.py, НЕ входит в b2_hit/basket \u2014 B2C1 (production) не\n"
        "   тронут ни в коде, ни в этом сравнении.\n\n"
        "Refs:  G:\\Claude\\research\\block_family_explore.py  \u00b7\n"
        "       G:\\ASVK\\lib\\\u0434\u0435\u0442\u0435\u043a\u0442\u043e\u0440\u044b\\mitigation_block.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=8.9, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B2C3_candidate_mitigation_FULL_DISP.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
