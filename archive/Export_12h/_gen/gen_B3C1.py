"""B3C1 rule (production, в basket) — maxV sweep(i-1) · Fractal Liquidity · TF=12h.

Правило (источник истины G:\\ASVK\\lib\\fractal12h\\b3_fractal_liquidity.py):
    SHORT (FH): h[i] > maxV(i-1)  AND  c[i] < maxV(i-1)   (прокол вверх + закрытие вниз)
    LONG  (FL): l[i] < maxV(i-1)  AND  c[i] > maxV(i-1)   (прокол вниз + закрытие вверх)
    БЕЗ depth-фильтра (в отличие от B9C2 = тот же sweep + depth/ATR14 >= 0.7).

    Домен: a_cand[a124_pool] — A1+A2+A4, БЕЗ A3 (a_cascade.py) — это ОТЛИЧАЕТСЯ
    от B1/B2/B9 (чистый A1 pre-w)! До 2026-07-24 здесь ошибочно стоял
    a4_body_wick — кумулятивный A1+A2+A3+A4, тащивший A3 внутрь домена
    вопреки команде. Исправлено в этой же сессии.

    maxV — level-1 shared indicator (lib/maxv.py): close 90-минутного LTF-окна
    (агрегат MAXV_LTF_MIN=90 мин) с абсолютным max объёмом внутри 12h бара.
    Направление (bull/bear) НЕ учитывается — просто какое окно собрало больше
    объёма. Walk-forward валидировано на BTC/ETH/SOL 2026-07-23.

Текущий результат на BTC: n=428, conf=324/428, WR=75.70%.

Refs:
  G:\\ASVK\\lib\\fractal12h\\b3_fractal_liquidity.py  — реализация
  G:\\ASVK\\lib\\maxv.py                              — источник maxV
  G:\\ASVK\\Export_12h\\_gen\\gen_B2C1.py              — шаблон вёрстки

Output:
  G:\\ASVK\\Export_12h\\B3C1.png
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
RED_EDGE = "#b30000"


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
    ax.text(0.5, 0.66, "B3C1 — maxV sweep(i-1) · Fractal Liquidity · TF=12h",
            ha="center", va="center", fontsize=21, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.28,
            "PRODUCTION (в basket)  ·  домен A1+A2+A4 без A3 (a124_pool)  ·  "
            "BTC n=428 WR=75.70%",
            ha="center", va="center", fontsize=11.2, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel_real(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup (схема)  ·  A1-pivot LOW \u2192 h[i]>maxV(i-1), c[i]<maxV(i-1)",
                 fontsize=12.0, fontweight="bold", pad=10, color="#333")

    level = 101.2
    x0, x_fire = 1.0, 8.5

    ax.plot([x0, x_fire + 1.5], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(x_fire + 1.7, level, "maxV(i-1)", fontsize=9.5,
            color=LEVEL_C, va="center", fontweight="bold")
    ax.text(4.5, level + 0.4, "close 90m-окна с max объёмом внутри бара i-1",
            fontsize=8.6, color=LEVEL_C, ha="center", style="italic")

    draw_candle(ax, x0, o=99.5, h=100.4, l=98.7, c=100.0, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    draw_candle(ax, x0 + 1.1, o=100.0, h=100.9, l=99.3, c=100.6, w=0.55, edge="#666",
                fill_up="#dcdce2", fill_dn="#8f8f96", lw=0.9)
    ax.text(x0 + 0.5, 97.9, "бар i-1  (формирует maxV)", fontsize=7.6,
            color="#555", ha="center", style="italic")

    _draw_ellipsis(ax, x0 + 2.0, x_fire - 1.3, level - 3.0)

    ax.plot(x0 + 0.3, 96.0, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x0 + 0.3, 95.2, "A1-pivot LOW\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    draw_candle(ax, x_fire, o=101.0, h=103.2, l=98.6, c=99.2,
                w=0.6, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 103.9, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 104.6, "FIRE", fontsize=9.5, fontweight="bold", color="#8a5a00", ha="center")

    ax.annotate("high > maxV(i-1)\n(прокол уровня вверх)",
                xy=(x_fire + 0.02, 103.15), xytext=(4.9, 107.0),
                fontsize=9.1, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4), ha="left", va="center")
    ax.annotate("close < maxV(i-1)\n(закрытие назад под уровень)",
                xy=(x_fire + 0.05, 99.2), xytext=(1.2, 93.8),
                fontsize=9.1, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4), ha="left", va="center")

    ax.set_xlim(-0.4, 12.5)
    ax.set_ylim(92.0, 108.0)
    ax.set_xlabel("12h bars \u2192", fontsize=9.5)
    ax.set_ylabel("price (условно)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel_real(ax) -> None:
    ax.set_title("LONG setup \u2014 РЕАЛЬНЫЙ пример: BTC 2026-07-17 12:00 UTC (confirmed)",
                 fontsize=12.0, fontweight="bold", pad=10, color="#333")

    level = 62888.00

    bars = [
        (-2, "07-16 12h", 64256.52, 64896.00, 63748.74, 63830.20),
        (-1, "07-17 00h", 63830.20, 64067.69, 62666.00, 63298.01),
        (0,  "07-17 12h", 63298.00, 64387.99, 62537.56, 63931.67),
        (1,  "07-18 00h", 63931.67, 64097.22, 63886.65, 64069.89),
        (2,  "07-18 12h", 64069.89, 64865.00, 63963.00, 64834.22),
    ]
    x0 = 3.4
    xs = {k: x0 + (k + 2) * 1.35 for k, *_ in bars}
    x_fire = xs[0]

    ax.plot([x0 - 1.3, xs[2] + 0.9], [level, level],
            ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=1.8, alpha=0.9)
    ax.text(xs[2] + 1.05, level, f"maxV(i-1)={level:,.0f}", fontsize=8.6,
            color=LEVEL_C, va="center", fontweight="bold")
    ax.text(xs[-2] - 1.1, 65900, "maxV сформирован внутри бара i-1\n(07-17 00h, 90m-окно с max объёмом)",
            fontsize=7.8, color=LEVEL_C, ha="left", style="italic")

    for k, lbl, o, h, l, c in bars:
        x = xs[k]
        is_fire = (k == 0)
        draw_candle(ax, x, o, h, l, c, w=0.75,
                    edge=FIRE_C if is_fire else "#333",
                    fill_up="white", fill_dn="#2a2a2a",
                    lw=1.6 if is_fire else 1.0)
        ax.text(x, l - 220, lbl, fontsize=7.6, color="#555", ha="center", rotation=0)

    ax.plot(x_fire, 61700, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire, 61380, "A1-pivot LOW (12h, l[i]<l[i-1] и l[i]<l[i-2])", fontsize=7.6,
            color=WARN, ha="center", fontweight="bold")

    ax.plot(x_fire, 65200, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 65500, "FIRE  (i, 2026-07-17 12:00)", fontsize=9, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate(f"low={62537.56:,.0f} < maxV(i-1)={level:,.0f}",
                xy=(x_fire - 0.05, 62537.56), xytext=(xs[-2] - 1.9, 61900),
                fontsize=8.4, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3), ha="left", va="center")
    ax.annotate(f"close={63931.67:,.0f} > {level:,.0f}",
                xy=(x_fire, 63931.67), xytext=(xs[1] - 0.1, 65000),
                fontsize=8.6, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3), ha="left", va="center")

    ax.annotate("Williams n=2 confirm: l[i+1]>l[i] и l[i+2]>l[i]  \u2192  CONFIRMED",
                xy=(xs[2], 63963.00), xytext=(xs[2] + 0.4, 64500),
                fontsize=8.3, color="#006400", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#006400", lw=1.2), ha="left", va="center")

    ax.set_xlim(x0 - 3.0, xs[2] + 3.0)
    ax.set_ylim(61100, 66400)
    ax.set_xlabel("12h bars \u2192  (реальные даты BTC)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.92, "Формальное правило (maxV sweep, без depth)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "SHORT (FH):  high[i] > maxV(i-1)   AND   close[i] < maxV(i-1)\n"
        "LONG  (FL):  low[i]  < maxV(i-1)   AND   close[i] > maxV(i-1)\n\n"
        "maxV(k) = close 90-минутного LTF-окна (MAXV_LTF_MIN=90) с абсолютным\n"
        "  MAX объёмом внутри 12h бара k. Направление (bull/bear) НЕ учитывается \u2014\n"
        "  просто какое окно собрало больше объёма. Level-1 shared indicator,\n"
        "  используется и другими блоками (lib/maxv.py).\n\n"
        "БЕЗ depth-фильтра \u2014 в отличие от B9C2 (тот же sweep + требование\n"
        "  depth/ATR14 >= 0.7 на пробой).\n\n"
        "Домен: a_cand[a124_pool] = A1+A2+A4, БЕЗ A3. НЕ совпадает с B1/B2/B9\n"
        "  (чистый A1) \u2014 это сознательный выбор канона для B3."
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
    ax.text(0.5, 0.92, "Canon-заметки \u00b7 история багфикса",
            ha="center", fontsize=12, fontweight="bold", color="#0057b8",
            transform=ax.transAxes)
    notes_text = (
        "\u2022  До 2026-07-24 в b3_fractal_liquidity.py ошибочно стоял\n"
        "   a4_body_wick (кумулятивный A1+A2+A3+A4) вместо a124_pool \u2014\n"
        "   A3 протаскивался в домен вопреки команде пользователя. Исправлено\n"
        "   в этой же сессии, число пересчитано (n=428, WR=75.70% на BTC).\n\n"
        "\u2022  В отличие от B1/B2/B9 (чистый A1 pre-w), B3 работает на a124_pool \u2014\n"
        "   A2 и A4 ЗДЕСЬ реальный фильтр, не informational only.\n\n"
        "\u2022  B3C2..B3C6 запланированы каноном, но не реализованы (как и в\n"
        "   WSL-источнике) \u2014 пока B3 = B3C1 единственно.\n\n"
        "\u2022  PRODUCTION: b3_hit входит в basket_hit напрямую (OR со всеми\n"
        "   другими блоками), живой сигнал в dashboard/asvk.py.\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b3_fractal_liquidity.py  \u00b7\n"
        "       G:\\ASVK\\lib\\maxv.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=8.9, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B3C1.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
