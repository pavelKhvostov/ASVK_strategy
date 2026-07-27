"""B9C1 rule — presentation-quality schematic + canon notes.

Правило: P11_count — доля 15m-баров в последних N×15m ПЕРЕД закрытием pivot-бара i,
направленных В ТУ ЖЕ сторону, что и ожидаемый разворот (contrarian к тренду, который
привёл к пивоту). 4 независимых окна (2h/3h/4h/6h), результат = OR (достаточно ОДНОГО
окна выше своего порога). Плюс overlay: close_match + range/ATR14 >= 1.2.

Source of truth (G:\\ASVK\\lib\\fractal12h\\b9_others.py):
    WINDOWS = [(8,0.65), (12,0.75), (16,0.65), (24,0.65)]   # (N х 15m, threshold)
    SHORT: cnt = count(15m close < open) в окне [pt_end - N*15m, pt_end)
    LONG:  cnt = count(15m close > open) в окне
    p11_or = OR по 4 окнам (ratio = cnt/N >= threshold)
    B9C1 = p11_or AND close_match AND range12[i]/ATR14[i] >= 1.2

Стиль 1-в-1 повторяет B1C1..B1C4/B9C2..B9C4 (шапка + 2 панели + 2 блока правил).

Refs:
  G:\\ASVK\\lib\\fractal12h\\b9_others.py   — реализация условия (b9c1)
  G:\\ASVK\\lib\\maxv.py                   — ATR14(12h), level-1 shared indicator

Output:
  Desktop/export_12h/B9C1.png
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

RED = "#c0392b"
RED_FILL = "#f5b7b1"
GRN = "#1e8449"
GRN_FILL = "#a9dfbf"
FIRE_C = "#111"
ACCENT = "#0057b8"
WARN = "#c0392b"
WIN_C = "#8e44ad"
OK_C = "#1e8449"
MUTE_C = "#999"


def draw_candle(ax, x, o, h, l, c, w=0.6, edge="#111",
                fill_up="white", fill_dn="#222", lw=1.0, alpha=1.0):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=2, alpha=alpha)
    color = fill_up if c >= o else fill_dn
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.03),
                            facecolor=color, edgecolor=edge, lw=lw,
                            zorder=3, alpha=alpha))


# is_red[i]: 15m-бар i (0=6h назад от close пивота ... 23=последний перед close)
# направлен В СТОРОНУ ожидаемого разворота (для SHORT-панели = bearish 15m бар)
SHORT_DIR = [True, False, True, True, False, True, True, False,      # 0-7   (N24 доп.)
             False, True, False, True,                               # 8-11  (N16 доп.)
             False, True, False, True,                               # 12-15 (N12 доп.)
             True, True, False, True, True, False, True, True]       # 16-23 (N8 окно)


def _window_stats(flags: list[bool]) -> dict:
    n = len(flags)
    out = {}
    for N, thr in [(8, 0.65), (12, 0.75), (16, 0.65), (24, 0.65)]:
        window = flags[n - N:]
        cnt = sum(window)
        ratio = cnt / N
        out[N] = (cnt, N, ratio, thr, ratio >= thr)
    return out


def _draw_15m_strip(ax, x0: float, y: float, flags: list[bool], tick_h: float = 0.9,
                    dir_fill=RED_FILL, dir_edge=RED, off_fill="#ecf0f1", off_edge="#95a5a6"):
    """24 узких плиток 15m-баров: закрашена (dir_fill) если бар "за" разворот, иначе серая."""
    for i, is_dir in enumerate(flags):
        x = x0 + i * 0.34
        fc, ec = (dir_fill, dir_edge) if is_dir else (off_fill, off_edge)
        ax.add_patch(Rectangle((x, y - tick_h / 2), 0.28, tick_h,
                                facecolor=fc, edgecolor=ec, lw=0.7, zorder=3))


def _draw_windows_brackets(ax, x0: float, y_top: float, stats: dict, step: float = 0.34):
    """4 вложенных bracket'а снизу вверх: N=8 (самый узкий, ближе к close) .. N=24 (самый широкий)."""
    x_end = x0 + 24 * step  # = pt_end (close пивота)
    order = [24, 16, 12, 8]
    for row, N in enumerate(order):
        cnt, n, ratio, thr, ok = stats[N]
        x_start = x0 + (24 - N) * step
        y = y_top - row * 0.92
        color = OK_C if ok else MUTE_C
        ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2.0 if ok else 1.1,
                                    alpha=1.0 if ok else 0.55))
        hrs = N * 15 // 60
        mark = "✓ FIRE" if ok else "✗"
        fw = "bold" if ok else "normal"
        ax.text(x_end + 0.35, y,
                f"N={N} ({hrs}h):  {cnt}/{n} = {ratio*100:.0f}%  (thr {thr*100:.0f}%)  {mark}",
                fontsize=8.6, color=color, va="center", fontweight=fw,
                alpha=1.0 if ok else 0.65)


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(17, 11.5), facecolor="white")
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.6, 4.6, 2.3],
                           hspace=0.32, wspace=0.10,
                           left=0.04, right=0.985, top=0.965, bottom=0.04)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#1c1c2e", edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.68, "B9C1 — P11 count · 4-window OR + overlay",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.42, "A1 anchor  (pool: все Williams-confirmable pivots)  ·  "
                       "rescued 2026-07-21 (было C2 → B3 → B7 → B9C1)",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="#ffc857", transform=ax.transAxes)
    ax.text(0.5, 0.15,
            "мини-структура ВНУТРИ последних часов pivot-бара (15m-грануляция)  ·  "
            "causal (всё окно ⊂ 12h бар i)",
            ha="center", va="center", fontsize=11, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup  ·  A1-pivot HIGH → contrarian 15m burst перед close",
                 fontsize=12.3, fontweight="bold", pad=10, color="#333")

    stats = _window_stats(SHORT_DIR)

    # 12h pivot-бар i — справа, "закрывается" в конце 15m-полосы
    x_strip0 = 0.6
    step = 0.34
    x_close = x_strip0 + 24 * step
    x_pivot = x_close + 2.0

    o_i, h_i, l_i, c_i = 104.5, 106.0, 99.0, 99.8
    rng = h_i - l_i
    atr14 = 5.4
    range_atr = rng / atr14

    draw_candle(ax, x_pivot, o=o_i, h=h_i, l=l_i, c=c_i, w=1.0,
                edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.6)
    ax.plot(x_pivot, h_i + 0.9, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_pivot, h_i + 1.9, "FIRE\n(close bar i)", fontsize=9.3, fontweight="bold",
            color="#8a5a00", ha="center")
    ax.plot(x_pivot, h_i + 4.2, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_pivot, h_i + 5.1, "A1-pivot HIGH", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    # 15m strip — последние 6h ДО close бара i
    y_strip = 97.5
    _draw_15m_strip(ax, x_strip0, y_strip, SHORT_DIR)
    ax.annotate("", xy=(x_close, y_strip), xytext=(x_pivot - 1.0, o_i),
                arrowprops=dict(arrowstyle="-", color="#999", lw=1.0, ls=(0, (2, 2))))
    ax.text(x_strip0, y_strip + 1.1, "15m бары  ·  красный = close < open (в сторону разворота)",
            fontsize=8.6, color=RED, fontweight="bold", ha="left")
    ax.text(x_strip0 - 0.1, y_strip - 1.3, "6h назад", fontsize=7.6, color="#666", ha="left")
    ax.text(x_close - 0.1, y_strip - 1.3, "close бара i", fontsize=7.6, color="#666", ha="right")

    # 4 вложенных окна снизу
    _draw_windows_brackets(ax, x_strip0, y_strip - 2.4, stats, step=step)

    # overlay checklist
    close_match = c_i < o_i
    bx, by = x_pivot - 1.9, 88.7
    ax.add_patch(FancyBboxPatch((bx, by), 4.1, 3.0, boxstyle="round,pad=0.08",
                                 facecolor="#eef6ff", edgecolor=ACCENT, lw=1.2, zorder=5))
    ax.text(bx + 2.05, by + 2.55, "overlay (сверх p11_or)", fontsize=8.4,
            color=ACCENT, ha="center", fontweight="bold")
    ax.text(bx + 0.2, by + 1.55,
            f"close_match:  c<o  →  {'✓' if close_match else '✗'}", fontsize=8.6,
            color=OK_C if close_match else WARN, fontweight="bold")
    ax.text(bx + 0.2, by + 0.55,
            f"range/ATR14 = {rng:.1f}/{atr14:.1f} = {range_atr:.2f}  (≥1.2 {'✓' if range_atr>=1.2 else '✗'})",
            fontsize=8.6, color=OK_C if range_atr >= 1.2 else WARN, fontweight="bold")

    ax.set_xlim(0.0, x_pivot + 2.3)
    ax.set_ylim(83.0, 112.0)
    ax.axis("off")


def _draw_long_panel(ax) -> None:
    ax.set_title("LONG setup  ·  A1-pivot LOW → contrarian 15m burst перед close",
                 fontsize=12.3, fontweight="bold", pad=10, color="#333")

    LONG_DIR = SHORT_DIR  # тот же паттерн долей, семантика зеркальна (bullish 15m вместо bearish)
    stats = _window_stats(LONG_DIR)

    x_strip0 = 0.6
    step = 0.34
    x_close = x_strip0 + 24 * step
    x_pivot = x_close + 2.0

    o_i, h_i, l_i, c_i = 95.5, 101.0, 94.0, 100.2
    rng = h_i - l_i
    atr14 = 5.4
    range_atr = rng / atr14

    draw_candle(ax, x_pivot, o=o_i, h=h_i, l=l_i, c=c_i, w=1.0,
                edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.6)
    ax.plot(x_pivot, l_i - 0.9, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_pivot, l_i - 1.9, "FIRE\n(close bar i)", fontsize=9.3, fontweight="bold",
            color="#8a5a00", ha="center")
    ax.plot(x_pivot, l_i - 4.2, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_pivot, l_i - 5.1, "A1-pivot LOW", fontsize=8,
            color=WARN, ha="center", fontweight="bold")

    y_strip = 103.0
    _draw_15m_strip(ax, x_strip0, y_strip, LONG_DIR, dir_fill=GRN_FILL, dir_edge=GRN)
    ax.annotate("", xy=(x_close, y_strip), xytext=(x_pivot - 1.0, o_i),
                arrowprops=dict(arrowstyle="-", color="#999", lw=1.0, ls=(0, (2, 2))))
    ax.text(x_strip0, y_strip + 1.1, "15m бары  ·  зелёный = close > open (в сторону разворота)",
            fontsize=8.6, color=GRN, fontweight="bold", ha="left")
    ax.text(x_strip0 - 0.1, y_strip - 1.3, "6h назад", fontsize=7.6, color="#666", ha="left")
    ax.text(x_close - 0.1, y_strip - 1.3, "close бара i", fontsize=7.6, color="#666", ha="right")

    _draw_windows_brackets(ax, x_strip0, 91.5, stats, step=step)

    close_match = c_i > o_i
    bx, by = x_pivot - 1.9, 83.3
    ax.add_patch(FancyBboxPatch((bx, by), 4.1, 3.0, boxstyle="round,pad=0.08",
                                 facecolor="#eef6ff", edgecolor=ACCENT, lw=1.2, zorder=5))
    ax.text(bx + 2.05, by + 2.55, "overlay (сверх p11_or)", fontsize=8.4,
            color=ACCENT, ha="center", fontweight="bold")
    ax.text(bx + 0.2, by + 1.55,
            f"close_match:  c>o  →  {'✓' if close_match else '✗'}", fontsize=8.6,
            color=OK_C if close_match else WARN, fontweight="bold")
    ax.text(bx + 0.2, by + 0.55,
            f"range/ATR14 = {rng:.1f}/{atr14:.1f} = {range_atr:.2f}  (≥1.2 {'✓' if range_atr>=1.2 else '✗'})",
            fontsize=8.6, color=OK_C if range_atr >= 1.2 else WARN, fontweight="bold")

    ax.set_xlim(0.0, x_pivot + 2.3)
    ax.set_ylim(80.0, 109.0)
    ax.axis("off")


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.93, "Формальное правило (OR по 4 окнам + overlay, causal)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "Для pivot bar i (close = pt_end = ts_pivot + 12h):\n"
        "  для каждого окна N × 15m ∈ {8, 12, 16, 24}  (= 2h/3h/4h/6h):\n"
        "    window = 15m-бары в [pt_end - N×15m, pt_end)\n"
        "    SHORT (FH):  cnt = count(15m close < open)\n"
        "    LONG  (FL):  cnt = count(15m close > open)\n"
        "    P11_N = cnt / N\n\n"
        "  p11_or = (P11_8≥0.65) ∨ (P11_12≥0.75) ∨ (P11_16≥0.65) ∨ (P11_24≥0.65)\n"
        "  (P11_12 порог выше — 3h окно шумнее, чем 2h/4h/6h)\n\n"
        "  Overlay:  close_match  AND  range12[i] / ATR14[i] >= 1.2\n\n"
        "B9C1 = p11_or AND close_match AND range_atr>=1.2\n"
        "Causal: всё окно ⊂ [born, close] бара i, ATR14 читается из data/maxv/ (level-1)."
    )
    ax.text(0.04, 0.82, rule_text, fontsize=8.9, color="#222",
            ha="left", va="top", transform=ax.transAxes,
            family="DejaVu Sans Mono")


def _draw_notes_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#fff5f0", edgecolor="#c0392b",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.93, "Canon-заметки  ·  rescued 2026-07-21",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "•  Единственное условие всей B1/B9-семьи, работающее на 15m-грануляции —\n"
        "   остальные (B1, B9C2..C4) читают только 12h OHLC. Видит \"внутреннюю кухню\"\n"
        "   последних часов бара, которую 12h-свеча сама по себе не показывает.\n\n"
        "•  OR, не AND: достаточно ОДНОГО из 4 окон выше порога. На схеме — намеренно\n"
        "   показан пример, где срабатывает только N=8 (2h), а более широкие окна (3h/4h/6h)\n"
        "   разбавлены более ранней активностью и порог не проходят.\n\n"
        "•  История имени: был C2 (2026-06 условие №2) → мигрировал в B3 → в B7 →\n"
        "   финально осел в B9C1 (2026-07-21). Semantic не менялся, менялось только место\n"
        "   в структуре басket'а по мере роста числа блоков.\n\n"
        "•  Overlay (close_match + range/ATR14≥1.2) добавлен при rescue — без него голый\n"
        "   p11_or ловил слишком много шумных случаев без реального импульса в баре i.\n\n"
        "•  Причинность: [pt_end - 6h, pt_end) — самое широкое окно (N=24) всё ещё строго\n"
        "   внутри 12h бара i, ничего из будущего не подсматривается.\n\n"
        "Refs:  G:\\ASVK\\lib\\fractal12h\\b9_others.py (b9c1)  ·  G:\\ASVK\\lib\\maxv.py (atr14)"
    )
    ax.text(0.04, 0.82, notes_text, fontsize=8.85, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/c/Users/Вадим/Desktop/export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B9C1.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
