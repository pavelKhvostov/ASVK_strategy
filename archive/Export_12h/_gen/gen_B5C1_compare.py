"""B5C1 -- SWEEP vs FULL_DISP, два РЕАЛЬНЫХ примера 2026 года (BTC).

Левая панель: 2026-07-22 12:00 UTC -- SWEEP сработал, FULL_DISP нет. Пивот НЕ
  подтвердился (l[i+1] пробил l[i] вниз) -- ровно тот случай, который старое
  правило пропускало как сигнал, а он оказался слабым.
Правая панель: 2026-07-17 12:00 UTC -- сработали и SWEEP, и FULL_DISP. Пивот
  подтвердился по Williams n=2.

Output: G:\\ASVK\\Export_12h\\B5C1_sweep_vs_fulldisp.png
"""
from __future__ import annotations
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

FIRE_C, ACCENT, WARN, LEVEL_C = "#111", "#0057b8", "#c0392b", "#7a2ba0"


def draw_candle(ax, x, o, h, l, c, w=0.55, edge="#111", fill_up="white", fill_dn="#222", lw=1.5):
    ax.plot([x, x], [l, h], color=edge, lw=lw, zorder=2)
    color = fill_up if c >= o else fill_dn
    body = max(abs(c - o), (h - l) * 0.03)
    ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, body, facecolor=color, edgecolor=edge, lw=lw, zorder=3))


def draw_example(ax, title, bars, vwap, margin, fires_full, confirmed, dates_note):
    thr = vwap + margin
    xs = list(range(1, 6))

    ax.plot([0.5, 5.5], [vwap, vwap], ls=(0, (5, 2, 1, 2)), color=LEVEL_C, lw=2)
    ax.text(5.6, vwap, f"VWAP={vwap:,.0f}", color=LEVEL_C, fontsize=9.5, fontweight="bold", va="center")

    ax.plot([0.5, 5.5], [thr, thr], ls=(0, (2, 2)), color=ACCENT, lw=1.6)
    ax.text(5.6, thr, f"+margin={thr:,.0f}", color=ACCENT, fontsize=9, fontweight="bold", va="center")

    y_lo = min(min(b[3] for b in bars), vwap) - 250
    y_hi = max(max(b[2] for b in bars), thr) + 400
    ax.axhspan(vwap, thr, xmin=0.02, xmax=0.98, color=WARN, alpha=0.08, zorder=0)

    for i, (lbl, o, h, l, c) in zip(xs, bars):
        is_fire = (i == 3)
        draw_candle(ax, i, o, h, l, c, edge=FIRE_C if is_fire else "#333", lw=2.0 if is_fire else 1.2)
        ax.text(i, y_lo + 60, lbl, ha="center", fontsize=9,
                fontweight="bold" if is_fire else "normal", color=FIRE_C if is_fire else "#666")

    star_color = "#ffb400" if confirmed else "#999"
    ax.plot(3, y_hi - 150, marker="*", markersize=24, color=star_color,
            markeredgecolor="#8a5a00" if confirmed else "#555", markeredgewidth=1, zorder=6)

    verdict = "SWEEP: да  |  FULL_DISP: да" if fires_full else "SWEEP: да  |  FULL_DISP: нет"
    v_color = "#006400" if fires_full else "#666"
    ax.text(3, y_hi - 320, verdict, ha="center", fontsize=10, fontweight="bold", color=v_color)

    result = "CONFIRMED (сигнал сработал)" if confirmed else "НЕ подтвердился (l[i+1] пробил l[i])"
    r_color = "#006400" if confirmed else WARN
    ax.text(3, y_lo - 250, result, ha="center", fontsize=10.5, fontweight="bold", color=r_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=r_color, linewidth=1.3))

    ax.set_title(title, fontsize=12, fontweight="bold", color="#222", pad=10)
    ax.set_xlim(0.3, 6.6)
    ax.set_ylim(y_lo - 550, y_hi + 100)
    ax.set_xlabel(dates_note, fontsize=8.5, color="#666")
    ax.grid(alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor="white")
fig.suptitle("B5C1 -- два реальных примера 2026: SWEEP vs FULL_DISP (BTC, LONG)",
             fontsize=17, fontweight="bold", color="#1c1c2e", y=0.99)

bars_left = [
    ("21-12h", 66345.59, 66956.15, 66052.63, 66556.16),
    ("22-00h", 66556.15, 66739.89, 65701.00, 66013.36),
    ("22-12h\n(FIRE)", 66013.36, 66384.00, 65553.67, 66114.49),
    ("23-00h", 66114.50, 66313.14, 65351.02, 65555.21),
    ("23-12h", 65555.21, 65589.41, 64650.00, 65098.97),
]
draw_example(axes[0], "2026-07-22 12:00 -- SWEEP сработал, FULL_DISP нет",
             bars_left, vwap=65758.49, margin=554.91,
             fires_full=False, confirmed=False,
             dates_note="close=66,114 > VWAP, но не дотянул до +margin=66,313")

bars_right = [
    ("16-12h", 64256.52, 64896.00, 63748.74, 63830.20),
    ("17-00h", 63830.20, 64067.69, 62666.00, 63298.01),
    ("17-12h\n(FIRE)", 63298.00, 64387.99, 62537.56, 63931.67),
    ("18-00h", 63931.67, 64097.22, 63886.65, 64069.89),
    ("18-12h", 64069.89, 64865.00, 63963.00, 64834.22),
]
draw_example(axes[1], "2026-07-17 12:00 -- SWEEP и FULL_DISP сработали оба",
             bars_right, vwap=62880.86, margin=556.89,
             fires_full=True, confirmed=True,
             dates_note="close=63,932 > +margin=63,438 -- решительный отскок")

fig.text(0.5, 0.015,
         "SWEEP: high[i]>VWAP AND close[i]>VWAP   |   FULL_DISP: high[i]>VWAP AND close[i]>VWAP+0.5xATR14   "
         "|   BTC 72.97%->81.86%  ETH 64.94%->75.77%  SOL 66.92%->75.71%",
         ha="center", fontsize=9.5, family="monospace", color="#333")

fig.tight_layout(rect=[0, 0.04, 1, 0.94])

out = pathlib.Path("/mnt/g/ASVK/Export_12h/B5C1_sweep_vs_fulldisp.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
