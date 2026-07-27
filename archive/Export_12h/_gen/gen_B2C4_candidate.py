"""B2C4 (KANDIDAT, ne v production/basket) - FULL_DISP * Rejection Block (rb) * TF=12h.

Sibling naxodki B2C3 (mitigation_block) - ta zhe механика FULL_DISP, tot zhe TF=12h,
tot zhe A1-domen, no drugoy istochnik zony: rb (odin bar s dominiruyushchim fitilem),
ne mnogoshagovaya struktura kak mitigation_block/OB.

Pravilo (b2_ob.py poka NE soderzhit etu mekhaniku - eto research-naxodka,
G:\\Claude\\research\\block_family_explore.py::mech_full_disp):
    w = zone_hi - zone_lo
    SHORT: high[k] >= zone_hi   AND  close[k] < zone_lo - w
    LONG:  low[k]  <= zone_lo   AND  close[k] > zone_hi + w
    Zone istochnik: rb (lib/detektory/rb.py) - "Rejection Block":
        body>0; upper>=2*lower AND upper>=3*body -> TOP RB (short/resistance)
                lower>=2*upper AND lower>=3*body -> BOTTOM RB (long/support)
        Zone: LONG=(low, body_bottom) support     SHORT=(body_top, high) resistance
    TF: 12h TOLKO (multi-TF union zdes' xuzhe, kak i u mitigation_block).
    Domen: chistyy A1 pre-w pool, kak ves' B2.

Kross-proverka (n / WR, FULL_DISP@12h):
    BTC  n=85  WR=81.18%      ETH  n=81  WR=90.12%      SOL  n=63  WR=84.13%
  (dlya sravneniya B2C1=70-81%, B2C2=63-75%, B2C3(mitigation)=83.8-88.0%)

Pravaya (LONG) panel - REALNYY primer, ne sxema: BTC, rb-zona (support) rodilas'
2024-04-02 00:00 UTC (odin dominant-wick bar), signal zazhegsya 2026-04-07 12:00 UTC
(i-bar, ogromnyy diapazon H-L), Williams n=2 podtverzhdenie po i+1/i+2 vidno na grafike.

Refs:
  G:\\Claude\\research\\block_family_explore.py  - istochnik mekhaniki/skana
  G:\\ASVK\\Export_12h\\_gen\\gen_B2C3_candidate.py - identichnyy shablon (mitigation_block)
  G:\\ASVK\\lib\\detektory\\rb.py                  - istochnik zon

Output:
  G:\\ASVK\\Export_12h\\B2C4_candidate_rb_FULL_DISP.png
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
    ax.text(0.5, 0.66, "B2C4 (kandidat) - FULL_DISP . Rejection Block (rb) . TF=12h",
            ha="center", va="center", fontsize=21, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.28,
            "NE v production/basket - research-naxodka (sibling B2C3)  .  "
            "BTC n=85 WR=81.18%  .  ETH n=81 WR=90.12%  .  SOL n=63 WR=84.13%",
            ha="center", va="center", fontsize=11.2, color="#c8c8d4",
            transform=ax.transAxes, style="italic")

    _draw_short_panel(fig.add_subplot(gs[1, 0]))
    _draw_long_panel_real(fig.add_subplot(gs[1, 1]))
    _draw_rule_box(fig.add_subplot(gs[2, 0]))
    _draw_notes_box(fig.add_subplot(gs[2, 1]))

    return fig


def _draw_rb_bar(ax, x, zone_lo, zone_hi, direction, color_edge):
    """Odin bar s dominiruyushchim fitilem - sam yavlyaetsya istochnikom zony."""
    if direction == "short":
        o, c = zone_lo + (zone_hi - zone_lo) * 0.15, zone_lo + (zone_hi - zone_lo) * 0.55
        h, l = zone_hi, zone_lo - (zone_hi - zone_lo) * 1.4
    else:
        o, c = zone_hi - (zone_hi - zone_lo) * 0.55, zone_hi - (zone_hi - zone_lo) * 0.15
        h, l = zone_hi + (zone_hi - zone_lo) * 1.4, zone_lo
    draw_candle(ax, x, o, h, l, c, w=0.55, edge=color_edge,
                fill_up="white", fill_dn="#555", lw=1.3, zorder=4)


def _draw_short_panel(ax) -> None:
    ax.set_title("SHORT setup (skhema)  .  A1-pivot LOW -> full displacement (TOP RB)",
                 fontsize=12.2, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 100.6, 102.2
    w = zone_hi - zone_lo
    x0, x_fire = 1.6, 8.5
    close_thr = zone_lo - w

    ax.add_patch(Rectangle((x0, zone_lo), (x_fire + 0.6) - x0, zone_hi - zone_lo,
                            facecolor=RED_FILL, edgecolor=RED_EDGE, linewidth=1.4,
                            alpha=0.85, zorder=1))
    ax.text(4.8, zone_hi + 0.35, "rb zone (body_top, high)  .  TOP RB  .  TF=12h",
            fontsize=9.2, color="#800000", fontweight="bold", ha="center")

    ax.plot([x0, x_fire + 0.6], [close_thr, close_thr],
            ls=(0, (5, 2, 1, 2)), color=FAR_C, lw=1.4, alpha=0.9)
    ax.text(x_fire + 0.75, close_thr, "zone_lo - w\n(close-porog)", fontsize=8,
            color=FAR_C, va="center", fontweight="bold")

    ax.text(x_fire + 1.55, zone_hi, "zone_hi", fontsize=8.3, color=RED_EDGE, va="center", fontweight="bold")
    ax.text(x_fire + 1.55, zone_lo, "zone_lo", fontsize=8.3, color=RED_EDGE, va="center", fontweight="bold")
    ax.annotate("", xy=(x_fire + 1.35, zone_hi), xytext=(x_fire + 1.35, zone_lo),
                arrowprops=dict(arrowstyle="<->", color=RED_EDGE, lw=1.2))
    ax.text(x_fire + 1.9, (zone_hi + zone_lo) / 2, "w", fontsize=9, color=RED_EDGE,
            va="center", fontweight="bold")

    _draw_rb_bar(ax, x0, zone_lo, zone_hi, "short", RED_EDGE)
    ax.annotate("upper >= 2*lower\nupper >= 3*body", xy=(x0, zone_hi + 0.9),
                xytext=(x0 - 1.3, 105.2), fontsize=7.8, color="#800000", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#800000", lw=1.2), ha="center")

    _draw_ellipsis(ax, x0 + 0.9, x_fire - 1.3, zone_lo + 0.8)

    ax.plot(x0 + 0.9, 96.3, marker="v", markersize=13, color=ACCENT,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x0 + 0.9, 95.5, "A1-pivot LOW\n(12h fractal)", fontsize=8,
            color=ACCENT, ha="center", fontweight="bold")

    draw_candle(ax, x_fire, o=101.4, h=102.9, l=97.3, c=98.0,
                w=0.6, edge=FIRE_C, fill_up="white", fill_dn="#2a2a2a", lw=1.4)
    ax.plot(x_fire, 103.6, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 104.3, "FIRE", fontsize=9.5, fontweight="bold", color="#8a5a00", ha="center")

    ax.annotate("high >= zone_hi\n(proboy DALNEGO kraya)",
                xy=(x_fire + 0.02, 102.85), xytext=(5.2, 106.7),
                fontsize=9.1, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4), ha="left", va="center")
    ax.annotate("close < zone_lo - w\n(vynos na tseluyu shirinu zony)",
                xy=(x_fire + 0.05, 98.0), xytext=(1.2, 93.5),
                fontsize=9.1, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4), ha="left", va="center")

    ax.set_xlim(-0.4, 12.0)
    ax.set_ylim(91.5, 108.0)
    ax.set_xlabel("12h bars ->", fontsize=9.5)
    ax.set_ylabel("price (uslovno)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_long_panel_real(ax) -> None:
    ax.set_title("LONG setup - REALNYY primer: BTC 2026-04-07 12:00 UTC (confirmed)",
                 fontsize=12.2, fontweight="bold", pad=10, color="#333")

    zone_lo, zone_hi = 68062.86, 69460.00
    w = zone_hi - zone_lo
    close_thr = zone_hi + w

    bars = [
        (-2, "04-06 12h", 69614.91, 70351.46, 68300.00, 68853.66),
        (-1, "04-07 00h", 68853.66, 69247.90, 68071.96, 68345.75),
        (0,  "04-07 12h", 68345.75, 72761.00, 67732.01, 71924.22),
        (1,  "04-08 00h", 71924.22, 72110.65, 71240.99, 71675.12),
        (2,  "04-08 12h", 71675.12, 72857.00, 70707.23, 71069.93),
    ]
    x0 = 4.0
    xs = {k: x0 + (k + 2) * 1.35 for k, *_ in bars}
    x_fire = xs[0]

    ax.add_patch(Rectangle((x0 - 0.9, zone_lo), (xs[2] + 0.9) - (x0 - 0.9), w,
                            facecolor=GRN_FILL, edgecolor=GRN_EDGE, linewidth=1.4,
                            alpha=0.85, zorder=1))
    ax.text(xs[-1], zone_lo - 700, "rb zone (support)\nborn 2024-04-02 00:00 UTC",
            fontsize=8.6, color="#004400", fontweight="bold", ha="left")

    ax.plot([x0 - 0.9, xs[2] + 0.9], [close_thr, close_thr],
            ls=(0, (5, 2, 1, 2)), color=FAR_C, lw=1.4, alpha=0.9)
    ax.text(xs[2] + 1.05, close_thr, f"zone_hi+w={close_thr:,.0f}\n(close-porog)", fontsize=7.8,
            color=FAR_C, va="center", fontweight="bold")

    _draw_ellipsis(ax, x0 - 2.2, x0 - 0.9, (zone_lo + zone_hi) / 2)

    for k, lbl, o, h, l, c in bars:
        x = xs[k]
        is_fire = (k == 0)
        draw_candle(ax, x, o, h, l, c, w=0.75,
                    edge=FIRE_C if is_fire else "#333",
                    fill_up="white", fill_dn="#2a2a2a",
                    lw=1.6 if is_fire else 1.0)
        ax.text(x, l - 550, lbl, fontsize=7.6, color="#555", ha="center", rotation=0)

    ax.plot(x_fire, 66550, marker="^", markersize=13, color=WARN,
            zorder=5, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(x_fire, 65950, "A1-pivot LOW (12h, l[i]<l[i-1] i l[i]<l[i-2])", fontsize=7.6,
            color=WARN, ha="center", fontweight="bold")

    ax.plot(x_fire, 73400, marker="*", markersize=22, color="#ffb400",
            markeredgecolor="#8a5a00", markeredgewidth=0.9, zorder=6)
    ax.text(x_fire, 73900, "FIRE  (i, 2026-04-07 12:00)", fontsize=9, fontweight="bold",
            color="#8a5a00", ha="center")

    ax.annotate(f"low={67732.01:,.0f} <= zone_lo={zone_lo:,.0f}",
                xy=(x_fire - 0.05, 67900), xytext=(xs[-2] - 1.7, 66550),
                fontsize=8.4, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3), ha="left", va="center")
    ax.annotate(f"close={71924.22:,.0f} > {close_thr:,.0f}",
                xy=(x_fire, 71924.22), xytext=(xs[1] - 0.2, 73200),
                fontsize=8.6, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3), ha="left", va="center")

    ax.annotate("Williams n=2 confirm: l[i+1]>l[i] i l[i+2]>l[i]  ->  CONFIRMED",
                xy=(xs[2], 70707.23), xytext=(xs[2] + 0.4, 69700),
                fontsize=8.3, color="#006400", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#006400", lw=1.2), ha="left", va="center")

    ax.set_xlim(x0 - 3.4, xs[2] + 3.0)
    ax.set_ylim(65200, 75200)
    ax.set_xlabel("12h bars ->  (realnye daty BTC)", fontsize=9.5)
    ax.grid(alpha=0.25)


def _draw_rule_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                 boxstyle="round,pad=0.02",
                                 facecolor="#f6f7fb", edgecolor="#2c3e50",
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.5, 0.92, "Formalnoe pravilo FULL_DISP (pervoe qualifying k)",
            ha="center", fontsize=12, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    rule_text = (
        "w = zone_hi - zone_lo\n\n"
        "SHORT:  high[k] >= zone_hi   AND   close[k] < zone_lo - w\n"
        "LONG:   low[k]  <= zone_lo   AND   close[k] > zone_hi + w\n\n"
        "Zone-istochnik = rb (Rejection Block, ODIN bar):\n"
        "  body>0; upper>=2*lower AND upper>=3*body  -> TOP RB (resistance)\n"
        "          lower>=2*upper AND lower>=3*body  -> BOTTOM RB (support)\n"
        "  Zone: LONG=(low, body_bottom)   SHORT=(body_top, high)\n\n"
        "TF: 12h TOLKO. Domen: a_cand[a1_pre_w] - chistyy A1, kak ves' B2.\n"
        "Fire = (bar_idx, direction), pervoe k, odna zona fires MAKS odin raz."
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
    ax.text(0.5, 0.92, "Kak eto nashli . status",
            ha="center", fontsize=12, fontweight="bold", color="#c0392b",
            transform=ax.transAxes)
    notes_text = (
        "*  Sibling naxodki B2C3 (mitigation_block) - identichnaya mekhanika\n"
        "   FULL_DISP@12h, drugoy element (rb = odin dominant-wick bar).\n\n"
        "*  rb daet SAMYY vysokiy WR sredi 4 provereninykh elementov na ETH\n"
        "   (90.12%, n=81) i stabilno vyshe bazlayna na vsekh 3 aktivakh.\n\n"
        "*  Realnyy primer sprava: odin ogromnyy bar (H-L ~5000$) srazu i\n"
        "   probivaet zonu, i zakryvaetsya za dalnim porogom - klassicheskiy\n"
        "   \"vynos+razvorot\" posle 2 let sushchestvovaniya zony (2024-04 -> 2026-04).\n\n"
        "*  STATUS: research-naxodka v G:\\Claude\\research\\, NE portirovana\n"
        "   v ASVK b2_ob.py, NE vxodit v b2_hit/basket - B2C1 (production) ne\n"
        "   tronut ni v kode, ni v etom sravnenii.\n\n"
        "Refs:  G:\\Claude\\research\\block_family_explore.py  .\n"
        "       G:\\ASVK\\lib\\detektory\\rb.py"
    )
    ax.text(0.04, 0.80, notes_text, fontsize=8.9, color="#333",
            ha="left", va="top", transform=ax.transAxes)


def main() -> None:
    out_dir = pathlib.Path("/mnt/g/ASVK/Export_12h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "B2C4_candidate_rb_FULL_DISP.png"

    fig = build_figure()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
