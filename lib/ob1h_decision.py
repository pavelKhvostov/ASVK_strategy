"""ob1h_decision — «Модуль Арденский 2»: слой решения над 1h-каскадами OB (ASVK-portable).

Тот же принцип, что в fractal12h/decision.py («Модуль Арденский 1»), но на другом
источнике сигналов — интрадей-каскады Liq_OB1h_VC и FVG_OB1h_VC. Они уже дают
OB-сигналы с подтверждением (vc_rb / vc_fvg / vc_snr), но без слоя решения, грейда и
живого вердикта. Здесь мы:

  1) объединяем два семейства (Liq_OB = снятие ликвидности, FVG_OB = заливка имбаланса);
  2) считаем confluence = число сработавших VC-типов (0..3) + согласие семейств
     (рядом по времени тот же сигнал в другом семействе → сильнее);
  3) грейдим уверенность 1..5 (как в Модуле 1);
  4) меряем ЧЕСТНЫЙ исход follow-through (без tp/sl): пошла ли цена в сторону сигнала
     за H часов после подтверждения — прямой аналог `confirmed` из Модуля 1;
  5) даём живой вердикт LONG/SHORT/FLAT по последнему 1h-сигналу.

Reads:
  data/liq_ob1h_vc/ob_stage4_race_canonical_{SYM}.parquet
  data/fvg_ob1h_vc/ob_stage4_race_canonical_{SYM}.parquet
  data/{SYM}USDT_1m.csv
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent / "fractal12h"))
from common import load_1m, agg_tf   # переиспользуем загрузчик/агрегатор из Модуля 1

BASE = pathlib.Path(__file__).resolve().parent.parent / "data"
TF_1H_MS = 3600 * 1000
CROSS_WINDOW_MS = 6 * TF_1H_MS       # окно «согласия семейств» ±6ч
FOLLOW_H = 8                          # горизонт исхода: 8 × 1h = 8ч
GRADE_LABEL = {1: "LOW", 2: "MED", 3: "HIGH", 4: "V.HIGH", 5: "PREMIUM"}


def load_family(sym: str, fam_dir: str, family: str) -> pd.DataFrame:
    p = BASE / fam_dir / f"ob_stage4_race_canonical_{sym}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_parquet(p)
    d = d[d["vc_any"]].copy()
    d["family"] = family
    d["vc_count"] = d[["vc_rb", "vc_fvg", "vc_snr"]].sum(axis=1).astype("int8")
    # момент подтверждения = самый ранний из vc_*_ts (0 = нет)
    tcols = ["vc_rb_ts", "vc_fvg_ts", "vc_snr_ts"]
    vt = d[tcols].where(d[tcols] > 0)
    d["conf_ts"] = vt.min(axis=1).fillna(d["ob_ts"]).astype("int64")
    return d[["ob_ts", "conf_ts", "direction", "family", "vc_count",
              "vc_rb", "vc_fvg", "vc_snr", "vc_triple"]]


def compute_decision_ob(sym: str) -> pd.DataFrame:
    liq = load_family(sym, "liq_ob1h_vc", "liq")
    fvg = load_family(sym, "fvg_ob1h_vc", "fvg")
    both = pd.concat([liq, fvg], ignore_index=True).sort_values("conf_ts").reset_index(drop=True)

    # согласие семейств: есть ли сигнал ДРУГОГО семейства того же направления в окне ±6ч
    cross = np.zeros(len(both), dtype=bool)
    ct = both["conf_ts"].to_numpy()
    dirn = both["direction"].to_numpy()
    famn = both["family"].to_numpy()
    for i in range(len(both)):
        lo, hi = ct[i] - CROSS_WINDOW_MS, ct[i] + CROSS_WINDOW_MS
        near = (ct >= lo) & (ct <= hi) & (dirn == dirn[i]) & (famn != famn[i])
        cross[i] = bool(near.any())
    both["cross_family"] = cross

    # грейд 1..5: база по числу VC + бонусы (triple, согласие семейств)
    score = both["vc_count"].to_numpy().astype(int) + both["cross_family"].astype(int) \
            + both["vc_triple"].astype(int)
    grade = np.clip(score, 1, 5)
    both["signal_grade"] = grade.astype("int8")
    return both


def add_outcome(dec: pd.DataFrame, sym: str, H: int = FOLLOW_H) -> pd.DataFrame:
    df_1h = agg_tf(load_1m(sym), TF_1H_MS, 0)
    ts = df_1h["ts"].to_numpy()
    c = df_1h["close"].to_numpy()
    T = len(ts)
    # индекс 1h-бара подтверждения (floor conf_ts к 1h)
    conf_bucket = (dec["conf_ts"].to_numpy() // TF_1H_MS) * TF_1H_MS
    idx = np.searchsorted(ts, conf_bucket, side="right") - 1
    won = np.zeros(len(dec), dtype=bool)
    ok = np.zeros(len(dec), dtype=bool)
    for k, (i, d) in enumerate(zip(idx, dec["direction"].to_numpy())):
        if 0 <= i < T - H:
            ok[k] = True
            won[k] = (c[i + H] < c[i]) if d == "short" else (c[i + H] > c[i])
    dec = dec.copy()
    dec["confirmable"] = ok
    dec["confirmed"] = won            # follow-through: цена пошла в сторону сигнала за H часов
    return dec


def print_stats(dec: pd.DataFrame, sym: str) -> None:
    def wr(df):
        cm = df[df["confirmable"]]
        n = len(cm); w = int(cm["confirmed"].sum())
        return (100.0 * w / n if n else 0.0), n
    print(f"\n==================== {sym} (Модуль Арденский 2, 1h) ====================",
          file=sys.stderr, flush=True)
    a_wr, a_n = wr(dec)
    print(f"  все 1h-сигналы (Liq+FVG): n={a_n}  follow-through WR={a_wr:.1f}%  (H={FOLLOW_H}ч)",
          file=sys.stderr, flush=True)
    print(f"  ── по грейду уверенности ──", file=sys.stderr, flush=True)
    for g in range(1, 6):
        w, n = wr(dec[dec["signal_grade"] == g])
        if n:
            print(f"    грейд {g} {GRADE_LABEL[g]:8s} n={n:>4}  WR={w:5.1f}%", file=sys.stderr, flush=True)
    print(f"  ── по направлению ──", file=sys.stderr, flush=True)
    for d in ("long", "short"):
        w, n = wr(dec[dec["direction"] == d])
        print(f"    {d:5s}  n={n:>4}  WR={w:5.1f}%", file=sys.stderr, flush=True)
    print(f"  ── согласие семейств Liq+FVG ──", file=sys.stderr, flush=True)
    w, n = wr(dec[dec["cross_family"]])
    print(f"    cross_family  n={n:>4}  WR={w:5.1f}%", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    args = ap.parse_args()
    t0 = time.time()
    dec = compute_decision_ob(args.symbol)
    dec = add_outcome(dec, args.symbol)
    print_stats(dec, args.symbol)
    out = BASE / "fractal12h" / f"ob1h_decision_{args.symbol}.parquet"
    dec.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"  written: {out.name}  ({len(dec)} rows)  {time.time()-t0:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
