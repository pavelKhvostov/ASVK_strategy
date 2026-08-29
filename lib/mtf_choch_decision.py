"""mtf_choch_decision — «Модуль Арденский 2»: CHoCH-развороты на 2h/3h/4h/6h.

То же ядро, что у Модуля 1 (fractal12h): фрактал Уильямса-3 → исход `confirmed`
(экстремум устоял 2 бара) → слом структуры CHoCH (b_structure) → грейд → живой вердикт.
Отличие: применяется к средним ТФ (2/3/4/6h) и использует ЛЁГКОЕ, чисто-OHLC слияние
(структура + RSI-дивергенция + объём + тело/фитиль), т.к. тяжёлые B1..B9 (e12d/s7d)
для этих ТФ не считаются.

Ключевая находка (прогон BTC/ETH/SOL 2020-2026): премиум-сигнал CHoCH держит 86-93% WR
на 2-6h — как на 12h — но срабатывает в 5-10 раз чаще (~60/год против ~4/год на 12h).
Именно CHoCH — production-сигнал Модуля 2; conf≥3 (~58%) — вспомогательный средний слой.

Оговорка: CHoCH и `confirmed` частично коррелируют по построению (оба ловят решительную
разворотную свечу) — часть WR механически ожидаема, но паттерн причинный и торгуемый.

Reads:  data/{SYM}USDT_1m.csv
Writes: data/fractal12h/m2_choch_{SYM}_{TF}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent / "fractal12h"))
from common import load_1m, agg_tf, DATA_OUT
import a_cascade, b_structure, b6_divergence, b7_money_hands

TFS = {"2h": 2, "3h": 3, "4h": 4, "6h": 6}
CONFLUENCE_COLS = ["a2_indep", "a4_indep", "bstruct_bos", "b6_hit", "b7_hit"]
GRADE_LABEL = {1: "LOW", 2: "MED", 3: "HIGH", 4: "V.HIGH", 5: "PREMIUM"}
MSK_OFFSET_MS = 3 * 3600 * 1000
RECENCY_BARS = 2   # на средних ТФ сигнал «текущий» ≤ 2 баров


def _msk(ts_ms: int) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp((ts_ms + MSK_OFFSET_MS) / 1000).strftime("%Y-%m-%d %H:%M MSK")


def compute_module2(sym: str, tf_h: int, df_1m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = agg_tf(df_1m, tf_h * 3600 * 1000, 0)
    cand = a_cascade.compute_a_cascade(df)
    bs = b_structure.compute_bstruct(cand, df)[["pivot_open_ts_ms", "direction", "bstruct_choch", "bstruct_bos"]]
    b6 = b6_divergence.compute_b6(cand, df)[["pivot_open_ts_ms", "direction", "b6_hit"]]
    b7 = b7_money_hands.compute_b7(cand, df)[["pivot_open_ts_ms", "direction", "b7_hit"]]
    m = (cand.merge(bs, on=["pivot_open_ts_ms", "direction"], how="left")
             .merge(b6, on=["pivot_open_ts_ms", "direction"], how="left")
             .merge(b7, on=["pivot_open_ts_ms", "direction"], how="left"))
    for c in ["bstruct_choch", "bstruct_bos", "b6_hit", "b7_hit"]:
        m[c] = m[c].fillna(False).astype(bool)

    m["confluence"] = m[CONFLUENCE_COLS].sum(axis=1).astype("int8")
    m["m2_choch"]   = m["bstruct_choch"]                 # PRODUCTION-сигнал (премиум)
    m["m2_conf3"]   = m["confluence"] >= 3               # средний слой
    m["m2_hit"]     = m["m2_choch"] | m["m2_conf3"]      # общий поток

    # грейд: CHoCH → 5; иначе по числу факторов
    grade = np.where(m["confluence"] >= 3, 3, np.where(m["confluence"] == 2, 2, 1))
    grade = np.where(m["bstruct_choch"], 5, grade)
    m["signal_grade"] = grade.astype("int8")
    return m, df


def wr(df: pd.DataFrame) -> tuple[float, int]:
    cm = df[df["confirmable"]]
    n = len(cm); w = int(cm["confirmed"].sum())
    return (100.0 * w / n if n else 0.0), n


def live_verdict(m: pd.DataFrame, df_tf: pd.DataFrame, tf_ms: int) -> dict:
    ts = df_tf["ts"].to_numpy()
    latest_ts = int(ts[-1]); latest_idx = len(ts) - 1
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}
    out = {"latest_bar_msk": _msk(latest_ts), "verdict": "FLAT", "grade": None,
           "direction": None, "age_bars": None, "signal_msk": None}
    sig = m[m["m2_hit"]]
    if sig.empty:
        return out
    row = sig.loc[sig["pivot_open_ts_ms"].idxmax()]
    p_idx = ts_to_idx.get(int(row["pivot_open_ts_ms"]))
    if p_idx is None:
        return out
    age = latest_idx - p_idx
    if age <= RECENCY_BARS:
        out.update({
            "verdict": "LONG" if row["direction"] == "long" else "SHORT",
            "grade": int(row["signal_grade"]), "direction": row["direction"],
            "age_bars": int(age), "on_current_candle": bool(age == 0),
            "signal_msk": _msk(int(row["pivot_open_ts_ms"])),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    df_1m = load_1m(args.symbol)

    print(f"\n============ Модуль Арденский 2 · {args.symbol} ============", file=sys.stderr, flush=True)
    print(f"  ТФ  | CHoCH WR (n)      | conf≥3 WR (n)    | вердикт сейчас", file=sys.stderr, flush=True)
    for name, hh in TFS.items():
        m, df = compute_module2(args.symbol, hh, df_1m)
        cw, cn = wr(m[m["m2_choch"]])
        c3w, c3n = wr(m[m["m2_conf3"]])
        v = live_verdict(m, df, hh * 3600 * 1000)
        vtxt = "— FLAT" if v["verdict"] == "FLAT" else \
               f"{'▲LONG' if v['verdict']=='LONG' else '▼SHORT'} г{v['grade']}/5 " \
               f"{'НА СВЕЧЕ✓' if v.get('on_current_candle') else str(v['age_bars'])+'б'}"
        print(f"  {name:3} | {cw:5.1f}% ({cn:>4})     | {c3w:5.1f}% ({c3n:>5})  | {vtxt}",
              file=sys.stderr, flush=True)
        if args.save:
            out = DATA_OUT / f"m2_choch_{args.symbol}_{name}.parquet"
            m.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
