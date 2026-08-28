"""verdict — живой направленный вердикт «сейчас LONG / SHORT / FLAT» (ASVK-portable).

Отвечает на вопрос в стиле ASVK: на ТЕКУЩЕЙ (последней закрытой) 12h-свече система
считает лонг или шорт? Сигнал fractal12h рождается на пивоте-фрактале (FH → short,
FL → long) и подтверждается слоем decision. Вердикт = направление самого свежего
активного сигнала.

Иерархия tier'ов (сильный → слабый), первый сработавший даёт вердикт:
    A  decision_choch  ~95% WR   (премиум-разворот)
    B  decision_hit    79-86%    (зона&подтв ∪ CHoCH) — основной поток
    C  decision_pot    83-86%    (зона&ликв&подтв) — консервативный

Свежесть: пивот на 12h подтверждается через 2 бара (Williams n=2), поэтому сигнал
«текущий», пока прошло ≤ RECENCY_BARS баров и цена не выбила пивот. На баре age=0
сигнал только что сформировался — «на этой свече».

Вывод (stdout, машиночитаемый + человекочитаемый), для потребления asvk.py TUI.

Reads:
  data/fractal12h/decision_hits_{SYM}_{start}_{end}.parquet
  data/{SYM}USDT_1m.csv   (последняя закрытая 12h-свеча)
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_1m, agg_12h, DATA_OUT, TF_12H_MS

RECENCY_BARS = 4      # сигнал считается «текущим» ≤ 4 баров (2 суток) после пивота
MSK_OFFSET_MS = 3 * 3600 * 1000

# tier → (колонка, метка, ориентир WR)
TIERS = [
    ("decision_choch", "A·CHoCH",  "~95%"),
    ("decision_hit",   "B·setup",  "79-86%"),
    ("decision_pot",   "C·pot",    "83-86%"),
]


def _msk(ts_ms: int) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp((ts_ms + MSK_OFFSET_MS) / 1000).strftime("%Y-%m-%d %H:%M MSK")


def compute_verdict(symbol: str, start: str, end: str) -> dict:
    dpath = DATA_OUT / f"decision_hits_{symbol}_{start}_{end}.parquet"
    if not dpath.exists():
        raise FileNotFoundError(f"Missing: {dpath} (запусти run_fractal12h / decision)")
    dec = pd.read_parquet(dpath)

    df_12h = agg_12h(load_1m(symbol))
    ts = df_12h["ts"].to_numpy()
    latest_ts = int(ts[-1])
    latest_close_ms = latest_ts + TF_12H_MS
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}
    latest_idx = len(ts) - 1

    out = {
        "symbol": symbol,
        "latest_bar_ts_ms": latest_ts,
        "latest_bar_msk": _msk(latest_ts),
        "verdict": "FLAT",
        "tier": None,
        "on_current_candle": False,
        "age_bars": None,
        "signal_ts_ms": None,
        "signal_msk": None,
        "tiers": {},
    }

    best = None  # (priority, pivot_idx, direction, tier_label, wr)
    for pri, (col, label, wr) in enumerate(TIERS):
        if col not in dec.columns:
            continue
        sig = dec[dec[col]]
        if sig.empty:
            out["tiers"][col] = None
            continue
        # самый свежий сигнал этого tier
        row = sig.loc[sig["pivot_open_ts_ms"].idxmax()]
        p_ts = int(row["pivot_open_ts_ms"])
        p_idx = ts_to_idx.get(p_ts, None)
        age = (latest_idx - p_idx) if p_idx is not None else None
        rec = {"direction": row["direction"], "signal_ts_ms": p_ts,
               "signal_msk": _msk(p_ts), "age_bars": age, "wr": wr}
        out["tiers"][col] = rec
        if age is not None and age <= RECENCY_BARS:
            if best is None or pri < best[0] or (pri == best[0] and p_idx > best[1]):
                best = (pri, p_idx, row["direction"], label, wr, p_ts, age)

    if best is not None:
        _, p_idx, direction, label, wr, p_ts, age = best
        out["verdict"] = "LONG" if direction == "long" else "SHORT"
        out["tier"] = label
        out["wr"] = wr
        out["age_bars"] = int(age)
        out["on_current_candle"] = bool(age == 0)
        out["signal_ts_ms"] = p_ts
        out["signal_msk"] = _msk(p_ts)
    return out


def print_verdict(v: dict) -> None:
    arrow = {"LONG": "▲ LONG", "SHORT": "▼ SHORT", "FLAT": "— FLAT"}[v["verdict"]]
    print(f"\n═══ ВЕРДИКТ {v['symbol']} · свеча закрыта {v['latest_bar_msk']} ═══", file=sys.stderr, flush=True)
    if v["verdict"] == "FLAT":
        print(f"  СЕЙЧАС: {arrow}  — нет свежего сигнала (≤{RECENCY_BARS} баров)", file=sys.stderr, flush=True)
    else:
        when = "НА ЭТОЙ СВЕЧЕ ✓" if v["on_current_candle"] else f"{v['age_bars']} бар(а/ов) назад"
        print(f"  СЕЙЧАС: {arrow}  [tier {v['tier']}, WR {v.get('wr','?')}]  сигнал {when}",
              file=sys.stderr, flush=True)
        print(f"          пивот {v['signal_msk']}", file=sys.stderr, flush=True)
    # что говорит каждый tier (последний сигнал)
    for col, label, _ in TIERS:
        rec = v["tiers"].get(col)
        if rec:
            d = "LONG" if rec["direction"] == "long" else "SHORT"
            fresh = "" if (rec["age_bars"] is not None and rec["age_bars"] <= RECENCY_BARS) else "  (устарел)"
            print(f"    {label:9s} последний: {d:5s}  {rec['age_bars']} бар назад{fresh}",
                  file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    ap.add_argument("--json", action="store_true", help="печатать JSON в stdout")
    args = ap.parse_args()

    t0 = time.time()
    v = compute_verdict(args.symbol, args.start, args.end)
    print_verdict(v)
    if args.json:
        print(json.dumps(v, ensure_ascii=False))
    print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
