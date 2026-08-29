"""direction — направленческая машина состояний LONG / SHORT / FLAT (ASVK-portable).

Усиливает live-логику «сейчас лонг или шорт» поверх слоя decision. В отличие от
verdict.py (простое окно свежести ≤N баров), здесь стойка держится по СМЫСЛУ метода
Арденского, без tp/sl — только направление:

  • сигнал на пивоте задаёт стойку (FH→SHORT с якорем = вершина, FL→LONG с якорем = дно);
  • стойка живёт, пока цена не ИНВАЛИДИРОВАЛА пивот — закрытие за ложный вынос:
        SHORT снимается, если close пробил выше якоря-вершины;
        LONG  снимается, если close пробил ниже якоря-дна;
  • встречный сигнал ПЕРЕВОРАЧИВАЕТ стойку (flip);
  • одноимённый сигнал ОБНОВЛЯЕТ якорь на более свежий пивот;
  • приоритет силы: сигнал tier A (CHoCH) перебивает слабый противоположный.

Причинность строгая: состояние на баре i считается только по сигналам и ценам до i
включительно — значит, серия состояний сама честно бэктестится (стойка на каждом баре
была известна на его закрытии). Живой вердикт = состояние на последнем закрытом баре.

Reads:
  data/fractal12h/decision_hits_{SYM}_{start}_{end}.parquet
  data/{SYM}USDT_1m.csv
Writes (опц.):
  data/fractal12h/direction_state_{SYM}_{start}_{end}.parquet
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

MSK_OFFSET_MS = 3 * 3600 * 1000

# tier → приоритет (меньше = сильнее) и метка. Стойку задаёт самый сильный доступный сигнал.
TIER_PRI = {"decision_choch": 0, "decision_hit": 1, "decision_pot": 2}
TIER_LABEL = {"decision_choch": "A·CHoCH", "decision_hit": "B·setup", "decision_pot": "C·pot"}


def _msk(ts_ms: int) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp((ts_ms + MSK_OFFSET_MS) / 1000).strftime("%Y-%m-%d %H:%M MSK")


def compute_state(dec: pd.DataFrame, df_12h: pd.DataFrame) -> pd.DataFrame:
    """Причинная серия состояний по барам 12h."""
    ts = df_12h["ts"].to_numpy()
    h = df_12h["high"].to_numpy()
    l = df_12h["low"].to_numpy()
    c = df_12h["close"].to_numpy()
    T = len(ts)
    ts_to_idx = {int(t): k for k, t in enumerate(ts)}

    # сигналы по бару: bar_idx -> (direction, priority, tier, anchor)
    sig = {}
    for _, r in dec.iterrows():
        i = ts_to_idx.get(int(r["pivot_open_ts_ms"]))
        if i is None:
            continue
        for col, pri in TIER_PRI.items():
            if col in dec.columns and bool(r.get(col, False)):
                cur = sig.get(i)
                if cur is None or pri < cur[1]:
                    anchor = h[i] if r["direction"] == "short" else l[i]
                    sig[i] = (r["direction"], pri, col, float(anchor))
                break  # взяли сильнейший tier этого пивота

    state = np.array(["flat"] * T, dtype=object)
    src_pri = np.full(T, 9, dtype=int)
    src_col = np.array([""] * T, dtype=object)
    src_ts = np.zeros(T, dtype=np.int64)
    anchor = np.full(T, np.nan)

    cur_state, cur_anchor, cur_pri, cur_col, cur_ts = "flat", np.nan, 9, "", 0
    for i in range(T):
        s = sig.get(i)
        if s is not None:
            d, pri, col, anc = s
            want = "short" if d == "short" else "long"
            if cur_state == "flat":
                cur_state, cur_anchor, cur_pri, cur_col, cur_ts = want, anc, pri, col, int(ts[i])
            elif cur_state == want:
                cur_anchor, cur_pri, cur_col, cur_ts = anc, pri, col, int(ts[i])  # refresh якоря
            else:
                # встречный: переворот, если он не слабее текущего (pri <= cur_pri)
                if pri <= cur_pri:
                    cur_state, cur_anchor, cur_pri, cur_col, cur_ts = want, anc, pri, col, int(ts[i])
        # инвалидация закрытием за якорь
        if cur_state == "short" and np.isfinite(cur_anchor) and c[i] > cur_anchor:
            cur_state, cur_anchor, cur_pri, cur_col, cur_ts = "flat", np.nan, 9, "", 0
        elif cur_state == "long" and np.isfinite(cur_anchor) and c[i] < cur_anchor:
            cur_state, cur_anchor, cur_pri, cur_col, cur_ts = "flat", np.nan, 9, "", 0
        state[i], anchor[i], src_pri[i], src_col[i], src_ts[i] = cur_state, cur_anchor, cur_pri, cur_col, cur_ts

    return pd.DataFrame({
        "ts": ts, "high": h, "low": l, "close": c,
        "state": state, "anchor": anchor, "src_tier": src_col, "src_ts": src_ts,
    })


def live_verdict(st: pd.DataFrame) -> dict:
    last = st.iloc[-1]
    ts_last = int(last["ts"])
    v = {
        "latest_bar_ts_ms": ts_last, "latest_bar_msk": _msk(ts_last),
        "state": last["state"], "verdict": {"long": "LONG", "short": "SHORT", "flat": "FLAT"}[last["state"]],
    }
    if last["state"] != "flat":
        # сколько баров держится текущая стойка
        run = 0
        for j in range(len(st) - 1, -1, -1):
            if st.iloc[j]["state"] == last["state"] and int(st.iloc[j]["src_ts"]) == int(last["src_ts"]):
                run += 1
            else:
                break
        anc = float(last["anchor"])
        px = float(last["close"])
        dist_pct = (anc - px) / px * 100 if last["state"] == "short" else (px - anc) / px * 100
        v.update({
            "tier": TIER_LABEL.get(last["src_tier"], last["src_tier"]),
            "signal_ts_ms": int(last["src_ts"]), "signal_msk": _msk(int(last["src_ts"])),
            "held_bars": run, "anchor": anc,
            "dist_to_invalidation_pct": round(dist_pct, 2),
        })
    return v


def print_state_stats(st: pd.DataFrame, sym: str) -> None:
    T = len(st)
    n_long = int((st["state"] == "long").sum())
    n_short = int((st["state"] == "short").sum())
    n_flat = int((st["state"] == "flat").sum())
    flips = int((st["state"].values[1:] != st["state"].values[:-1]).sum())
    print(f"\n=== DIRECTION state {sym} (баров {T:,}) ===", file=sys.stderr, flush=True)
    print(f"  LONG {100*n_long/T:4.1f}%  SHORT {100*n_short/T:4.1f}%  FLAT {100*n_flat/T:4.1f}%  "
          f"переключений: {flips}", file=sys.stderr, flush=True)


def eval_direction(st: pd.DataFrame, horizons=(1, 2, 4)) -> None:
    """Объективная метрика без tp/sl: совпадает ли стойка с последующим движением цены.
    Для бара со стойкой correct = цена через H баров пошла в сторону стойки.
    База = доля роста рынка за H (up-bias). SHORT-точность — главный тест (против тренда)."""
    c = st["close"].to_numpy()
    state = st["state"].to_numpy()
    T = len(c)
    print(f"  H │ стойка(all)  LONG      SHORT     │ база↑", file=sys.stderr, flush=True)
    for H in horizons:
        up = c[H:] > c[:-H]                      # рынок вырос за H (для базы)
        base_up = 100.0 * up.mean()
        tot = cl = ln = lc = sn = sc = 0
        for i in range(T - H):
            s = state[i]
            if s == "flat":
                continue
            win = (c[i + H] > c[i]) if s == "long" else (c[i + H] < c[i])
            tot += 1; cl += int(win)
            if s == "long":  ln += 1; lc += int(win)
            else:            sn += 1; sc += int(win)
        wr = 100.0 * cl / tot if tot else 0.0
        wl = 100.0 * lc / ln if ln else 0.0
        ws = 100.0 * sc / sn if sn else 0.0
        print(f"  {H} │ {wr:5.1f}% n={tot:<4} {wl:5.1f}% n={ln:<4} {ws:5.1f}% n={sn:<4} │ {base_up:5.1f}%",
              file=sys.stderr, flush=True)


def print_verdict(v: dict, sym: str) -> None:
    arrow = {"LONG": "▲ LONG", "SHORT": "▼ SHORT", "FLAT": "— FLAT"}[v["verdict"]]
    print(f"\n═══ ВЕРДИКТ {sym} · свеча закрыта {v['latest_bar_msk']} ═══", file=sys.stderr, flush=True)
    if v["verdict"] == "FLAT":
        print(f"  СЕЙЧАС: {arrow}  — нет активной стойки", file=sys.stderr, flush=True)
    else:
        print(f"  СЕЙЧАС: {arrow}  [tier {v['tier']}]  держится {v['held_bars']} бар(а/ов)",
              file=sys.stderr, flush=True)
        print(f"          сигнал {v['signal_msk']}; до инвалидации {v['dist_to_invalidation_pct']}% "
              f"(закрытие за {v['anchor']:.2f})", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", required=True)
    ap.add_argument("--save", action="store_true", help="сохранить серию состояний в parquet")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--eval", action="store_true", help="метрика направления (без tp/sl)")
    args = ap.parse_args()

    t0 = time.time()
    dpath = DATA_OUT / f"decision_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    if not dpath.exists():
        raise FileNotFoundError(f"Missing: {dpath} (запусти run_fractal12h / decision)")
    dec = pd.read_parquet(dpath)
    df_12h = agg_12h(load_1m(args.symbol))

    st = compute_state(dec, df_12h)
    print_state_stats(st, args.symbol)
    if args.eval:
        eval_direction(st)
    v = live_verdict(st)
    print_verdict(v, args.symbol)

    if args.save:
        out = DATA_OUT / f"direction_state_{args.symbol}_{args.start}_{args.end}.parquet"
        st.to_parquet(out, index=False, compression="zstd", compression_level=9)
        print(f"  written: {out.name}", file=sys.stderr, flush=True)
    if args.json:
        print(json.dumps(v, ensure_ascii=False))
    print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
