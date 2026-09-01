"""arden_signals_v2 — самодостаточный движок сигналов по методу Р. Арденского, версия 2.

Что нового в v2 (по сравнению с arden_signals.py):
  Грейд и сигнал теперь считаются по СИЛЕ факторов, а не по их числу. Причина —
  стратификация показала, что из пяти OHLC-факторов два (ext5, тело/фитиль) — почти
  шум (~42-44% WR), и подсчёт «в лоб» (conf>=3) забивался ими → всего ~60%. Ранжируем
  факторы по реальному WR и берём силу СИЛЬНЕЙШЕГО присутствующего:

    сила 5  choch (слом структуры)   ~93% WR   ← premium
    сила 4  money_hands (объём+погл.) ~77% WR   ← strong (production)
    сила 3  rsi_div (дивергенция)     ~66%
    сила 2  bos (sweep+rejection)     ~58%
    сила 1  ext5 / тело-фитиль        ~42-44%  (шум-база)

  Это переносит на самодостаточный OHLC-движок находку «возврата сильных сигналов»:
  сильное подтверждение само по себе достаточно, и его не нужно смешивать со слабыми.

СИГНАЛЫ v2 (production):
  premium = сила 5 (CHoCH)             — 90-98% WR (12h/2-6h), редкий;
  strong  = сила >= 4 (CHoCH|money)    — 75-83% WR, в ~2.7x больше сигналов (основной поток);
  wide    = сила >= 3 (+rsi_div)       — ~68% WR, для максимального охвата (опция, <75%).

Всё причинно (только данные до и включая бар-кандидат). Метрика `confirmed` = разворот
устоял 2 бара (Williams n=2), без tp/sl. Оговорка: choch и confirmed частично коррелируют
по построению — часть высокого WR механически ожидаема, но паттерн причинный и торгуемый.

ЗАПУСК:
  python arden_signals_v2.py --csv path/BTCUSDT_1m.csv --tf 12h --name BTC --live
  python arden_signals_v2.py --csv path/SOLUSDT_1m.csv --tf 4h --live
CSV: open_time(ISO8601), open, high, low, close, volume (Binance-формат).
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import pandas as pd

TF_MS = {"2h": 2, "3h": 3, "4h": 4, "6h": 6, "12h": 12, "1d": 24}
MSK_OFFSET_MS = 3 * 3600 * 1000
STRENGTH_LABEL = {1: "шум", 2: "bos", 3: "rsi-div", 4: "money", 5: "CHoCH"}

A2_LEFT_N = 5
A4_BODY_MAX = 0.80
A4_WICK_MIN = 0.03
RSI_N = 14
VOL_N = 20
VOL_K = 1.5


# ─────────────────────────── данные ───────────────────────────

def load_1m(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"open": "float64", "high": "float64",
                                      "low": "float64", "close": "float64", "volume": "float64"})
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def agg_tf(df_1m: pd.DataFrame, tf_h: int) -> pd.DataFrame:
    tf = tf_h * 3600 * 1000
    b = (df_1m["ts"].values // tf) * tf
    g = df_1m.assign(bucket=b).groupby("bucket", sort=True, as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"))
    return g.rename(columns={"bucket": "ts"})


# ─────────────────────── детекторы (причинно) ───────────────────────

def fractals(h, l):
    T = len(h); fh = np.zeros(T, bool); fl = np.zeros(T, bool)
    fh[2:] = (h[2:] > h[1:-1]) & (h[2:] > h[:-2])
    fl[2:] = (l[2:] < l[1:-1]) & (l[2:] < l[:-2])
    return fh, fl


def confirmed_mask(h, l):
    T = len(h); cs = np.zeros(T, bool); cl = np.zeros(T, bool)
    cs[:-2] = (h[1:-1] < h[:-2]) & (h[2:] < h[:-2])
    cl[:-2] = (l[1:-1] > l[:-2]) & (l[2:] > l[:-2])
    return cs, cl, (np.arange(T) <= T - 3)


def prev_swing(is_swing):
    T = len(is_swing); out = np.full(T, -1, np.int64); last = -1
    for i in range(T):
        out[i] = last
        if is_swing[i]:
            last = i
    return out


def wilder_rsi(c, n=RSI_N):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0.0); ls = np.where(d < 0, -d, 0.0)
    T = len(c); ag = np.full(T, np.nan); al = np.full(T, np.nan)
    if T <= n:
        return np.full(T, np.nan)
    ag[n] = g[1:n + 1].mean(); al[n] = ls[1:n + 1].mean()
    for i in range(n + 1, T):
        ag[i] = (ag[i - 1] * (n - 1) + g[i]) / n
        al[i] = (al[i - 1] * (n - 1) + ls[i]) / n
    rs = np.where(al > 0, ag / al, np.inf)
    return 100.0 - 100.0 / (1.0 + rs)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].values; h = df["high"].values; l = df["low"].values
    c = df["close"].values; v = df["volume"].values; ts = df["ts"].values
    T = len(c)
    fh, fl = fractals(h, l)
    cs, cl, conf_ok = confirmed_mask(h, l)
    pfh = prev_swing(fh); pfl = prev_swing(fl)
    rsi = wilder_rsi(c)
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o) / rng
    up_w = (h - np.maximum(o, c)) / rng
    lo_w = (np.minimum(o, c) - l) / rng
    avgv = np.full(T, np.nan)
    for i in range(1, T):
        a = max(0, i - VOL_N)
        if i > a:
            avgv[i] = v[a:i].mean()

    rows = []
    for i in range(T):
        for is_p, direction in ((fh[i], "short"), (fl[i], "long")):
            if not is_p:
                continue
            sh = direction == "short"
            a2 = (h[i] > h[i - A2_LEFT_N:i].max()) if (sh and i >= A2_LEFT_N) else \
                 ((l[i] < l[i - A2_LEFT_N:i].min()) if (not sh and i >= A2_LEFT_N) else False)
            wick = up_w[i] if sh else lo_w[i]
            a4 = (body[i] <= A4_BODY_MAX) and (wick >= A4_WICK_MIN)
            js, jl = pfh[i], pfl[i]
            if sh:
                bos = js >= 0 and h[i] > h[js] and c[i] < h[js]
                choch = jl >= 0 and c[i] < l[jl]
                rdiv = js >= 0 and np.isfinite(rsi[i]) and np.isfinite(rsi[js]) and h[i] > h[js] and rsi[i] < rsi[js]
            else:
                bos = jl >= 0 and l[i] < l[jl] and c[i] > l[jl]
                choch = js >= 0 and c[i] > h[js]
                rdiv = jl >= 0 and np.isfinite(rsi[i]) and np.isfinite(rsi[jl]) and l[i] < l[jl] and rsi[i] > rsi[jl]
            ratio = v[i] / avgv[i] if (np.isfinite(avgv[i]) and avgv[i] > 0) else 0.0
            mid = (h[i] + l[i]) / 2.0
            mh = (ratio >= VOL_K) and ((c[i] < mid) if sh else (c[i] > mid))

            # v2: СИЛА = сила сильнейшего присутствующего фактора (по реальному WR)
            strength = 1 if (a2 or a4) else 0
            if bos:   strength = max(strength, 2)
            if rdiv:  strength = max(strength, 3)
            if mh:    strength = max(strength, 4)
            if choch: strength = 5
            rows.append({
                "ts": int(ts[i]), "bar": i, "direction": direction,
                "confirmable": bool(conf_ok[i]),
                "confirmed": bool((cs[i] if sh else cl[i]) and conf_ok[i]),
                "a2": bool(a2), "a4": bool(a4), "bos": bool(bos), "rsi_div": bool(rdiv),
                "money_hands": bool(mh), "choch": bool(choch),
                "strength": int(strength),
                "premium": bool(strength == 5),
                "strong": bool(strength >= 4),   # ← production
                "wide": bool(strength >= 3),
            })
    return pd.DataFrame(rows)


# ─────────────────────────── отчёт ───────────────────────────

def _wr(df):
    cm = df[df["confirmable"]]; n = len(cm)
    return (100.0 * cm["confirmed"].mean() if n else 0.0), n


def _msk(ts_ms):
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp((ts_ms + MSK_OFFSET_MS) / 1000).strftime("%Y-%m-%d %H:%M MSK")


def report(sig, tf, name):
    b, bn = _wr(sig)
    print(f"\n===== {name} · TF={tf} · v2 =====", file=sys.stderr, flush=True)
    print(f"  фрактал (база)        n={bn:>5}  WR={b:5.1f}%", file=sys.stderr, flush=True)
    for col, lbl in [("wide", "wide  (сила≥3)"), ("strong", "strong(сила≥4) ← production"),
                     ("premium", "premium(CHoCH)")]:
        w, n = _wr(sig[sig[col]])
        print(f"  {lbl:28} n={n:>5}  WR={w:5.1f}%", file=sys.stderr, flush=True)
    print(f"  ── по силе фактора ──", file=sys.stderr, flush=True)
    for s in range(1, 6):
        w, n = _wr(sig[sig["strength"] == s])
        if n:
            print(f"    сила {s} {STRENGTH_LABEL[s]:8s} n={n:>5}  WR={w:5.1f}%", file=sys.stderr, flush=True)


def live_verdict(sig, df):
    last_bar = df.shape[0] - 1
    fired = sig[sig["strong"]]
    if fired.empty:
        print("  вердикт: — FLAT", file=sys.stderr, flush=True); return
    row = fired.iloc[fired["bar"].values.argmax()]
    age = last_bar - int(row["bar"])
    if age <= 2:
        d = "▲ LONG" if row["direction"] == "long" else "▼ SHORT"
        when = "НА ЭТОЙ СВЕЧЕ ✓" if age == 0 else f"{age} бар(а) назад"
        s = int(row["strength"])
        print(f"  вердикт: СЕЙЧАС {d}  сила {s}/5 {STRENGTH_LABEL[s]} "
              f"({_msk(int(row['ts']))}, {when})", file=sys.stderr, flush=True)
    else:
        print(f"  вердикт: — FLAT (последний сигнал {age} баров назад)", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Сигналы по методу Арденского v2 (грейд по силе факторов)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="12h", choices=list(TF_MS))
    ap.add_argument("--name", default="")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    t0 = time.time()
    df = agg_tf(load_1m(args.csv), TF_MS[args.tf])
    sig = compute(df)
    report(sig, args.tf, args.name or args.csv)
    if args.live:
        live_verdict(sig, df)
    if args.save:
        sig.to_parquet(args.save, index=False)
        print(f"  saved: {args.save}", file=sys.stderr, flush=True)
    print(f"  ({time.time()-t0:.1f}s, баров {len(df):,})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
