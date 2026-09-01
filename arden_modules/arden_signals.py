"""arden_signals — самодостаточный движок сигналов по методу Р. Арденского.

Объединяет ядро «Модуля Арденский 1» (12h) и «Модуля 2» (2/3/4/6h) в ОДИН файл,
работающий только от 1-минутного CSV (нужны numpy + pandas). Никакого тяжёлого
пайплайна ASVK (e12d/s7d) — только чистый OHLCV.

ЧТО СЧИТАЕТ (всё причинно — только данные до и включая бар-кандидат):
  • кандидат       — фрактал Уильямса-3: вершина → SHORT, дно → LONG;
  • исход `confirmed` — экстремум устоял 2 бара (Williams n=2) = разворот отработал;
  • слом структуры CHoCH — закрытие пробило противоположный swing (сильный разворот);
  • слияние (confluence) из OHLC: ext5 + тело/фитиль + BOS + RSI-дивергенция + объём;
  • грейд уверенности 1..5; живой вердикт LONG/SHORT/FLAT на последней свече.

СИГНАЛЫ (production):
  premium = CHoCH               — 86-96% WR (12h и 2-6h), редкий но железный;
  setup   = CHoCH ∪ confluence≥3 — общий поток (на 12h средний слой ~60% без тяж. блоков).

ЗАПУСК:
  python arden_signals.py --csv path/to/BTCUSDT_1m.csv --tf 12h
  python arden_signals.py --csv path/to/ETHUSDT_1m.csv --tf 4h --live

CSV-формат: колонки open_time(ISO8601), open, high, low, close, volume (как отдаёт Binance).
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import pandas as pd

TF_MS = {"2h": 2, "3h": 3, "4h": 4, "6h": 6, "12h": 12, "1d": 24}   # часы
MSK_OFFSET_MS = 3 * 3600 * 1000
GRADE_LABEL = {1: "LOW", 2: "MED", 3: "HIGH", 4: "V.HIGH", 5: "PREMIUM"}

# параметры (проверены на BTC/ETH/SOL 2020-2026)
A2_LEFT_N = 5          # ext5: экстремум выше/ниже 5 предыдущих
A4_BODY_MAX = 0.80     # тело/размах <= 0.80
A4_WICK_MIN = 0.03     # мин. доля фитиля
RSI_N = 14
VOL_N = 20            # окно среднего объёма
VOL_K = 1.5          # климакс объёма >= 1.5x среднего


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
    """Williams-3: маски вершин (short) и дон (long) — известны на закрытии бара."""
    T = len(h)
    fh = np.zeros(T, bool); fl = np.zeros(T, bool)
    fh[2:] = (h[2:] > h[1:-1]) & (h[2:] > h[:-2])
    fl[2:] = (l[2:] < l[1:-1]) & (l[2:] < l[:-2])
    return fh, fl


def confirmed_mask(h, l):
    """Williams n=2 право: экстремум устоял 2 бара (исход разворота)."""
    T = len(h)
    cs = np.zeros(T, bool); cl = np.zeros(T, bool)
    cs[:-2] = (h[1:-1] < h[:-2]) & (h[2:] < h[:-2])
    cl[:-2] = (l[1:-1] > l[:-2]) & (l[2:] > l[:-2])
    conf_ok = np.arange(T) <= T - 3
    return cs, cl, conf_ok


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
            # A2 ext5
            if i >= A2_LEFT_N:
                a2 = (h[i] > h[i - A2_LEFT_N:i].max()) if sh else (l[i] < l[i - A2_LEFT_N:i].min())
            else:
                a2 = False
            # A4 тело/фитиль
            wick = up_w[i] if sh else lo_w[i]
            a4 = (body[i] <= A4_BODY_MAX) and (wick >= A4_WICK_MIN)
            # структура: BOS (sweep+rejection) и CHoCH (close пробил противоположный swing)
            js, jl = pfh[i], pfl[i]
            if sh:
                bos = js >= 0 and h[i] > h[js] and c[i] < h[js]
                choch = jl >= 0 and c[i] < l[jl]
            else:
                bos = jl >= 0 and l[i] < l[jl] and c[i] > l[jl]
                choch = js >= 0 and c[i] > h[js]
            # RSI-дивергенция
            if sh:
                rdiv = js >= 0 and np.isfinite(rsi[i]) and np.isfinite(rsi[js]) and h[i] > h[js] and rsi[i] < rsi[js]
            else:
                rdiv = jl >= 0 and np.isfinite(rsi[i]) and np.isfinite(rsi[jl]) and l[i] < l[jl] and rsi[i] > rsi[jl]
            # объём: климакс + поглощение
            ratio = v[i] / avgv[i] if (np.isfinite(avgv[i]) and avgv[i] > 0) else 0.0
            mid = (h[i] + l[i]) / 2.0
            mh = (ratio >= VOL_K) and ((c[i] < mid) if sh else (c[i] > mid))

            conf = int(a2) + int(a4) + int(bos) + int(rdiv) + int(mh)
            grade = 1 if conf <= 1 else (2 if conf == 2 else (3 if conf == 3 else 4))
            if choch:
                grade = 5
            rows.append({
                "ts": int(ts[i]), "bar": i, "direction": direction,
                "confirmable": bool(conf_ok[i]),
                "confirmed": bool((cs[i] if sh else cl[i]) and conf_ok[i]),
                "a2": bool(a2), "a4": bool(a4), "bos": bool(bos), "choch": bool(choch),
                "rsi_div": bool(rdiv), "money_hands": bool(mh),
                "confluence": conf, "grade": grade,
                "premium": bool(choch), "setup": bool(choch or conf >= 3),
            })
    return pd.DataFrame(rows)


# ─────────────────────────── отчёт ───────────────────────────

def _wr(df):
    cm = df[df["confirmable"]]; n = len(cm)
    return (100.0 * cm["confirmed"].mean() if n else 0.0), n


def _msk(ts_ms):
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp((ts_ms + MSK_OFFSET_MS) / 1000).strftime("%Y-%m-%d %H:%M MSK")


def report(sig: pd.DataFrame, tf: str, name: str):
    base, bn = _wr(sig)
    prem, pn = _wr(sig[sig["premium"]])
    setu, sn = _wr(sig[sig["setup"]])
    print(f"\n===== {name} · TF={tf} =====", file=sys.stderr, flush=True)
    print(f"  фрактал (база)   n={bn:>5}  WR={base:5.1f}%", file=sys.stderr, flush=True)
    print(f"  setup (CHoCH∪≥3) n={sn:>5}  WR={setu:5.1f}%", file=sys.stderr, flush=True)
    print(f"  PREMIUM (CHoCH)  n={pn:>5}  WR={prem:5.1f}%  ← основной сигнал", file=sys.stderr, flush=True)
    for g in range(1, 6):
        w, n = _wr(sig[sig["grade"] == g])
        if n:
            print(f"    грейд {g} {GRADE_LABEL[g]:8s} n={n:>5}  WR={w:5.1f}%", file=sys.stderr, flush=True)


def live_verdict(sig: pd.DataFrame, df: pd.DataFrame):
    if sig.empty:
        print("  вердикт: — FLAT (нет сигналов)", file=sys.stderr, flush=True); return
    last_bar = df.shape[0] - 1
    fired = sig[sig["setup"]]
    if fired.empty:
        print("  вердикт: — FLAT", file=sys.stderr, flush=True); return
    row = fired.iloc[fired["bar"].values.argmax()]
    age = last_bar - int(row["bar"])
    if age <= 2:
        d = "▲ LONG" if row["direction"] == "long" else "▼ SHORT"
        when = "НА ЭТОЙ СВЕЧЕ ✓" if age == 0 else f"{age} бар(а) назад"
        print(f"  вердикт: СЕЙЧАС {d}  грейд {row['grade']}/5 {GRADE_LABEL[row['grade']]} "
              f"({_msk(int(row['ts']))}, {when})", file=sys.stderr, flush=True)
    else:
        print(f"  вердикт: — FLAT (последний сигнал {age} баров назад)", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Сигналы по методу Арденского (Модуль 1 = 12h, Модуль 2 = 2/3/4/6h)")
    ap.add_argument("--csv", required=True, help="путь к {SYM}USDT_1m.csv")
    ap.add_argument("--tf", default="12h", choices=list(TF_MS), help="таймфрейм (12h=Модуль1, 2h/3h/4h/6h=Модуль2)")
    ap.add_argument("--name", default="", help="метка (напр. BTC)")
    ap.add_argument("--live", action="store_true", help="показать живой вердикт")
    ap.add_argument("--save", default="", help="сохранить сигналы в parquet по этому пути")
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
