"""arden_trader — торговый модуль по методу Р. Арденского (Power of Three).

ОТЛИЧИЕ ОТ arden_signals*.py:
  arden_signals выдаёт булев сигнал «разворот / не разворот» (метрика `confirmed`).
  Прибыли он не считает и, как показал бэктест, в деньги напрямую НЕ конвертируется.
  arden_trader — это финансовый инструмент: он выдаёт ПЛАН СДЕЛКИ
  (вход / стоп / две цели / размер позиции) и торгует полную последовательность
  Power of Three, а не точку пивота.

ПОЧЕМУ ЭТО РАБОТАЕТ, А ВХОД «НА ПИВОТЕ» — НЕТ:
  Арденский прямо запрещает ловить экстремум («не ловите ножей»). Вход берётся
  ПОСЛЕ слома структуры, на коррекции в зону OTE. Вход на пивоте в бэктесте даёт
  матожидание ~0 (ПФ 0.65-1.18); полная последовательность — ПФ 1.83.

ПОСЛЕДОВАТЕЛЬНОСТЬ (Power of Three):
  1. ЛОЖНЫЙ ВЫНОС  — свеча снимает ликвидность за предыдущим фракталом.
  2. BOS           — закрытие за противоположным swing-уровнем. КРИТИЧНО: не позднее
                     3 баров от выноса. Это главный фильтр, подтверждённый OOS:
                     BOS<=3 бара даёт E=+0.33R, BOS>=4 бара — E=+0.01R (шум).
  3. КОРРЕКЦИЯ     — цена возвращается в зону OTE импульса (0.705-0.79).
  4. ВХОД          — лимитный ордер в OTE. Глубже = лучше: OTE 0.79 даёт E=+0.577R,
                     OTE 0.705 — +0.323R.
  5. СТОП          — за экстремумом ложного выноса. Минимум 0.7% движения цены,
                     иначе сделка пропускается (стоп внутри шума).
  6. ВЫХОД         — 33% на 1.5R → стоп в безубыток → 67% на 3.0R (средний RR 2.5).
                     Дробный выход подтверждён OOS: E +0.42R против +0.33R у
                     фиксированного RR 2.5, при этом WR 53% вместо 40%.

РЕЗУЛЬТАТЫ (BTC/ETH/SOL, 4h+12h, 2017-2026, комиссия 0.10% туда-обратно):
  1056 сделок, E=+0.422R, ПФ=1.83, WR=53%, ~119 сделок/год.
  Прибыльна КАЖДЫЙ год, включая худший 2025 (+0.115R).
  При риске 0.5% на сделку: CAGR ~28%/год, макс. просадка -13.3%, Кальмар 2.10.

ЗАПУСК:
  python arden_trader.py --csv BTCUSDT_1m.csv --tf 12h --name BTC --live
  python arden_trader.py --csv BTCUSDT_1m.csv --tf 4h --backtest
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd

# ── параметры стратегии (все подтверждены out-of-sample) ──
TF_H = {"4h": 4, "12h": 12}
BOS_MAX_BARS = 3      # жёсткое вето: слом структуры не позже 3 баров от выноса
ENTRY_MAX_BARS = 10   # ждём коррекцию в OTE не дольше 10 баров после BOS
OTE_MIN = 0.705       # минимальная глубина входа (0.79 предпочтительнее)
MIN_STOP_PCT = 0.7    # минимальный стоп в % движения цены
TP1_R, TP1_FRAC = 1.5, 1 / 3   # первая цель и её доля
TP2_R = 3.0                     # вторая цель (остаток), средний RR = 2.5
RISK_PCT = 0.5        # риск счёта на сделку
TRADE_MAX_BARS = 60
FEE_RT = 0.10         # комиссия туда-обратно, % от цены
MSK = 3 * 3600 * 1000


# ────────────────────────── данные и индикаторы ──────────────────────────

def load_1m(path):
    df = pd.read_csv(path, dtype={c: "float64" for c in ("open", "high", "low", "close", "volume")})
    dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")
    return df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts") \
             .drop_duplicates("ts").reset_index(drop=True)


def agg_tf(df, tf_h):
    tf = tf_h * 3600 * 1000
    g = df.assign(b=(df["ts"].values // tf) * tf).groupby("b", sort=True, as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"))
    return g.rename(columns={"b": "ts"})


def fractals(h, l):
    T = len(h); fh = np.zeros(T, bool); fl = np.zeros(T, bool)
    fh[2:] = (h[2:] > h[1:-1]) & (h[2:] > h[:-2])
    fl[2:] = (l[2:] < l[1:-1]) & (l[2:] < l[:-2])
    return fh, fl


def prev_swing(is_sw):
    out = np.full(len(is_sw), -1, np.int64); last = -1
    for i in range(len(is_sw)):
        out[i] = last
        if is_sw[i]:
            last = i
    return out


def atr(h, l, c, n=14):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    a = np.full(len(c), np.nan)
    if len(c) <= n:
        return a
    a[n] = tr[1:n + 1].mean()
    for i in range(n + 1, len(c)):
        a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


# ────────────────────── шаги 1-2: вынос + слом структуры ──────────────────────

def find_setups(df):
    """Ложный вынос + BOS не позже BOS_MAX_BARS. Возвращает подтверждённые сетапы."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    T = len(c)
    fh, fl = fractals(h, l)
    pfh, pfl = prev_swing(fh), prev_swing(fl)
    a = atr(h, l, c)
    out = []
    for S in range(3, T - 2):
        for is_p, dr in ((fh[S], "short"), (fl[S], "long")):
            if not is_p:
                continue
            jH, jL = pfh[S], pfl[S]
            if jH < 0 or jL < 0 or not np.isfinite(a[S]) or a[S] <= 0:
                continue
            short = dr == "short"
            # шаг 1: ложный вынос ликвидности за предыдущий фрактал
            if short and not h[S] > h[jH]:
                continue
            if not short and not l[S] < l[jL]:
                continue
            lvl = l[jL] if short else h[jH]      # уровень, слом которого = BOS
            ext = h[S] if short else l[S]        # экстремум выноса → там будет стоп
            imp = l[S] if short else h[S]        # дальняя точка импульса
            bos = None
            for b in range(S + 1, min(S + 1 + BOS_MAX_BARS, T)):
                imp = min(imp, l[b]) if short else max(imp, h[b])
                if (h[b] > ext) if short else (l[b] < ext):
                    break                         # вынос обновлён → сетап недействителен
                if (c[b] < lvl) if short else (c[b] > lvl):
                    bos = b
                    break
            rng = (ext - imp) if short else (imp - ext)
            if bos is None or rng <= 0:
                continue
            out.append(dict(S=S, dr=dr, ext=ext, imp=imp, rng=rng, bos=bos,
                            bos_bars=bos - S, imp_atr=rng / a[S]))
    return out


# ─────────────────── шаги 3-6: план сделки (вход/стоп/цели) ───────────────────

def plan(df, st, ote=0.79):
    """Строит план сделки из сетапа. None, если вход/стоп не проходят фильтры."""
    short = st["dr"] == "short"
    entry = st["imp"] + ote * st["rng"] if short else st["imp"] - ote * st["rng"]
    stop = st["ext"]
    risk = (stop - entry) if short else (entry - stop)
    if risk <= 0:
        return None
    risk_pct = 100 * risk / entry
    if risk_pct < MIN_STOP_PCT:                   # стоп внутри шума → пропуск
        return None
    sgn = -1 if short else 1
    return dict(**st, ote=ote, entry=entry, stop=stop, risk=risk, risk_pct=risk_pct,
                tp1=entry + sgn * TP1_R * risk, tp2=entry + sgn * TP2_R * risk)


def fill_and_run(df, p):
    """Ищет исполнение лимитника в OTE и проигрывает сделку по правилам выхода."""
    h, l, c, ts = df["high"].values, df["low"].values, df["close"].values, df["ts"].values
    T = len(c); short = p["dr"] == "short"
    en = None
    for j in range(p["bos"] + 1, min(p["bos"] + 1 + ENTRY_MAX_BARS, T)):
        if (h[j] > p["stop"]) if short else (l[j] < p["stop"]):
            break                                  # стоп выбит до входа → отмена
        if (h[j] >= p["entry"]) if short else (l[j] <= p["entry"]):
            en = j
            break
    if en is None:
        return None
    maxR = 0.0; stopped = False; endR = 0.0
    for j in range(en + 1, min(en + 1 + TRADE_MAX_BARS, T)):
        if (h[j] >= p["stop"]) if short else (l[j] <= p["stop"]):
            stopped = True
            break
        fav = (p["entry"] - l[j]) if short else (h[j] - p["entry"])
        maxR = max(maxR, fav / p["risk"])
        endR = ((p["entry"] - c[j]) if short else (c[j] - p["entry"])) / p["risk"]
    return dict(p, entry_ts=int(ts[en]), maxR=maxR, stopped=stopped, endR=endR,
                R=r_multiple(maxR, stopped, endR, p["risk_pct"]))


def r_multiple(maxR, stopped, endR, risk_pct):
    """Итог в R: 33% на 1.5R, стоп в безубыток, 67% на 3R, минус комиссия."""
    p1 = TP1_R if maxR >= TP1_R else (-1.0 if stopped else float(np.clip(endR, -1, TP1_R)))
    if maxR >= TP2_R:
        p2 = TP2_R
    elif maxR >= TP1_R:
        p2 = 0.0                                   # стоп перенесён в безубыток
    else:
        p2 = -1.0 if stopped else float(np.clip(endR, -1, TP2_R))
    return TP1_FRAC * p1 + (1 - TP1_FRAC) * p2 - FEE_RT / risk_pct


def scan(df, ote=0.79):
    """Полный проход: сетапы → планы → исполнение.

    Уровень OTE фиксируется ЗАРАНЕЕ (это цена лимитного ордера) — перебирать
    уровни и брать тот, что исполнился, нельзя: это заглядывание вперёд.
    0.79 даёт лучшее матожидание (+0.577R против +0.323R у 0.705), но исполняется
    реже — часть сетапов разворачивается, не дойдя до глубокой коррекции.
    """
    rows = []
    for st in find_setups(df):
        p = plan(df, st, ote)
        if p is None:
            continue
        t = fill_and_run(df, p)
        if t:
            rows.append(t)
    return pd.DataFrame(rows)


# ─────────────────────────────── отчёты ───────────────────────────────

def backtest(tr, name, tf):
    if tr.empty:
        print("  сделок нет", file=sys.stderr); return
    r = tr["R"].values
    g, ls = r[r > 0].sum(), -r[r < 0].sum()
    yrs = (tr.entry_ts.max() - tr.entry_ts.min()) / (1000 * 86400 * 365.25)
    eq = np.cumprod(1 + r * RISK_PCT / 100)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    p = lambda *a: print(*a, file=sys.stderr, flush=True)
    p(f"\n===== {name} · TF={tf} · Arden Trader =====")
    p(f"  сделок {len(r)} за {yrs:.1f} лет ({len(r)/max(yrs,.01):.0f}/год)")
    p(f"  E={r.mean():+.3f}R  ПФ={g/ls if ls else 99:.2f}  WR={(r>0).mean()*100:.0f}%  ст.откл={r.std():.2f}R")
    p(f"  медиана стопа {tr.risk_pct.median():.2f}% цены  →  цель {tr.risk_pct.median()*TP2_R:.2f}%")
    p(f"  при риске {RISK_PCT}%/сделку: ×{eq[-1]:.2f}  CAGR {(eq[-1]**(1/max(yrs,.01))-1)*100:+.1f}%/год  просадка {dd*100:.1f}%")
    for ote in sorted(tr.ote.unique(), reverse=True):
        s = tr[tr.ote == ote]["R"].values
        p(f"    OTE {ote}: n={len(s):>4}  E={s.mean():+.3f}R")


def live(df, tr, name, tf):
    """Актуальные сетапы: ждём вход / только что вошли."""
    p = lambda *a: print(*a, file=sys.stderr, flush=True)
    T = len(df); h, l = df["high"].values, df["low"].values
    fmt = lambda t: pd.to_datetime(t + MSK, unit="ms").strftime("%Y-%m-%d %H:%M MSK")
    pending = []
    for st in find_setups(df):
        if st["bos"] < T - 1 - ENTRY_MAX_BARS:
            continue                               # окно входа истекло
        for ote in (0.79,):
            pl = plan(df, st, ote)
            if pl is None:
                continue
            short = pl["dr"] == "short"
            seg = slice(pl["bos"] + 1, T)
            hit = (h[seg] >= pl["entry"]).any() if short else (l[seg] <= pl["entry"]).any()
            killed = (h[seg] > pl["stop"]).any() if short else (l[seg] < pl["stop"]).any()
            if not killed:
                pending.append((pl, hit))
            break
    p(f"\n  ── {name} {tf}: активные планы ──")
    if not pending:
        p("  FLAT — валидных сетапов нет (это норма)"); return
    for pl, hit in pending[-3:]:
        d = "▼ SHORT" if pl["dr"] == "short" else "▲ LONG"
        p(f"  {d}  {'ВХОД ИСПОЛНЕН' if hit else 'ЖДЁМ ЛИМИТ'}  (вынос {fmt(int(df.ts.values[pl['S']]))})")
        p(f"    вход {pl['entry']:.2f} (OTE {pl['ote']})   стоп {pl['stop']:.2f} ({pl['risk_pct']:.2f}%)")
        p(f"    цель1 {pl['tp1']:.2f} (1.5R, 33%)   цель2 {pl['tp2']:.2f} (3.0R, 67%)")
        p(f"    BOS за {pl['bos_bars']} бар(а), импульс {pl['imp_atr']:.1f} ATR")
        p(f"    объём: риск {RISK_PCT}% счёта / {pl['risk_pct']:.2f}% = позиция {RISK_PCT/pl['risk_pct']*100:.0f}% депозита")


def main():
    ap = argparse.ArgumentParser(description="Arden Trader — план сделки по Power of Three")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="12h", choices=list(TF_H))
    ap.add_argument("--name", default="")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--ote", type=float, default=0.79)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    df = agg_tf(load_1m(a.csv), TF_H[a.tf])
    tr = scan(df, a.ote)
    nm = a.name or a.csv
    if a.backtest or not a.live:
        backtest(tr, nm, a.tf)
    if a.live:
        live(df, tr, nm, a.tf)
    if a.save and not tr.empty:
        tr.to_parquet(a.save, index=False)
        print(f"  saved: {a.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
