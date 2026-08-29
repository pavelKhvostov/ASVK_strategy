"""session_sweep — «Модуль Арденский 3»: сессионный вынос ликвидности (ASVK-portable).

ПРИНЦИПИАЛЬНО ДРУГАЯ логика, чем Модули 1/2 (там — фрактал+структура). Здесь движок —
ВРЕМЯ СУТОК + снятие диапазона, из сетапа №1 Арденского («торговля от сессий», модель
London Test / Judas swing, market-sessions.pdf + sistema.pdf):

  1) за каждый день строим ДИАПАЗОН АЗИИ  = [min low, max high] в окне 02:00–09:00 MSK;
  2) в окне ФРАНКФУРТ/ЛОНДОН 09:00–13:00 MSK ищем ВЫНОС экстремума диапазона с возвратом:
        SHORT: high > asia_high  И  close < asia_high   (сняли ликвидность сверху → вниз)
        LONG:  low  < asia_low   И  close > asia_low    (сняли снизу → вверх)
     берём ПЕРВЫЙ такой бар за день на каждую сторону;
  3) исход БЕЗ tp/sl — follow-through: пошла ли цена в сторону сигнала за H баров.

ТФ детекции — 15m. Причинность: диапазон Азии закрыт до окна Лондона; вынос — по бару окна.
Крипта торгуется 24/7, сессии слабее, чем на форексе — результат может оказаться слабым.

Reads:  data/{SYM}USDT_1m.csv
Writes: data/fractal12h/m3_session_{SYM}.parquet
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

TF_15M_MS = 15 * 60 * 1000
MSK_OFF = 3 * 3600 * 1000
DAY_MS = 24 * 3600 * 1000
HOUR_MS = 3600 * 1000
ASIA = range(2, 9)        # 02:00–09:00 MSK
LONDON = range(9, 13)     # 09:00–13:00 MSK (Франкфурт+Лондон killzone)
HORIZONS = (4, 8, 16)     # 1ч / 2ч / 4ч в 15m-барах


def _msk(ts_ms: int) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp((ts_ms + MSK_OFF) / 1000).strftime("%Y-%m-%d %H:%M MSK")


def compute_session(sym: str, df_1m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = agg_tf(df_1m, TF_15M_MS, 0).reset_index(drop=True)
    ts = df["ts"].to_numpy()
    msk = ts + MSK_OFF
    df["day"] = (msk // DAY_MS).astype("int64")
    df["hour"] = ((msk % DAY_MS) // HOUR_MS).astype("int64")
    h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()

    rows = []
    for day, g in df.groupby("day", sort=True):
        asia = g[g["hour"].isin(list(ASIA))]
        lon = g[g["hour"].isin(list(LONDON))].sort_values("ts")
        if len(asia) < 4 or lon.empty:
            continue
        a_hi = asia["high"].max(); a_lo = asia["low"].min()
        # первый вынос вверх и первый вынос вниз в окне Лондона
        for side, cond, direction in (
            ("hi", (lon["high"].to_numpy() > a_hi) & (lon["close"].to_numpy() < a_hi), "short"),
            ("lo", (lon["low"].to_numpy() < a_lo) & (lon["close"].to_numpy() > a_lo), "long"),
        ):
            if cond.any():
                j = int(np.argmax(cond))
                brow = lon.iloc[j]
                rows.append({
                    "signal_ts": int(brow["ts"]), "direction": direction,
                    "asia_hi": float(a_hi), "asia_lo": float(a_lo),
                    "close": float(brow["close"]),
                })
    sig = pd.DataFrame(rows).sort_values("signal_ts").reset_index(drop=True)

    # исход follow-through по 15m
    idx = np.searchsorted(ts, sig["signal_ts"].to_numpy(), side="left")
    for H in HORIZONS:
        won = np.zeros(len(sig), dtype=bool); ok = np.zeros(len(sig), dtype=bool)
        for k, (i, d) in enumerate(zip(idx, sig["direction"])):
            if 0 <= i < len(ts) - H:
                ok[k] = True
                won[k] = (c[i + H] < c[i]) if d == "short" else (c[i + H] > c[i])
        sig[f"ok{H}"] = ok; sig[f"win{H}"] = won
    return sig, df


def print_stats(sig: pd.DataFrame, sym: str) -> None:
    print(f"\n============ Модуль Арденский 3 · {sym} (сессионный вынос) ============",
          file=sys.stderr, flush=True)
    print(f"  сигналов всего: {len(sig)}  "
          f"(SHORT {int((sig.direction=='short').sum())}, LONG {int((sig.direction=='long').sum())})",
          file=sys.stderr, flush=True)
    print(f"  H(15m) │ все        LONG       SHORT", file=sys.stderr, flush=True)
    for H in HORIZONS:
        def wr(m):
            s = sig[m & sig[f"ok{H}"]]
            return (100.0 * s[f"win{H}"].mean() if len(s) else 0.0), int(len(s))
        aw, an = wr(pd.Series(True, index=sig.index))
        lw, ln = wr(sig.direction == "long")
        sw, sn = wr(sig.direction == "short")
        print(f"  {H:2} ({H*15//60}ч) │ {aw:4.1f}% n={an:<4} {lw:4.1f}% n={ln:<4} {sw:4.1f}% n={sn:<4}",
              file=sys.stderr, flush=True)


def live_verdict(sig: pd.DataFrame, df: pd.DataFrame) -> None:
    last_ts = int(df["ts"].to_numpy()[-1])
    if sig.empty:
        return
    row = sig.iloc[-1]
    age_h = (last_ts - int(row["signal_ts"])) / HOUR_MS
    fresh = age_h <= 8
    d = "▲ LONG" if row["direction"] == "long" else "▼ SHORT"
    print(f"  вердикт: {'СЕЙЧАС ' + d if fresh else '— FLAT'}  "
          f"(последний вынос: {_msk(int(row['signal_ts']))}, {age_h:.0f}ч назад)",
          file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    sig, df = compute_session(args.symbol, load_1m(args.symbol))
    print_stats(sig, args.symbol)
    live_verdict(sig, df)
    if args.save:
        out = DATA_OUT / f"m3_session_{args.symbol}.parquet"
        sig.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"  ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
