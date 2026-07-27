"""B4 MA-family — sub-basket B4C1..B4C6, все на механике FULL_DISP.

B4C1/B4C2 портированы из ~/smc-warehouse/scripts/фрактал-12h/b4_hma.py (WSL,
read-only источник), изначально с механикой SWEEP (WR 57-70%, слишком слабо, B4C1
даже не переносился). Разносторонний research (G:\\Claude\\research\\ma_family_explore.py
— 6 типов MA × 7 длин × 5 TF на BTC, затем кросс-проверка на ETH/SOL) показал: причина
слабости — механика, не индикатор/длина/тип. С FULL_DISP (решительное смещение close
за уровень на >=0.5×ATR14, вместо простого касания) практически любая MA даёт высокий
WR. Жадный отбор (маргинальная новизна поверх уже выбранного) на BTC дал 4 доп.
кандидата (B4C3-B4C6), которые почти удваивают объём блока. Развёрнуто 2026-07-25.

Canonical semantic (все шесть — FULL_DISP, margin = 0.5 * ATR14(12h)[i]):
  SHORT: high12[i] > MA_prev  AND  close12[i] < MA_prev - margin
  LONG:  low12[i]  < MA_prev  AND  close12[i] > MA_prev + margin

  B4C1  HMA-78   12h∪D  (multi-TF OR, canon-длина)
  B4C2  HMA-200  D      (canon-длина)
  B4C3  THMA-9   12h    (research: короткая длина, самый крупный по n прирост)
  B4C4  WMA-50   D      (research, плоская WMA без Hull-обёртки)
  B4C5  THMA-9   D
  B4C6  EHMA-20  D

b4_hit = b4c1 OR b4c2 OR ... OR b4c6 (все шесть — реальные production-условия).

ВАЖНО (предупреждение из research, не убирать): B4C3-B4C6 подобраны жадным
алгоритмом НА BTC — при кросс-проверке всех четырёх разом на остальных 10 активах
WR вырос только на 5 из 10 (AVAX/DOGE/LINK/LTC/XRP, +0.9..+2.4pp), а на 5 просел
(ETH/SOL/ADA/BNB/DOT, −1.6..−5.7pp) при том, что объём почти утраивается везде
одинаково. Т.е. эффект неоднородный по активам — решение развернуть принято
пользователем осознанно, несмотря на смешанный кросс-активный результат.

LIVE (past-only): MA_prev — значение mhull ПРЕДЫДУЩЕГО уже закрытого бара нужной TF.
Для TF=12h — тот же ряд, что и pivot (prev = i-1). Для TF=D — didx = searchsorted(td,
ts_pivot, 'right')-1 (D-бар, которому принадлежит pivot), затем prev = didx-1 (день ДО
дня pivot-бара — на момент закрытия 12h-бара текущий D-бар ещё не закрыт). Проверено
на реальных датах (00:00 UTC -> зазор 0ч, 12:00 UTC -> зазор 12ч) — оба случая
используют последний ФАКТИЧЕСКИ закрытый D-бар, без lookahead.

Pool: A1+A2+A4, без A3 (a_cand[a124_pool], см. a_cascade.py) — как у B3, по явной
инструкции пользователя (A3 исключён из домена).

MA — level-1 shared indicator, см. G:\\ASVK\\lib\\trendline.py (variants 12h78, D78,
D200, 12h9Thma, D50Wma, D9Thma, D20Ehma). ATR14(12h) — level-1 формула (lib/maxv.py).

Reads:
  G:\\ASVK\\data\\fractal12h\\a_candidates_{SYM}_{start}_{end}.parquet
  G:\\ASVK\\data\\trendline\\trendline_{SYM}_{variant}_*.parquet (последний по дате)
Writes:
  G:\\ASVK\\data\\fractal12h\\b4_hits_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — для trendline (level-1)
from common import load_1m, agg_12h, DATA_OUT
from trendline import latest_trendline_path


# (b4cN, variant, TF="12h" использует тот же 12h-ряд, что и pivot; иначе — прошлый закрытый D-бар)
SUB_CONDITIONS = [
    ("b4c1a", "12h78", "12h"),
    ("b4c1b", "D78", "D"),
    ("b4c2", "D200", "D"),
    ("b4c3", "12h9Thma", "12h"),
    ("b4c4", "D50Wma", "D"),
    ("b4c5", "D9Thma", "D"),
    ("b4c6", "D20Ehma", "D"),
]


def load_ma_variant(symbol: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (ts, mhull) для заданного trendline-варианта."""
    tl = pd.read_parquet(latest_trendline_path(symbol, variant=variant), columns=["ts", "mhull"])
    return tl["ts"].to_numpy(), tl["mhull"].to_numpy()


def atr14_sma(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """ATR14 = SMA(TR, 14) — canon-совместимо (не Wilder EWM), см. lib/maxv.py."""
    n = len(h)
    c_prev = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - c_prev), np.abs(l - c_prev)])
    tr[0] = 0.0
    csum = np.cumsum(tr)
    atr = np.full(n, np.nan)
    atr[14:] = (csum[14:] - csum[:n - 14]) / 14
    return atr


def _full_disp(bar_h, bar_l, bar_c, hv, direction, margin) -> bool:
    if hv is None or (isinstance(hv, float) and np.isnan(hv)):
        return False
    if direction == "short":
        return bar_h > hv and bar_c < hv - margin
    return bar_l < hv and bar_c > hv + margin


def _live_prev(t_tf: np.ndarray, ma_arr: np.ndarray, ts_pivot: int, same_series: bool,
               pi: int) -> float:
    """LIVE-safe значение MA на предыдущем ЗАКРЫТОМ баре нужной TF."""
    if same_series:
        prev = pi - 1
    else:
        idx = int(np.searchsorted(t_tf, ts_pivot, side="right")) - 1
        prev = idx - 1
    if prev < 0 or prev >= len(ma_arr):
        return float("nan")
    return ma_arr[prev]


def compute_b4(a_cand: pd.DataFrame, df_12h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    t12 = df_12h["ts"].to_numpy()
    h12 = df_12h["high"].to_numpy()
    l12 = df_12h["low"].to_numpy()
    c12 = df_12h["close"].to_numpy()
    atr12 = atr14_sma(h12, l12, c12)

    series = {}
    for name, variant, tf in SUB_CONDITIONS:
        t_tf, ma_arr = load_ma_variant(symbol, variant)
        series[name] = (t_tf, ma_arr, tf == "12h")

    ts_to_idx12 = {int(t): k for k, t in enumerate(t12)}
    pool = a_cand[a_cand["a124_pool"]].copy()
    rows = []
    for _, row in pool.iterrows():
        ts_pivot = int(row["pivot_open_ts_ms"])
        direction = row["direction"]
        pi = ts_to_idx12.get(ts_pivot)
        if pi is None:
            continue
        bar_h, bar_l, bar_c = h12[pi], l12[pi], c12[pi]
        margin = 0.5 * atr12[pi] if not np.isnan(atr12[pi]) else 0.0

        fires = {}
        for name, variant, tf in SUB_CONDITIONS:
            t_tf, ma_arr, same_series = series[name]
            hv = _live_prev(t_tf, ma_arr, ts_pivot, same_series, pi)
            fires[name] = _full_disp(bar_h, bar_l, bar_c, hv, direction, margin)

        b4c1 = fires["b4c1a"] or fires["b4c1b"]
        b4c2 = fires["b4c2"]
        b4c3 = fires["b4c3"]
        b4c4 = fires["b4c4"]
        b4c5 = fires["b4c5"]
        b4c6 = fires["b4c6"]

        rows.append({
            "pivot_open_ts_ms": ts_pivot,
            "direction": direction,
            "confirmable": bool(row["confirmable"]),
            "confirmed": bool(row["confirmed"]),
            "b4c1": b4c1,
            "b4c2": b4c2,
            "b4c3": b4c3,
            "b4c4": b4c4,
            "b4c5": b4c5,
            "b4c6": b4c6,
            "b4_hit": b4c1 or b4c2 or b4c3 or b4c4 or b4c5 or b4c6,
        })
    return pd.DataFrame(rows)


def print_stats(hits: pd.DataFrame) -> None:
    for col in ["b4c1", "b4c2", "b4c3", "b4c4", "b4c5", "b4c6", "b4_hit"]:
        m = hits[col]
        n = int(m.sum())
        cm = m & hits["confirmable"]
        n_c = int(cm.sum())
        n_conf = int(hits.loc[cm, "confirmed"].sum())
        wr = 100.0 * n_conf / n_c if n_c else 0.0
        marker = "  ← B4" if col == "b4_hit" else ""
        print(f"  {col:8s}  n={n:>4,}  conf={n_conf:>3,}/{n_c:>3,}  WR={wr:5.2f}%{marker}",
              file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-08")
    args = ap.parse_args()

    print(f"b4_hma (ASVK-portable, B4C1..B4C6): {args.symbol} {args.start} → {args.end}",
          file=sys.stderr, flush=True)
    t0 = time.time()

    a_path = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    a_cand = pd.read_parquet(a_path)
    print(f"  A1+A2+A4 candidates domain: {int(a_cand['a124_pool'].sum()):,}",
          file=sys.stderr, flush=True)

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)

    hits = compute_b4(a_cand, df_12h, args.symbol)
    print_stats(hits)

    out = DATA_OUT / f"b4_hits_{args.symbol}_{args.start}_{args.end}.parquet"
    hits.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(hits):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
