"""A cascade — оркестратор независимых A-фильтров для Williams n=2 фрактал-пивотов (12h).

Каждый A-фильтр — отдельный независимый скрипт (a1_filter.py..a4_filter.py, см. их
докстринги): чистый per-bar геометрический тест, не зависящий от того, прошли ли
предыдущие стадии. Это НЕ то же самое, что старая (до 2026-07-23) слитная версия
этого файла, где `a4_body_wick` физически была кумулятивной `A1 AND A2 AND A3 AND
A4` — из-за чего B3 (`b3_fractal_liquidity.py`) и B4 (`b4_hma.py`), фильтруя по
`a4_body_wick`, случайно тащили в свой домен ещё и A3, хотя по явной инструкции
пользователя (проверено по транскрипту прошлой сессии: "какие результаты без A3
фильтра, только на A1+A2+A4?", "A3 исключили пока из расчетов", "Фильтры a1 a2 a4
остаются") их домен должен быть A1+A2+A4, БЕЗ A3.

Причина бага (диагноз пользователя): A-фильтры должны быть независимыми скриптами,
а не слитой цепочкой AND — так их можно комбинировать под нужды каждого B-блока
явно, а не получать случайные комбинации через побочный эффект кумулятивных колонок.

Выходные колонки a_candidates (одна строка на A1-пивот):
    a1_pre_w    — база (всегда True — иначе строки бы не было)
    a2_indep    — независимый тест A2 (ext_5), БЕЗ AND с a1
    a3_indep    — независимый тест A3 (color), БЕЗ AND с a1/a2
    a4_indep    — независимый тест A4 (body/wick), БЕЗ AND с a1/a2/a3
    a124_pool   — a1_pre_w & a2_indep & a4_indep (без a3) — рабочий домен B3/B4
    confirmable/confirmed — Williams n=2 right-confirmation (см. a1_filter.py)

B1/B2/B9 используют `a1_pre_w` (не изменилось). B3/B4 используют `a124_pool`
(раньше — ошибочно — `a4_body_wick`).

Reads:  G:\\ASVK\\data\\{SYM}USDT_1m.csv
Writes: G:\\ASVK\\data\\fractal12h\\a_candidates_{SYM}_{start}_{end}.parquet
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_1m, agg_12h, DATA_OUT
import a1_filter
import a2_filter
import a3_filter
import a4_filter


def compute_a_cascade(df_12h: pd.DataFrame) -> pd.DataFrame:
    """Независимые A1..A4 + confirmation, собранные в один DataFrame (одна строка на A1-пивот)."""
    ts = df_12h["ts"].to_numpy()

    a1 = a1_filter.compute_a1(df_12h)
    conf = a1_filter.compute_confirmation(df_12h)
    a2 = a2_filter.compute_a2(df_12h)
    a3 = a3_filter.compute_a3(df_12h)
    a4 = a4_filter.compute_a4(df_12h)

    frames = []
    for direction in ("short", "long"):
        idx = np.flatnonzero(a1[direction])
        a124 = a1[direction] & a2[direction] & a4[direction]
        wick_pct = a4["up_wick_pct"] if direction == "short" else a4["lo_wick_pct"]
        df = pd.DataFrame({
            "pivot_open_ts_ms": ts[idx].astype(np.int64),
            "direction":        direction,
            "a1_pre_w":         True,
            "a2_indep":         a2[direction][idx],
            "a3_indep":         a3[direction][idx],
            "a4_indep":         a4[direction][idx],
            "a124_pool":        a124[idx],
            "confirmable":      conf["confirmable"][idx],
            "confirmed":        conf[direction][idx] & conf["confirmable"][idx],
            "body_pct":         a4["body_pct"][idx],
            "wick_pct":         wick_pct[idx],
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values("pivot_open_ts_ms").reset_index(drop=True)
    return out


def print_stage_stats(cand: pd.DataFrame) -> None:
    """Печать статистики: база A1, независимые A2/A3/A4 (справочно — каждый САМ ПО СЕБЕ
    поверх A1, не кумулятивно друг с другом), и рабочий домен a124_pool (реальный
    вход для B3/B4)."""
    def _line(label, mask):
        m = mask & cand["a1_pre_w"]
        c_mask = m & cand["confirmable"]
        n_conf = int(cand.loc[c_mask, "confirmed"].sum())
        n_total_c = int(c_mask.sum())
        wr = 100.0 * n_conf / n_total_c if n_total_c else 0.0
        print(f"  {label:24s}  n={int(m.sum()):>5,}  conf={n_conf:>5,}/{n_total_c:>5,}  WR={wr:5.2f}%",
              file=sys.stderr, flush=True)

    _line("a1_pre_w (база)", cand["a1_pre_w"])
    _line("a2_indep (справочно)", cand["a2_indep"])
    _line("a3_indep (справочно)", cand["a3_indep"])
    _line("a4_indep (справочно)", cand["a4_indep"])
    _line("a124_pool (B3/B4 домен)", cand["a124_pool"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--start", default="2020-01-01",
                    help="lower bound (inclusive) для pivot ts; 12h aggregation считается по всем данным до filter")
    ap.add_argument("--end", default="2026-07-08",
                    help="upper bound (exclusive)")
    args = ap.parse_args()

    UTC = timezone.utc
    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms   = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)

    print(f"a_cascade: {args.symbol} {args.start} → {args.end}", file=sys.stderr, flush=True)
    t0 = time.time()

    df_1m = load_1m(args.symbol)
    df_12h = agg_12h(df_1m)  # ВСЯ история → корректный left-lookback
    print(f"  12h bars total: {len(df_12h):,}", file=sys.stderr, flush=True)

    cand = compute_a_cascade(df_12h)

    # Фильтр по окну (после cascade, чтобы left-lookback не терял контекст)
    mask = (cand["pivot_open_ts_ms"] >= start_ms) & (cand["pivot_open_ts_ms"] < end_ms)
    cand = cand[mask].reset_index(drop=True)

    n_bars_in_window = int(((df_12h["ts"] >= start_ms) & (df_12h["ts"] < end_ms)).sum())
    print(f"  12h bars in window: {n_bars_in_window:,}", file=sys.stderr, flush=True)
    print_stage_stats(cand)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / f"a_candidates_{args.symbol}_{args.start}_{args.end}.parquet"
    cand.to_parquet(out, index=False, compression="zstd", compression_level=9)
    print(f"written: {out.name}  ({len(cand):,} rows)  total {time.time()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
