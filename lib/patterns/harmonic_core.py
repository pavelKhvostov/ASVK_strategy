"""harmonic_core — общее ядро для 4 bearish harmonic-паттернов (Gartley/Bat/
Butterfly/Crab), X-A-B-C-D, только SHORT (D — вершина). Портировано verbatim
из канон-верифицированного G:\\Claude\\patterns\\harmonic\\harmonic_core.py
("полноценный эксперимент" 2026-07-25: Gartley 254, Bat 77, Butterfly 133,
Crab 116 уникальных паттернов, BTCUSDT 1h 2020-01 → 2026-07) на ASVK-portable
контракт (df_1h передаётся вызывающим кодом).

ВАЖНО (сознательное решение, не менять молча "для единообразия" с
top-family/five_point): вход СРАЗУ в точке D, БЕЗ FVG и БЕЗ отдельного
пробоя (BR) — harmonic-паттерны торгуются от завершения структуры, а не от
подтверждённого слома уровня. MA-фильтр (use_ma_filter) по умолчанию
ВЫКЛЮЧЕН — это наш добавленный слой (2a/2c/2d), не часть канона harmonic
trading, и был диагностирован как причина искусственного 4-паттернового
провала до "полного эксперимента" (см. память asvk-short-pattern-scanners-
status.md).

Жёстко проверяются (одинаково определены во всех источниках):
  AB/XA в [ab_lo, ab_hi]; BC/AB в [bc_lo, bc_hi] (обычно широкий, 0.382-0.886
  у всех 4); D/XA в [d_xa_lo, d_xa_hi] — ГЛАВНЫЙ коэффициент паттерна.
  d_beyond_x=True требует D>X (Butterfly/Crab, экстеншн); False требует D<X
  (Gartley/Bat, ретрейс внутри X-A).
CD/BC считается, но НЕ фильтруется (определение расходится между
источниками). Плюс структура B<X, C>A и per-leg true-extremum на каждом
отрезке (тот же приём, что в Double/Triple Top).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from common import TF_1H_MS
from block5_common import fractals, load_hma_wma, ma_filter_short_ok

MAX_SPAN_XD_DEFAULT = 504   # ~3 недели — harmonic-паттерны разворачиваются дольше top/треугольников


@dataclass
class HarmonicParams:
    ab_lo: float
    ab_hi: float
    bc_lo: float = 0.382
    bc_hi: float = 0.886
    cd_lo: float = 0.0
    cd_hi: float = float("inf")
    d_xa_lo: float = 0.0
    d_xa_hi: float = 0.0
    d_beyond_x: bool = False
    max_span: int = MAX_SPAN_XD_DEFAULT
    use_ma_filter: bool = False
    n_workers: int = 30


def _empty_stats() -> dict:
    return {'n_fh': 0, 'n_fl': 0, 'passed_geometry': 0, 'passed_canon': 0}


def _process_x_chunk(x_chunk: list[int], h: np.ndarray, l: np.ndarray, c: np.ndarray,
                      hma_mhull: Optional[np.ndarray], hma_shull: Optional[np.ndarray],
                      wma50: Optional[np.ndarray],
                      fh_list: list[int], fl_list: list[int], p: HarmonicParams) -> list[dict]:
    out: list[dict] = []
    for i_x in x_chunk:
        i_x = int(i_x)
        X = h[i_x]

        a_cands = [i for i in fl_list if i_x < i <= i_x + p.max_span]
        for i_a in a_cands:
            A = l[i_a]
            XA = X - A
            if XA <= 0:
                continue
            if h[i_x:i_a + 1].max() > X:
                continue
            if l[i_x:i_a + 1].min() < A:
                continue

            b_cands = [i for i in fh_list if i_a < i <= i_x + p.max_span]
            for i_b in b_cands:
                B = h[i_b]
                if B >= X:
                    continue
                ab_ratio = (B - A) / XA
                if not (p.ab_lo <= ab_ratio <= p.ab_hi):
                    continue
                if h[i_a:i_b + 1].max() > B:
                    continue
                if l[i_a:i_b + 1].min() < A:
                    continue

                c_cands = [i for i in fl_list if i_b < i <= i_x + p.max_span]
                for i_c in c_cands:
                    C = l[i_c]
                    if C <= A:
                        continue
                    AB = B - A
                    bc_ratio = (B - C) / AB
                    if not (p.bc_lo <= bc_ratio <= p.bc_hi):
                        continue
                    if l[i_b:i_c + 1].min() < C:
                        continue
                    if h[i_b:i_c + 1].max() > B:
                        continue

                    d_cands = [i for i in fh_list if i_c < i <= i_x + p.max_span]
                    for i_d in d_cands:
                        D = h[i_d]
                        if p.d_beyond_x and D <= X:
                            continue
                        if not p.d_beyond_x and D >= X:
                            continue
                        d_xa_ratio = (D - A) / XA
                        if not (p.d_xa_lo <= d_xa_ratio <= p.d_xa_hi):
                            continue
                        if h[i_c:i_d + 1].max() > D:
                            continue
                        if l[i_c:i_d + 1].min() < C:
                            continue

                        BC = B - C
                        cd_ratio = (D - C) / BC if BC > 0 else float('nan')
                        if not (p.cd_lo <= cd_ratio <= p.cd_hi):
                            continue

                        if p.use_ma_filter:
                            if hma_mhull is None or not ma_filter_short_ok(
                                    i_d, c, hma_mhull, hma_shull, wma50):
                                continue

                        out.append({
                            'i_x': i_x, 'i_a': i_a, 'i_b': i_b, 'i_c': i_c, 'i_d': i_d,
                            'X': float(X), 'A': float(A), 'B': float(B), 'C': float(C), 'D': float(D),
                            'ab_ratio': float(ab_ratio), 'bc_ratio': float(bc_ratio),
                            'cd_ratio': float(cd_ratio), 'd_xa_ratio': float(d_xa_ratio),
                            'width_bars': int(i_d - i_x),
                        })
    return out


def detect_harmonic_geometry(h: np.ndarray, l: np.ndarray, c: np.ndarray, ts: np.ndarray,
                              params: HarmonicParams,
                              hma_mhull: Optional[np.ndarray] = None,
                              hma_shull: Optional[np.ndarray] = None,
                              wma50: Optional[np.ndarray] = None) -> tuple[list[dict], dict]:
    n_bars = len(c)
    stats = _empty_stats()
    if n_bars < 20:
        return [], stats

    h = h.astype(np.float64); l = l.astype(np.float64); c = c.astype(np.float64)

    FH = fractals(h, 2, 'high')
    FL = fractals(l, 2, 'low')
    stats['n_fh'] = len(FH); stats['n_fl'] = len(FL)
    fh_list = FH.tolist(); fl_list = FL.tolist()

    n_workers = max(1, min(params.n_workers, os.cpu_count() or 4))
    x_chunks = np.array_split(FH, n_workers)
    x_chunks = [chunk.tolist() for chunk in x_chunks if len(chunk) > 0]

    if len(x_chunks) > 1:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_x_chunk)(chunk, h, l, c, hma_mhull, hma_shull, wma50,
                                       fh_list, fl_list, params)
            for chunk in x_chunks
        )
    else:
        chunk_results = [_process_x_chunk(chunk, h, l, c, hma_mhull, hma_shull, wma50,
                                           fh_list, fl_list, params)
                          for chunk in x_chunks]

    patterns: list[dict] = []
    for res in chunk_results:
        patterns.extend(res)

    for p in patterns:
        p['ts_d'] = int(ts[p['i_d']])

    seen: set[int] = set(); uniq: list[dict] = []
    for p in sorted(patterns, key=lambda p: p['ts_d']):
        if p['ts_d'] not in seen:
            seen.add(p['ts_d']); uniq.append(p)
    stats['passed_geometry'] = len(uniq)
    stats['passed_canon'] = len(uniq)   # нет отдельного FVG/BR-слоя у harmonic

    return uniq, stats


def compute_harmonic(pattern_name: str, params: HarmonicParams, df_1h: pd.DataFrame,
                      symbol: str) -> tuple[pd.DataFrame, dict]:
    h = df_1h["high"].to_numpy(); l = df_1h["low"].to_numpy(); c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    hma_mhull = hma_shull = wma50 = None
    if params.use_ma_filter:
        hma_mhull, hma_shull, wma50 = load_hma_wma(symbol, t_arr)

    patterns, stats = detect_harmonic_geometry(h, l, c, t_arr, params, hma_mhull, hma_shull, wma50)

    rows = []
    for p in patterns:
        rows.append({
            "signal_ts": p['ts_d'] + 3 * TF_1H_MS,
            "direction": "short",
            "pattern": pattern_name,
            "ts_d": p['ts_d'],
            "X": p['X'], "A": p['A'], "B": p['B'], "C": p['C'], "D": p['D'],
            "ab_ratio": p['ab_ratio'], "bc_ratio": p['bc_ratio'],
            "cd_ratio": p['cd_ratio'], "d_xa_ratio": p['d_xa_ratio'],
            "width_bars": p['width_bars'],
            "status": "CONFIRMED",
        })
    return pd.DataFrame(rows), stats


def print_stats(stats: dict, label: str) -> None:
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry (= passed_canon, нет FVG/BR у harmonic)   n={stats['passed_geometry']}  ← {label}",
          file=sys.stderr, flush=True)
