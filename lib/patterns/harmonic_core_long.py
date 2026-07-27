"""harmonic_core_long — общее ядро для 4 bullish harmonic-паттернов
(Gartley/Bat/Butterfly/Crab), X-A-B-C-D, только LONG (D — дно, PRZ).

Зеркало harmonic_core.py: все роли высоких и низких точек инвертированы.
  SHORT: X=FH, A=FL, B=FH, C=FL, D=FH  (вход SHORT от пика D)
  LONG:  X=FL, A=FH, B=FL, C=FH, D=FL  (вход LONG от дна D)

Коэффициенты идентичны SHORT — описывают размеры ног, не направление:
  AB/XA = (A-B)/(A-X)    — ретрейс XA вниз к B
  BC/AB = (C-B)/(A-B)    — ретрейс AB вверх к C
  CD/BC = (C-D)/(C-B)    — финальная нога к PRZ
  D/XA  = (A-D)/(A-X)    — глубина PRZ относительно XA

  d_beyond_x=False: D > X  (ретрейс — Gartley/Bat)
  d_beyond_x=True:  D < X  (экстеншн за X — Butterfly/Crab)

Структурные проверки:
  B > X (B выше стартовой точки X)
  C < A (C ниже вершины A)
  Per-leg true-extremum: каждая точка — настоящий экстремум своего отрезка.

signal_ts = ts[i_d] + 3*TF_1H_MS  (D подтверждён как фрактал, вход +3h)
direction = "long"
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
from block5_common import fractals

MAX_SPAN_XD_DEFAULT = 504   # ~3 недели


@dataclass
class HarmonicLongParams:
    ab_lo: float
    ab_hi: float
    bc_lo: float = 0.382
    bc_hi: float = 0.886
    cd_lo: float = 0.0
    cd_hi: float = float("inf")
    d_xa_lo: float = 0.0
    d_xa_hi: float = 0.0
    d_beyond_x: bool = False   # False=ретрейс D>X; True=экстеншн D<X
    max_span: int = MAX_SPAN_XD_DEFAULT
    n_workers: int = 30


def _empty_stats() -> dict:
    return {'n_fh': 0, 'n_fl': 0, 'passed_geometry': 0, 'passed_canon': 0}


def _process_x_chunk_long(x_chunk: list[int],
                           h: np.ndarray, l: np.ndarray, c: np.ndarray,
                           fh_list: list[int], fl_list: list[int],
                           p: HarmonicLongParams) -> list[dict]:
    """Один чанк FL-кандидатов для X (нижний пивот)."""
    out: list[dict] = []
    for i_x in x_chunk:
        i_x = int(i_x)
        X = l[i_x]   # X = FL (тrough)

        # A = FH после X (вверх — нога XA)
        a_cands = [i for i in fh_list if i_x < i <= i_x + p.max_span]
        for i_a in a_cands:
            A = h[i_a]
            XA = A - X
            if XA <= 0:
                continue
            # true-extremum XA: нет нового low ниже X, нет нового high выше A
            if l[i_x:i_a + 1].min() < X:
                continue
            if h[i_x:i_a + 1].max() > A:
                continue

            # B = FL после A (вниз — ретрейс AB)
            b_cands = [i for i in fl_list if i_a < i <= i_x + p.max_span]
            for i_b in b_cands:
                B = l[i_b]
                if B <= X:          # B должен быть выше X
                    continue
                ab_ratio = (A - B) / XA
                if not (p.ab_lo <= ab_ratio <= p.ab_hi):
                    continue
                # true-extremum AB
                if h[i_a:i_b + 1].max() > A:
                    continue
                if l[i_a:i_b + 1].min() < B:
                    continue

                # C = FH после B (вверх — ретрейс BC)
                c_cands = [i for i in fh_list if i_b < i <= i_x + p.max_span]
                for i_c in c_cands:
                    C = h[i_c]
                    if C >= A:          # C должен быть ниже A
                        continue
                    AB = A - B
                    bc_ratio = (C - B) / AB
                    if not (p.bc_lo <= bc_ratio <= p.bc_hi):
                        continue
                    # true-extremum BC
                    if l[i_b:i_c + 1].min() < B:
                        continue
                    if h[i_b:i_c + 1].max() > C:
                        continue

                    # D = FL после C (вниз — нога CD к PRZ)
                    d_cands = [i for i in fl_list if i_c < i <= i_x + p.max_span]
                    for i_d in d_cands:
                        D = l[i_d]
                        if p.d_beyond_x and D >= X:   # экстеншн: D должен быть НИЖЕ X
                            continue
                        if not p.d_beyond_x and D <= X:  # ретрейс: D должен быть ВЫШЕ X
                            continue
                        d_xa_ratio = (A - D) / XA
                        if not (p.d_xa_lo <= d_xa_ratio <= p.d_xa_hi):
                            continue
                        # true-extremum CD
                        if h[i_c:i_d + 1].max() > C:
                            continue
                        if l[i_c:i_d + 1].min() < D:
                            continue

                        BC = C - B
                        cd_ratio = (C - D) / BC if BC > 0 else float('nan')
                        if not (p.cd_lo <= cd_ratio <= p.cd_hi):
                            continue

                        out.append({
                            'i_x': i_x, 'i_a': i_a, 'i_b': i_b, 'i_c': i_c, 'i_d': i_d,
                            'X': float(X), 'A': float(A), 'B': float(B),
                            'C': float(C), 'D': float(D),
                            'ab_ratio': float(ab_ratio),
                            'bc_ratio': float(bc_ratio),
                            'cd_ratio': float(cd_ratio),
                            'd_xa_ratio': float(d_xa_ratio),
                            'width_bars': int(i_d - i_x),
                        })
    return out


def detect_harmonic_long_geometry(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                                   ts: np.ndarray,
                                   params: HarmonicLongParams) -> tuple[list[dict], dict]:
    n_bars = len(c)
    stats = _empty_stats()
    if n_bars < 20:
        return [], stats

    h = h.astype(np.float64)
    l = l.astype(np.float64)
    c = c.astype(np.float64)

    FH = fractals(h, 2, 'high')
    FL = fractals(l, 2, 'low')
    stats['n_fh'] = len(FH)
    stats['n_fl'] = len(FL)
    fh_list = FH.tolist()
    fl_list = FL.tolist()

    # X итерируем по FL (нижние пивоты — стартовая точка LONG-паттерна)
    n_workers = max(1, min(params.n_workers, os.cpu_count() or 4))
    x_chunks = np.array_split(FL, n_workers)
    x_chunks = [chunk.tolist() for chunk in x_chunks if len(chunk) > 0]

    if len(x_chunks) > 1:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(_process_x_chunk_long)(chunk, h, l, c, fh_list, fl_list, params)
            for chunk in x_chunks
        )
    else:
        chunk_results = [_process_x_chunk_long(chunk, h, l, c, fh_list, fl_list, params)
                          for chunk in x_chunks]

    patterns: list[dict] = []
    for res in chunk_results:
        patterns.extend(res)

    for pat in patterns:
        pat['ts_d'] = int(ts[pat['i_d']])

    # дедупликация по ts_d (один сигнал на один D-бар)
    seen: set[int] = set()
    uniq: list[dict] = []
    for pat in sorted(patterns, key=lambda p: p['ts_d']):
        if pat['ts_d'] not in seen:
            seen.add(pat['ts_d'])
            uniq.append(pat)

    stats['passed_geometry'] = len(uniq)
    stats['passed_canon'] = len(uniq)   # нет отдельного FVG/BR у harmonic (как SHORT)

    return uniq, stats


def compute_harmonic_long(pattern_name: str, params: HarmonicLongParams,
                           df_1h: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    h = df_1h["high"].to_numpy()
    l = df_1h["low"].to_numpy()
    c = df_1h["close"].to_numpy()
    t_arr = df_1h["ts"].to_numpy()

    patterns, stats = detect_harmonic_long_geometry(h, l, c, t_arr, params)

    rows = []
    for p in patterns:
        rows.append({
            "signal_ts": p['ts_d'] + 3 * TF_1H_MS,
            "direction": "long",
            "pattern": pattern_name,
            "ts_d": p['ts_d'],
            "X": p['X'], "A": p['A'], "B": p['B'], "C": p['C'], "D": p['D'],
            "ab_ratio": p['ab_ratio'],
            "bc_ratio": p['bc_ratio'],
            "cd_ratio": p['cd_ratio'],
            "d_xa_ratio": p['d_xa_ratio'],
            "width_bars": p['width_bars'],
            "status": "CONFIRMED",
        })
    return pd.DataFrame(rows), stats


def print_stats(stats: dict, label: str) -> None:
    import sys
    print(f"  FH={stats['n_fh']}  FL={stats['n_fl']}", file=sys.stderr, flush=True)
    print(f"  passed_geometry (=passed_canon, нет FVG/BR у harmonic)  n={stats['passed_geometry']}  ← {label}",
          file=sys.stderr, flush=True)
