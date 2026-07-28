"""block5_common — общие helpers для Block 5 (наши 12 паттернов: H&S уже был,
плюс 7 новых top-family + 4 harmonic). Портировано из G:\\Claude\\patterns\\
(WSL/joblib, статичный CSV, инлайн-FVG/HMA/WMA) на живой ASVK-portable
контракт: df_1h передаётся вызывающим кодом (run_patterns.py), FVG читается
из events_e12d (общий детектор проекта, как в hs_top.py), HMA(78)/WMA(50) —
из level-1 shared indicators (lib/trendline.py, lib/wma.py), а не своя
инлайн-формула.

Различие с бэктестом в G:\\Claude\\patterns\\: там FVG считался напрямую по
3 свечам (low[i-1]>high[i+1]) на том же датасете, здесь — через уже готовый
canon-детектор events_e12d (element=fvg). Числа поэтому будут БЛИЗКИМИ, но не
обязаны совпадать один-в-один — так же, как уже отмечено в hs_top.py при
переносе с WSL-локальной HMA на shared trendline.py.
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # G:\ASVK\lib — level-1
from common import latest_events_path
from trendline import latest_trendline_path
from wma import latest_wma_path


def fractals(arr: np.ndarray, n: int = 2, kind: str = 'high') -> np.ndarray:
    """Williams N=2 fractal indices (verbatim из всех scan_*.py в G:\\Claude\\patterns\\)."""
    N = len(arr)
    mask = np.zeros(N, dtype=bool)
    for i in range(n, N - n):
        if kind == 'high':
            if all(arr[i] >= arr[i-k] for k in range(1, n+1)) and \
               all(arr[i] >  arr[i+k] for k in range(1, n+1)):
                mask[i] = True
        else:
            if all(arr[i] <= arr[i-k] for k in range(1, n+1)) and \
               all(arr[i] <  arr[i+k] for k in range(1, n+1)):
                mask[i] = True
    return np.where(mask)[0]

def load_hma_wma(symbol: str, t_arr: np.ndarray):
    """Возвращает (hma_mhull, hma_shull, wma50) reindexed на t_arr — verbatim
    паттерн из hs_top.py. hma_shull[i] == hma_mhull[i-2] (уже сдвинуто)."""
    tl = pd.read_parquet(latest_trendline_path(symbol, variant="1h78"),
                          columns=["ts", "mhull", "shull"])
    hma_mhull = pd.Series(tl["mhull"].to_numpy(), index=tl["ts"].to_numpy()).reindex(t_arr).to_numpy()
    hma_shull = pd.Series(tl["shull"].to_numpy(), index=tl["ts"].to_numpy()).reindex(t_arr).to_numpy()

    wm = pd.read_parquet(latest_wma_path(symbol), columns=["ts", "wma50"])
    wma50 = pd.Series(wm["wma50"].to_numpy(), index=wm["ts"].to_numpy()).reindex(t_arr).to_numpy()
    return hma_mhull, hma_shull, wma50


def ma_filter_short_ok(i: int, closes: np.ndarray, hma_mhull: np.ndarray,
                        hma_shull: np.ndarray, wma50: np.ndarray,
                        use_2a: bool = True, use_2c: bool = True,
                        use_2d: bool = True) -> bool:
    """2a/2c/2d фильтры SHORT на баре i. Флаги use_* отключают условие
    (заменяют на True) — для паттернов где фильтр статистически избыточен:
      descending: use_2d=False (death-cross 42%%, форма редко в даунтренде);
      sym_tri:    use_2a=False, use_2c=False (MA ещё нейтральны при breakout);
      rising_channel: use_2c=False (HMA ещё растёт во время подъёма канала)."""
    hma_i, hma_i2, wma_i = hma_mhull[i], hma_shull[i], wma50[i]
    if any(np.isnan(x) for x in (hma_i, hma_i2, wma_i)):
        return False
    c = closes[i]
    cond_2a = ((wma_i > c) and (hma_i > c)) if use_2a else True
    cond_2c = (c < hma_i2)                  if use_2c else True
    cond_2d = (hma_i > wma_i)               if use_2d else True
    return cond_2a and cond_2c and cond_2d

def apply_fvg_ma_filter(patterns: list[dict], t_arr: np.ndarray, closes: np.ndarray,
                         fvg_from_key: str, i_br_key: str,
                         h_arr: np.ndarray, l_arr: np.ndarray, n_bars: int,
                         hma_mhull: np.ndarray, hma_shull: np.ndarray, wma50: np.ndarray,
                         stats: dict, use_2a: bool = True, use_2c: bool = True,
                         use_2d: bool = True) -> list[dict]:
    """FVG(2) inline bearish + MA SHORT-фильтр. FVG: l[i-1]>h[i+1] в (fvg_from, i_br)."""
    out = []
    for p in patterns:
        i_from = p[fvg_from_key]
        i_br = p[i_br_key]
        fvg_ok = False
        for i_fvg in range(i_from + 1, min(n_bars - 1, i_br)):
            if 0 < i_fvg < n_bars - 1:
                if l_arr[i_fvg - 1] > h_arr[i_fvg + 1]:
                    fvg_ok = True; break
        if not fvg_ok:
            stats['rej_fvg'] += 1
            continue

        if not ma_filter_short_ok(i_br, closes, hma_mhull, hma_shull, wma50,
                                   use_2a=use_2a, use_2c=use_2c, use_2d=use_2d):
            stats['rej_hma_wma'] += 1
            continue

        stats['passed_canon'] += 1
        out.append(p)
    return out


def ma_filter_long_ok(i: int, closes: np.ndarray, hma_mhull: np.ndarray,
                       hma_shull: np.ndarray, wma50: np.ndarray,
                       use_2a: bool = True, use_2c: bool = True,
                       use_2d: bool = True) -> bool:
    """2a/2c/2d фильтры LONG на баре i — зеркало ma_filter_short_ok для LONG-паттернов.
      2a_long: close > WMA50 AND close > HMA (цена выше обоих MA = бычий режим)
      2c_long: HMA растёт (mhull[i] > mhull[i-2] = shull[i])
      2d_long: HMA < WMA (pre-golden-cross: HMA ещё не пересёк WMA вверх)"""
    hma_i, hma_i2, wma_i = hma_mhull[i], hma_shull[i], wma50[i]
    if any(np.isnan(x) for x in (hma_i, hma_i2, wma_i)):
        return False
    c = closes[i]
    cond_2a = ((wma_i < c) and (hma_i < c)) if use_2a else True
    cond_2c = (c > hma_i2)                  if use_2c else True
    cond_2d = (hma_i < wma_i)               if use_2d else True
    return cond_2a and cond_2c and cond_2d


def apply_fvg_ma_filter_long(patterns: list[dict], t_arr: np.ndarray, closes: np.ndarray,
                               fvg_from_key: str, i_br_key: str,
                               h_arr: np.ndarray, l_arr: np.ndarray, n_bars: int,
                               hma_mhull: np.ndarray, hma_shull: np.ndarray, wma50: np.ndarray,
                               stats: dict, use_2a: bool = True, use_2c: bool = True,
                               use_2d: bool = True) -> list[dict]:
    """FVG(2) inline bullish + MA LONG-фильтр. FVG: h[i-1]<l[i+1] в (fvg_from, i_br)."""
    out = []
    for p in patterns:
        i_from = p[fvg_from_key]
        i_br = p[i_br_key]
        fvg_ok = False
        for i_fvg in range(i_from + 1, min(n_bars - 1, i_br)):
            if 0 < i_fvg < n_bars - 1:
                if h_arr[i_fvg - 1] < l_arr[i_fvg + 1]:
                    fvg_ok = True; break
        if not fvg_ok:
            stats['rej_fvg'] += 1
            continue

        if not ma_filter_long_ok(i_br, closes, hma_mhull, hma_shull, wma50,
                                  use_2a=use_2a, use_2c=use_2c, use_2d=use_2d):
            stats['rej_hma_wma'] += 1
            continue

        stats['passed_canon'] += 1
        out.append(p)
    return out
