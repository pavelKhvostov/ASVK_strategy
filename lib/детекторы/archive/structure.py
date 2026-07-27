"""Market Structure — canonical SMC/ICT детектор BOS/CHoCH/MSS + Fibonacci zones.

Синтезирует ICT-канон (innercircletrader.net + Huddleston notes), SMC-школу
(DailyPriceAction), Wyckoff-соответствия и Williams-fractal подход к real-time
swing detection.

Источник методологии: `~/smc-warehouse/литература/структура-bos-choch-fibonacci-справочник.md`.

Ключевые концепты:
- **Two-tier swings** (Williams fractal): internal (N=5) для micro-flow +
  swing (N=25) для истинной структуры. Оба одновременно.
- **BOS** — same-direction continuation (close body за swing в направлении тренда).
- **CHoCH** — opposite-direction warning (close body за internal swing против тренда,
  требует предварительно confirmed trend + as least один BOS в противоположном).
- **MSS** — confirmed reversal (close body за external swing + displacement +
  опционально prior liquidity sweep).
- **Fibonacci zones** — equilibrium (0.5), OTE (0.62 / 0.705 / 0.79), tp extensions
  (-0.27, -0.62). Rows canonical per ICT.

Anti-lookahead: swing at bar t validated at bar t+N. Уровень доступен для решений
строго с индекса t+N.

Canonical event schema (совместим с elements/common.py):
    {
        "ts": int (close_ts бара где произошло событие),
        "kind": "born",
        "type": "BOS" | "CHoCH" | "MSS",
        "direction": "long" | "short",
        "level": float (пробитый swing level),
        "zone_lo": float, "zone_hi": float (rectangle around pivot bar OHLC),
        "meta": {
            "tier": "internal" | "swing",
            "pivot_ts": int (когда родился swing which was broken),
            "pivot_bar_idx": int,
            "trend_state_before": "bull_confirmed" | "bear_confirmed" | ...,
            "displacement": bool,
            "wick_ratio": float,
            "prior_sweep": bool (для MSS),
        }
    }
"""
from __future__ import annotations
from typing import Literal
import numpy as np


# --------------------------- Fractal / Swing Detection ---------------------------

def _fractal_swings(h: np.ndarray, l: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Williams N-fractal swing detection.

    Возвращает:
      swing_hi_idx: массив индексов bars которые являются swing highs (confirmed)
      swing_lo_idx: массив индексов bars которые являются swing lows (confirmed)

    ⚠ Anti-lookahead: swing at bar t confirmed at bar t+N. Использовать значение
    swing.level в решениях можно только с индекса >= t+N.
    """
    n = len(h)
    if n < 2*N + 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    hi_mask = np.zeros(n, dtype=bool)
    lo_mask = np.zeros(n, dtype=bool)

    for t in range(N, n - N):
        wh = h[t-N:t+N+1]
        wl = l[t-N:t+N+1]
        if h[t] == wh.max():
            hi_mask[t] = True
        if l[t] == wl.min():
            lo_mask[t] = True

    # exclusive (защита от одновременного hi+lo на одном баре — редко, но возможно)
    both = hi_mask & lo_mask
    hi_mask &= ~both
    lo_mask &= ~both

    hi_idx = np.where(hi_mask)[0]
    lo_idx = np.where(lo_mask)[0]
    return hi_idx, lo_idx


def _alternate_swings(hi_idx: np.ndarray, lo_idx: np.ndarray,
                      h: np.ndarray, l: np.ndarray) -> list[tuple[int, int, float]]:
    """Строит alternating последовательность (label, bar_idx, price).

    label: +1 = swing high, -1 = swing low.
    Гарантирует что не идут два одинаковых подряд (де-дуп: оставляем более высокий hi
    и более низкий lo).
    """
    events = []
    for i in hi_idx:
        events.append((+1, int(i), float(h[i])))
    for i in lo_idx:
        events.append((-1, int(i), float(l[i])))
    events.sort(key=lambda x: x[1])

    # де-дуп: если два подряд одного знака — оставить более экстремальный
    out: list[tuple[int, int, float]] = []
    for e in events:
        if not out:
            out.append(e)
            continue
        if e[0] == out[-1][0]:
            # тот же знак: заменить если новый более экстремальный
            if (e[0] == +1 and e[2] > out[-1][2]) or (e[0] == -1 and e[2] < out[-1][2]):
                out[-1] = e
        else:
            out.append(e)
    return out


# --------------------------- Trend State Classifier ---------------------------

def _confirm_trend(alt_seq: list[tuple[int, int, float]]) -> Literal[
    "bull_confirmed", "bear_confirmed", "consolidation", "insufficient"
]:
    """Classify trend по последним 3-4 alternating swings.

    bull_confirmed: последние два highs растут (HH) AND последние два lows растут (HL).
                    Требует минимум 2 highs + 2 lows (4+ swings).
    bear_confirmed: последние два highs падают (LH) AND последние два lows падают (LL).
    """
    highs = [(t, p) for lbl, t, p in alt_seq if lbl == +1]
    lows  = [(t, p) for lbl, t, p in alt_seq if lbl == -1]
    if len(highs) < 2 or len(lows) < 2:
        return "insufficient"
    hh_up = highs[-1][1] > highs[-2][1]
    hl_up = lows[-1][1]  > lows[-2][1]
    lh_dn = highs[-1][1] < highs[-2][1]
    ll_dn = lows[-1][1]  < lows[-2][1]
    if hh_up and hl_up:
        return "bull_confirmed"
    if lh_dn and ll_dn:
        return "bear_confirmed"
    return "consolidation"


# --------------------------- Displacement & Sweep ---------------------------

def _atr14(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = len(h)
    tr = np.zeros(n)
    if n == 0:
        return tr
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    # SMA14
    atr = np.zeros(n)
    for i in range(n):
        s = max(0, i - 13)
        atr[i] = tr[s:i+1].mean()
    return atr


def _is_displacement(open_: float, high: float, low: float, close: float,
                     atr_val: float, atr_mult: float, max_wick_ratio: float) -> tuple[bool, float]:
    body = abs(close - open_)
    rng = high - low
    if rng <= 0:
        return False, 1.0
    wick_ratio = (rng - body) / rng
    ok = (body >= atr_mult * atr_val) and (wick_ratio <= max_wick_ratio)
    return ok, wick_ratio


def _had_prior_sweep_bull(h: np.ndarray, c: np.ndarray, o: np.ndarray,
                          level: float, t: int, lookback: int) -> bool:
    """Bullish MSS требует prior SSL sweep (bar с low ниже level но close выше)."""
    start = max(0, t - lookback)
    for k in range(start, t):
        if o[k] < level and c[k] > level:
            # wick down was below level but close recovered above — SSL sweep
            return True
    return False


def _had_prior_sweep_bear(l: np.ndarray, c: np.ndarray, o: np.ndarray,
                          level: float, t: int, lookback: int) -> bool:
    """Bearish MSS требует prior BSL sweep."""
    start = max(0, t - lookback)
    for k in range(start, t):
        if o[k] > level and c[k] < level:
            return True
    return False


# --------------------------- Main Detector ---------------------------

def detect(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    ts: np.ndarray, tf_ms: int,
    N_internal: int = 5,
    N_swing: int = 25,
    close_break: bool = True,
    displacement_atr_mult: float = 1.2,
    max_wick_ratio: float = 0.4,
    require_sweep_for_mss: bool = True,
    sweep_lookback: int = 20,
    choch_anchor: Literal["last_HL", "hl_before_last_bos"] = "last_HL",
) -> list[dict]:
    """Canonical BOS/CHoCH/MSS detector с two-tier swings.

    Args:
        o, h, l, c, ts: numpy arrays 1D одинакового размера
        tf_ms: bar timeframe в ms
        N_internal, N_swing: fractal window lengths (see docs)
        close_break: True = break body close vs level; False = high/low
        displacement_atr_mult: min body/ATR ratio for MSS displacement
        max_wick_ratio: max wick_ratio for MSS displacement candle
        require_sweep_for_mss: ICT strict = True; SMC generic = False
        sweep_lookback: bars window для prior sweep search
        choch_anchor: `last_HL` = LTF style (Fxnx); `hl_before_last_bos` = DPA strict

    Returns: list of events (see module docstring).
    """
    n = len(o)
    if n < 2 * N_swing + 1:
        return []

    # ── Tier 1: internal swings (N_internal) ─────────────────────────────
    hi_idx_int, lo_idx_int = _fractal_swings(h, l, N_internal)
    alt_internal = _alternate_swings(hi_idx_int, lo_idx_int, h, l)

    # ── Tier 2: swing/external (N_swing) ─────────────────────────────────
    hi_idx_sw, lo_idx_sw = _fractal_swings(h, l, N_swing)
    alt_swing = _alternate_swings(hi_idx_sw, lo_idx_sw, h, l)

    atr = _atr14(h, l, c)

    events: list[dict] = []
    zone_id = 0

    trend: str = "insufficient"
    last_bull_bos_anchor_lo: tuple[int, float] | None = None
    last_bear_bos_anchor_hi: tuple[int, float] | None = None

    # One-shot: pivot may be broken by BOS/CHoCH/MSS only once per (type, pivot_bar_idx)
    broken_bos: set[tuple[int, str]] = set()      # {(pivot_bar_idx, direction)}
    broken_choch: set[tuple[int, str]] = set()
    broken_mss: set[tuple[int, str]] = set()

    # Bar-by-bar loop
    for t in range(2 * N_swing, n):
        # ── доступные confirmed swings в [t] ─────────────────────────────
        # Anti-lookahead: swing at j доступен с j+N.
        alt_int_avail = [e for e in alt_internal if e[1] + N_internal <= t]
        alt_sw_avail  = [e for e in alt_swing    if e[1] + N_swing    <= t]

        if len(alt_int_avail) < 3 or len(alt_sw_avail) < 3:
            continue

        # Trend state по SWING (external) — определяет direction фильтр
        trend_ext = _confirm_trend(alt_sw_avail)
        # Trend state по INTERNAL — не используется прямо, но полезно для будущих extensions

        # Last confirmed pivots на каждом tier
        def _last_of(seq, label):
            for x in reversed(seq):
                if x[0] == label:
                    return x
            return None

        int_last_hi = _last_of(alt_int_avail, +1)
        int_last_lo = _last_of(alt_int_avail, -1)
        sw_last_hi  = _last_of(alt_sw_avail,  +1)
        sw_last_lo  = _last_of(alt_sw_avail,  -1)

        # ── break_price по типу close/wick ───────────────────────────────
        c_t = float(c[t])
        h_t = float(h[t])
        l_t = float(l[t])
        break_up   = c_t   if close_break else h_t
        break_down = c_t   if close_break else l_t

        atr_t = float(atr[t]) if not np.isnan(atr[t]) else 0.0
        disp_ok, wick_ratio = _is_displacement(float(o[t]), h_t, l_t, c_t,
                                                atr_t, displacement_atr_mult, max_wick_ratio)

        # ── BOS (continuation) — по internal tier ───────────────────────
        # Bullish BOS: bull trend confirmed AND close > last confirmed internal high
        if trend_ext == "bull_confirmed" and int_last_hi is not None:
            level = int_last_hi[2]
            pivot_bar = int_last_hi[1]
            key = (pivot_bar, "long")
            # bar t должен быть после подтверждения swing + one-shot per pivot
            if pivot_bar + N_internal <= t and break_up > level and key not in broken_bos:
                broken_bos.add(key)
                zone_id += 1
                zone_lo = float(min(l[pivot_bar-1], l[pivot_bar])) if pivot_bar >= 1 else float(l[pivot_bar])
                zone_hi = float(h[pivot_bar])
                events.append({
                    "ts": int(ts[t]) + tf_ms,
                    "kind": "born",
                    "type": "BOS",
                    "direction": "long",
                    "level": level,
                    "zone_lo": zone_lo,
                    "zone_hi": zone_hi,
                    "role": "continuation",
                    "zone_id": zone_id,
                    "meta": {
                        "tier": "internal",
                        "pivot_ts": int(ts[pivot_bar]),
                        "pivot_bar_idx": pivot_bar,
                        "trend_state_before": trend_ext,
                        "displacement": disp_ok,
                        "wick_ratio": wick_ratio,
                    },
                })
                # anchor для будущих CHoCH: last HL при BOS bullish
                if int_last_lo is not None:
                    last_bull_bos_anchor_lo = (int_last_lo[1], int_last_lo[2])

        # Bearish BOS
        if trend_ext == "bear_confirmed" and int_last_lo is not None:
            level = int_last_lo[2]
            pivot_bar = int_last_lo[1]
            key = (pivot_bar, "short")
            if pivot_bar + N_internal <= t and break_down < level and key not in broken_bos:
                broken_bos.add(key)
                zone_id += 1
                zone_lo = float(l[pivot_bar])
                zone_hi = float(max(h[pivot_bar-1], h[pivot_bar])) if pivot_bar >= 1 else float(h[pivot_bar])
                events.append({
                    "ts": int(ts[t]) + tf_ms,
                    "kind": "born",
                    "type": "BOS",
                    "direction": "short",
                    "level": level,
                    "zone_lo": zone_lo,
                    "zone_hi": zone_hi,
                    "role": "continuation",
                    "zone_id": zone_id,
                    "meta": {
                        "tier": "internal",
                        "pivot_ts": int(ts[pivot_bar]),
                        "pivot_bar_idx": pivot_bar,
                        "trend_state_before": trend_ext,
                        "displacement": disp_ok,
                        "wick_ratio": wick_ratio,
                    },
                })
                if int_last_hi is not None:
                    last_bear_bos_anchor_hi = (int_last_hi[1], int_last_hi[2])

        # ── CHoCH (warning of reversal) — по internal tier ──────────────
        # Bearish CHoCH (в bull_confirmed): close < internal low anchor
        if trend_ext == "bull_confirmed":
            # anchor selection
            if choch_anchor == "hl_before_last_bos" and last_bull_bos_anchor_lo is not None:
                anchor_bar, anchor_lvl = last_bull_bos_anchor_lo
            elif int_last_lo is not None:
                anchor_bar, anchor_lvl = int_last_lo[1], int_last_lo[2]
            else:
                anchor_bar, anchor_lvl = None, None

            key = (anchor_bar, "short") if anchor_bar is not None else None
            if (anchor_bar is not None and anchor_bar + N_internal <= t
                    and break_down < anchor_lvl and key not in broken_choch):
                broken_choch.add(key)
                zone_id += 1
                zone_lo = float(l[anchor_bar])
                zone_hi = float(max(h[anchor_bar-1], h[anchor_bar])) if anchor_bar >= 1 else float(h[anchor_bar])
                events.append({
                    "ts": int(ts[t]) + tf_ms,
                    "kind": "born",
                    "type": "CHoCH",
                    "direction": "short",
                    "level": float(anchor_lvl),
                    "zone_lo": zone_lo,
                    "zone_hi": zone_hi,
                    "role": "reversal_warning",
                    "zone_id": zone_id,
                    "meta": {
                        "tier": "internal",
                        "pivot_ts": int(ts[anchor_bar]),
                        "pivot_bar_idx": anchor_bar,
                        "trend_state_before": trend_ext,
                        "displacement": disp_ok,
                        "wick_ratio": wick_ratio,
                        "choch_anchor": choch_anchor,
                    },
                })

        # Bullish CHoCH (в bear_confirmed)
        if trend_ext == "bear_confirmed":
            if choch_anchor == "hl_before_last_bos" and last_bear_bos_anchor_hi is not None:
                anchor_bar, anchor_lvl = last_bear_bos_anchor_hi
            elif int_last_hi is not None:
                anchor_bar, anchor_lvl = int_last_hi[1], int_last_hi[2]
            else:
                anchor_bar, anchor_lvl = None, None

            key = (anchor_bar, "long") if anchor_bar is not None else None
            if (anchor_bar is not None and anchor_bar + N_internal <= t
                    and break_up > anchor_lvl and key not in broken_choch):
                broken_choch.add(key)
                zone_id += 1
                zone_lo = float(min(l[anchor_bar-1], l[anchor_bar])) if anchor_bar >= 1 else float(l[anchor_bar])
                zone_hi = float(h[anchor_bar])
                events.append({
                    "ts": int(ts[t]) + tf_ms,
                    "kind": "born",
                    "type": "CHoCH",
                    "direction": "long",
                    "level": float(anchor_lvl),
                    "zone_lo": zone_lo,
                    "zone_hi": zone_hi,
                    "role": "reversal_warning",
                    "zone_id": zone_id,
                    "meta": {
                        "tier": "internal",
                        "pivot_ts": int(ts[anchor_bar]),
                        "pivot_bar_idx": anchor_bar,
                        "trend_state_before": trend_ext,
                        "displacement": disp_ok,
                        "wick_ratio": wick_ratio,
                        "choch_anchor": choch_anchor,
                    },
                })

        # ── MSS (confirmed reversal) — по SWING tier + displacement + sweep ─
        # Bearish MSS (в bull_confirmed): close < last external swing low
        if trend_ext == "bull_confirmed" and sw_last_lo is not None:
            level = sw_last_lo[2]
            pivot_bar = sw_last_lo[1]
            mss_key = (pivot_bar, "short")
            if pivot_bar + N_swing <= t and break_down < level and disp_ok and mss_key not in broken_mss:
                sweep_ok = True
                if require_sweep_for_mss:
                    ext_hi_lvl = sw_last_hi[2] if sw_last_hi is not None else None
                    if ext_hi_lvl is not None:
                        sweep_ok = _had_prior_sweep_bear(l, c, o, ext_hi_lvl, t, sweep_lookback)
                    else:
                        sweep_ok = False
                if sweep_ok:
                    broken_mss.add(mss_key)
                    zone_id += 1
                    zone_lo = float(l[pivot_bar])
                    zone_hi = float(max(h[pivot_bar-1], h[pivot_bar])) if pivot_bar >= 1 else float(h[pivot_bar])
                    events.append({
                        "ts": int(ts[t]) + tf_ms,
                        "kind": "born",
                        "type": "MSS",
                        "direction": "short",
                        "level": level,
                        "zone_lo": zone_lo,
                        "zone_hi": zone_hi,
                        "role": "reversal_confirmed",
                        "zone_id": zone_id,
                        "meta": {
                            "tier": "swing",
                            "pivot_ts": int(ts[pivot_bar]),
                            "pivot_bar_idx": pivot_bar,
                            "trend_state_before": trend_ext,
                            "displacement": True,
                            "wick_ratio": wick_ratio,
                            "prior_sweep": sweep_ok,
                        },
                    })

        # Bullish MSS
        if trend_ext == "bear_confirmed" and sw_last_hi is not None:
            level = sw_last_hi[2]
            pivot_bar = sw_last_hi[1]
            mss_key = (pivot_bar, "long")
            if pivot_bar + N_swing <= t and break_up > level and disp_ok and mss_key not in broken_mss:
                sweep_ok = True
                if require_sweep_for_mss:
                    ext_lo_lvl = sw_last_lo[2] if sw_last_lo is not None else None
                    if ext_lo_lvl is not None:
                        sweep_ok = _had_prior_sweep_bull(h, c, o, ext_lo_lvl, t, sweep_lookback)
                    else:
                        sweep_ok = False
                if sweep_ok:
                    broken_mss.add(mss_key)
                    zone_id += 1
                    zone_lo = float(min(l[pivot_bar-1], l[pivot_bar])) if pivot_bar >= 1 else float(l[pivot_bar])
                    zone_hi = float(h[pivot_bar])
                    events.append({
                        "ts": int(ts[t]) + tf_ms,
                        "kind": "born",
                        "type": "MSS",
                        "direction": "long",
                        "level": level,
                        "zone_lo": zone_lo,
                        "zone_hi": zone_hi,
                        "role": "reversal_confirmed",
                        "zone_id": zone_id,
                        "meta": {
                            "tier": "swing",
                            "pivot_ts": int(ts[pivot_bar]),
                            "pivot_bar_idx": pivot_bar,
                            "trend_state_before": trend_ext,
                            "displacement": True,
                            "wick_ratio": wick_ratio,
                            "prior_sweep": sweep_ok,
                        },
                    })

    return events


# --------------------------- Fibonacci Zones ---------------------------

def fib_zones(swing_low: float, swing_high: float,
              direction: Literal["long", "short"] = "long") -> dict:
    """ICT-canonical Fibonacci zones для dealing range.

    Bullish (long setup): fib от swing_low (0.0) → swing_high (1.0).
      Retracement идёт вниз; discount = < equilibrium.

    Bearish (short setup): fib от swing_high (0.0) → swing_low (1.0).
      Retracement идёт вверх; premium = > equilibrium.

    Returns dict with:
      equilibrium: float (0.5)
      ote_upper / ote_sweet / ote_lower: OTE границы (0.62 / 0.705 / 0.79)
      tp1 / tp2: extension targets (-0.27 / -0.62 в ICT нотации)
      discount / premium: (lo, hi) ranges
    """
    L, H = float(swing_low), float(swing_high)
    leg = H - L
    eq = (H + L) / 2.0
    if direction == "long":
        return {
            "equilibrium": eq,
            "ote_upper":  H - 0.62  * leg,
            "ote_sweet":  H - 0.705 * leg,
            "ote_lower":  H - 0.79  * leg,
            "tp1":        H + 0.27  * leg,
            "tp2":        H + 0.62  * leg,
            "discount":   (L, eq),
            "premium":    (eq, H),
        }
    else:  # short
        return {
            "equilibrium": eq,
            "ote_lower":  L + 0.62  * leg,
            "ote_sweet":  L + 0.705 * leg,
            "ote_upper":  L + 0.79  * leg,
            "tp1":        L - 0.27  * leg,
            "tp2":        L - 0.62  * leg,
            "premium":    (eq, H),
            "discount":   (L, eq),
        }


def is_in_discount(price: float, swing_low: float, swing_high: float) -> bool:
    """Проверка: находится ли price в discount zone (< equilibrium)."""
    return price < (swing_low + swing_high) / 2.0


def is_in_premium(price: float, swing_low: float, swing_high: float) -> bool:
    """Проверка: находится ли price в premium zone (> equilibrium)."""
    return price > (swing_low + swing_high) / 2.0


def is_in_ote(price: float, swing_low: float, swing_high: float,
              direction: Literal["long", "short"] = "long") -> bool:
    """Проверка: находится ли price в OTE zone (0.62 - 0.79 retracement)."""
    z = fib_zones(swing_low, swing_high, direction)
    if direction == "long":
        return z["ote_lower"] <= price <= z["ote_upper"]
    else:
        return z["ote_lower"] <= price <= z["ote_upper"]


# --------------------------- Regime Helper ---------------------------

def get_regime_at(events: list[dict], T: int) -> Literal[
    "bull_trending", "bear_trending", "bull_reversal_warning", "bear_reversal_warning",
    "bull_reversal_confirmed", "bear_reversal_confirmed", "neutral"
]:
    """Определяет текущий regime по последнему event до момента T.

    Использование: для filter в chain scan — LONG сделки только когда bull_trending
    or bull_reversal_confirmed; SHORT наоборот.
    """
    latest = None
    for e in events:
        if e["ts"] > T:
            break
        latest = e
    if latest is None:
        return "neutral"

    t = latest["type"]
    d = latest["direction"]
    if t == "BOS":
        return "bull_trending" if d == "long" else "bear_trending"
    if t == "CHoCH":
        return "bull_reversal_warning" if d == "long" else "bear_reversal_warning"
    if t == "MSS":
        return "bull_reversal_confirmed" if d == "long" else "bear_reversal_confirmed"
    return "neutral"


def get_last_swing_range(events: list[dict], T: int) -> tuple[float, float] | None:
    """Ищет последний dealing range (swing_lo, swing_hi) до момента T
    из событий структуры (по meta zones).

    Возвращает None если недостаточно events.
    """
    los = []
    his = []
    for e in events:
        if e["ts"] > T:
            break
        if e.get("meta", {}).get("tier") == "swing":
            if e["direction"] == "long":
                his.append(e["zone_hi"])
            else:
                los.append(e["zone_lo"])
    if not los or not his:
        return None
    return (los[-1], his[-1])
