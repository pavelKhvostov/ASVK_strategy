"""Contract for element detectors.

Каждый detector.detect(o, h, l, c, ts, tf_ms, **extras) -> list[dict], где event:
    ts:         int   — момент события (ms UTC), bar close = ts_bar_open + tf_ms
    kind:       str   — "born" | "fill_partial" | "retire"
    direction:  str   — "long" | "short"
    zone_lo:    float — исходные границы (immutable)
    zone_hi:    float
    active_lo:  float — residual границы после fill_partial (born: = zone_lo)
    active_hi:  float — residual (born: = zone_hi)
    role:       str   — "support" | "resistance" | "sweep_high" | "sweep_low"
    zone_id:    int   — локальный id (e12d ренумерует глобально)
    meta:       dict  — element-specific

Direction canon:
    long  = zone под ценой (support, тестируется wick'ом сверху)
    short = zone над ценой (resistance, тестируется wick'ом снизу)

Для sweep-элементов (fractal, marubozu):
    zone_lo == zone_hi == level
    long = FL (sweep_low), short = FH (sweep_high)

element, tf, symbol заполняются e12d, а не детектором.
"""
