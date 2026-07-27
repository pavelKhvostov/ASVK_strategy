"""Проверка гипотезы: пользовательские референсные значения maxV совпадают
с VIC ASVK (индикаторы/vic_asvk.py) на auto-LTF (mlt=45 -> 16m), а НЕ с
1m-версией из B3C1_maxv_sweep.py (которая используется во всём basket-канон).

Read-only: только читает ~/smc-warehouse/график/BTCUSDT_1m.csv и импортирует
~/smc-warehouse/индикаторы/vic_asvk.py как есть, ничего не пишет обратно.
"""
from __future__ import annotations
import pathlib
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

WAREHOUSE = pathlib.Path.home() / "smc-warehouse"
sys.path.insert(0, str(WAREHOUSE / "индикаторы"))
from vic_asvk import calculate_vic_series, auto_ltf_minutes

MSK = timezone(timedelta(hours=3))

print(f"auto_ltf_minutes(htf_min=720, mlt=45) = {auto_ltf_minutes(720, 45)} min", file=sys.stderr)

df = pd.read_csv(WAREHOUSE / "график" / "BTCUSDT_1m.csv",
                  dtype={"open": "float64", "high": "float64", "low": "float64",
                        "close": "float64", "volume": "float64"})
dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")

# ограничим окно для скорости — с запасом (нужно 17-07..22-07 MSK)
start_ms = int(datetime(2026, 7, 10, tzinfo=timezone.utc).timestamp() * 1000)
end_ms = int(datetime(2026, 7, 24, tzinfo=timezone.utc).timestamp() * 1000)
window = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
ltf_data = list(zip(window["ts"], window["open"], window["high"],
                    window["low"], window["close"], window["volume"]))
print(f"1m bars in window: {len(ltf_data):,}", file=sys.stderr)

vic_bars = calculate_vic_series(ltf_data, htf_min=720, mlt=45)

user_ref = {
    "2026-07-17 03:00": 62940, "2026-07-17 15:00": 63247,
    "2026-07-18 03:00": 63947, "2026-07-18 15:00": 64517,
    "2026-07-19 03:00": 64891, "2026-07-19 15:00": 64457,
    "2026-07-20 03:00": 63970, "2026-07-20 15:00": 65597,
    "2026-07-21 03:00": 65534, "2026-07-21 15:00": 66329,
    "2026-07-22 03:00": 66389, "2026-07-22 15:00": 66026,
}

print(f"\n{'open MSK':17s} {'VIC 16m maxV':>13s} {'user ref':>10s} {'diff':>9s} {'diff%':>7s}")
for b in vic_bars:
    dt_msk = datetime.fromtimestamp(b.htf_open_ms / 1000, tz=MSK).strftime("%Y-%m-%d %H:%M")
    if dt_msk not in user_ref:
        continue
    ref = user_ref[dt_msk]
    diff = b.maxV - ref
    print(f"{dt_msk:17s} {b.maxV:>13.2f} {ref:>10.2f} {diff:>+9.2f} {100*diff/ref:>6.3f}%")
