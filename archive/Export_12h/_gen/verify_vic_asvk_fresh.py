"""Та же проверка, но на СВЕЖИХ данных G:\\ASVK\\data\\BTCUSDT_1m.csv (WSL-копия
устарела, обрывается 19 июля) — покрывает все 12 дат из референса пользователя.
"""
from __future__ import annotations
import pathlib
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vic_asvk import calculate_vic_series

MSK = timezone(timedelta(hours=3))

df = pd.read_csv("G:/ASVK/data/BTCUSDT_1m.csv",
                  dtype={"open": "float64", "high": "float64", "low": "float64",
                        "close": "float64", "volume": "float64"})
dt = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
df["ts"] = dt.astype("datetime64[ms, UTC]").astype("int64")

start_ms = int(datetime(2026, 7, 15, tzinfo=timezone.utc).timestamp() * 1000)
end_ms = int(datetime(2026, 7, 24, tzinfo=timezone.utc).timestamp() * 1000)
window = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
ltf_data = list(zip(window["ts"], window["open"], window["high"],
                    window["low"], window["close"], window["volume"]))
print(f"1m bars in window: {len(ltf_data):,}", file=sys.stderr)

user_ref = {
    "2026-07-17 03:00": 62940, "2026-07-17 15:00": 63247,
    "2026-07-18 03:00": 63947, "2026-07-18 15:00": 64517,
    "2026-07-19 03:00": 64891, "2026-07-19 15:00": 64457,
    "2026-07-20 03:00": 63970, "2026-07-20 15:00": 65597,
    "2026-07-21 03:00": 65534, "2026-07-21 15:00": 66329,
    "2026-07-22 03:00": 66389, "2026-07-22 15:00": 66026,
}
mine_1m = {
    "2026-07-17 03:00": 62837.00, "2026-07-17 15:00": 63178.00,
    "2026-07-18 03:00": 63941.84, "2026-07-18 15:00": 64756.75,
    "2026-07-19 03:00": 64705.15, "2026-07-19 15:00": 64302.56,
    "2026-07-20 03:00": 64200.52, "2026-07-20 15:00": 65202.48,
    "2026-07-21 03:00": 65553.00, "2026-07-21 15:00": 66094.67,
    "2026-07-22 03:00": 65893.99, "2026-07-22 15:00": 66184.70,
}

for mlt in (100, 45, 1000):  # mlt=1000 -> LTF=1m (форсируем, как у пользователя "LTF: 1 минута")
    vic_bars = calculate_vic_series(ltf_data, htf_min=720, mlt=mlt)
    ltf_min = None
    from vic_asvk import auto_ltf_minutes
    ltf_min = auto_ltf_minutes(720, mlt)
    print(f"\n=== mlt={mlt}  (LTF={ltf_min}m) ===", file=sys.stderr)
    print(f"{'open MSK':17s} {'VIC maxV':>11s} {'mine 1m':>11s} {'user ref':>10s} "
          f"{'|VIC-ref|':>10s} {'|1m-ref|':>10s} {'closer':>8s}", file=sys.stderr)
    sum_vic = 0.0; sum_1m = 0.0; n = 0
    for b in vic_bars:
        dt_msk = datetime.fromtimestamp(b.htf_open_ms / 1000, tz=MSK).strftime("%Y-%m-%d %H:%M")
        if dt_msk not in user_ref:
            continue
        ref = user_ref[dt_msk]
        m1 = mine_1m[dt_msk]
        d_vic = abs(b.maxV - ref)
        d_1m = abs(m1 - ref)
        sum_vic += d_vic; sum_1m += d_1m; n += 1
        closer = "VIC" if d_vic < d_1m else "1m"
        print(f"{dt_msk:17s} {b.maxV:>11.2f} {m1:>11.2f} {ref:>10.2f} "
              f"{d_vic:>10.2f} {d_1m:>10.2f} {closer:>8s}", file=sys.stderr)
    if n:
        print(f"AVG abs diff:  VIC={sum_vic/n:.2f}   1m={sum_1m/n:.2f}   (n={n})", file=sys.stderr)
