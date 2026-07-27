"""Backfill warehouse CSV from Binance API for BTC/ETH/SOL.
Downloads bars from (last_ts_in_csv + 1min) до now (closed bars only).
"""
import csv
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

WAREHOUSE = Path(__file__).resolve().parent.parent / "data"  # ASVK-standalone
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL_MS = 60_000
BATCH = 1000
BINANCE_URL = "https://api.binance.com/api/v3/klines"


def read_last_ts_ms(csv_path: Path) -> int | None:
    if not csv_path.exists(): return None
    with open(csv_path, "rb") as f:
        f.seek(0, 2); size = f.tell()
        f.seek(max(0, size - 8192))
        tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
    for line in reversed(tail):
        parts = line.split(",")
        if len(parts) < 6 or parts[0] == "open_time": continue
        try:
            dt = datetime.fromisoformat(parts[0])
            return int(dt.timestamp() * 1000)
        except: continue
    return None


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(BINANCE_URL, params={
            "symbol": symbol, "interval": "1m",
            "startTime": cursor, "endTime": end_ms - 1, "limit": BATCH,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data: break
        all_bars.extend(data)
        cursor = int(data[-1][0]) + INTERVAL_MS
        if len(data) < BATCH: break
        time.sleep(0.1)  # rate limit
    return all_bars


def append_bars(csv_path: Path, bars: list, last_ts_ms: int) -> int:
    now_ms = int(time.time() * 1000)
    clean = [b for b in bars if int(b[0]) > last_ts_ms and int(b[0]) + INTERVAL_MS <= now_ms]
    if not clean: return 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for b in clean:
            dt = datetime.fromtimestamp(int(b[0])/1000, tz=timezone.utc)
            iso = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            w.writerow([iso, str(b[1]), str(b[2]), str(b[3]), str(b[4]), str(b[5])])
    return len(clean)


def main() -> None:
    for sym in SYMBOLS:
        csv_path = WAREHOUSE / f"{sym}_1m.csv"
        last_ts = read_last_ts_ms(csv_path)
        if last_ts is None:
            print(f"{sym}: NO CSV / empty — skip"); continue
        last_dt = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)
        now_ms = int(time.time() * 1000)
        gap_min = (now_ms - last_ts) / 60_000
        print(f"{sym}: last {last_dt.strftime('%Y-%m-%d %H:%M UTC')}, gap {gap_min:.0f} min")
        if gap_min < 2:
            print(f"  already fresh, skip"); continue
        start_ms = last_ts + INTERVAL_MS
        end_ms = (now_ms // INTERVAL_MS) * INTERVAL_MS
        t0 = time.time()
        bars = fetch_klines(sym, start_ms, end_ms)
        n = append_bars(csv_path, bars, last_ts)
        dt = time.time() - t0
        print(f"  fetched {len(bars)}, appended {n} in {dt:.1f}s")


if __name__ == "__main__":
    main()
