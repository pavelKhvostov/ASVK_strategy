"""ASVK 1m data collector — Binance BTC/ETH/SOL.

Один файл, Windows native. 15-мин цикл. Rich TUI + TG queue + toast fallback.

Layout:
  G:\\ASVK\\
    data\\{SYMBOL}USDT_1m.csv   (append mode)
    logs\\asvk.log
    logs\\tg_pending.json       (retry queue)
    config.txt                  (TG_TOKEN, TG_CHAT)
    asvk.py                     (this file)
    ASVK.bat                    (launcher)
"""
from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group

try:
    from winotify import Notification, audio
    HAVE_WINOTIFY = True
except Exception:
    HAVE_WINOTIFY = False


# ══════════════════════════ CONFIG ══════════════════════════

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
LOG_DIR = BASE / "logs"
CONFIG_FILE = BASE / "config.txt"
TG_QUEUE_FILE = LOG_DIR / "tg_pending.json"
LOG_FILE = LOG_DIR / "asvk.log"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "1m"
INTERVAL_MS = 60_000
CYCLE_SEC = 900          # 15 min
CYCLE_BUFFER_SEC = 30    # +30s чтобы биржа закрыла свечи
HEARTBEAT_SEC = 3600     # раз в час TG "жив"
TG_FAIL_ALERT_MIN = 30   # если TG в down > 30 мин → toast + sound
MSK = timezone(timedelta(hours=3))
UTC = timezone.utc

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_LIMIT = 1000     # max bars per request
FETCH_TIMEOUT = 15
FETCH_RETRIES = 5


# ══════════════════════════ CONFIG LOAD ══════════════════════════

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {"TG_TOKEN": "", "TG_CHAT": ""}
    cfg = {}
    for line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


CFG = load_config()


# ══════════════════════════ TG QUEUE ══════════════════════════

class TGSender:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.queue: list[dict] = []
        self.last_success: float | None = None
        self.last_fail_reason: str = ""
        self.total_sent = 0
        self.load_queue()

    def enabled(self) -> bool:
        return bool(self.token) and bool(self.chat_id)

    def load_queue(self):
        if TG_QUEUE_FILE.exists():
            try:
                self.queue = json.loads(TG_QUEUE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.queue = []

    def save_queue(self):
        TG_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TG_QUEUE_FILE.write_text(json.dumps(self.queue, ensure_ascii=False, indent=2), encoding="utf-8")

    def send_one(self, text: str) -> bool:
        if not self.enabled():
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if r.status_code == 200 and r.json().get("ok"):
                self.last_success = time.time()
                self.total_sent += 1
                return True
            self.last_fail_reason = f"HTTP {r.status_code}: {r.text[:150]}"
            logging.warning(f"TG send failed: {self.last_fail_reason}")
            return False
        except Exception as e:
            self.last_fail_reason = f"{type(e).__name__}: {str(e)[:150]}"
            logging.warning(f"TG send exception: {self.last_fail_reason}")
            return False

    def queue_msg(self, text: str, tag: str = "info"):
        entry = {"ts": time.time(), "text": text, "tag": tag, "tries": 0}
        self.queue.append(entry)
        self.save_queue()

    def flush(self, max_tries: int = 3):
        if not self.queue:
            return
        remaining = []
        for entry in self.queue:
            if self.send_one(entry["text"]):
                logging.info(f"TG sent (queue): {entry['text'][:60]}")
            else:
                entry["tries"] = entry.get("tries", 0) + 1
                if entry["tries"] < max_tries or entry.get("tag") == "critical":
                    remaining.append(entry)
                else:
                    logging.warning(f"TG dropped after {max_tries}: {entry['text'][:60]}")
        self.queue = remaining
        self.save_queue()

    def notify(self, text: str, tag: str = "info"):
        if not self.enabled():
            return
        if self.send_one(text):
            return
        self.queue_msg(text, tag)

    def health(self) -> tuple[str, str]:
        if not self.enabled():
            return ("OFF", "TG отключён")
        if not self.last_success:
            return ("?", "ещё не отправляли")
        age_min = (time.time() - self.last_success) / 60
        if age_min > TG_FAIL_ALERT_MIN:
            return ("DOWN", f"нет успеха {age_min:.0f} мин")
        if self.queue:
            return ("WARN", f"в очереди {len(self.queue)}")
        return ("OK", f"{self.total_sent} отправлено")


TG = TGSender(CFG.get("TG_TOKEN", ""), CFG.get("TG_CHAT", ""))


# ══════════════════════════ TOAST FALLBACK ══════════════════════════

def toast(title: str, msg: str, urgent: bool = False):
    if not HAVE_WINOTIFY:
        return
    try:
        n = Notification(app_id="ASVK", title=title, msg=msg)
        if urgent:
            n.set_audio(audio.LoopingAlarm, loop=False)
        n.show()
    except Exception:
        pass


# ══════════════════════════ CSV IO ══════════════════════════

def csv_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}_{INTERVAL}.csv"


def read_last_ts_ms(symbol: str) -> int | None:
    """Быстро читаем last open_time (ms) через seek от конца."""
    p = csv_path(symbol)
    if not p.exists() or p.stat().st_size < 100:
        return None
    with open(p, "rb") as f:
        size = p.stat().st_size
        f.seek(-min(8192, size), 2)
        tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
    for line in reversed(tail):
        parts = line.split(",")
        if len(parts) < 6 or parts[0] == "open_time":
            continue
        try:
            dt = datetime.fromisoformat(parts[0])
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    return None


def append_bars(symbol: str, bars: list[list]) -> int:
    """Append klines в CSV. bars = list of [open_time_ms, o, h, l, c, v, ...]."""
    if not bars:
        return 0
    p = csv_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists() or p.stat().st_size == 0
    written = 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for b in bars:
            open_ms = int(b[0])
            dt = datetime.fromtimestamp(open_ms / 1000, tz=UTC)
            iso = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            w.writerow([iso, str(b[1]), str(b[2]), str(b[3]), str(b[4]), str(b[5])])
            written += 1
    return written


# ══════════════════════════ BINANCE ══════════════════════════

def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Fetch klines в диапазоне [start_ms, end_ms), батчами."""
    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": BINANCE_LIMIT,
        }
        data = None
        for retry in range(FETCH_RETRIES):
            try:
                r = requests.get(BINANCE_URL, params=params, timeout=FETCH_TIMEOUT)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 5))
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                logging.warning(f"{symbol} fetch retry {retry+1}/{FETCH_RETRIES}: {e}")
                time.sleep(min(2 ** retry, 30))
        if data is None:
            raise RuntimeError(f"{symbol} fetch failed after {FETCH_RETRIES} retries")
        if not data:
            break
        all_bars.extend(data)
        last_open = int(data[-1][0])
        cursor = last_open + INTERVAL_MS
        # если binance вернул меньше limit — либо конец, либо gap → просто цикл
        if len(data) < BINANCE_LIMIT:
            break
    return all_bars


# ══════════════════════════ STATE ══════════════════════════

class SymbolState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.last_bar_ms: int | None = None
        self.gap_min: float = 0.0
        self.added_last_cycle: int = 0
        self.added_total: int = 0
        self.status: str = "init"
        self.last_error: str = ""
        self.data_ok: bool = False  # last N bars integrity (no gaps, no zeros)
        self.integrity_msg: str = ""

    def refresh_from_disk(self):
        self.last_bar_ms = read_last_ts_ms(self.symbol)
        if self.last_bar_ms is None:
            self.gap_min = -1
            self.status = "no_data"
        else:
            expected = int(time.time() * 1000) - INTERVAL_MS
            self.gap_min = max(0.0, (expected - self.last_bar_ms) / 60_000)
            if self.gap_min <= 2:
                self.status = "OK"
            elif self.gap_min <= 30:
                self.status = "lag"
            else:
                self.status = "GAP"

    def check_integrity(self, n_bars: int = 200):
        """Read last N bars from CSV, check gaps (60s spacing) + zero values."""
        try:
            csv_path = DATA_DIR / f"{self.symbol}_1m.csv"
            if not csv_path.exists():
                self.data_ok = False; self.integrity_msg = "no file"; return
            # Read tail — last N+1 rows
            with csv_path.open("rb") as f:
                f.seek(0, 2)
                filesize = f.tell()
                read_bytes = min(filesize, n_bars * 100)  # ~100 bytes per row estimate
                f.seek(filesize - read_bytes)
                tail = f.read().decode("utf-8", errors="ignore")
            lines = tail.splitlines()
            if len(lines) < 2:
                self.data_ok = False; self.integrity_msg = "too few rows"; return
            # Skip first line (might be partial), take last n_bars rows
            data_rows = lines[-n_bars:]
            # Header parsing: open_time,open,high,low,close,volume (or similar)
            # Parse each row → ts_ms, open, high, low, close
            from datetime import datetime as _dt, timezone as _tz
            parsed = []
            for line in data_rows:
                parts = line.split(",")
                if len(parts) < 5: continue
                try:
                    ts_str = parts[0]
                    ts_dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts_ms = int(ts_dt.timestamp() * 1000)
                    o, h, l, c = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    v = float(parts[5]) if len(parts) > 5 else 1.0
                    parsed.append((ts_ms, o, h, l, c, v))
                except Exception:
                    continue
            if len(parsed) < 10:
                self.data_ok = False; self.integrity_msg = "parse fail"; return
            # Check zeros
            zeros = sum(1 for row in parsed if any(v == 0 or v is None for v in row[1:5]))  # OHLC only, skip volume
            if zeros > 0:
                self.data_ok = False; self.integrity_msg = f"{zeros} zero values"; return
            # Check gaps (each ts should be prev + 60000ms)
            gaps = 0
            for i in range(1, len(parsed)):
                if parsed[i][0] - parsed[i-1][0] != 60_000:
                    gaps += 1
            if gaps > 0:
                self.data_ok = False; self.integrity_msg = f"{gaps} gaps"; return
            # OHLC consistency check
            bad_ohlc = sum(1 for row in parsed if row[2] < row[3] or row[2] < row[1] or row[2] < row[4] or row[3] > row[1] or row[3] > row[4])
            if bad_ohlc > 0:
                self.data_ok = False; self.integrity_msg = f"{bad_ohlc} bad OHLC"; return
            self.data_ok = True; self.integrity_msg = f"OK ({len(parsed)} bars)"
        except Exception as e:
            self.data_ok = False; self.integrity_msg = f"ERR {type(e).__name__}"


STATES = {s: SymbolState(s) for s in SYMBOLS}
EVENTS: deque[tuple[float, str]] = deque(maxlen=15)
STATS = {
    "start_ts": time.time(),
    "cycles": 0,
    "errors": 0,
    "last_cycle_ts": None,
    "next_cycle_ts": None,
    "last_heartbeat_ts": time.time(),  # первый heartbeat через час после старта, не сразу
}


def event(msg: str):
    EVENTS.append((time.time(), msg))
    logging.info(msg)


# ══════════════════════════ CYCLE ══════════════════════════

def update_symbol(state: SymbolState) -> None:
    state.refresh_from_disk()
    if state.last_bar_ms is None:
        state.status = "ERR"
        state.last_error = "no CSV / empty"
        state.data_ok = False
        state.integrity_msg = "no CSV"
        return
    now_ms = int(time.time() * 1000)
    # bars из start_ms = last_bar_ms + 1min; end закрытая свеча = (now // 60k) * 60k
    start_ms = state.last_bar_ms + INTERVAL_MS
    end_ms = (now_ms // INTERVAL_MS) * INTERVAL_MS
    if start_ms >= end_ms:
        state.added_last_cycle = 0
        state.status = "OK"
        return
    try:
        bars = fetch_klines(state.symbol, start_ms, end_ms)
    except Exception as e:
        state.status = "ERR"
        state.last_error = str(e)[:100]
        state.data_ok = False
        state.integrity_msg = f"fetch ERR: {type(e).__name__}"
        STATS["errors"] += 1
        return
    # фильтр: только закрытые (open_time + interval <= now) и дедуп по last_bar_ms
    clean = [b for b in bars if int(b[0]) > state.last_bar_ms and int(b[0]) + INTERVAL_MS <= now_ms]
    written = append_bars(state.symbol, clean)
    state.added_last_cycle = written
    state.added_total += written
    state.refresh_from_disk()
    state.check_integrity()



# ══════════════════════════ PIPELINE TRIGGER ══════════════════════════

PIPELINE_LOCK = BASE / "pipeline_running.lock"


def trigger_pipeline():
    """Fire-and-forget: spawn WSL pipeline. Skip if already running OR too recent."""
    # Throttle: только 1 run per 15-min quarter, on first ASVK cycle после quarter boundary MSK
    try:
        ps_path = BASE / "pipeline_status.json"
        if ps_path.exists():
            ps = json.loads(ps_path.read_text(encoding="utf-8"))
            fin_str = ps.get("finished_at", "")
            if fin_str and ps.get("state") == "success":
                fin_dt = datetime.fromisoformat(fin_str)
                now_dt = datetime.now(MSK)
                # Same 15-min quarter MSK — уже отработали
                fin_q = (fin_dt.year, fin_dt.month, fin_dt.day, fin_dt.hour, fin_dt.minute // 15)
                now_q = (now_dt.year, now_dt.month, now_dt.day, now_dt.hour, now_dt.minute // 15)
                if fin_q == now_q:
                    event(f"Pipeline throttled — уже отработал в {fin_dt.strftime('%H:%M')} MSK")
                    return
    except Exception:
        pass
    # Check lockfile
    if PIPELINE_LOCK.exists():
        try:
            mtime = PIPELINE_LOCK.stat().st_mtime
            age_s = time.time() - mtime
            if age_s < 1800:  # 30 min max stuck
                event("Pipeline already running (skip trigger)")
                return
        except Exception:
            pass
        try:
            PIPELINE_LOCK.unlink()
        except Exception:
            pass
    # Touch lock
    PIPELINE_LOCK.write_text(str(time.time()))
    # Spawn WSL pipeline detached
    try:
        cmd = [str(BASE / "python" / "python.exe"), str(BASE / "asvk_pipeline.py")]
        subprocess.Popen(cmd, creationflags=0x08000008, close_fds=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # DETACHED_PROCESS + CREATE_NO_WINDOW
        event("Pipeline triggered via python.exe")
    except Exception as e:
        event(f"Pipeline trigger FAILED: {e}")
        try: PIPELINE_LOCK.unlink()
        except Exception: pass


def run_cycle():
    STATS["cycles"] += 1
    for sym in SYMBOLS:
        st = STATES[sym]
        update_symbol(st)
    STATS["last_cycle_ts"] = time.time()
    added = sum(STATES[s].added_last_cycle for s in SYMBOLS)
    errs = [f"{s}:{STATES[s].last_error}" for s in SYMBOLS if STATES[s].status == "ERR"]
    if errs:
        event(f"cycle #{STATS['cycles']} errors: {'; '.join(errs)}")
        TG.notify(f"⚠️ ASVK: {'; '.join(errs)}", tag="warn")
    else:
        event(f"cycle #{STATS['cycles']} +{added} bars")
        if added > 0:
            trigger_pipeline()
    # heartbeat
    if time.time() - STATS["last_heartbeat_ts"] >= HEARTBEAT_SEC:
        parts = []
        for s in SYMBOLS:
            st = STATES[s]
            parts.append(f"{s[:3]}:{st.status}")
        TG.notify(f"❤️ ASVK жив. {' | '.join(parts)}", tag="info")
        STATS["last_heartbeat_ts"] = time.time()
    # TG queue flush
    TG.flush()
    # Escalate: если TG down > 30 мин → toast + sound
    if TG.enabled() and TG.last_success and (time.time() - TG.last_success) / 60 > TG_FAIL_ALERT_MIN:
        toast("ASVK — TG down", f"нет доставки {(time.time() - TG.last_success)/60:.0f} мин", urgent=True)


def next_cycle_ts() -> float:
    now = time.time()
    return ((int(now) // CYCLE_SEC) + 1) * CYCLE_SEC + CYCLE_BUFFER_SEC


# ══════════════════════════ TUI ══════════════════════════

console = Console()


def fmt_dt_msk(ms: int | None) -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000, tz=MSK).strftime("%m-%d %H:%M")



# Блок 3 (Liq_OB4h_VC + FVG_OB4h_VC) и Блок 4 (Liq_OB1h_VC + FVG_OB1h_VC) читают
# canonical race parquet из разных папок — держим списки раздельно, чтобы TUI-панели
# каждого блока показывали только свои сигналы, не вперемешку.
OB4H_ALGOS = [("L4h", "liq_ob4h_vc"), ("F4h", "fvg_ob4h_vc")]
OB1H_ALGOS = [("L1h", "liq_ob1h_vc"), ("F1h", "fvg_ob1h_vc")]


def _read_ob_signals(algos: list[tuple[str, str]], limit: int, empty_hint: str) -> list[str]:
    """Собрать canonical setups (Liq/FVG race) × BTC/ETH/SOL для заданного набора algo/folder.
    Sorted по ob_ts (birth) desc, top limit.
    Формат: {ob_ts YYYY-MM-DD HH:MM} — {algo} · {sym} · {DIR} · VC:{types} {first_vc_ts MM-DD HH:MM}"""
    import pandas as pd
    rows = []  # (ob_ts, first_vc_ts, algo, sym, direction, vc_types_str)
    _debug_paths = []
    for algo, folder in algos:
        for sym in SYMBOLS:
            s = sym.replace("USDT", "")
            parquet = BASE / "data" / folder / f"ob_stage4_race_canonical_{s}.parquet"
            _debug_paths.append(f"{parquet}={parquet.exists()}")
            if not parquet.exists(): continue
            try:
                df = pd.read_parquet(parquet, columns=["ob_ts","direction","vc_rb","vc_fvg","vc_snr","vc_any","vc_rb_ts","vc_fvg_ts","vc_snr_ts"])
                _debug_paths.append(f"  READ len={len(df)} vc_any={(df['vc_any']==True).sum()}")
            except Exception as _e:
                _debug_paths.append(f"  EXC: {type(_e).__name__}: {str(_e)[:200]}")
                continue
            df = df[df["vc_any"] == True]
            if len(df) == 0:
                _debug_paths.append(f"  FILTERED to 0")
                continue
            df = df.sort_values("ob_ts", ascending=False).head(limit)
            for _, r in df.iterrows():
                vc_types = []
                if bool(r["vc_rb"]):  vc_types.append("RB")
                if bool(r["vc_fvg"]): vc_types.append("FVG")
                if bool(r["vc_snr"]): vc_types.append("SNR")
                vc_ts_list = [int(r[c]) for c in ("vc_rb_ts","vc_fvg_ts","vc_snr_ts") if int(r[c]) > 0]
                first_vc_ts = min(vc_ts_list) if vc_ts_list else 0
                rows.append((int(r["ob_ts"]), first_vc_ts, algo, s, str(r["direction"]),
                             "+".join(vc_types) or "-"))
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:limit]
    try:
        (BASE / "logs" / "_signals_debug.log").write_text(
            f"rows={len(rows)} sample={rows[:2]}\nBASE={BASE}\nSYMBOLS={SYMBOLS}\n" + "\n".join(_debug_paths),
            encoding="utf-8")
    except Exception:
        pass
    if not rows:
        return [f"Signals  [dim]— {empty_hint} —[/]"]
    lines = []
    for ob_ts, vc_ts, algo, sym, dir_, vcs in rows:
        ob_dt = datetime.fromtimestamp(ob_ts/1000, tz=MSK).strftime("%Y-%m-%d %H:%M")
        vc_dt = datetime.fromtimestamp(vc_ts/1000, tz=MSK).strftime("%m-%d %H:%M") if vc_ts else "--"
        dir_str = "[green]LONG [/]" if dir_ == "long" else "[red]SHORT[/]"
        lines.append(f"  {ob_dt} — {algo} · {sym} · {dir_str} · VC:{vcs} {vc_dt}")
    return lines


def read_ob4h_signals(limit: int = 15) -> list[str]:
    """Блок 3: Liq_OB4h_VC + FVG_OB4h_VC × BTC/ETH/SOL."""
    return _read_ob_signals(OB4H_ALGOS, limit, "первый прогон Liq_OB4h/FVG_OB4h ещё не закончен")


def read_ob1h_signals(limit: int = 15) -> list[str]:
    """Блок 4: Liq_OB1h_VC + FVG_OB1h_VC × BTC/ETH/SOL."""
    return _read_ob_signals(OB1H_ALGOS, limit, "первый прогон Liq_OB1h/FVG_OB1h ещё не закончен")


_FRACTAL_COND_RE = re.compile(r"^b\d+c\d+$")


def read_latest_fractal_signals(limit: int = 15) -> list[str]:
    """Собрать basket (B1∪B9) hits 12h-фрактал стратегии × BTC/ETH/SOL.
    Sorted по pivot_open_ts_ms (birth) desc, top limit.
    Формат: {pivot_ts YYYY-MM-DD HH:MM} — {sym} · {DIR} · {conditions} · {status}
    conditions — КАЖДОЕ сработавшее под-условие (B1C1, B9C2, ...), не просто блок,
    т.к. на один сигнал может сработать сразу несколько условий (в т.ч. из разных блоков)."""
    import pandas as pd
    fractal_dir = BASE / "data" / "fractal12h"
    rows = []  # (pivot_ts, sym, direction, conditions_str, confirmable, confirmed)
    for sym in SYMBOLS:
        s = sym.replace("USDT", "")
        candidates = sorted(fractal_dir.glob(f"basket_hits_{s}_*.parquet"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            continue
        try:
            df = pd.read_parquet(candidates[-1])
        except Exception:
            continue
        cond_cols = sorted(c for c in df.columns if _FRACTAL_COND_RE.match(c))
        df = df[df["basket_hit"] == True]
        if len(df) == 0:
            continue
        df = df.sort_values("pivot_open_ts_ms", ascending=False).head(limit)
        for _, r in df.iterrows():
            fired = [c.upper() for c in cond_cols if bool(r.get(c, False))]
            rows.append((int(r["pivot_open_ts_ms"]), s, str(r["direction"]),
                        "+".join(fired) or "-", bool(r["confirmable"]), bool(r["confirmed"])))
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:limit]
    if not rows:
        return ["Signals  [dim]— пока пусто (первый прогон fractal12h ещё не закончен) —[/]"]
    lines = []
    for pivot_ts, sym, dir_, conditions, confirmable, confirmed in rows:
        pivot_dt = datetime.fromtimestamp(pivot_ts/1000, tz=MSK).strftime("%Y-%m-%d %H:%M")
        dir_str = "[green]LONG [/]" if dir_ == "long" else "[red]SHORT[/]"
        if not confirmable:
            status = "[yellow]PENDING[/]"
        elif confirmed:
            status = "[green]CONFIRMED[/]"
        else:
            status = "[dim]FAILED[/]"
        lines.append(f"  {pivot_dt} — {sym} · {dir_str} · {conditions} · {status}")
    return lines


def read_latest_pattern_signals(limit: int = 15) -> list[str]:
    """Собрать patterns_hits (22 паттерна Block 5) × BTC/ETH/SOL.
    Sorted по signal_ts desc, top limit.
    Формат: {signal_dt YYYY-MM-DD HH:MM} — {sym} · {DIR} · {pattern} · {status}"""
    import pandas as pd
    patterns_dir = BASE / "data" / "patterns"
    syms = [s.replace("USDT", "") for s in SYMBOLS]
    rows = []
    for sym in syms:
        candidates = sorted(patterns_dir.glob(f"patterns_hits_{sym}_*.parquet"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            continue
        try:
            df = pd.read_parquet(candidates[-1])
        except Exception:
            continue
        if len(df) == 0:
            continue
        df = df.drop_duplicates(subset=["signal_ts", "direction", "pattern", "status"])
        for _, r in df.iterrows():
            rows.append((int(r["signal_ts"]), sym, str(r["direction"]),
                         str(r["pattern"]), str(r["status"])))
    if not rows:
        return ["Signals  [dim]— пока пусто (первый прогон patterns ещё не закончен) —[/]"]
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:limit]
    lines = []
    for sig_ts, sym, direction, pattern, status in rows:
        signal_dt = datetime.fromtimestamp(sig_ts / 1000, tz=MSK).strftime("%Y-%m-%d %H:%M")
        dir_str = "[green]LONG [/]" if direction == "long" else "[red]SHORT[/]"
        pat_str = pattern.upper()
        if status == "CONFIRMED":
            status_str = "[green]CONFIRMED[/]"
        elif status == "PENDING":
            status_str = "[yellow]PENDING[/]"
        else:
            status_str = "[dim]FAILED[/]"
        lines.append(f"  {signal_dt} — {sym} · {dir_str} · {pat_str} · {status_str}")
    return lines


def render():
    now_msk = datetime.now(MSK).strftime("%H:%M:%S  %Y-%m-%d  MSK")

    # ── Блок 1: данные (backfill) + e12d/s7d ──
    symbol_shorts = [s.replace("USDT", "") for s in SYMBOLS]
    all_have_bar = all(STATES[s].last_bar_ms for s in SYMBOLS)
    if all_have_bar:
        min_bar_ms = min(STATES[s].last_bar_ms for s in SYMBOLS)
        data_time = datetime.fromtimestamp(min_bar_ms/1000, tz=MSK).strftime("%H:%M %Y-%m-%d MSK")
        all_data_ok = all(STATES[s].data_ok for s in SYMBOLS)
        data_dot = "[green]●[/]" if all_data_ok else "[red]●[/]"
        line1 = f"{' · '.join(symbol_shorts)}  {data_dot}  {data_time}"
    else:
        line1 = f"[dim]{' · '.join(symbol_shorts)}[/]  [red]●[/]  no data"

    line2 = "E12D S7D  [red]●[/]  — no runs yet —"
    try:
        ps = json.loads((BASE / "pipeline_status.json").read_text(encoding="utf-8", errors="replace"))
        state = ps.get("state", "?")
        fin_str = ps.get("finished_at", "")
        start_str = ps.get("started_at", "")
        if state == "running" and start_str:
            start_dt = datetime.fromisoformat(start_str)
            age_min = (time.time() - start_dt.timestamp()) / 60
            n_ok = sum(1 for v in ps.get("steps",{}).values() if v.get("ok"))
            line2 = f"E12D S7D  [yellow]●[/]  RUNNING {age_min:.0f}m ({n_ok} steps OK)"
        elif fin_str:
            fin_dt = datetime.fromisoformat(fin_str)
            age_min = (time.time() - fin_dt.timestamp()) / 60
            pipe_ok = state == "success" and age_min < 90
            pipe_dot = "[green]●[/]" if pipe_ok else "[red]●[/]"
            line2 = f"E12D S7D  {pipe_dot}  {fin_dt.strftime('%H:%M %Y-%m-%d MSK')}"
    except Exception:
        pass

    block1_content = Group(Text.from_markup(line1), Text.from_markup(line2))
    panel_block1 = Panel(block1_content, border_style="cyan",
                         title=f"[bold cyan]{now_msk}[/]", title_align="right")

    # ── Блок 2: 12h-фрактал (A-фильтры + B1..B9) ──
    fractal_lines = read_latest_fractal_signals(limit=15)
    fractal_content = Group(Text.from_markup("Signals:"),
                            *[Text.from_markup(sl) for sl in fractal_lines])
    panel_block2 = Panel(fractal_content, border_style="cyan",
                         title="[bold cyan]FRACTAL 12h · BTC/ETH/SOL[/]", title_align="right")

    # ── Блок 3: Liq_OB4h_VC + FVG_OB4h_VC ──
    ob4h_lines = read_ob4h_signals(limit=15)
    ob4h_content = Group(Text.from_markup("Signals:"),
                         *[Text.from_markup(sl) for sl in ob4h_lines])
    panel_block3 = Panel(ob4h_content, border_style="cyan",
                         title="[bold cyan]Volume Confirmation 4h · BTC/ETH/SOL[/]", title_align="right")

    # ── Блок 4: Liq_OB1h_VC + FVG_OB1h_VC ──
    ob1h_lines = read_ob1h_signals(limit=15)
    ob1h_content = Group(Text.from_markup("Signals:"),
                         *[Text.from_markup(sl) for sl in ob1h_lines])
    panel_block4 = Panel(ob1h_content, border_style="cyan",
                         title="[bold cyan]Volume Confirmation 1h · BTC/ETH/SOL[/]", title_align="right")

    # ── Блок 5: паттерны (H&S TOP + Wedge falling) ──
    pattern_lines = read_latest_pattern_signals(limit=15)
    pattern_content = Group(Text.from_markup("Signals:"),
                            *[Text.from_markup(sl) for sl in pattern_lines])
    panel_block5 = Panel(pattern_content, border_style="cyan",
                         title="[bold cyan]PATTERNS 1h · BTC/ETH/SOL[/]", title_align="right")

    return Group(panel_block1, panel_block2, panel_block3, panel_block4, panel_block5)


# ══════════════════════════ MAIN ══════════════════════════

def setup_logging():
    from logging.handlers import RotatingFileHandler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")  # 5MB × 3 backup = 20MB max
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def startup_report():
    parts = []
    for s in SYMBOLS:
        st = STATES[s]
        st.refresh_from_disk()
        st.check_integrity()
        if st.last_bar_ms is None:
            parts.append(f"{s}: НЕТ данных")
        else:
            last = fmt_dt_msk(st.last_bar_ms)
            parts.append(f"{s}: last {last} MSK, gap {st.gap_min:.0f}m")
    msg = "🚀 ASVK запущен\n" + "\n".join(parts)
    event("ASVK started")
    TG.notify(msg, tag="info")


def main():
    setup_logging()
    logging.info("=" * 60)
    logging.info("ASVK startup")
    if not TG.enabled():
        event("⚠ TG disabled (config.txt пуст или без TG_TOKEN/TG_CHAT)")

    startup_report()

    STATS["next_cycle_ts"] = time.time()  # первый цикл сразу

    # screen=True — альтернативный буфер терминала (полноценный double-buffer). С
    # screen=False Rich каждую секунду стирает и перерисовывает построчно поверх основного
    # буфера; с ростом высоты вывода (теперь 3 полные панели вместо 1) это стало заметно
    # мигать. screen=True устраняет мерцание полностью — цена: финальные event()-сообщения
    # после выхода из Live (Ctrl+C) не остаются на экране (буфер восстанавливается), но они
    # всё равно попадают в лог и в "=== ASVK stopped ===" от ASVK.bat.
    with Live(auto_refresh=False, console=console, screen=True) as live:
        try:
            while True:
                now = time.time()
                if now >= STATS["next_cycle_ts"]:
                    try:
                        run_cycle()
                    except Exception:
                        logging.error("run_cycle crashed:\n" + traceback.format_exc())
                        STATS["errors"] += 1
                        event("cycle crashed (см. log)")
                    STATS["next_cycle_ts"] = next_cycle_ts()
                live.update(render(), refresh=True)
                time.sleep(1.0)  # 1 render/sec — довольно для MSK second display
        except KeyboardInterrupt:
            event("stopped by user (Ctrl+C)")
            TG.notify("🛑 ASVK остановлен пользователем", tag="info")
            TG.flush()
            logging.info("ASVK stopped clean")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.error("FATAL:\n" + traceback.format_exc())
        toast("ASVK CRASHED", "См. logs/asvk.log", urgent=True)
        raise
