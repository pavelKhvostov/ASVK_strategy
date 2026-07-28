#!/usr/bin/env python3
"""ASVK self-contained pipeline. Runs INSIDE G:\\ASVK\\ using bundled Python.

All paths ASVK-relative. All Python packages bundled. No external dependencies.

Block structure (matches daemon TUI panels 1:1):
    Блок 1 — данные (backfill) + e12d/s7d + shared level-1 индикаторы (maxv,
             trendline, vwap_anchors)
    Блок 2 — 12h-фрактал: A-фильтры (a1..a4) + B1..B9 условия
    Блок 3 — Liq_OB4h_VC + FVG_OB4h_VC (почасовой gate)
    Блок 4 — Liq_OB1h_VC + FVG_OB1h_VC (каждый 15-мин цикл)
    Блок 5 — Паттерны (H&S TOP + Wedge falling)
Блоки только переставлены/сгруппированы; каждый шаг внутри вызывает тот же
скрипт с теми же аргументами, что и раньше — поведение не менялось.

Блоки 2/3/4/5 независимы от вывода друг друга — запускаются параллельно.
Символы BTC/ETH/SOL независимы внутри каждого блока — тоже параллельно.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

ASVK_BASE = Path(__file__).resolve().parent
PYTHON_EXE = ASVK_BASE / "python" / "python.exe"
LIB_DIR = ASVK_BASE / "lib"
DATA_DIR = ASVK_BASE / "data"
EVENTS_DIR = DATA_DIR / "events"
S7D_DIR = DATA_DIR / "s7d"
SNAPSHOTS_DIR = ASVK_BASE / "snapshots"
STATUS_FILE = ASVK_BASE / "pipeline_status.json"
LOG_FILE = ASVK_BASE / "logs" / "pipeline.log"
MSK = timezone(timedelta(hours=3))
SYMBOLS = ["BTC", "ETH", "SOL"]

# MA-варианты, которые читает lib/fractal12h/b4_hma.py (SUB_CONDITIONS) помимо D200.
# Список должен зеркалить SUB_CONDITIONS там; при добавлении нового B4Cn на новом
# MA-варианте — добавить и сюда, иначе latest_trendline_path() тихо продолжит отдавать
# старый файл вместо ошибки (найдено при аудите 2026-07-24).
B4_MA_VARIANTS = [
    ("12h", 78, "Hma"),   # B4C1a
    ("D",   78, "Hma"),   # B4C1b
    ("12h",  9, "Thma"),  # B4C3
    ("D",   50, "Wma"),   # B4C4
    ("D",    9, "Thma"),  # B4C5
    ("D",   20, "Ehma"),  # B4C6
]

for d in [EVENTS_DIR, S7D_DIR, SNAPSHOTS_DIR, LOG_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)

# Force UTF-8 stdout для non-ASCII в log messages
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Lock for thread-safe log and status writes
_WRITE_LOCK = threading.Lock()


def log(msg: str):
    ts = datetime.now(MSK).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} MSK] {msg}"
    print(line, flush=True)
    with _WRITE_LOCK:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_block(title: str):
    log(f"═══ {title} ═══")


def write_status(data: dict):
    with _WRITE_LOCK:
        tmp = STATUS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATUS_FILE)


def run_python(script: str, args: list, timeout: int = 300) -> tuple[bool, float, list]:
    """Robust bundled Python subprocess.
       stdout/stderr → tempfiles (не pipe → нет buffer deadlock).
       Hard kill на timeout.
       Возврат (ok, duration_s, tail 6 строк).
    """
    import tempfile
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{LIB_DIR};{LIB_DIR / 'детекторы'}" if os.name == "nt" else f"{LIB_DIR}:{LIB_DIR / 'детекторы'}"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [str(PYTHON_EXE), script] + args

    tmp_dir = ASVK_BASE / "logs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Use thread id for uniqueness when called from multiple threads simultaneously
    tid = threading.get_ident()
    stdout_path = tmp_dir / f"_sub_{os.getpid()}_{tid}_{int(t0*1000)}_out.tmp"
    stderr_path = tmp_dir / f"_sub_{os.getpid()}_{tid}_{int(t0*1000)}_err.tmp"

    rc = -1
    killed = False
    try:
        with open(stdout_path, "wb") as fout, open(stderr_path, "wb") as ferr:
            CREATE_NO_WINDOW = 0x08000000
            proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, env=env, creationflags=CREATE_NO_WINDOW)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                killed = True
                proc.kill()
                try:
                    rc = proc.wait(timeout=5)
                except Exception:
                    rc = -9

        dt = round(time.time() - t0, 1)
        def read_tail(p):
            try:
                data = p.read_bytes()[-3000:]
                return data.decode("utf-8", errors="replace").splitlines()[-3:]
            except Exception:
                return []
        tail = read_tail(stdout_path) + read_tail(stderr_path)
        if killed:
            tail.insert(0, f"[KILLED after {timeout}s timeout]")
    finally:
        for p in (stdout_path, stderr_path):
            try: p.unlink()
            except Exception: pass

    return (rc == 0), dt, tail


def run_step(script: Path, args: list, status: dict, key: str, label: str, timeout: int = 300):
    """Запустить один шаг, записать результат в status[key] + лог + write_status."""
    ok, dt, tail = run_python(str(script), args, timeout=timeout)
    status["steps"][key] = {"ok": ok, "duration_s": dt, "output_tail": tail}
    log(f"  {label}: {'OK' if ok else 'FAIL'} in {dt}s")
    write_status(status)
    return ok


def run_stage_chain(base_dir: Path, stages: list[tuple[str, str, int]], sym: str,
                     end_date: str, status: dict, key_prefix: str):
    """Запустить цепочку stage1..stageN для одного символа; при FAIL — остальные stage SKIP."""
    chain_broken = False
    for stage_key, script_name, tmo in stages:
        if chain_broken:
            status["steps"][f"{key_prefix}_{stage_key}_{sym}"] = {
                "ok": False, "duration_s": 0, "output_tail": ["[SKIPPED — earlier stage FAIL]"]
            }
            log(f"  {key_prefix} {stage_key} {sym}: SKIP (earlier stage FAIL)")
            write_status(status)
            continue
        ok = run_step(base_dir / script_name, ["--symbol", sym, "--end", end_date],
                       status, f"{key_prefix}_{stage_key}_{sym}", f"{key_prefix} {stage_key} {sym}",
                       timeout=tmo)
        if not ok:
            chain_broken = True


def _run_parallel(tasks: list, label: str = "parallel"):
    """Run list of (callable, *args) concurrently via threads. Re-raises first exception."""
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(fn, *args) for fn, *args in tasks]
        errors = []
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                log(f"  {label} task ERROR: {e}")
                errors.append(e)
        if errors:
            raise errors[0]


# ══════════════════════════ Блок 1: данные + e12d/s7d/индикаторы ══════════════════════════

def block1_data_and_indicators(status: dict, end_date: str):
    """Backfill 1m + e12d (зоны интереса) + s7d (снапшоты) + shared level-1
    индикаторы (maxv/ATR14, trendline D200 + B4_MA_VARIANTS + 1h78, wma,
    vwap_anchors), которые читают детекторы блоков 2-5."""
    log_block("Блок 1: данные + e12d/s7d/индикаторы")

    run_step(LIB_DIR / "binance_backfill.py", [], status, "backfill", "backfill")

    for sym in SYMBOLS:
        run_step(LIB_DIR / "e12d.py", ["--symbol", sym, "--start", "2018-01-01", "--end", end_date],
                  status, f"e12d_{sym}", f"e12d {sym}")

    for sym in SYMBOLS:
        run_step(LIB_DIR / "s7d.py", ["--symbol", sym, "--start", "2018-01-01", "--end", end_date],
                  status, f"s7d_{sym}", f"s7d {sym}")

    # Batch: trendline(8 вариантов) + maxv + wma + vwap_anchors — загружает CSV один раз на символ
    for sym in SYMBOLS:
        run_step(LIB_DIR / "trendline_batch.py",
                  ["--symbol", sym, "--end", end_date],
                  status, f"trendline_batch_{sym}", f"trendline_batch {sym}")


# ══════════════════════════ Блок 2: 12h-фрактал ══════════════════════════

def _cold_start_flags(now_msk: datetime) -> tuple[bool, bool]:
    """Определяет нужен ли принудительный запуск заблокированных по времени блоков.

    Возвращает (force_hourly, force_12h):
    - force_hourly: блоки 3/5 запустить сейчас даже если минута >= 15
    - force_12h:   блок 2 запустить сейчас даже если UTC час не 0/12

    Логика: смотрим финальный timestamp предыдущего прогона.
    Если прошло > 20 мин — значит пропустили хотя бы один hourly цикл.
    Если прошло > 12h — значит пропустили хотя бы один 12h цикл.
    Дополнительно: если data/fractal12h/ не существует — блок 2 никогда не считался.
    """
    force_hourly = False
    force_12h = False
    try:
        if STATUS_FILE.exists():
            prev = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            finished_str = prev.get("finished_at")
            if finished_str:
                finished = datetime.fromisoformat(finished_str)
                downtime_s = (now_msk - finished).total_seconds()
                if downtime_s > 20 * 60:
                    force_hourly = True
                    log(f"  cold-start: {downtime_s/60:.0f} мин простоя → force_hourly блоки 3/5")
                if downtime_s > 12 * 3600:
                    force_12h = True
                    log(f"  cold-start: {downtime_s/3600:.1f}h простоя → force_12h блок 2")
            else:
                force_hourly = force_12h = True
                log("  cold-start: нет finished_at → force all gates")
        else:
            force_hourly = force_12h = True
            log("  cold-start: нет pipeline_status.json → force all gates")
    except Exception as e:
        force_hourly = force_12h = True
        log(f"  cold-start: ошибка чтения статуса ({e}) → force all gates")

    fractal_dir = DATA_DIR / "fractal12h"
    if not fractal_dir.exists() or not any(fractal_dir.glob("basket_hits_BTC_*.parquet")):
        if not force_12h:
            force_12h = True
            log("  cold-start: data/fractal12h/ пуст → force_12h блок 2")

    return force_hourly, force_12h


def block2_fractal12h(status: dict, end_date: str, now_utc: datetime, force: bool = False):
    """12h gate: запускается в 15-мин окне после закрытия 12h-бара (00:00 и 12:00 UTC).
    force=True при cold-start: пропускает gate.
    Параллельно по символам."""
    log_block("Блок 2: 12h-фрактал (A + B1..B9)")
    if not force and (now_utc.hour not in (0, 12) or now_utc.minute >= 15):
        log(f"  fractal12h: SKIPPED (UTC {now_utc.strftime('%H:%M')}, 12h gate — только 00:00-00:14 и 12:00-12:14 UTC)")
        return
    if force:
        log(f"  fractal12h: FORCED (cold-start, UTC {now_utc.strftime('%H:%M')})")
    FRACTAL12H_DIR = LIB_DIR / "fractal12h"
    tasks = [
        (run_step,
         FRACTAL12H_DIR / "run_fractal12h.py",
         ["--symbol", sym, "--start", "2020-01-01", "--end", end_date],
         status, f"fractal12h_{sym}", f"fractal12h {sym}", 600)
        for sym in SYMBOLS
    ]
    _run_parallel(tasks, "fractal12h")


# ══════════════════════════ Блок 3: Liq_OB4h_VC + FVG_OB4h_VC ══════════════════════════

def block3_ob4h(status: dict, end_date: str, now_msk: datetime, force: bool = False):
    """Почасовой gate — запускается только в первый 15-мин cycle нового часа MSK
    (минута < 15), т.к. входной 4h OB canonical не меняется чаще раза в час.
    force=True при cold-start: пропускает gate.
    Символы и цепочки (liq + fvg) запускаются параллельно."""
    log_block("Блок 3: Liq_OB4h_VC + FVG_OB4h_VC")
    if not force and now_msk.minute >= 15:
        log(f"  ob4h_vc: SKIPPED (MSK {now_msk.strftime('%H:%M')}, hourly gate — только :00-:14)")
        return
    if force and now_msk.minute >= 15:
        log(f"  ob4h_vc: FORCED (cold-start, MSK {now_msk.strftime('%H:%M')})")

    LIQ4H_DIR = LIB_DIR / "Liq_OB4h_VC"
    LIQ4H_STAGES = [
        ("stage1", "ob_canonical_4h.py",         600),
        ("stage2", "ob_stage2_fvg_container.py", 900),
        ("stage3", "ob_stage3_1h_vc.py",         600),
        ("stage4", "ob_stage4_race.py",         1200),
    ]
    FVG4H_DIR = LIB_DIR / "FVG_OB4h_VC"
    FVG4H_STAGES = [
        ("stage1", "fvg_ob_canonical_4h.py", 600),
        ("stage3", "ob_stage3_1h_vc.py",     600),
        ("stage4", "ob_stage4_race.py",     1200),
    ]

    log(f"  liq_ob4h_vc + fvg_ob4h_vc: RUNNING parallel {len(SYMBOLS)} syms × 2 chains"
        f" (MSK {now_msk.strftime('%H:%M')})")
    tasks = (
        [(run_stage_chain, LIQ4H_DIR, LIQ4H_STAGES, sym, end_date, status, "liq_ob4h_vc")
         for sym in SYMBOLS] +
        [(run_stage_chain, FVG4H_DIR, FVG4H_STAGES, sym, end_date, status, "fvg_ob4h_vc")
         for sym in SYMBOLS]
    )
    _run_parallel(tasks, "ob4h")


# ══════════════════════════ Блок 4: Liq_OB1h_VC + FVG_OB1h_VC ══════════════════════════

def block4_ob1h(status: dict, end_date: str, now_msk: datetime):
    """Каждый 15-мин cycle — символы и цепочки (liq + fvg) параллельно."""
    log_block("Блок 4: Liq_OB1h_VC + FVG_OB1h_VC")

    LIQ1H_DIR = LIB_DIR / "Liq_OB1h_VC"
    LIQ1H_STAGES = [
        ("stage1", "ob_canonical_1h.py",         600),
        ("stage2", "ob_stage2_fvg_container.py", 900),
        ("stage3", "ob_stage3_15m_vc.py",        600),
        ("stage4", "ob_stage4_race.py",         1200),
    ]
    FVG1H_DIR = LIB_DIR / "FVG_OB1h_VC"
    FVG1H_STAGES = [
        ("stage1", "fvg_ob_canonical_1h.py", 600),
        ("stage3", "ob_stage3_15m_vc.py",    600),
        ("stage4", "ob_stage4_race.py",     1200),
    ]

    log(f"  liq_ob1h_vc + fvg_ob1h_vc: RUNNING parallel {len(SYMBOLS)} syms × 2 chains"
        f" (MSK {now_msk.strftime('%H:%M')})")
    tasks = (
        [(run_stage_chain, LIQ1H_DIR, LIQ1H_STAGES, sym, end_date, status, "liq_ob1h_vc")
         for sym in SYMBOLS] +
        [(run_stage_chain, FVG1H_DIR, FVG1H_STAGES, sym, end_date, status, "fvg_ob1h_vc")
         for sym in SYMBOLS]
    )
    _run_parallel(tasks, "ob1h")


# ══════════════════════════ Блок 5: Паттерны ══════════════════════════

def block5_patterns(status: dict, end_date: str, now_msk: datetime, force: bool = False):
    """Hourly gate: запускается в 15-мин окне после закрытия 1h-бара.
    22 паттерна — инкрементальный скан последних 600 1h-баров. Параллельно по символам.
    force=True при cold-start: пропускает gate."""
    log_block("Блок 5: Паттерны (22 паттерна: SHORT + LONG)")
    if not force and now_msk.minute >= 15:
        log(f"  patterns: SKIPPED (MSK {now_msk.strftime('%H:%M')}, hourly gate — только :00-:14)")
        return
    if force and now_msk.minute >= 15:
        log(f"  patterns: FORCED (cold-start, MSK {now_msk.strftime('%H:%M')})")
    PATTERNS_DIR = LIB_DIR / "patterns"
    tasks = [
        (run_step,
         PATTERNS_DIR / "run_patterns.py",
         ["--symbol", sym, "--start", "2020-01-01", "--end", end_date],
         status, f"patterns_{sym}", f"patterns {sym}", 120)
        for sym in SYMBOLS
    ]
    _run_parallel(tasks, "patterns")


# ══════════════════════════ Export + finalize ══════════════════════════

def export_live_snapshot(status: dict):
    import pandas as pd
    summary = {"exported_at": datetime.now(MSK).isoformat(), "symbols": {}}
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for sym in SYMBOLS:
        candidates = sorted(S7D_DIR.glob(f"snapshots_s7d_{sym}_*.parquet"))
        if not candidates: continue
        df = pd.read_parquet(candidates[-1])
        df = df[df["anchor_ts"] <= now_ms]
        if len(df) == 0: continue
        max_anchor = df["anchor_ts"].max()
        latest = df[df["anchor_ts"] == max_anchor].copy()
        out = SNAPSHOTS_DIR / f"{sym}_latest.parquet"
        latest.to_parquet(out, index=False)
        summary["symbols"][sym] = {
            "anchor_ts_ms": int(max_anchor),
            "anchor_msk": datetime.fromtimestamp(max_anchor/1000, tz=MSK).strftime("%Y-%m-%d %H:%M MSK"),
            "current_price": float(latest["current_price"].iloc[0]) if "current_price" in latest.columns else None,
            "n_zones": len(latest),
        }
    (SNAPSHOTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    status["steps"]["export"] = {"ok": True, "summary": summary}
    log(f"  export: OK anchors {[v.get('anchor_msk') for v in summary['symbols'].values()]}")


def finalize(status: dict, global_t0: float) -> int:
    total = round(time.time() - global_t0, 1)
    all_ok = all(v.get("ok", False) for v in status["steps"].values())
    status["state"] = "success" if all_ok else "partial_fail"
    status["finished_at"] = datetime.now(MSK).isoformat()
    status["total_duration_s"] = total
    write_status(status)
    log(f"═══ ASVK pipeline finished {total}s | {status['state']} ═══")
    try:
        (ASVK_BASE / "pipeline_running.lock").unlink(missing_ok=True)
    except Exception:
        pass
    return 0 if all_ok else 1


def main():
    global_t0 = time.time()
    started_at = datetime.now(MSK).isoformat()
    end_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    now_utc = datetime.now(timezone.utc)
    now_msk = datetime.now(MSK)
    log(f"═══ ASVK pipeline started (end={end_date}) ═══")

    status = {"state": "running", "mode": "asvk-standalone", "started_at": started_at,
              "end_date": end_date, "steps": {}}
    write_status(status)

    block1_data_and_indicators(status, end_date)

    # Блоки 2/3/4/5 не зависят от вывода друг друга — запускаем параллельно
    # Cold-start: если демон только запустился или долго не работал — форсируем gate-блоки
    force_hourly, force_12h = _cold_start_flags(now_msk)
    log("═══ Блоки 2/3/4/5: параллельный старт ═══")
    _run_parallel([
        (block2_fractal12h, status, end_date, now_utc, force_12h),
        (block3_ob4h,       status, end_date, now_msk, force_hourly),
        (block4_ob1h,       status, end_date, now_msk),
        (block5_patterns,   status, end_date, now_msk, force_hourly),
    ], "blocks2-5")

    export_live_snapshot(status)
    return finalize(status, global_t0)


if __name__ == "__main__":
    sys.exit(main())
