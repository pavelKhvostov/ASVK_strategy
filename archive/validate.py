#!/usr/bin/env python3
"""Regression validation для ASVK.

Сравнивает текущие выходные данные с зафиксированным baseline.
Запускать ПОСЛЕ полного pipeline прогона с тем же --end date что в baseline.

Использование:
    python3 validate.py                          # авто-найти последний baseline
    python3 validate.py --date 2026-07-28        # конкретный baseline
    python3 validate.py --run-pipeline           # сначала запустить pipeline, потом проверить

Выход:
    0 — PASS (всё совпадает)
    1 — FAIL (есть расхождения)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BDIR = BASE / "baseline"
PYTHON = BASE / "python" / "python.exe"
SYMS = ["BTC", "ETH", "SOL"]

# ──────────────────────────────────────────────────────────────
def stats(path: Path) -> dict:
    df = pd.read_parquet(path)
    s: dict = {"rows": len(df)}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s["col0_sum"] = float(round(df[col].sum(), 4))
            s["col0_name"] = col
            break
    return s


def compare(label: str, baseline: dict, current: dict, fails: list) -> None:
    diffs = []
    for k in ("rows", "col0_sum"):
        if k not in baseline or k not in current:
            continue
        b, c = baseline[k], current[k]
        if k == "rows":
            if b != c:
                diffs.append(f"rows {b}→{c} (delta {c-b:+d})")
        else:
            # float: allow tiny rounding, but same integer part
            if round(b) != round(c):
                diffs.append(f"col0_sum {b:.2e}→{c:.2e}")
    if diffs:
        print(f"  FAIL  {label}: {', '.join(diffs)}")
        fails.append(label)
    else:
        print(f"  OK    {label}: rows={current['rows']}")


# ──────────────────────────────────────────────────────────────
def validate(baseline_date: str) -> int:
    summary_path = BDIR / f"summary_{baseline_date}.json"
    if not summary_path.exists():
        print(f"ERROR: baseline не найден: {summary_path}", file=sys.stderr)
        return 1

    baseline = json.loads(summary_path.read_text())
    fails: list[str] = []

    print(f"\n=== ASVK Regression Validate (baseline {baseline_date}) ===\n")

    # ── e12d ──
    print("── e12d events ──")
    for sym in SYMS:
        key = sym
        bl = baseline["sections"]["e12d"].get(key)
        if bl is None:
            continue
        cands = sorted((DATA / "events").glob(f"events_e12d_{sym}_*_{baseline_date}.parquet"))
        if not cands:
            print(f"  MISS  e12d_{sym}: файл не найден (pipeline не запускался?)")
            fails.append(f"e12d_{sym}_missing")
            continue
        compare(f"e12d_{sym}", bl, stats(cands[-1]), fails)

    # ── s7d ──
    print("\n── s7d snapshots ──")
    for sym in SYMS:
        key = sym
        bl = baseline["sections"]["s7d"].get(key)
        if bl is None:
            continue
        cands = sorted((DATA / "s7d").glob(f"snapshots_s7d_{sym}_*_{baseline_date}.parquet"))
        if not cands:
            batch = DATA / "s7d" / f"snapshots_s7d_{sym}_batch.parquet"
            if batch.exists():
                cands = [batch]
        if not cands:
            print(f"  MISS  s7d_{sym}: файл не найден")
            fails.append(f"s7d_{sym}_missing")
            continue
        compare(f"s7d_{sym}", bl, stats(cands[-1]), fails)

    # ── fractal12h ──
    print("\n── fractal12h ──")
    bl_f = baseline["sections"]["fractal12h"]
    for sym in SYMS:
        key = f"a_cand_{sym}"
        if key in bl_f:
            cands = sorted((DATA / "fractal12h").glob(f"a_candidates_{sym}_*_{baseline_date}.parquet"))
            if cands:
                compare(key, bl_f[key], stats(cands[-1]), fails)
    for bn in [1, 2, 3, 4, 5, 8, 9]:
        for sym in SYMS:
            key = f"b{bn}_{sym}"
            if key not in bl_f:
                continue
            p = DATA / "fractal12h" / f"b{bn}_hits_{sym}_2020-01-01_{baseline_date}.parquet"
            if not p.exists():
                print(f"  MISS  {key}")
                fails.append(f"{key}_missing")
                continue
            compare(key, bl_f[key], stats(p), fails)

    # ── OB stage4 (compare against saved copies) ──
    print("\n── OB stage4 race ──")
    bl_ob = baseline["sections"]["ob_stage4"]
    algos = ["liq_ob4h_vc", "liq_ob1h_vc", "fvg_ob4h_vc", "fvg_ob1h_vc"]
    for algo in algos:
        for sym in SYMS:
            for kind in ["canonical", "sweep_only"]:
                key = f"{algo}/{kind}_{sym}"
                if key not in bl_ob:
                    continue
                cur_p = DATA / algo / f"ob_stage4_race_{kind}_{sym}.parquet"
                if not cur_p.exists():
                    print(f"  MISS  {key}")
                    fails.append(f"{key}_missing")
                    continue
                compare(key, bl_ob[key], stats(cur_p), fails)

    # ── Patterns ──
    print("\n── Patterns ──")
    bl_pat = baseline["sections"]["patterns"]
    for key, bl in bl_pat.items():
        # key = 'bat_BTC' → файл bat_hits_BTC_2020-01-01_{date}.parquet
        parts = key.rsplit("_", 1)
        pat_name, sym = parts[0], parts[1]
        cands = sorted((DATA / "patterns").glob(f"{pat_name}_hits_{sym}_*_{baseline_date}.parquet"))
        if not cands:
            print(f"  MISS  patterns/{key}")
            fails.append(f"patterns/{key}_missing")
            continue
        compare(f"patterns/{key}", bl, stats(cands[-1]), fails)

    # ── Summary ──
    print(f"\n{'='*50}")
    if fails:
        print(f"FAIL — {len(fails)} расхождений:")
        for f in fails:
            print(f"  • {f}")
        return 1
    else:
        print(f"PASS — все {len(bl_ob) + len(bl_f) + 3 + 3 + len(bl_pat)} проверок совпали с baseline")
        return 0


# ──────────────────────────────────────────────────────────────
def run_pipeline(end_date: str) -> bool:
    """Запускает asvk_pipeline.py с фиксированным end_date."""
    # Temporarily patch end_date — pipeline вычисляет его сам внутри main()
    # Используем переменную окружения чтобы не трогать код
    import os
    env = os.environ.copy()
    env["ASVK_END_DATE_OVERRIDE"] = end_date
    result = subprocess.run(
        [str(PYTHON), str(BASE / "asvk_pipeline.py")],
        env=env, cwd=str(BASE)
    )
    return result.returncode == 0


# ──────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="ASVK regression validator")
    p.add_argument("--date", help="Baseline date (default: последний найденный)")
    p.add_argument("--run-pipeline", action="store_true",
                   help="Запустить pipeline перед валидацией")
    args = p.parse_args()

    # Найти baseline date
    if args.date:
        bdate = args.date
    else:
        files = sorted(BDIR.glob("summary_*.json"))
        if not files:
            print("ERROR: baseline не найден в baseline/", file=sys.stderr)
            return 1
        bdate = files[-1].stem.replace("summary_", "")
        print(f"Использую baseline: {bdate}")

    if args.run_pipeline:
        print(f"Запускаю pipeline (end_date={bdate})...")
        if not run_pipeline(bdate):
            print("ERROR: pipeline завершился с ошибкой", file=sys.stderr)
            return 1

    return validate(bdate)


if __name__ == "__main__":
    sys.exit(main())
