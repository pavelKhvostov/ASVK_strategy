"""arden_live — живой раннер стратегии Арденского с оповещениями в Telegram.

ЧТО ДЕЛАЕТ:
  1. Каждые N секунд тянет свечи с Binance по BTC/ETH/SOL на ТФ 2h/4h/6h/12h.
  2. Ищет сетапы Power of Three (ложный вынос → слом структуры ≤3 баров → коррекция в OTE 0.79).
  3. Как только сетап сформировался — присылает в Telegram план сделки: вход, стоп, две цели.
  4. Обновляет локальный дашборд dashboard.html (график + активные сигналы + статистика).
  5. При первом запуске прописывает себя в автозагрузку Windows — работает после перезагрузки.

ЗАПУСК:
  python arden_live.py                 обычный запуск (и установка автозапуска)
  python arden_live.py --once          один проход и выход (для проверки)
  python arden_live.py --test-telegram проверить связь с ботом
  python arden_live.py --no-autostart  не прописывать автозагрузку
  python arden_live.py --uninstall     убрать из автозагрузки и выйти

НАСТРОЙКА: при первом запуске рядом создаётся config.json — впиши в него токен бота и id чата.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arden_trader import (find_setups, plan, TP1_R, TP2_R, TP1_FRAC,
                          MIN_STOP_PCT, ENTRY_MAX_BARS, BOS_MAX_BARS)

CONFIG_PATH = os.path.join(HERE, "config.json")
STATE_PATH = os.path.join(HERE, "live_state.json")
LOG_PATH = os.path.join(HERE, "arden_live.log")
DASH_PATH = os.path.join(HERE, "dashboard.html")
TEMPLATE_PATH = os.path.join(HERE, "dashboard_template.html")
MSK = timezone(timedelta(hours=3))

DEFAULT_CONFIG = {
    "_комментарий": "Впиши bot_token и admin_chat_id. Токен берётся у @BotFather, id чата — у @userinfobot.",
    "telegram": {
        "bot_token": "",
        "admin_chat_id": ""
    },
    "trading": {
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "timeframes": ["2h", "4h", "6h", "12h"],
        "ote": 0.79,
        "min_stop_percent": 0.7,
        "risk_percent_per_trade": 0.5,
        "max_open_positions": 10,
        "tp1_R": 1.5,
        "tp2_R": 3.0
    },
    "runtime": {
        "poll_seconds": 120,
        "notify_on_fill": True,
        "autostart": True
    }
}


# ─────────────────────────────── служебное ───────────────────────────────

def log(msg, quiet=False):
    line = f"[{datetime.now(MSK):%Y-%m-%d %H:%M:%S}] {msg}"
    if not quiet:
        print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 66)
        print("  Создан файл настроек:")
        print("  " + CONFIG_PATH)
        print()
        print("  Открой его и впиши два значения:")
        print('    "bot_token"     — токен бота от @BotFather')
        print('    "admin_chat_id" — id твоего чата от @userinfobot')
        print()
        print("  Потом запусти снова.")
        print("=" * 66 + "\n")
        return None
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in DEFAULT_CONFIG.items():          # дозаполняем недостающее
        if isinstance(v, dict):
            cfg.setdefault(k, {})
            for kk, vv in v.items():
                cfg[k].setdefault(kk, vv)
        else:
            cfg.setdefault(k, v)
    return cfg


def apply_targets(cfg):
    """Цели берём из конфига — arden_trader.plan() считает tp1/tp2 по своим константам."""
    import arden_trader as AT
    AT.TP1_R = float(cfg["trading"].get("tp1_R", 1.5))
    AT.TP2_R = float(cfg["trading"].get("tp2_R", 3.0))
    return AT.TP1_R, AT.TP2_R


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {"sent": {}}


def save_state(st):
    cutoff = time.time() - 30 * 86400                 # чистим записи старше 30 дней
    st["sent"] = {k: v for k, v in st["sent"].items() if v > cutoff}
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f)


# ─────────────────────────────── данные ───────────────────────────────

def klines(symbol, interval, limit=400, retries=3):
    """Свечи с Binance. ПОСЛЕДНЯЯ (незакрытая) свеча отбрасывается — причинность."""
    url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
           f"&interval={interval}&limit={limit}")
    last_err = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "arden-live/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = json.load(r)
            break
        except Exception as e:                        # noqa: BLE001 — сеть капризна
            last_err = e
            time.sleep(2 * (a + 1))
    else:
        raise RuntimeError(f"Binance недоступен для {symbol} {interval}: {last_err}")
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume",
                                    "ct", "qv", "n", "tb", "tq", "ig"])
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(
        {"ts": "int64", "open": "float64", "high": "float64",
         "low": "float64", "close": "float64", "volume": "float64"})
    return df.iloc[:-1].reset_index(drop=True)        # без формирующейся свечи


# ─────────────────────────────── поиск сигналов ───────────────────────────────

def scan(df, sym, tf, ote, min_stop):
    """Актуальные планы сделок: ждём вход / вход исполнен. Прошедшие — отбрасываем."""
    h, l = df["high"].values, df["low"].values
    ts = df["ts"].values
    T = len(df)
    out = []
    for st in find_setups(df):
        if st["bos"] < T - 1 - ENTRY_MAX_BARS:        # окно входа истекло
            continue
        p = plan(df, st, ote)
        if p is None or p["risk_pct"] < min_stop:
            continue
        short = p["dr"] == "short"
        seg = slice(p["bos"] + 1, T)
        killed = bool((h[seg] > p["stop"]).any() if short else (l[seg] < p["stop"]).any())
        if killed:
            continue                                  # стоп выбит до входа — сетап мёртв
        filled = bool((h[seg] >= p["entry"]).any() if short else (l[seg] <= p["entry"]).any())
        out.append({
            "sym": sym.replace("USDT", ""), "pair": sym, "tf": tf,
            "dir": p["dr"], "status": "FILLED" if filled else "PENDING",
            "sweep_ts": int(ts[p["S"]]), "bos_ts": int(ts[p["bos"]]),
            "entry": float(p["entry"]), "stop": float(p["stop"]),
            "tp1": float(p["tp1"]), "tp2": float(p["tp2"]),
            "risk_pct": float(p["risk_pct"]), "bos_bars": int(p["bos_bars"]),
            "imp_atr": float(p["imp_atr"]),
            "last": float(df["close"].values[-1]),
            "bars_since_bos": int(T - 1 - p["bos"]),
        })
    return out


def price_fmt(x):
    if x >= 1000:
        return f"{x:,.2f}".replace(",", " ")
    if x >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.6f}"


def _t1():
    import arden_trader as AT
    return AT.TP1_R


def _t2():
    import arden_trader as AT
    return AT.TP2_R


def signal_text(s, risk_pct_acct):
    arrow = "🔻 SHORT" if s["dir"] == "short" else "🔺 LONG"
    head = "НОВЫЙ СЕТАП — выставить лимит" if s["status"] == "PENDING" else "ВХОД ИСПОЛНЕН"
    size = risk_pct_acct / s["risk_pct"] * 100
    d1 = abs(s["tp1"] - s["entry"]) / s["entry"] * 100
    d2 = abs(s["tp2"] - s["entry"]) / s["entry"] * 100
    return (
        f"<b>Стратегия Арденского</b>\n"
        f"{head}\n\n"
        f"<b>{s['sym']}/USDT · {s['tf']} · {arrow}</b>\n\n"
        f"<code>вход: {price_fmt(s['entry'])}</code>\n"
        f"<code>сл:   {price_fmt(s['stop'])}  (−{s['risk_pct']:.2f}%)</code>\n"
        f"<code>тп1:  {price_fmt(s['tp1'])}  (+{d1:.2f}%) — 1.5R, закрыть 33%</code>\n"
        f"<code>тп2:  {price_fmt(s['tp2'])}  (+{d2:.2f}%) — 3.0R, закрыть 67%</code>\n\n"
        f"После тп1 — стоп в безубыток.\n"
        f"Объём: риск {risk_pct_acct}% счёта ÷ {s['risk_pct']:.2f}% = <b>{size:.0f}%</b> депозита.\n\n"
        f"<i>слом за {s['bos_bars']} бар(а), импульс {s['imp_atr']:.1f} ATR · "
        f"цена {price_fmt(s['last'])}</i>"
    )


# ─────────────────────────────── telegram ───────────────────────────────

def tg_send(token, chat_id, text):
    payload = json.dumps({"chat_id": str(chat_id), "text": text,
                          "parse_mode": "HTML",
                          "disable_web_page_preview": True}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r).get("ok", False)
    except urllib.error.HTTPError as e:
        log(f"  Telegram отказал: {e.code} {e.read().decode('utf-8', 'replace')[:200]}")
    except Exception as e:                            # noqa: BLE001
        log(f"  Telegram недоступен: {e}")
    return False


# ─────────────────────────────── автозапуск ───────────────────────────────

def startup_vbs_path():
    return os.path.join(os.environ.get("APPDATA", ""),
                        r"Microsoft\Windows\Start Menu\Programs\Startup", "ArdenLive.vbs")


def install_autostart():
    if os.name != "nt":
        log("Автозапуск настраивается только на Windows — пропущено.")
        return False
    vbs = startup_vbs_path()
    if os.path.exists(vbs):
        return True
    exe = sys.executable.replace("python.exe", "pythonw.exe")
    if not os.path.exists(exe):
        exe = sys.executable
    script = os.path.abspath(__file__)
    body = ('Set S = CreateObject("WScript.Shell")\r\n'
            f'S.CurrentDirectory = "{HERE}"\r\n'
            f'S.Run """{exe}"" ""{script}""", 0, False\r\n')
    try:
        os.makedirs(os.path.dirname(vbs), exist_ok=True)
        with open(vbs, "w", encoding="utf-8") as f:
            f.write(body)
        log(f"Автозапуск установлен: {vbs}")
        log("  Раннер будет стартовать при входе в Windows, окно скрыто.")
        return True
    except OSError as e:
        log(f"Не удалось установить автозапуск: {e}")
        return False


def uninstall_autostart():
    vbs = startup_vbs_path()
    if os.path.exists(vbs):
        os.remove(vbs)
        log(f"Автозапуск удалён: {vbs}")
    else:
        log("Автозапуск и не был установлен.")


# ─────────────────────────────── дашборд ───────────────────────────────

PORTFOLIO = {
    "n": 1095, "years": 8.9, "per_year": 123, "E": 0.491, "PF": 2.01, "WR": 55,
    "sumR": 537, "RR": 1.63, "cagr": 34.5, "dd": -8.4, "cagr1": 79.4, "dd1": -16.4,
    "ci": [0.395, 0.581], "median_stop": 1.19, "max_concurrent": 10,
    "train": {"n": 716, "E": 0.482, "PF": 2.00},
    "test": {"n": 379, "E": 0.507, "PF": 2.04},
    "by_tf": {"2h": {"n": 465, "E": 0.465, "PF": 1.93, "sumR": 216},
              "4h": {"n": 302, "E": 0.555, "PF": 2.22, "sumR": 167},
              "6h": {"n": 219, "E": 0.384, "PF": 1.77, "sumR": 84},
              "12h": {"n": 109, "E": 0.638, "PF": 2.36, "sumR": 70}},
    "by_sym": {"BTC": {"n": 315, "E": 0.544, "PF": 2.19},
               "ETH": {"n": 403, "E": 0.395, "PF": 1.75},
               "SOL": {"n": 377, "E": 0.548, "PF": 2.18}},
    "by_year": {"2017": 0.299, "2018": 0.437, "2019": 0.539, "2020": 0.495,
                "2021": 0.494, "2022": 0.398, "2023": 0.603, "2024": 0.393,
                "2025": 0.362, "2026": 1.077},
}


def write_dashboard(candles, signals, cfg):
    if not os.path.exists(TEMPLATE_PATH):
        return
    data = {
        "updated": datetime.now(MSK).strftime("%Y-%m-%d %H:%M МСК"),
        "portfolio": PORTFOLIO,
        "signals": sorted(signals, key=lambda s: -s["bos_ts"]),
        "candles": candles,
        "risk": cfg["trading"]["risk_percent_per_trade"],
        "tfs": cfg["trading"]["timeframes"],
        "syms": [s.replace("USDT", "") for s in cfg["trading"]["symbols"]],
    }
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(DASH_PATH, "w", encoding="utf-8") as f:
        f.write(html)


# ─────────────────────────────── основной цикл ───────────────────────────────

def cycle(cfg, state, notify=True):
    tr = cfg["trading"]
    tg = cfg["telegram"]
    risk = tr["risk_percent_per_trade"]
    all_sig, candles, sent = [], {}, 0
    for pair in tr["symbols"]:
        sym = pair.replace("USDT", "")
        candles.setdefault(sym, {})
        for tf in tr["timeframes"]:
            try:
                df = klines(pair, tf)
            except RuntimeError as e:
                log(f"  {e}")
                continue
            sigs = scan(df, pair, tf, tr["ote"], tr["min_stop_percent"])
            all_sig += sigs
            tail = df.tail(220)
            candles[sym][tf] = [{"t": int(t // 1000), "o": o, "h": h, "l": l, "c": c}
                                for t, o, h, l, c in zip(tail.ts, tail.open, tail.high,
                                                         tail.low, tail.close)]
            for s in sigs:
                key = f"{s['pair']}|{tf}|{s['sweep_ts']}|{s['dir']}|{s['status']}"
                if key in state["sent"]:
                    continue
                if s["status"] == "FILLED" and not cfg["runtime"]["notify_on_fill"]:
                    continue
                log(f"  СИГНАЛ {s['sym']} {tf} {s['dir'].upper()} {s['status']} "
                    f"вход {price_fmt(s['entry'])} стоп {price_fmt(s['stop'])}")
                if notify and tg["bot_token"] and tg["admin_chat_id"]:
                    if tg_send(tg["bot_token"], tg["admin_chat_id"], signal_text(s, risk)):
                        state["sent"][key] = time.time()
                        sent += 1
                else:
                    state["sent"][key] = time.time()
            time.sleep(0.25)                          # вежливость к API
    write_dashboard(candles, all_sig, cfg)
    save_state(state)
    pend = sum(1 for s in all_sig if s["status"] == "PENDING")
    fill = sum(1 for s in all_sig if s["status"] == "FILLED")
    log(f"  проход завершён: активных планов {len(all_sig)} "
        f"(ждут входа {pend}, в позиции {fill}), отправлено новых {sent}")
    return all_sig


def main():
    ap = argparse.ArgumentParser(description="Живой раннер стратегии Арденского")
    ap.add_argument("--once", action="store_true", help="один проход и выход")
    ap.add_argument("--test-telegram", action="store_true", help="проверить связь с ботом")
    ap.add_argument("--no-autostart", action="store_true", help="не прописывать автозапуск")
    ap.add_argument("--uninstall", action="store_true", help="убрать автозапуск и выйти")
    a = ap.parse_args()

    if a.uninstall:
        uninstall_autostart()
        return

    cfg = load_config()
    if cfg is None:
        return
    tg = cfg["telegram"]
    if not tg["bot_token"] or not tg["admin_chat_id"]:
        print("\n  В config.json пустые bot_token или admin_chat_id.")
        print("  Впиши их и запусти снова. Дашборд пока будет обновляться без Telegram.\n")

    if a.test_telegram:
        if not tg["bot_token"] or not tg["admin_chat_id"]:
            print("  Нечего проверять — заполни config.json."); return
        ok = tg_send(tg["bot_token"], tg["admin_chat_id"],
                     "<b>Стратегия Арденского</b>\nСвязь установлена. "
                     "Сигналы будут приходить сюда.")
        print("  Отправлено ✓" if ok else "  Не отправилось — проверь токен и id чата.")
        return

    log("=" * 60)
    log("Arden Live запущен")
    log(f"  пары: {', '.join(cfg['trading']['symbols'])}")
    log(f"  таймфреймы: {', '.join(cfg['trading']['timeframes'])}")
    log(f"  вход OTE {cfg['trading']['ote']}, мин. стоп {cfg['trading']['min_stop_percent']}%, "
        f"риск {cfg['trading']['risk_percent_per_trade']}% на сделку")
    log(f"  дашборд: {DASH_PATH}")

    if cfg["runtime"]["autostart"] and not a.no_autostart and not a.once:
        install_autostart()

    state = load_state()
    if a.once:
        cycle(cfg, state)
        return

    period = max(30, int(cfg["runtime"]["poll_seconds"]))
    while True:
        try:
            cycle(cfg, state)
        except KeyboardInterrupt:
            log("Остановлено пользователем."); return
        except Exception:                             # noqa: BLE001 — цикл не должен падать
            log("ОШИБКА в проходе:\n" + traceback.format_exc())
        time.sleep(period)


if __name__ == "__main__":
    main()
