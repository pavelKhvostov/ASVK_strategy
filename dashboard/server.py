"""ASVK Chart — портативный FastAPI-сервер для одного красивого BTC-графика.

Полностью внутри G:\\ASVK: bundled python (fastapi/uvicorn поставлены в
G:\\ASVK\\python), lightweight-charts.js — локальная копия (static/), никакого
CDN/интернета в рантайме. Данные — только G:\\ASVK\\data\\BTCUSDT_1m.csv (тот же
файл, что пишет asvk.py daemon).

Live-обновление без перечитывания всего CSV на каждый тик: tail_lines() читает
файл с конца (seek), а не целиком — файл на десятки млн строк, poll каждые 5с.

Запуск:
    G:\\ASVK\\python\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8080
    (или через G:\\ASVK\\open_chart.bat — поднимает сервер и открывает браузер)
"""
from __future__ import annotations
import asyncio
import io
import pathlib
import re
import sys
import time
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
CSV_PATH = DATA_DIR / "BTCUSDT_1m.csv"
MSK = timezone(timedelta(hours=3))
TF_4H_MS = 4 * 60 * 60 * 1000

# Блок 3 (Liq_OB4h_VC + FVG_OB4h_VC) — та же пара algo/folder, что в asvk.py TUI
# (_read_ob_signals/OB4H_ALGOS), чтобы сигнал не "говорил" разное в TUI и в браузере.
OB4H_SOURCES = [("L4h", "liq_ob4h_vc"), ("F4h", "fvg_ob4h_vc")]

sys.path.insert(0, str(BASE / "lib"))  # переиспользуем level-1 индикаторы напрямую
from wma import latest_wma_path                          # noqa: E402
from trendline import latest_trendline_path               # noqa: E402

TF_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000, "1d": 24 * 60 * 60_000}

app = FastAPI(title="ASVK Chart — BTC")
STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════ tail-read (без чтения всего файла) ═══════════════════════
def tail_lines(path: pathlib.Path, n_lines: int) -> list[bytes]:
    """Последние n_lines строк файла — seek с конца, без загрузки всего файла в память."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        block_size = 65536
        blocks: list[bytes] = []
        pos = file_size
        newline_count = 0
        while pos > 0 and newline_count <= n_lines:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            block = f.read(read_size)
            blocks.append(block)
            newline_count += block.count(b"\n")
        data = b"".join(reversed(blocks))
    lines = data.split(b"\n")
    lines = [ln for ln in lines if ln.strip()]
    return lines[-n_lines:] if len(lines) > n_lines else lines


def header_line(path: pathlib.Path) -> bytes:
    with open(path, "rb") as f:
        return f.readline().strip()


def load_1m_tail(n_lines: int) -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()
    hdr = header_line(CSV_PATH)
    body = tail_lines(CSV_PATH, n_lines)
    raw = b"\n".join([hdr, *body])
    df = pd.read_csv(io.BytesIO(raw))
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="ISO8601")
    return df.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)


def agg_tf(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df1m.empty:
        return df1m
    rule = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}.get(tf, "1h")
    df = df1m.set_index("open_time").sort_index()
    agg = df.resample(rule).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return agg.reset_index()


def _to_epoch(ts) -> int:
    return int(pd.Timestamp(ts).timestamp())


# ═══════════════════════ ROUTES ═══════════════════════
@app.get("/", response_class=HTMLResponse)
def index():
    """Отдаёт index.html с cache-busting query (?v=mtime) на static assets — иначе
    Chrome иногда переиспользует закешированный dashboard.js/style.css при обычной
    навигации (не hard-refresh), и правки в коде визуально "не применяются"."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for name in ("dashboard.js", "style.css"):
        f = STATIC_DIR / name
        v = int(f.stat().st_mtime) if f.exists() else 0
        html = html.replace(f"/static/{name}\"", f"/static/{name}?v={v}\"")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


def load_1h_indicators(t_ms: np.ndarray) -> dict[str, np.ndarray]:
    """WMA-50 и HMA-78 (mhull/upper/lower/color) на 1h — level-1 файлы, пересчитываются
    пайплайном каждые 15 мин (т.е. значение для только что закрывшегося часа появится с
    лагом до 15 мин — "обновляется каждый час" с запасом). Reindex по ts (ms) — та же
    сетка, что у свечей.

    upper/lower = max/min(mhull, shull) — canon-лента как в исходном
    ~/traid-bot/research/asvk_trend_line/plot_asvk_trend_line.py (fill_between, не
    жирная линия). color — та же логика (close > shull → up)."""
    wm = pd.read_parquet(latest_wma_path("BTC"), columns=["ts", "wma50"])
    wma_s = pd.Series(wm["wma50"].to_numpy(), index=wm["ts"].to_numpy())

    tl = pd.read_parquet(latest_trendline_path("BTC", variant="1h78"),
                          columns=["ts", "mhull", "upper", "lower", "color"])
    idx = tl["ts"].to_numpy()
    out = {"wma50": wma_s.reindex(t_ms).to_numpy()}
    for col in ("mhull", "upper", "lower", "color"):
        out[col] = pd.Series(tl[col].to_numpy(), index=idx).reindex(t_ms).to_numpy()
    return out


@app.get("/api/candles")
def candles(tf: str = "1h", limit: int = 500):
    n_days = {"15m": 8, "1h": 30, "4h": 90, "1d": 365}.get(tf, 30)
    n_lines = n_days * 1440 + 10
    df1m = load_1m_tail(n_lines)
    df = agg_tf(df1m, tf).tail(limit)

    ind = None
    if tf == "1h" and len(df):
        # open_time может быть datetime64[us] или [ns] в зависимости от версии pandas —
        # явно приводим к [ms] перед astype(int64), иначе деление на 1e6 даёт секунды, не мс.
        t_ms = df["open_time"].dt.tz_localize(None).astype("datetime64[ms]").astype("int64").to_numpy()
        try:
            ind = load_1h_indicators(t_ms)
        except FileNotFoundError:
            pass   # level-1 файлы ещё не посчитаны первым циклом пайплайна

    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        row = {"time": _to_epoch(r["open_time"]), "open": float(r["open"]), "high": float(r["high"]),
               "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"])}
        if ind is not None:
            if not np.isnan(ind["wma50"][i]):
                row["wma50"] = float(ind["wma50"][i])
            if not np.isnan(ind["mhull"][i]):
                row["hma78"] = float(ind["mhull"][i])
                row["hma78_upper"] = float(ind["upper"][i])
                row["hma78_lower"] = float(ind["lower"][i])
                if ind["color"][i]:
                    row["hma78_dir"] = str(ind["color"][i])
        out.append(row)
    return out


@app.get("/api/price")
def price():
    df1m = load_1m_tail(1450)  # чуть больше суток — на 24h change
    if df1m.empty:
        return {"price": None}
    last = df1m.iloc[-1]
    ago = last["open_time"] - pd.Timedelta(hours=24)
    past_rows = df1m[df1m["open_time"] >= ago]
    past = past_rows.iloc[0] if len(past_rows) else df1m.iloc[0]
    pct = (last["close"] - past["open"]) / past["open"] * 100
    return {
        "price": float(last["close"]),
        "change_pct_24h": round(float(pct), 2),
        "ts_ms": int(last["open_time"].timestamp() * 1000),
    }


_FRACTAL_COND_RE = re.compile(r"^b\d+c\d+$")  # то же, что asvk.py:_FRACTAL_COND_RE


@app.get("/api/signals/fractal12h")
def signals_fractal12h(symbol: str = "BTC", limit: int = 200):
    """12h-фрактал basket (B1∪B9) hits — для маркеров на графике. Тот же источник
    (basket_hits_{symbol}_*.parquet) и та же логика статуса, что в asvk.py TUI
    (read_latest_fractal_signals) — сигнал не должен "говорить" разное в TUI и в
    браузере. time отдаём честным UTC (как /api/candles) — сдвиг на MSK клиент
    делает сам, тем же MSK_SHIFT_SEC, что и для свечей."""
    fractal_dir = DATA_DIR / "fractal12h"
    candidates = sorted(fractal_dir.glob(f"basket_hits_{symbol}_*.parquet"),
                         key=lambda p: p.stat().st_mtime)
    if not candidates:
        return []
    try:
        df = pd.read_parquet(candidates[-1])
    except Exception:
        return []
    df = df[df["basket_hit"] == True]
    if df.empty:
        return []
    cond_cols = sorted(c for c in df.columns if _FRACTAL_COND_RE.match(c))
    df = df.sort_values("pivot_open_ts_ms", ascending=False).head(limit)

    out = []
    for _, r in df.iterrows():
        fired = [c.upper() for c in cond_cols if bool(r.get(c, False))]
        if not bool(r["confirmable"]):
            status = "pending"
        elif bool(r["confirmed"]):
            status = "confirmed"
        else:
            status = "failed"
        out.append({
            "time": int(r["pivot_open_ts_ms"]) // 1000,
            "direction": str(r["direction"]),
            "status": status,
            "conditions": fired,
        })
    return out


@app.get("/api/signals/ob4h")
def signals_ob4h(symbol: str = "BTC"):
    """Block 3 (Liq_OB4h_VC + FVG_OB4h_VC) — OB-зона + звёздочка на VC, для маркеров
    на графике. Тот же источник (ob_stage4_race_canonical_{symbol}.parquet) и тот же
    vc_any-фильтр, что в asvk.py TUI (_read_ob_signals) — сигнал не должен "говорить"
    разное в TUI и в браузере.

    OB-зона по canonical ob_zone() (см. lib/Liq_OB4h_VC/ob_stage4_race.py, идентично
    в FVG_OB4h_VC — сверено построчно):
      LONG:  [min(prev.low, cur.low), prev.open]
      SHORT: [prev.open, max(prev.high, cur.high)]
    Прямоугольник рисуется от начала OB-свечи (prev_open = ob_ts − 2×TF_4H) до
    момента первой сработавшей VC (min из vc_rb_ts/vc_fvg_ts/vc_snr_ts > 0) —
    звёздочка ставится в этой точке."""
    out = []
    for algo, folder in OB4H_SOURCES:
        p = DATA_DIR / folder / f"ob_stage4_race_canonical_{symbol}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        df = df[df["vc_any"] == True]
        for _, r in df.iterrows():
            vc_ts_list = [int(r[c]) for c in ("vc_rb_ts", "vc_fvg_ts", "vc_snr_ts") if int(r[c]) > 0]
            if not vc_ts_list:
                continue
            vc_ts = min(vc_ts_list)
            ob_ts = int(r["ob_ts"])
            prev_open = ob_ts - 2 * TF_4H_MS
            direction = str(r["direction"])
            if direction == "long":
                zone_lo = float(min(r["prev_l"], r["cur_l"]))
                zone_hi = float(r["prev_o"])
            else:
                zone_lo = float(r["prev_o"])
                zone_hi = float(max(r["prev_h"], r["cur_h"]))
            out.append({
                "algo": algo,
                "direction": direction,
                "zone_lo": zone_lo,
                "zone_hi": zone_hi,
                "zone_start": prev_open // 1000,
                "vc_time": vc_ts // 1000,
            })
    return out


def _live_candle(tf: str) -> dict | None:
    """Текущий (ещё не закрытый) бар TF — для live-обновления последней свечи графика."""
    tf_ms = TF_MS.get(tf, TF_MS["1h"])
    n_lines = (tf_ms // 60_000) + 10
    df1m = load_1m_tail(n_lines)
    if df1m.empty:
        return None
    ts_ms = df1m["open_time"].dt.tz_localize(None).astype("datetime64[ms]").astype("int64")
    bucket_ms = (ts_ms // tf_ms) * tf_ms
    cur_bucket = bucket_ms.iloc[-1]
    cur = df1m[bucket_ms == cur_bucket]
    if cur.empty:
        return None
    return {
        "time": int(cur_bucket // 1000),
        "open": float(cur["open"].iloc[0]), "high": float(cur["high"].max()),
        "low": float(cur["low"].min()), "close": float(cur["close"].iloc[-1]),
        "volume": float(cur["volume"].sum()),
    }


# ═══════════════════════ WebSocket — live tick ═══════════════════════
class Broadcast:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def register(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def unregister(self, ws: WebSocket):
        self.clients.discard(ws)

    async def push(self, msg: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)


hub = Broadcast()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.register(ws)
    try:
        while True:
            await ws.receive_text()  # ping keepalive
    except WebSocketDisconnect:
        hub.unregister(ws)


async def broadcaster():
    while True:
        try:
            await hub.push({"type": "tick", "price": price(), "candle": _live_candle("1h")})
        except Exception as e:
            print(f"broadcast err: {e}")
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcaster())
