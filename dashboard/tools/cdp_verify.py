"""cdp_verify — headless-Chrome скриншот + консольный лог ASVK Chart (диагностика).

Открывает http://127.0.0.1:8080/ в headless Chrome через DevTools Protocol,
делает скриншот всей страницы + скриншот с наведённым courser'ом (crosshair-легенда/
тултип), плюс печатает всё, что попало в консоль браузера (ошибки/исключения).

Требует уже запущенный dashboard-сервер (G:\\ASVK\\open_chart.bat) и bundled python
с пакетом `websockets` (уже стоит — тянется как зависимость uvicorn[standard]).

Usage:
    G:\\ASVK\\python\\python.exe tools\\cdp_verify.py
Output:
    tools\\out\\verify_full.png
    tools\\out\\verify_crosshair.png   (курсор наведён на предпоследнюю свечу)
"""
import asyncio
import base64
import json
import pathlib
import subprocess
import time
import urllib.request

import websockets

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9334
URL = "http://127.0.0.1:8080/"
HOVER_XY = (1618, 480)   # координаты курсора для crosshair-скриншота (правый край графика)

OUT.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen([
    CHROME, "--headless=new", "--disable-gpu",
    f"--remote-debugging-port={PORT}",
    "--window-size=1920,1080",
    f"--user-data-dir={OUT}\\chrome-cdp-profile",
])
time.sleep(2)


async def main():
    ver = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version").read())
    ws_url = ver["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url, max_size=None) as ws:
        _id = 0
        console_msgs = []

        async def send(method, params=None, session_id=None):
            nonlocal _id
            _id += 1
            msg = {"id": _id, "method": method, "params": params or {}}
            if session_id:
                msg["sessionId"] = session_id
            await ws.send(json.dumps(msg))
            while True:
                resp = json.loads(await ws.recv())
                if resp.get("id") == _id:
                    return resp

        created = await send("Target.createTarget", {"url": "about:blank"})
        target_id = created["result"]["targetId"]
        attached = await send("Target.attachToTarget", {"targetId": target_id, "flatten": True})
        session_id = attached["result"]["sessionId"]

        await send("Page.enable", session_id=session_id)
        await send("Runtime.enable", session_id=session_id)
        await send("Network.enable", session_id=session_id)
        await send("Network.setCacheDisabled", {"cacheDisabled": True}, session_id=session_id)

        async def listener():
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("sessionId") != session_id:
                    continue
                m = msg.get("method")
                if m == "Runtime.consoleAPICalled":
                    args = [a.get("value", a.get("description", "")) for a in msg["params"]["args"]]
                    console_msgs.append((msg["params"]["type"], " ".join(str(a) for a in args)))
                elif m == "Runtime.exceptionThrown":
                    console_msgs.append(("exception", msg["params"]["exceptionDetails"].get("text", "")))

        listen_task = asyncio.ensure_future(listener())

        await send("Page.navigate", {"url": URL}, session_id=session_id)
        await asyncio.sleep(4)

        shot = await send("Page.captureScreenshot", {"format": "png"}, session_id=session_id)
        (OUT / "verify_full.png").write_bytes(base64.b64decode(shot["result"]["data"]))

        x, y = HOVER_XY
        await send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y}, session_id=session_id)
        await asyncio.sleep(1)
        shot2 = await send("Page.captureScreenshot", {"format": "png"}, session_id=session_id)
        (OUT / "verify_crosshair.png").write_bytes(base64.b64decode(shot2["result"]["data"]))

        listen_task.cancel()
        print("CONSOLE:")
        for t, txt in console_msgs:
            print(f"  [{t}] {txt}")
        print(f"\nsaved: {OUT / 'verify_full.png'}")
        print(f"saved: {OUT / 'verify_crosshair.png'}")


asyncio.run(main())
proc.terminate()
