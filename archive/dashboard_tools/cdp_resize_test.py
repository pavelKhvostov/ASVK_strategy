"""cdp_resize_test — репродукция resize/fullscreen поведения ASVK Chart (диагностика).

Открывает страницу в маленьком headless-окне (700x500), ждёт полной загрузки
(boot() + rolling window), скриншотит, затем реально ресайзит окно браузера через
Browser.setWindowBounds (не просто меняет viewport — настоящий OS-level resize,
как при разворачивании окна в fullscreen/maximize) и скриншотит снова.

Использовался для диагностики двух багов (оба исправлены в dashboard.js):
  1. rightOffset не давал отступа справа при явном setVisibleRange (нужен
     setVisibleLogicalRange).
  2. followingLive rolling-window не переприменялся на resize (неделя "плыла" в
     20 дней при разворачивании окна).

Requires: запущенный dashboard-сервер (G:\\ASVK\\open_chart.bat).
Usage:
    G:\\ASVK\\python\\python.exe tools\\cdp_resize_test.py
Output:
    tools\\out\\step1_small.png    (700x500, сразу после загрузки)
    tools\\out\\step2_after_resize.png   (после resize в 1920x1080)
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
PORT = 9333
URL = "http://127.0.0.1:8080/"

OUT.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen([
    CHROME, "--headless=new", "--disable-gpu",
    f"--remote-debugging-port={PORT}",
    "--window-size=700,500",
    f"--user-data-dir={OUT}\\chrome-cdp-profile-resize",
])
time.sleep(2)


async def main():
    ver = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version").read())
    ws_url = ver["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url, max_size=None) as ws:
        _id = 0

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

        created = await send("Target.createTarget", {"url": URL})
        target_id = created["result"]["targetId"]
        attached = await send("Target.attachToTarget", {"targetId": target_id, "flatten": True})
        session_id = attached["result"]["sessionId"]

        await send("Page.enable", session_id=session_id)
        await asyncio.sleep(3)  # маленькое окно, ждём boot() + rolling window

        shot = await send("Page.captureScreenshot", {"format": "png"}, session_id=session_id)
        (OUT / "step1_small.png").write_bytes(base64.b64decode(shot["result"]["data"]))

        win_id_resp = await send("Browser.getWindowForTarget", {"targetId": target_id})
        window_id = win_id_resp["result"]["windowId"]
        await send("Browser.setWindowBounds", {
            "windowId": window_id,
            "bounds": {"width": 1920, "height": 1080},
        })
        await asyncio.sleep(2)

        shot2 = await send("Page.captureScreenshot", {"format": "png"}, session_id=session_id)
        (OUT / "step2_after_resize.png").write_bytes(base64.b64decode(shot2["result"]["data"]))
        print(f"saved: {OUT / 'step1_small.png'}")
        print(f"saved: {OUT / 'step2_after_resize.png'}")


asyncio.run(main())
proc.terminate()
