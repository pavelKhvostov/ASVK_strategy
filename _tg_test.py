"""Диагностика TG для ASVK.
Запускается вручную: `py -3 _tg_test.py` в G:\\ASVK\\.
"""
from pathlib import Path
import sys
import requests

BASE = Path(__file__).resolve().parent
cfg = {}
for line in (BASE / "config.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

token = cfg.get("TG_TOKEN", "")
chat = cfg.get("TG_CHAT", "")

print(f"Python: {sys.version}")
print(f"requests: {requests.__version__}")
print(f"Token: {token[:15]}...{token[-6:] if len(token) > 20 else ''}")
print(f"Chat: {chat}")
print()

# Test 1: getMe
print("[1] getMe...")
try:
    r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
    print(f"    HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"    EXCEPTION: {type(e).__name__}: {e}")

# Test 2: plain text sendMessage
print("\n[2] sendMessage plain...")
try:
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": "debug plain"},
        timeout=10,
    )
    print(f"    HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"    EXCEPTION: {type(e).__name__}: {e}")

# Test 3: with HTML parse_mode
print("\n[3] sendMessage HTML...")
try:
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": "debug <b>HTML</b>", "parse_mode": "HTML"},
        timeout=10,
    )
    print(f"    HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"    EXCEPTION: {type(e).__name__}: {e}")

# Test 4: emoji
print("\n[4] sendMessage emoji...")
try:
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": "🚀 debug emoji"},
        timeout=10,
    )
    print(f"    HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"    EXCEPTION: {type(e).__name__}: {e}")

print("\nDone.")
