ASVK — 1m data collector (Binance BTC/ETH/SOL)
===============================================

ПЕРВЫЙ ЗАПУСК (одноразово):
  1. Двойной клик по _setup.bat  → установит requests/rich/winotify
  2. Двойной клик по ASVK.bat    → запустит daemon

ПОСЛЕ ЭТОГО:
  Клик по ASVK.bat — единственное что нужно.
  Закрыть окно = остановить daemon.
  Ctrl+C в окне = clean stop (шлёт TG "остановлен").

ЧТО ДЕЛАЕТ:
  Каждые 15 мин fetch'ит новые 1m свечи с Binance для BTC/ETH/SOL,
  дописывает в data/{SYMBOL}USDT_1m.csv (без дублей).
  При старте — auto backfill за весь пропущенный интервал
  (сколько бы daemon ни стоял).

TELEGRAM:
  Настраивается в config.txt (TG_TOKEN, TG_CHAT).
  Уведомления: startup, backfill done, heartbeat раз в час, ошибки.
  Если TG API down → сообщения в очередь (logs/tg_pending.json),
  ретрай следующим циклом.
  Если TG down > 30 мин → Windows toast + звук.

ФАЙЛЫ:
  asvk.py           — код daemon
  config.txt        — TG creds
  ASVK.bat          — кнопка запуска
  _setup.bat        — одноразовая установка pip зависимостей
  data\             — CSV с 1m свечами
  logs\asvk.log     — полный log
  logs\tg_pending.json — retry queue для TG (если API падал)

ДИАГНОСТИКА:
  data\ пусто?      → нужно seed'ить: скопировать CSV из warehouse.
  TG "OFF"?         → пустой config.txt.
  TG "DOWN"?        → сеть / bot revoked / wrong chat_id.
  Символ "GAP"?     → > 30 мин без новых данных. Проверь связь.
  Символ "ERR"?     → см. logs\asvk.log последнюю строку.
