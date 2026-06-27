#!/bin/sh
# 終端機直接啟動 (非 Docker)。先 `cp .env.example .env` 並填好 token。
set -e
cd "$(dirname "$0")"

# 首次或缺套件時自動安裝
python3 -c "import pandas,numpy,requests" 2>/dev/null || pip3 install -r requirements.txt

echo "啟動策略機器人 (含儀表板 + 黃金/台指/穀物背景監控)..."
echo "停止: Ctrl+C"
exec python3 telegram_bot.py
