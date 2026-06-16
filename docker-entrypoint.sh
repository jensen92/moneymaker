#!/bin/sh
# 啟動腳本: 先補齊股價資料 (增量, 已有檔案自動跳過), 再啟動 Telegram 機器人。
set -e

if [ "${MM_SKIP_BOOT_DOWNLOAD}" = "1" ]; then
    echo "MM_SKIP_BOOT_DOWNLOAD=1 → 跳過啟動時資料下載"
else
    echo "啟動前更新股價資料 (增量, 首次需數十分鐘)..."
    python download_all.py || echo "資料更新失敗, 以現有資料啟動"
fi

echo "啟動 Telegram 策略機器人..."
exec python telegram_bot.py
