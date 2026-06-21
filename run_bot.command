#!/bin/bash
# macOS 一鍵啟動 Telegram 策略機器人。
# 用法: 在 Finder 雙擊本檔, 或在 Terminal 執行  bash run_bot.command
#
# 功能:
#   1. 每次啟動先 git pull 最新策略 (雲端更新 → 本機同步落地)
#   2. 啟動互動機器人 (long-polling, /scan /futures /analyze /menu ...)
#   3. 機器人若意外結束會自動重啟; 按 Ctrl+C 兩次可停止
#
# 首次使用前: 複製 .env.example 為 .env, 填入你的 TOKEN / CHAT_ID / 資料夾。

cd "$(dirname "$0")" || exit 1

# 載入本機密鑰 (.env 不會進 git)
if [ -f .env ]; then
    set -a; . ./.env; set +a
else
    echo "⚠️  找不到 .env, 請先複製 .env.example 為 .env 並填入 TOKEN/CHAT_ID"
fi

echo "================ MoneyMaker 機器人啟動器 ================"
while true; do
    echo "[$(date '+%F %T')] 拉取雲端最新程式碼..."
    git pull --rebase origin main || echo "git pull 失敗, 用現有版本繼續"

    echo "[$(date '+%F %T')] 啟動機器人 (Ctrl+C 結束)"
    python3 telegram_bot.py
    code=$?

    echo "[$(date '+%F %T')] 機器人結束 (code=$code), 5 秒後重啟... (Ctrl+C 取消)"
    sleep 5
done
