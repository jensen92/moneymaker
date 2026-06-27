# Telegram 策略機器人 — 24h 雲端常駐 (Railway / Fly.io / 任何 Docker 主機)
FROM python:3.11-slim

# 設為台灣時區, 讓機器人的時間判斷與台股一致
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 先裝相依 (利用 Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 資料目錄: 建議掛持久化 volume 到 /app/data_adj 以免每次重啟重抓
ENV MM_DATA_DIR=/app/data_adj \
    MM_AUTO_DOWNLOAD=1 \
    MM_WEB_PORT=8800 \
    PYTHONUNBUFFERED=1

# 儀表板由 telegram_bot 以背景執行緒啟動 (BOT_WEB=1 預設開); 對外映此埠即可開圖表
EXPOSE 8800

RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
