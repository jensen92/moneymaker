# Telegram 策略機器人 24h 部署用映像
FROM python:3.11-slim

WORKDIR /app

# 先裝相依 (利用 Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 資料目錄: 建議掛持久化 volume 到 /app/data_adj 以免每次重啟重抓
ENV MM_DATA_DIR=/app/data_adj \
    MM_AUTO_DOWNLOAD=1 \
    PYTHONUNBUFFERED=1

RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
