# Telegram 策略機器人 — 24h 雲端常駐 (Railway / Fly.io / 任何 Docker 主機)
FROM python:3.11-slim

# 設為台灣時區, 讓機器人的時間判斷與台股一致
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 啟動機器人 (長輪詢, 不需開 port)。
# 首次啟動會自動下載全市場資料到 data_adj/ (數分鐘)。
CMD ["python3", "telegram_bot.py"]
