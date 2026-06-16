# Telegram 策略機器人 — 24 小時雲端部署指南

本機器人 (`telegram_bot.py`) 用純 HTTP 長輪詢 (long-polling)，
**不需要 webhook、不需要對外開 port**，只要能連外即可。
適合掛在任何能跑 Python 的雲端：Railway / Fly.io / Render / 一般 VPS。

> 補充：每日 08:00 的「正式選股推播 + 存檔」已由 GitHub Actions
> (`.github/workflows/daily-scan.yml`) 處理。這隻雲端機器人主要負責
> **互動指令 / 按鈕選單**（隨時掃描、查本年度進出、看策略說明、個股分析）。
> 若不想收到兩次 08:00 推播，見下方「避免重複推播」。

---

## 1. 必要環境變數

| 變數 | 說明 |
|------|------|
| `TELEGRAM_BOT_TOKEN` | BotFather 給的 token (**勿** commit) |
| `TELEGRAM_CHAT_ID`   | 你的 chat id (只有這個 id 能下指令) |
| `MM_DATA_DIR`        | 股價資料夾，雲端用 `/app/data_adj` |
| `MM_AUTO_DOWNLOAD`   | `1` = 每日掃描前自動增量更新資料 (雲端建議開) |
| `MM_SKIP_BOOT_DOWNLOAD` | `1` = 啟動時不下載 (有持久化 volume 時可加速重啟) |

取得 `chat id`：先把機器人加好友、傳一句話，機器人會回你 `你的 chat id: xxxxx`。

---

## 2. 用 Docker 部署 (推薦)

專案已附 `Dockerfile` 與 `docker-entrypoint.sh`：啟動時先增量補資料、再啟動機器人。

```bash
docker build -t moneymaker-bot .
docker run -d --name mmbot --restart unless-stopped \
  -e TELEGRAM_BOT_TOKEN="123456:ABC..." \
  -e TELEGRAM_CHAT_ID="123456789" \
  -v mmdata:/app/data_adj \          # 持久化資料，避免每次重啟重抓
  moneymaker-bot
```

`-v mmdata:/app/data_adj` 把資料存進 Docker volume；下次重啟有 `MM_SKIP_BOOT_DOWNLOAD=1`
可跳過首次大下載 (首次仍需數十分鐘抓 15 年歷史)。

---

## 3. Railway

1. Railway → New Project → Deploy from GitHub repo，選本 repo。
2. Railway 會自動偵測 `Dockerfile` 並 build。
3. Variables 加上 `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`
   (`MM_DATA_DIR` / `MM_AUTO_DOWNLOAD` 已在 Dockerfile 設好)。
4. 在 Settings → Volumes 掛一個 volume 到 `/app/data_adj` (資料持久化)。
5. Deploy。第一次會花時間下載資料，之後常駐。

> Railway 免費額度有用量上限；長期常駐建議用付費方案或 Fly.io。

---

## 4. Fly.io

```bash
fly launch --no-deploy            # 產生 fly.toml (選 Dockerfile)
fly volumes create mmdata --size 1
# 在 fly.toml 的 [mounts] 設定: source="mmdata" destination="/app/data_adj"
fly secrets set TELEGRAM_BOT_TOKEN="123456:ABC..." TELEGRAM_CHAT_ID="123456789"
fly deploy
```

注意：機器人是長輪詢、沒有對外服務，`fly.toml` 不需要 `[http_service]`／健康檢查。
若 Fly 要求至少一個 service，可移除或設 `auto_stop_machines = false` 確保不被休眠。

---

## 5. 一般 VPS (systemd)

```bash
sudo apt install -y python3-pip
pip3 install -r requirements.txt

# /etc/systemd/system/mmbot.service
[Unit]
Description=MoneyMaker Telegram Bot
After=network-online.target

[Service]
WorkingDirectory=/opt/moneymaker
Environment=TELEGRAM_BOT_TOKEN=123456:ABC...
Environment=TELEGRAM_CHAT_ID=123456789
Environment=MM_DATA_DIR=/opt/moneymaker/data_adj
Environment=MM_AUTO_DOWNLOAD=1
ExecStart=/usr/bin/python3 /opt/moneymaker/telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mmbot
journalctl -u mmbot -f          # 看 log
```

資料更新：開了 `MM_AUTO_DOWNLOAD=1` 後機器人會在每天 08:00 掃描前增量補資料；
也可另外用 cron 跑 `python3 download_all.py`。

---

## 6. 啟動後怎麼用

在 Telegram 對機器人輸入 `/menu`，會跳出按鈕選單，點一下直接執行：

- 📋 全市場掃描 — 掃 PA/PB/D/C，列出符合標的 + 進出場點位
- 📈 本年度進出清單 — 回測本年已平倉交易 + 勝率/總R/損益
- 🧠 策略設計說明 — 進出場邏輯說明
- 📊 機器人狀態 — 資料夾 / 股票數

也可直接打指令：`/scan`、`/year C,D`、`/info`、`/analyze 2330`、`/backtest C,D`、`/status`。

---

## 7. 避免重複推播

每日 08:00 有兩個來源：
- **GitHub Actions** (`daily-scan.yml`) — 正式選股報告 + 存 `picks/`
- **雲端機器人** (`scheduler_loop`) — 用 tier/波動濾網的掃描

若只想留一個：
- 想只用 GitHub Actions：把雲端機器人的 `TELEGRAM_CHAT_ID` 留著供互動指令用，
  但可在 `telegram_bot.py` 註解掉 `threading.Thread(target=scheduler_loop...)` 那行。
- 想只用機器人：在 GitHub repo 的 Actions 頁面停用該 workflow。
