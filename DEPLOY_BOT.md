# Telegram 策略機器人 — 雲端 24h 部署指南

這支機器人 (`telegram_bot.py`) 是**互動式按鈕選單**：在 Telegram 點按鈕即可
執行「全市場掃描 / 本年度進出清單 / 策略設計說明」。它用長輪詢監聽你的點擊，
必須跑在一台**一直開機**的機器上（與每日自動掃描的 GitHub Actions 不同）。

選單功能（點按鈕直接執行）：

| 按鈕 | 功能 |
|------|------|
| 🔍 全市場掃描 | 掃全台股，列出符合 PA/PB/D 的標的 + 進場價/停損/建議股數 |
| 📅 本年度進出清單 | 回測 PA/PB/D，列出今年所有進出場交易與損益 |
| 📖 策略設計方式 | 說明 PA/PB/D 怎麼選股、怎麼設計進出場 |

也可打字指令：`/menu`（叫出選單）、`/analyze 2330`（個股分析）、`/scan`、`/year`、`/doc`。

---

## 需要的環境變數

| 變數 | 說明 |
|------|------|
| `TELEGRAM_BOT_TOKEN` | BotFather 給的 token（與每日掃描同一個即可） |
| `TELEGRAM_CHAT_ID` | 你的 chat id（只有你能用機器人，其他人會被擋） |

> 首次啟動會自動下載全市場資料到 `data_adj/`（約數分鐘），之後常駐記憶體。

---

## 方案 A：Railway（最簡單，推薦）

1. 到 https://railway.app 用 GitHub 登入。
2. **New Project → Deploy from GitHub repo →** 選 `jensen92/moneymaker`。
3. Railway 會自動偵測 `Dockerfile` 並建置。
4. 進專案 **Variables**，新增 `TELEGRAM_BOT_TOKEN` 與 `TELEGRAM_CHAT_ID`。
5. Deploy 完成後，到 Telegram 對機器人送 `/menu`，按鈕就會出現。

Railway 免費額度有限；長期掛機建議升級到 Hobby（約 $5/月）。

## 方案 B：Fly.io（有免費額度）

```bash
# 安裝 flyctl 後
fly launch --no-deploy        # 偵測 Dockerfile, 產生 fly.toml (選最小機型)
fly secrets set TELEGRAM_BOT_TOKEN=<token> TELEGRAM_CHAT_ID=<chat_id>
fly deploy
```

> 機器人不需對外開 port，若 `fly.toml` 有 `[http_service]` 區塊可整段刪除。

## 方案 C：自己的 VPS / Oracle Cloud 免費機

```bash
git clone https://github.com/jensen92/moneymaker.git
cd moneymaker
docker build -t mm-bot .
docker run -d --restart=always \
  -e TELEGRAM_BOT_TOKEN=<token> \
  -e TELEGRAM_CHAT_ID=<chat_id> \
  --name mm-bot mm-bot
```

不用 Docker 也行：

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=<token>
export TELEGRAM_CHAT_ID=<chat_id>
nohup python3 telegram_bot.py &        # 或用 systemd / tmux 常駐
```

---

## 常見問題

- **按鈕沒反應**：確認機器人程序還活著、token/chat_id 正確（值前後不要有空白）。
- **資料太舊**：機器人啟動時下載一次；要更新可重啟容器，或加 `BOT_DAILY_SCAN=1`
  讓它每天 08:00 自己重掃推播（預設關閉，因為 GitHub Actions 已負責每日推播，
  開啟會重複）。
- **掃描/回測很慢**：全市場讀取需數十秒～數分鐘，屬正常；機器人會即時回報進度。
