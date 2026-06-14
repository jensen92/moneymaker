"""Telegram 策略機器人: 在 Telegram 直接下指令跑選股 / 回測, 結果即時回傳.

設計: 純 HTTP 長輪詢 (long-polling), 只需 requests, 不用 webhook / 伺服器.
重活在背景 thread 序列化執行, 不阻塞指令接收.

啟動前設定環境變數 (token 切勿寫進程式或 commit):
    export TELEGRAM_BOT_TOKEN="<BotFather 給的 token>"
    export TELEGRAM_CHAT_ID="<你的 chat id>"      # 限定只有你能用; 留空則第一次 /start 會顯示
    export MM_DATA_DIR="/Users/jensenchen/money maker/adjusted_data"   # 資料夾

執行:
    python3 telegram_bot.py

指令:
    /picks            今日 C / D 選股
    /backtest [C,D]   組合回測 (慢, 數分鐘)
    /status           機器人 / 資料狀態
    /help             指令說明
"""
import os
import subprocess
import threading
import time

import requests

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
DATA_DIR = os.environ.get("MM_DATA_DIR", "").strip()
HERE = os.path.dirname(os.path.abspath(__file__))
API = f"https://api.telegram.org/bot{TOKEN}"

# 一次只跑一個重活 (回測/選股很吃 CPU), 用鎖序列化
_job_lock = threading.Lock()


def send(chat_id, text):
    """送訊息, 過長自動分段 (Telegram 上限 4096)."""
    for i in range(0, len(text), 3900):
        chunk = text[i:i + 3900]
        try:
            requests.post(f"{API}/sendMessage",
                          data={"chat_id": chat_id, "text": chunk},
                          timeout=30)
        except requests.RequestException as e:
            print("send error:", e)


def run_script(args):
    """以子行程跑腳本, 回傳合併後的 stdout/stderr."""
    env = dict(os.environ)
    if DATA_DIR:
        env["MM_DATA_DIR"] = DATA_DIR
    try:
        p = subprocess.run(["python3", *args], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=3600)
        out = p.stdout or ""
        if p.returncode != 0:
            out += f"\n[錯誤 returncode={p.returncode}]\n{p.stderr[-1500:]}"
        return out.strip() or "(無輸出)"
    except subprocess.TimeoutExpired:
        return "[逾時] 執行超過 60 分鐘已中止"
    except Exception as e:                       # noqa: BLE001
        return f"[例外] {e}"


def heavy_job(chat_id, label, args):
    """背景執行重活: 取鎖 → 跑 → 回傳."""
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        send(chat_id, f"⏳ {label} 執行中, 請稍候...")
        t0 = time.time()
        out = run_script(args)
        mins = (time.time() - t0) / 60
        send(chat_id, f"✅ {label} 完成 (耗時 {mins:.1f} 分)\n\n{out}")
    finally:
        _job_lock.release()


HELP = (
    "📈 策略機器人指令:\n"
    "/picks  - 今日 C / D 選股\n"
    "/backtest [C,D] - 組合回測 (慢, 數分鐘)\n"
    "/status - 資料 / 機器人狀態\n"
    "/help   - 顯示此說明"
)


def handle(chat_id, text):
    parts = text.strip().split()
    cmd = parts[0].lower().lstrip("/").split("@")[0] if parts else ""
    args = parts[1:]

    if cmd in ("start", "help"):
        send(chat_id, f"你的 chat id: {chat_id}\n\n{HELP}")
    elif cmd == "status":
        dd = DATA_DIR or "(未設定 MM_DATA_DIR)"
        n = (len([f for f in os.listdir(DATA_DIR)
                  if f.endswith(".csv") and not f.startswith("_")])
             if DATA_DIR and os.path.isdir(DATA_DIR) else "?")
        send(chat_id, f"🟢 機器人運作中\n資料夾: {dd}\n股票數: {n}")
    elif cmd == "picks":
        threading.Thread(target=heavy_job,
                         args=(chat_id, "今日選股", ["daily_picks.py"]),
                         daemon=True).start()
    elif cmd == "backtest":
        strats = args[0] if args else "C,D"
        threading.Thread(
            target=heavy_job,
            args=(chat_id, f"回測 {strats}",
                  ["backtest.py", "--strategies", strats,
                   "--data-dir", DATA_DIR or "data", "--leverage", "1"]),
            daemon=True).start()
    else:
        send(chat_id, f"未知指令: {text}\n\n{HELP}")


def main():
    if not TOKEN:
        raise SystemExit("請先設定環境變數 TELEGRAM_BOT_TOKEN")
    print("策略機器人啟動, 等待指令... (Ctrl+C 結束)")
    # 清空積壓的舊訊息, 避免啟動後重播歷史指令
    try:
        r0 = requests.get(f"{API}/getUpdates", params={"offset": -1}, timeout=10)
        updates0 = r0.json().get("result", [])
        offset = updates0[-1]["update_id"] + 1 if updates0 else None
    except Exception:  # noqa: BLE001
        offset = None
    while True:
        try:
            r = requests.get(f"{API}/getUpdates",
                             params={"timeout": 30, "offset": offset},
                             timeout=40)
            updates = r.json().get("result", [])
        except (requests.RequestException, ValueError) as e:
            print("poll error:", e)
            time.sleep(3)
            continue

        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("edited_message")
            if not msg or "text" not in msg:
                continue
            chat_id = str(msg["chat"]["id"])
            if ALLOWED_CHAT and chat_id != ALLOWED_CHAT:
                send(chat_id, "⛔ 未授權的使用者")
                print(f"拒絕未授權 chat id: {chat_id}")
                continue
            print(f"指令 from {chat_id}: {msg['text']}")
            handle(chat_id, msg["text"])


if __name__ == "__main__":
    main()
