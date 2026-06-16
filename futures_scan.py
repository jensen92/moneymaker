"""GitHub Actions 每日期貨訊號: 穀物期貨 (ZS/ZW/ZC) 策略掃描 → Telegram + picks/.

跑最新一根 K 棒, 列出明日開盤可進場的部位 (進場 = 次一交易日開盤)。
與 futures_signals.py 同邏輯, 但格式化後發 Telegram 並存檔。

環境變數:
  TELEGRAM_BOT_TOKEN  Telegram bot token
  TELEGRAM_CHAT_ID    你的 chat id
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from futures_backtest import load_all, RISK_PCT, INIT_CAPITAL
from futures_data import FUTURES
from futures_strategies import STRATEGIES, STRATEGY_NAMES

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
SCAN_KEYS = os.environ.get("FUT_STRATEGIES", "M,D,S").split(",")
PICKS_DIR = Path(__file__).parent / "picks"
PICKS_DIR.mkdir(exist_ok=True)


def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram] 未設定 token/chat_id，跳過發送")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text,
                                     "parse_mode": "HTML"}, timeout=15)
        if not r.ok:
            print(f"[Telegram] API 錯誤 {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"[Telegram] 發送失敗: {e}")


def main():
    print(f"=== 期貨每日訊號 {datetime.now():%Y-%m-%d %H:%M} ===")
    data = load_all()
    if not data:
        print("無期貨資料, 請先跑 futures_data.py")
        return 1

    last_date = max(df["date"].iloc[-1] for df in data.values())
    today_str = last_date.strftime("%Y-%m-%d")

    hits = []
    for key in SCAN_KEYS:
        key = key.strip()
        fn = STRATEGIES.get(key)
        if fn is None:
            continue
        for market, df in data.items():
            i = len(df) - 1
            s = fn(df, i)
            if not s:
                continue
            row = df.iloc[i]
            ref = row["close"]
            atr = row["atr20"]
            stop = ref - s["dir"] * s["stop_atr"] * atr
            stop_dist = abs(ref - stop)
            pv = FUTURES[market]["point_value"]
            contracts = max(1, int(INIT_CAPITAL * RISK_PCT / (stop_dist * pv)))
            risk_usd = contracts * stop_dist * pv
            hits.append((key, market, s["dir"], ref, stop, contracts, risk_usd))

    lines = [f"🌾 <b>期貨每日訊號 {today_str}</b>　策略 {'/'.join(SCAN_KEYS)}", ""]
    if not hits:
        lines.append("今日無新進場訊號")
        lines.append("(持有中部位請依各策略出場規則管理)")
    else:
        for key, m, d, ref, stop, n, risk in hits:
            side = "🟢做多" if d > 0 else "🔴做空"
            lines.append(
                f"<b>{FUTURES[m]['name']}</b> {side} · {STRATEGY_NAMES[key]}\n"
                f"  參考{ref:.2f} 停損{stop:.2f}　{n}口 風險${risk:,.0f}")
        lines.append("")
        lines.append("進場=次一交易日開盤；停損/口數於開盤後依實際價重算")
    lines.append("")
    lines.append("⚠️ 量化訊號僅供參考，請自行評估風險")

    output = "\n".join(lines)
    send_telegram(output)
    print("[Telegram] 已發送")

    plain = output.replace("<b>", "").replace("</b>", "")
    picks_file = PICKS_DIR / f"futures_{today_str}.md"
    picks_file.write_text(plain, encoding="utf-8")
    print(f"[Picks] 已存 {picks_file}")

    for old in sorted(PICKS_DIR.glob("futures_*.md"))[:-30]:
        old.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
