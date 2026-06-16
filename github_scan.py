"""GitHub Actions 每日選股腳本: PA + PB + D 三策略掃描, 結果發 Telegram + 存 picks/.

環境變數:
  TELEGRAM_BOT_TOKEN  Telegram bot token
  TELEGRAM_CHAT_ID    你的 chat id
  MM_DATA_DIR         資料目錄 (預設 data_adj/)
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import backtest as bt

DATA_DIR = os.environ.get("MM_DATA_DIR",
                          os.path.join(os.path.dirname(__file__), "data_adj"))
bt.DATA_DIR = DATA_DIR

from backtest import load_regime, INIT_CAPITAL, RISK_PCT, PICKS_PER_DAY, compute_rs_rank
from strategies import STRATEGIES, add_indicators

# .strip() 防止 Secret 值前後夾帶空白/tab 導致 Telegram API 拒絕
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
SCAN_KEYS = ["PA", "PB", "D"]
PICKS_DIR = Path(__file__).parent / "picks"
PICKS_DIR.mkdir(exist_ok=True)


def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram] 未設定 token/chat_id，跳過發送")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chunk in _split(text, 4000):
        try:
            r = requests.post(url, json={"chat_id": CHAT_ID, "text": chunk,
                                         "parse_mode": "HTML"}, timeout=15)
            if not r.ok:
                # 印出 Telegram 回傳的錯誤 (不含敏感資訊) 方便診斷
                print(f"[Telegram] API 錯誤 {r.status_code}: {r.text[:300]}")
        except Exception as e:
            print(f"[Telegram] 發送失敗: {e}")
        time.sleep(0.3)


def _split(text, limit):
    """把長文切成 < limit 字的段落。"""
    lines, chunk, chunks = text.split("\n"), [], []
    for ln in lines:
        if sum(len(l)+1 for l in chunk) + len(ln) > limit:
            chunks.append("\n".join(chunk))
            chunk = []
        chunk.append(ln)
    if chunk:
        chunks.append("\n".join(chunk))
    return chunks or [text]


def load_all_data():
    all_data = {}
    uni_path = os.path.join(DATA_DIR, "_universe_all.json")
    names = {}
    if os.path.exists(uni_path):
        with open(uni_path) as f:
            raw = json.load(f)
        names = {s["code"]: s.get("name", "") for s in raw} if isinstance(raw, list) else {}

    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, fn), parse_dates=["date"])
            df = add_indicators(df).reset_index(drop=True)
            if len(df) >= 210:
                all_data[code] = df
        except Exception:
            pass
    return all_data, names


def main():
    print(f"=== 每日選股掃描 {datetime.now():%Y-%m-%d %H:%M} ===")

    all_data, names = load_all_data()
    print(f"載入 {len(all_data)} 檔股票")

    rs = compute_rs_rank(all_data)
    risk_on = load_regime()

    candidates = {k: [] for k in SCAN_KEYS}
    latest_date = None

    for code, df in all_data.items():
        i = len(df) - 1
        d = df["date"].iloc[i]
        if latest_date is None or d > latest_date:
            latest_date = d
        try:
            rank = rs.at[d, code]
        except KeyError:
            rank = None
        if rank is None or (isinstance(rank, float) and np.isnan(rank)):
            continue
        for key in SCAN_KEYS:
            s = STRATEGIES[key](df, i, rs_rank=rank)
            if s:
                candidates[key].append((s["score"], code, df, s))

    today_str = latest_date.strftime("%Y-%m-%d") if latest_date else datetime.now().strftime("%Y-%m-%d")
    regime_ok = latest_date in risk_on

    # ── 組成輸出內容 ──
    lines = []
    lines.append(f"📊 <b>每日選股報告 {today_str}</b>")
    lines.append(f"大盤狀態: {'✅ 多頭 (可進場)' if regime_ok else '⛔ 偏弱 (今日不進場)'}")
    lines.append("")

    label = {"PA": "PA 最嚴 (PF3.4/勝率57%)",
             "PB": "PB 平衡 (PF2.5/勝率51%)",
             "D":  "D  原版 (PF2.0/勝率53%)"}

    for key in SCAN_KEYS:
        lines.append(f"<b>【策略 {label[key]}】</b>")
        if not regime_ok:
            lines.append("  今日大盤不符合進場條件")
            lines.append("")
            continue
        picks = sorted(candidates[key], reverse=True)[:PICKS_PER_DAY]
        if not picks:
            lines.append("  (今日無訊號)")
            lines.append("")
            continue
        for score, code, df, s in picks:
            row = df.iloc[-1]
            ref = row["close"]
            stop_pct = s.get("stop_pct", 0.08)
            stop = ref * (1 - stop_pct)
            risk_amt = INIT_CAPITAL * RISK_PCT
            shares = int(risk_amt / max(ref - stop, 0.01) / 1000) * 1000
            name = names.get(code, "")
            gain_cap = s.get("gain_cap", 9.99)
            gain_str = f"突破漲幅上限 {gain_cap:.0%}" if gain_cap < 9 else "讓利潤奔跑"
            lines.append(
                f"  <b>{code} {name}</b>\n"
                f"    參考進場: {ref:.2f}  停損: {stop:.2f} (-{stop_pct:.0%})\n"
                f"    建議股數: {shares:,} 股  RS: {score:.2f}\n"
                f"    最長持有: {s['max_hold']} 日  出場: {gain_str}"
            )
        lines.append("")

    lines.append("─" * 30)
    lines.append("⚠️ 以上為量化訊號, 僅供參考, 請自行評估風險。")

    output = "\n".join(lines)

    # ── 發 Telegram ──
    send_telegram(output)
    print("[Telegram] 已發送")

    # ── 存 picks/ 檔案 ──
    plain = (output.replace("<b>", "").replace("</b>", "")
                   .replace("<i>", "").replace("</i>", ""))
    picks_file = PICKS_DIR / f"picks_{today_str}.md"
    picks_file.write_text(plain, encoding="utf-8")
    print(f"[Picks] 已存 {picks_file}")

    # 保留最近 30 天的 picks 檔案
    all_picks = sorted(PICKS_DIR.glob("picks_*.md"))
    for old in all_picks[:-30]:
        old.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
