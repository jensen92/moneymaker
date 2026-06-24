"""每日盤後模擬交易紀錄 (paper trading log): 收盤後執行, 把策略 PA/PB/K/L/D
本年度「已平倉」交易寫入 picks/year_log_<year>.csv, 逐日累加、絕不覆寫已寫入的
舊紀錄 (以 code+strategy+entry_date+exit_date 為唯一鍵去重)。

與 /year (telegram_bot._year_trades_live) 的差異:
  /year 每次呼叫都即時重跑整年回測, 若策略邏輯或股價資料事後修正, 過去顯示的
  結果可能跟著變動; 本檔產生的是「凍結」紀錄, 一旦某筆交易被寫入就不再因為
  之後的程式碼/資料變動而改變, 是真正逐日累積的模擬交易簿。

用法: GitHub Actions 每個交易日 15:05 (盤後) 跑一次 `python sim_log.py`,
新增的已平倉交易會推播 Telegram 通知, 並 commit 回 picks/。

環境變數:
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (可留空, 留空則只更新檔案不推播)
  MM_DATA_DIR  資料目錄 (預設 data_adj/)
"""
import csv
import datetime as dt
import os

import pandas as pd
import requests

import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MM_DATA_DIR", os.path.join(HERE, "data_adj"))
bt.DATA_DIR = DATA_DIR

from backtest import (load_all, load_regime, load_regime_tiers,
                      load_vol_scalars, collect_signals, build_entry_map,
                      run_sub, INIT_CAPITAL)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
SCAN_KEYS = ["PA", "PB", "K", "L", "D"]
LEDGER_FIELDS = ["code", "name", "strategy", "entry_date", "entry",
                 "exit_date", "exit", "pnl", "r"]


def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram] 未設定 token/chat_id, 跳過發送")
        return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=30)
    except requests.RequestException as e:
        print("send_telegram error:", e)


def _ledger_path(year):
    return os.path.join(HERE, "picks", f"year_log_{year}.csv")


def main():
    print("載入全市場資料...")
    data, names = load_all()
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    init_eq = INIT_CAPITAL / len(SCAN_KEYS)

    year = dt.date.today().year
    ystart = pd.Timestamp(year, 1, 1)

    all_tr = []
    for k in SCAN_KEYS:
        print(f"回測策略 {k}...")
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        entry_map = build_entry_map(sigs, data)
        trades, _ = run_sub(data, entry_map, k, 1.0, init_eq,
                            regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        all_tr.extend(trades)

    yt = [t for t in all_tr
          if t["entry_date"] >= ystart or t["exit_date"] >= ystart]

    path = _ledger_path(year)
    existing = []
    seen = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.append(row)
                seen.add((row["code"], row["strategy"],
                          row["entry_date"], row["exit_date"]))

    new_rows = []
    for t in yt:
        key = (t["code"], t["strategy"], t["entry_date"].strftime("%Y-%m-%d"),
               t["exit_date"].strftime("%Y-%m-%d"))
        if key in seen:
            continue
        seen.add(key)
        new_rows.append({
            "code": t["code"], "name": names.get(t["code"], ""),
            "strategy": t["strategy"],
            "entry_date": t["entry_date"].strftime("%Y-%m-%d"),
            "entry": f"{t['entry']:.2f}",
            "exit_date": t["exit_date"].strftime("%Y-%m-%d"),
            "exit": f"{t['exit']:.2f}",
            "pnl": f"{t['pnl']:.2f}", "r": f"{t['r']:.4f}",
        })

    if not new_rows:
        print("無新增已平倉交易")
        return

    rows = existing + new_rows
    rows.sort(key=lambda r: r["entry_date"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"新增 {len(new_rows)} 筆, 累計 {len(rows)} 筆 → {path}")

    lines = [f"📒 {year} 模擬交易紀錄 新增 {len(new_rows)} 筆 (盤後 15:00 結算)"]
    for r in new_rows:
        nm = f" {r['name']}" if r["name"] else ""
        emoji = "🟢" if float(r["pnl"]) > 0 else "🔴"
        lines.append(
            f"{emoji} [{r['strategy']}] {r['code']}{nm}\n"
            f"   進 {r['entry_date'][5:]} {float(r['entry']):.1f} → "
            f"出 {r['exit_date'][5:]} {float(r['exit']):.1f}  "
            f"{float(r['r']):+.1f}R ({float(r['pnl']):+,.0f})")
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
