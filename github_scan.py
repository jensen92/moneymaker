"""GitHub Actions 每日選股腳本: PA + PB + D + C 四策略掃描 + 近觸發觀察名單.

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

from backtest import (load_regime, load_regime_tiers, load_vol_scalars,
                      INIT_CAPITAL, RISK_PCT, PICKS_PER_DAY, compute_rs_rank,
                      suggested_position)
from strategies import STRATEGIES, add_indicators, _d_features

# .strip() 防止 Secret 值前後夾帶空白/tab 導致 Telegram API 拒絕
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
SCAN_KEYS = ["K", "D"]
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


def _build_watchlist(all_data, names, rs, latest_date):
    """找出通過趨勢模板、RS > 0.75，但收縮尚未達標 (0.80–0.85) 的股票."""
    WATCH_RS = 0.75
    WATCH_CONTRACTION = 0.85

    candidates = []
    for code, df in all_data.items():
        i = len(df) - 1
        d = df["date"].iloc[i]
        if latest_date and d < latest_date:
            continue
        try:
            rk = rs.at[d, code]
        except KeyError:
            continue
        if rk is None or (isinstance(rk, float) and np.isnan(rk)):
            continue
        if rk < WATCH_RS:
            continue

        feat = _d_features(df, i, rk)
        if feat is None:
            continue

        # 已有任一策略訊號 → 已在主清單，不重複列入
        already = any(STRATEGIES[k](df, i, rs_rank=rk) for k in SCAN_KEYS)
        if already:
            continue

        ctr = feat["contraction"]
        if ctr > WATCH_CONTRACTION:
            continue

        row = df.iloc[i]
        prior_high20 = df["high"].iloc[i - 20: i].max()
        near_breakout = row["close"] >= prior_high20 * 0.95

        candidates.append({
            "code":        code,
            "name":        names.get(code, ""),
            "rs":          rk,
            "contraction": ctr,
            "threshold":   0.80,
            "vol_surge":   feat["vol_surge"],
            "breakout":    feat["breakout"],
            "near_bo":     near_breakout,
        })

    candidates.sort(key=lambda x: (-int(x["near_bo"]), -x["rs"]))
    return candidates[:10]


def main():
    print(f"=== 每日選股掃描 {datetime.now():%Y-%m-%d %H:%M} ===")

    all_data, names = load_all_data()
    print(f"載入 {len(all_data)} 檔股票")

    rs = compute_rs_rank(all_data)
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()

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
    # 市況分級 + 波動標量 → 與 run_sub 一致的有效風險 (建議股數用)
    tier = regime_tiers.get(latest_date, 2)
    vol_scalar = vol_scalars.get(latest_date, 1.0)

    # ── 組成輸出內容 ──
    lines = []
    lines.append(f"📊 <b>每日選股 {today_str}</b>　"
                 f"大盤{'✅多頭' if regime_ok else '⛔偏弱'}")
    lines.append("")

    label = {"PA": "PA 最嚴", "PB": "PB 平衡", "K": "K 策略", "L": "L 策略",
             "D": "D 原版", "C": "C 寬鬆"}
    stats = {"PA": "PF2.84 勝54%", "PB": "PF2.25 勝47%",
             "K": "PF1.93 勝51%", "L": "PF1.97 勝52%",
             "D": "PF1.46 勝47%", "C": "PF1.58 勝49%"}

    if not regime_ok:
        lines.append("今日大盤偏弱，PA/PB/D 均不進場")
        lines.append("")
    else:
        for key in SCAN_KEYS:
            picks = sorted(candidates[key], reverse=True)[:PICKS_PER_DAY]
            if not picks:
                lines.append(f"<b>▍{label[key]}</b> {stats[key]}　(無訊號)")
                continue
            s0 = picks[0][3]
            stop_pct = s0.get("stop_pct", 0.08)
            gain_cap = s0.get("gain_cap", 9.99)
            gain_str = f"突破上限{gain_cap:.0%}" if gain_cap < 9 else "讓利潤奔跑"
            lines.append(
                f"<b>▍{label[key]}</b> {stats[key]} · 停損-{stop_pct:.0%} · "
                f"持有{s0['max_hold']}日 · {gain_str}")
            for score, code, df, s in picks:
                ref = df.iloc[-1]["close"]
                # 與 run_sub 一致: 有效風險(市況×波動) + minervini 25% 上限
                pos = suggested_position(INIT_CAPITAL, ref, s,
                                         tier=tier, vol_scalar=vol_scalar)
                name = names.get(code, "")
                lines.append(
                    f"  {code} {name}　進{ref:.1f} 損{pos['stop']:.1f}　"
                    f"{pos['shares']:,}股 RS{score:.2f}")
            lines.append("")

    # ── 近觸發觀察名單 ──
    watchlist = _build_watchlist(all_data, names, rs, latest_date)
    lines.append("<b>▍近觸發觀察名單</b> RS&gt;0.75 模板✅ 收縮未達")
    if watchlist:
        for w in watchlist[:8]:
            lines.append(
                f"  {w['code']} {w['name']}　RS{w['rs']:.2f} "
                f"收縮{w['contraction']:.2f} 量{w['vol_surge']:.1f}x "
                f"突破{'✅' if w['breakout'] else '❌'}")
    else:
        lines.append("  (今日無近觸發候選)")
    lines.append("")
    lines.append("⚠️ 量化訊號僅供參考，請自行評估風險")

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
