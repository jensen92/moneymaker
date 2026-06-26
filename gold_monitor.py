"""黃金期貨 (GC=F) 順勢突破策略 — 背景即時監控 (供 telegram_bot.py 輪詢呼叫).

每次呼叫 check_live():
  1. 抓「即時」價格 (Yahoo 1 分鐘線最後一筆), 不必等小時收盤。
  2. 若小時資料已過期 (> STALE_MIN 分鐘) 才重抓 GC_60m.csv (省流量)。
  3. 用「已收盤」小時資料算出目前突破價 (過去 breakout 小時高點) 與
     (持倉中時) ATR 移動停損價。
  4. 與即時價比較, 狀態轉換 (無倉→進場 / 持倉→停損出場) 才回傳通知,
     並把倉位狀態存進 gold_state.json (跨輪詢/重啟不重複通知)。

回傳: (alert_text or None, state)
"""
import json
import os
import time

import numpy as np

import gold_strategies as gs
from gold_signals import fetch_gc_hourly

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "gold_state.json")
STALE_MIN = 30   # 小時資料超過此分鐘數才重抓 (節省 Yahoo 請求)

_last_hourly_fetch = 0.0


def _load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"pos": 0, "entry": None, "trail": None}


def _save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def fetch_live_price():
    """即時參考價: Yahoo 1 分鐘線最後一筆收盤 (近乎即時, 比小時線快)。"""
    import requests
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F",
                     params={"range": "1d", "interval": "1m"},
                     headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                     timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    for v in reversed(closes):
        if v is not None:
            return float(v)
    raise ValueError("無有效即時價")


def check_live():
    global _last_hourly_fetch
    cfg = gs.CONFIG
    if time.time() - _last_hourly_fetch > STALE_MIN * 60:
        if fetch_gc_hourly():
            _last_hourly_fetch = time.time()

    dt, o, h, l, c = gs.load_bars()
    i = len(c) - 1
    a = gs.atr(h, l, c, cfg["atr_n"])
    if np.isnan(a[i]):
        return None, _load_state()
    bo = cfg["breakout"]
    hh = h[i - bo:i].max()
    ll = l[i - bo:i].min()
    atr_now = a[i]

    price = fetch_live_price()
    state = _load_state()
    alert = None

    if state["pos"] == 0:
        if price > hh:
            stop = price - cfg["atr_stop"] * atr_now
            state = {"pos": 1, "entry": price, "trail": stop}
            alert = (
                f"🟢 黃金突破進場訊號 (做多)\n"
                f"進場參考價 {price:,.2f}  (突破過去{bo}H高點 {hh:,.2f})\n"
                f"初始停損 {stop:,.2f}  (風險 {price - stop:,.2f} 點 ≈ "
                f"${(price - stop) * gs.POINT_VALUE:,.0f}/口)\n"
                f"出場規則: 價格回落跌破移動停損即出場 (無固定停利, 隨ATR追蹤)\n"
                f"策略回測: 勝率48.3% PF1.83 最大DD$29,700 MAR6.07 (147筆/2024-2026)"
            )
        elif cfg["allow_short"] and price < ll:
            stop = price + cfg["atr_stop"] * atr_now
            state = {"pos": -1, "entry": price, "trail": stop}
            alert = (
                f"🔴 黃金突破進場訊號 (做空)\n"
                f"進場參考價 {price:,.2f}  (跌破過去{bo}H低點 {ll:,.2f})\n"
                f"初始停損 {stop:,.2f}"
            )
    elif state["pos"] == 1:
        new_trail = max(state["trail"], price - cfg["atr_stop"] * atr_now)
        if price <= new_trail:
            pnl = (price - state["entry"]) * gs.POINT_VALUE
            alert = (
                f"⛔ 黃金多單觸發移動停損, 出場\n"
                f"進場 {state['entry']:,.2f} → 出場 {price:,.2f}  "
                f"損益 ${pnl:+,.0f}/口 (未計滑價手續費)"
            )
            state = {"pos": 0, "entry": None, "trail": None}
        else:
            state["trail"] = new_trail
    elif state["pos"] == -1:
        new_trail = min(state["trail"], price + cfg["atr_stop"] * atr_now)
        if price >= new_trail:
            pnl = (state["entry"] - price) * gs.POINT_VALUE
            alert = (
                f"⛔ 黃金空單觸發移動停損, 出場\n"
                f"進場 {state['entry']:,.2f} → 出場 {price:,.2f}  "
                f"損益 ${pnl:+,.0f}/口 (未計滑價手續費)"
            )
            state = {"pos": 0, "entry": None, "trail": None}
        else:
            state["trail"] = new_trail

    _save_state(state)
    return alert, state


if __name__ == "__main__":
    a, s = check_live()
    print(a or f"無新訊號, 目前狀態: {s}")
