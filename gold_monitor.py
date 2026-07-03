"""黃金期貨 (GC=F) 順勢突破策略 — 背景即時監控 (供 telegram_bot.py 輪詢呼叫).

每次呼叫 check_live():
  1. 抓「即時」價 (Yahoo 1 分鐘線最後一筆), 不必等小時收盤。
  2. 若小時資料過期 (> STALE_MIN 分鐘) 才重抓 GC_60m.csv (省流量); 資料只含
     『已完成整點棒』(load_bars 已濾掉即時快照與非整點棒)。
  3. 進場分兩種通知:
       🟡 即時突破預警 — 即時價越過『過去 bo 根已完成棒最高價』(形成中下一根的
          突破參考價), 但本小時尚未收盤 → 預警, 供提前準備下單 (每個突破價只報一次)。
       🟢 確認進場 — 出現『新的已完成整點棒』且其收盤符合回測突破條件 → 建立部位
          (與回測一致: 進場價 = 該棒收盤, 停損 = 收盤 - atr_stop×ATR)。
  4. 出場: 部位的 ATR 移動停損在每根新完成棒以收盤更新, 並以即時價即時觸發 (較敏感、
     偏保守), 跌破即出場。
  5. 倉位/去重狀態存 gold_state.json, 跨輪詢/重啟不重複通知。

回傳: (alert_text or None, state)
"""
import json
import os
import time

import numpy as np

import gold_strategies as gs
from gold_signals import fetch_gc_hourly, fetch_live_price

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "gold_state.json")
STALE_MIN = 30   # 小時資料超過此分鐘數才重抓 (節省 Yahoo 請求)

_last_hourly_fetch = 0.0


def _perf_line():
    """動態跑一次回測算績效, 確保通知裡的數字與當前策略/資料同步
    (避免寫死的快照隨資料增長而失真; 與 gold_signals 同一來源)。"""
    try:
        m = gs.metrics(gs.backtest())
        return (f"回測 {m['n']}筆｜PF {m['pf']:.2f}｜勝率 {m['win']:.0%}｜"
                f"DD ${m['dd']/1000:.0f}k｜報酬/回撤比 {m['mar']:.1f}")
    except Exception:
        return "回測: 順勢突破 (詳見 /gold)"


_PERF = _perf_line()


def _load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                s = json.load(f)
                s.setdefault("last_bar", None)
                s.setdefault("alerted_level", None)
                s.setdefault("notified_trail", s.get("trail"))
                s.setdefault("units", 1 if s.get("pos") else 0)
                s.setdefault("risk0", None)
                s.setdefault("entry2", None)
                return s
        except Exception:
            pass
    return {"pos": 0, "entry": None, "trail": None,
            "last_bar": None, "alerted_level": None}


def _save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def check_live():
    global _last_hourly_fetch
    cfg = gs.CONFIG
    if time.time() - _last_hourly_fetch > STALE_MIN * 60:
        if fetch_gc_hourly():
            _last_hourly_fetch = time.time()

    dt, o, h, l, c = gs.load_bars()           # 只含已完成整點棒
    i = len(c) - 1
    a = gs.atr(h, l, c, cfg["atr_n"])
    if i < cfg["breakout"] + cfg["atr_n"] + 1 or np.isnan(a[i]):
        return None, _load_state()
    bo = cfg["breakout"]
    bar_ts = dt[i]
    bar_close = c[i]
    atr_now = a[i]
    up_level = gs.breakout_level(h, i, bo)                 # 形成中下一根多單突破參考價
    dn_level = float(l[i - bo + 1:i + 1].min())            # 空單參考價 (含本根)

    price = fetch_live_price()
    if price is None:
        return None, _load_state()
    state = _load_state()
    alert = None

    # 首次啟動: 對齊到目前最後完成棒, 不補發歷史訊號
    if state["last_bar"] is None and state["pos"] == 0:
        state["last_bar"] = bar_ts
        _save_state(state)
        return None, state

    new_bar = bar_ts != state["last_bar"]

    # 追補中間棒: 兩次輪詢間若一次出現多根已完成棒 (例如停機/斷網後), 先以收盤
    # 逐根處理中間棒 (不含最後一根, 由下方主邏輯處理), 確保不漏進場、移動停損不跳過
    # 中間高水位、且中間棒收盤觸發的停損會被認列。中間棒狀態更新但不另發通知。
    if new_bar and state["last_bar"] in dt:
        li = dt.index(state["last_bar"])
        for k in range(li + 1, i):
            if state["pos"] == 0:
                sk = gs.signal_breakout(h, l, c, k, cfg, a)
                if sk == 1:
                    state = {"pos": 1, "entry": float(c[k]),
                             "trail": float(c[k] - cfg["atr_stop"] * a[k]),
                             "last_bar": dt[k], "alerted_level": None}
                elif cfg["allow_short"] and sk == -1:
                    state = {"pos": -1, "entry": float(c[k]),
                             "trail": float(c[k] + cfg["atr_stop"] * a[k]),
                             "last_bar": dt[k], "alerted_level": None}
            elif state["pos"] == 1:
                state["trail"] = max(state["trail"], float(c[k] - cfg["atr_stop"] * a[k]))
                if c[k] <= state["trail"]:
                    state = {"pos": 0, "entry": None, "trail": None,
                             "last_bar": dt[k], "alerted_level": None}
            elif state["pos"] == -1:
                state["trail"] = min(state["trail"], float(c[k] + cfg["atr_stop"] * a[k]))
                if c[k] >= state["trail"]:
                    state = {"pos": 0, "entry": None, "trail": None,
                             "last_bar": dt[k], "alerted_level": None}
            state["last_bar"] = dt[k]
        new_bar = bar_ts != state["last_bar"]     # 重算 (中間棒處理後)

    if state["pos"] == 0:
        sig = gs.signal_breakout(h, l, c, i, cfg, a) if new_bar else None
        if sig == 1:
            stop = bar_close - cfg["atr_stop"] * atr_now
            state = {"pos": 1, "entry": bar_close, "trail": stop,
                     "last_bar": bar_ts, "alerted_level": None}
            rk = f"風險{bar_close - stop:.0f}點/${(bar_close - stop) * gs.POINT_VALUE:,.0f}口"
            state["notified_trail"] = stop
            state["units"] = 1
            state["risk0"] = bar_close - stop
            state["entry2"] = None
            alert = (
                f"🟢 黃金確認進場（做多） {bar_ts[5:]} 台北\n"
                f"進場 {bar_close:,.1f}｜停損 {stop:,.1f}（{rk}）\n"
                f"👉 操作: 立即市價買進1口, 同時掛停損賣單 {stop:,.1f}\n"
                f"浮盈達+1R(收盤≥{bar_close + (bar_close - stop):,.1f})將發🔺加碼通知;\n"
                f"🔼通知=上移停損單, ⛔通知=平倉\n{_PERF}"
            )
        elif cfg["allow_short"] and sig == -1:
            stop = bar_close + cfg["atr_stop"] * atr_now
            state = {"pos": -1, "entry": bar_close, "trail": stop,
                     "last_bar": bar_ts, "alerted_level": None}
            rk = f"風險{stop - bar_close:.0f}點/${(stop - bar_close) * gs.POINT_VALUE:,.0f}口"
            alert = (
                f"🔴 黃金確認進場（做空） {bar_ts[5:]} 台北\n"
                f"進場 {bar_close:,.1f}｜停損 {stop:,.1f}（{rk}）\n"
                f"停利 無 · ATR移動停損鎖利, 突破即出\n{_PERF}"
            )
        elif price > up_level and state["alerted_level"] != round(up_level, 1):
            # 即時突破預警 (本小時尚未收盤, 待收盤確認), 每個突破價只報一次
            stop = price - cfg["atr_stop"] * atr_now
            state["alerted_level"] = round(up_level, 1)
            state["last_bar"] = bar_ts
            rk = f"風險{price - stop:.0f}點/${(price - stop) * gs.POINT_VALUE:,.0f}口"
            alert = (
                f"🟡 黃金即時突破預警（待本小時收盤確認）\n"
                f"即時 {price:,.1f} 已越門檻 {up_level:,.1f}\n"
                f"進場(預估) {price:,.1f}｜停損(預估) {stop:,.1f}（{rk}）\n"
                f"👉 操作: 先別進場, 備妥委託即可; 收盤站上門檻會發🟢確認通知再買"
            )
        else:
            state["last_bar"] = bar_ts
    elif state["pos"] == 1:
        if new_bar:
            state["trail"] = max(state["trail"], bar_close - cfg["atr_stop"] * atr_now)
            state["last_bar"] = bar_ts
            # 🔺 加碼: 浮盈達 +1R 且僅 1 口 → 加碼 1 口 (雙框架驗證: 總利+40%,
            #    報酬/回撤比不劣化; 只加一次, 不收緊停損, 加碼風險由浮盈支付)
            if (state.get("units", 1) == 1 and state.get("risk0")
                    and bar_close >= state["entry"] + state["risk0"]):
                state["units"] = 2
                state["entry2"] = bar_close
                alert = (
                    f"🔺 黃金加碼（浮盈達+1R） {bar_ts[5:]} 台北\n"
                    f"加碼價 {bar_close:,.1f}（原進場 {state['entry']:,.1f}）\n"
                    f"👉 操作: 市價再買1口(共2口), 停損單改為2口掛 {state['trail']:,.1f}\n"
                    f"之後🔼通知照樣上移(2口一起), ⛔通知2口全平"
                )
                _save_state(state)
                return alert, state
        if price <= state["trail"]:
            pnl = (price - state["entry"]) * gs.POINT_VALUE
            units = state.get("units", 1)
            if units == 2 and state.get("entry2"):
                pnl2 = (price - state["entry2"]) * gs.POINT_VALUE
                pnl_txt = (f"損益 第1口${pnl:+,.0f} + 加碼口${pnl2:+,.0f} "
                           f"= ${pnl + pnl2:+,.0f}（未計滑價手續費）")
                op = "立即市價平倉2口、取消停損單。空手等下一次🟢訊號"
            else:
                pnl_txt = f"損益 ${pnl:+,.0f}/口（未計滑價手續費）"
                op = "立即市價平倉、取消原停損單。空手等下一次🟢訊號"
            alert = (
                f"⛔ 黃金多單停損出場\n"
                f"進場 {state['entry']:,.1f} → 出場 {price:,.1f}（停損 {state['trail']:,.1f}）\n"
                f"{pnl_txt}\n"
                f"👉 操作: {op}"
            )
            state = {"pos": 0, "entry": None, "trail": None,
                     "last_bar": bar_ts, "alerted_level": None}
        elif state["trail"] > state.get("notified_trail", state["trail"]) + 0.3 * atr_now:
            # 移動停損有感上移 (>0.3×ATR) → 通知上移停損單, 否則實單停損永遠停在初始價
            locked = (state["trail"] - state["entry"]) * gs.POINT_VALUE
            alert = (
                f"🔼 黃金停損上移 {state.get('notified_trail', 0):,.1f} → {state['trail']:,.1f}\n"
                f"👉 操作: 把停損賣單改到 {state['trail']:,.1f}"
                f"（已鎖{'利潤' if locked >= 0 else '風險'} ${locked:+,.0f}/口）, 其餘不動"
            )
            state["notified_trail"] = state["trail"]
    elif state["pos"] == -1:
        if new_bar:
            state["trail"] = min(state["trail"], bar_close + cfg["atr_stop"] * atr_now)
            state["last_bar"] = bar_ts
        if price >= state["trail"]:
            pnl = (state["entry"] - price) * gs.POINT_VALUE
            alert = (
                f"⛔ 黃金空單停損出場\n"
                f"進場 {state['entry']:,.1f} → 出場 {price:,.1f}（停損 {state['trail']:,.1f}）\n"
                f"損益 ${pnl:+,.0f}/口（未計滑價手續費）"
            )
            state = {"pos": 0, "entry": None, "trail": None,
                     "last_bar": bar_ts, "alerted_level": None}

    _save_state(state)
    return alert, state


if __name__ == "__main__":
    a, s = check_live()
    print(a or f"無新訊號, 目前狀態: {s}")
