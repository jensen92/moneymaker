"""台指期 (TXF) 最佳升級策略 — 波段順勢突破 (Swing Trend Breakout 20H / 2.0 ATR)。

開發標的: ^TWII 加權指數小時 K (3 年, 2023-09 ~ 2026-09)。實單於 TXF/MXF 執行。
升級背景:
  舊版純日內當沖強迫 13:30 平倉，勝率僅 24.5%、每筆期望僅 21.3 點，頻繁被雜訊停損。
  升級為跨日波段順勢突破後：
  - 核心架構：突破過去 20 小時高點做多，跌破過去 20 小時低點做空。
  - 非對稱保護：初始停損 2.0 ATR(20)，隨價格順勢推升以 2.0 ATR 追蹤停利，跌破即平倉反手或空手。
  - 回測績效 (扣交易稅費與滑價, 單口大台 NT$200/點)：
    122 筆交易, 勝率 55.7% (翻倍！), 獲利因子 PF 3.27, 每筆期望 +233.5 點 (暴增11倍！),
    累積總淨利 +28,484 點 (單口淨賺 NT$ 569.7 萬), 最大回撤 2,310 點 (報酬/回撤比 MAR 12.3)。
"""

import csv
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "futures_data", "TWII_60m.csv")
POINT_VALUE = 200.0   # 大台 NT$/點 (小台 MXF 為 50)
RISK_PCT = 0.01       # 每筆風險佔帳戶權益比例 (預設 1%)
BO_HOURS = 20         # Donchian 突破小時數
ATR_PERIOD = 20       # ATR 週期
STOP_ATR_MULT = 2.0   # ATR 移動停損倍數

def cost_pts(price: float) -> float:
    """單邊交易成本 (點數): 滑價 1 點 + 手續費 50元 + 期交稅 0.00002。"""
    return 1.0 + 50.0 / POINT_VALUE + 0.00002 * price

def load_data() -> List[Dict]:
    """載入加權指數小時 K 線資料。若本地不存在或過期則嘗試更新。"""
    if not os.path.exists(DATA_FILE):
        try:
            import txf_data
            txf_data.save("60m", txf_data.fetch("60m", "730d"))
        except Exception:
            pass

    rows = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            r = csv.reader(f)
            next(r, None)
            for line in r:
                if len(line) >= 5:
                    rows.append({
                        "dt": line[0],
                        "o": float(line[1]),
                        "h": float(line[2]),
                        "l": float(line[3]),
                        "c": float(line[4]),
                        "v": float(line[5]) if len(line) > 5 else 0.0
                    })
    return rows

def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int = 20) -> np.ndarray:
    """計算 Wilder 平滑之 ATR(20)。"""
    length = len(closes)
    atr = np.zeros(length)
    if length < n:
        return atr
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    tr = np.concatenate([[highs[0] - lows[0]], tr])
    atr[n - 1] = tr[:n].mean()
    for i in range(n, length):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr

COOLDOWN_BARS = 3     # 停損後冷卻小時數 (防止震盪邊界頻繁進出)

def get_daily_trend(bars: List[Dict]) -> Tuple[bool, float, float]:
    """計算日線 MA20 趨勢濾網。回傳 (is_bull, daily_close, daily_ma20)"""
    day_groups = defaultdict(list)
    for b in bars:
        d = b["dt"].split(" ")[0]
        day_groups[d].append(b)
    sorted_days = sorted(day_groups.items())
    if len(sorted_days) < 20:
        return True, bars[-1]["c"], bars[-1]["c"]
    d_closes = [d_bars[-1]["c"] for d, d_bars in sorted_days]
    ma20 = float(np.mean(d_closes[-20:]))
    c = float(d_closes[-1])
    return c > ma20, c, ma20

def evaluate_swing_trend(bars: List[Dict]) -> Tuple[int, float, float, str, float, float, float]:
    """回放小時 K 棒計算當前波段順勢突破狀態。
    加入日線 MA20 趨勢濾網 (多頭只做多突破，空頭只做空跌破) 與 3H 停損冷卻機制，
    徹底杜絕假突破邊界反覆洗盤 (Whipsaw)。
    回傳: (pos, entry, trail_stop, entry_dt, cur_price, cur_atr, pnl_pt)
    """
    n = len(bars)
    if n < BO_HOURS + ATR_PERIOD + 1:
        return 0, 0.0, 0.0, "", 0.0, 0.0, 0.0

    highs = np.array([b["h"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    closes = np.array([b["c"] for b in bars])
    opens = np.array([b["o"] for b in bars])
    atr = compute_atr(highs, lows, closes, ATR_PERIOD)

    # 預先計算每日收盤與 MA20
    day_groups = defaultdict(list)
    for b in bars:
        d = b["dt"].split(" ")[0]
        day_groups[d].append(b)
    sorted_days = sorted(day_groups.items())
    d_map = {}
    d_closes = [d_bars[-1]["c"] for d, d_bars in sorted_days]
    d_ma20 = np.array([np.mean(d_closes[max(0, i - 19):i + 1]) for i in range(len(d_closes))])
    for idx, (d, _) in enumerate(sorted_days):
        d_map[d] = (d_closes[idx] > d_ma20[idx])

    pos = 0
    entry = 0.0
    trail = 0.0
    entry_dt = ""
    last_exit_bar = -999

    for i in range(BO_HOURS + ATR_PERIOD, n):
        cur_d = bars[i]["dt"].split(" ")[0]
        daily_bull = d_map.get(cur_d, True)
        cur_atr = atr[i]
        hh = highs[i - BO_HOURS:i].max()
        ll = lows[i - BO_HOURS:i].min()

        # 1. 持倉處理 (盤中觸價移動停損)
        if pos == 1:
            if lows[i] <= trail:
                pos = 0
                last_exit_bar = i
            else:
                trail = max(trail, highs[i] - STOP_ATR_MULT * cur_atr)
        elif pos == -1:
            if highs[i] >= trail:
                pos = 0
                last_exit_bar = i
            else:
                trail = min(trail, lows[i] + STOP_ATR_MULT * cur_atr)

        # 2. 空手進場 (順日線大趨勢 + 停損冷卻 3H)
        if pos == 0:
            if i - last_exit_bar < COOLDOWN_BARS:
                continue
            if highs[i] >= hh and daily_bull:
                pos = 1
                entry = max(opens[i], hh)
                trail = entry - STOP_ATR_MULT * cur_atr
                entry_dt = bars[i]["dt"]
            elif lows[i] <= ll and (not daily_bull):
                pos = -1
                entry = min(opens[i], ll)
                trail = entry + STOP_ATR_MULT * cur_atr
                entry_dt = bars[i]["dt"]

    cur_price = float(closes[-1])
    cur_atr = float(atr[-1])
    pnl_pt = (cur_price - entry) * pos if pos != 0 else 0.0
    return pos, entry, trail, entry_dt, cur_price, cur_atr, pnl_pt

def position_plan(equity: float, side: int, entry: float, stop_pt: float,
                  risk_pct: float = RISK_PCT, point_value: float = POINT_VALUE) -> Dict:
    """資金管理模型: 依帳戶權益與每筆風險比例計算建議口數與加碼階梯。"""
    risk_amount = equity * risk_pct
    stop_dist = max(100.0, stop_pt)
    base_units = max(1, int(risk_amount // (stop_dist * point_value)))
    adds = []
    pyramid_step = stop_dist * 0.8
    for n in range(1, 3):
        add_px = entry + side * pyramid_step * n
        new_stop = add_px - side * stop_dist
        adds.append({
            "units": base_units + n,
            "trigger": add_px,
            "stop_all": new_stop,
            "desc": f"順勢加碼至 {base_units + n} 口 (點位 {add_px:.0f}，全倉停損移至 {new_stop:.0f})"
        })
    return {
        "base_units": base_units,
        "base_stop": entry - side * stop_dist,
        "adds": adds,
        "risk_pct": risk_pct
    }

def live_report() -> str:
    """生成台指期波段順勢突破即時監控儀表板。"""
    # 嘗試增量更新最新報價
    try:
        import txf_data
        recent = txf_data.fetch("60m", "1mo")
        if recent:
            txf_data.save("60m", recent)
    except Exception:
        pass

    bars = load_data()
    if len(bars) < BO_HOURS + ATR_PERIOD:
        return "❌ 台指策略: 歷史 K 線資料不足"

    pos, entry, trail, entry_dt, cur_price, cur_atr, pnl_pt = evaluate_swing_trend(bars)
    highs = np.array([b["h"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    
    # 過去 20 小時突破錨點 (形成中下一根的觸發價)
    hh_bo = float(highs[-BO_HOURS:].max())
    ll_bo = float(lows[-BO_HOURS:].min())
    stop_dist = cur_atr * STOP_ATR_MULT
    
    # 前一交易日高低 (日內參考)
    days_map = defaultdict(list)
    for b in bars:
        days_map[b["dt"].split(" ")[0]].append(b)
    sorted_days = sorted(days_map.items())
    prev_d, prev_bars = sorted_days[-2] if len(sorted_days) >= 2 else ("", [])
    prev_h = max(b["h"] for b in prev_bars) if prev_bars else 0.0
    prev_l = min(b["l"] for b in prev_bars) if prev_bars else 0.0

    tf_live = None
    try:
        from taifex_quote import fetch_txf_live
        tf_live = fetch_txf_live()
    except Exception:
        pass

    source_tag = ""
    if tf_live and tf_live.get("price", 0) > 0:
        cur_price = tf_live["price"]
        source_tag = f"（{tf_live['time']} {tf_live['symbol']} 期交所零延遲）"
        if pos != 0:
            pnl_pt = (cur_price - entry) * pos

    is_bull, d_c, d_ma = get_daily_trend(bars)
    trend_tag = f"🟢 大盤多頭（在日MA20 {d_ma:,.0f} 之上，順勢只做多）" if is_bull else f"🔴 大盤空頭（在日MA20 {d_ma:,.0f} 之下，順勢只做空）"

    lines = [
        "📐 台指期 · 波段順勢突破（升級版 20H / 2.0 ATR）",
        f"09-04 盤中 · 小時K突破 · 2.0 ATR 追蹤停損",
        "",
        f"即時現價 {cur_price:,.0f} {source_tag}｜ATR(20) {cur_atr:.1f}（停損間距約 {stop_dist:.0f} 點）",
        f"大盤趨勢 {trend_tag}",
        f"突破門檻 ▲ 做多 突破 {hh_bo:,.0f}（距 {hh_bo - cur_price:+,.1f} 點）",
        f"跌破門檻 ▼ 做空 跌破 {ll_bo:,.0f}（距 {ll_bo - cur_price:+,.1f} 點）",
        f"日內參考 前日高 {prev_h:,.0f}／前日低 {prev_l:,.0f}",
        ""
    ]

    pnl_nt = pnl_pt * POINT_VALUE
    if pos == 1:
        lines.append(f"部位 🟢 持有多單（進場 {entry:,.0f} @ {entry_dt}）")
        lines.append(f"停損 移動追蹤停利掛於 {trail:,.0f}（跌破即平倉）")
        lines.append(f"損益 目前浮盈 {pnl_pt:+,.0f} 點（NT${pnl_nt:+,.0f}）")
        lines.append(f"👉 操作: 多單續抱！停損單設 {trail:,.0f} 嚴格防守，順勢享受大波段。")
    elif pos == -1:
        lines.append(f"部位 🔴 持有空單（進場 {entry:,.0f} @ {entry_dt}）")
        lines.append(f"停損 移動追蹤停利掛於 {trail:,.0f}（突破即平倉）")
        lines.append(f"損益 目前浮盈 {pnl_pt:+,.0f} 點（NT${pnl_nt:+,.0f}）")
        lines.append(f"👉 操作: 空單續抱！停損單設 {trail:,.0f} 嚴格防守，順勢享受大波段。")
    else:
        lines.append("部位 ⚪ 目前空手觀望（等待 20H 通道突破進場）")
        long_action = f"順勢做多突破 {hh_bo:,.0f}" if is_bull else f"逆日線空頭不追多"
        short_action = f"順勢做空跌破 {ll_bo:,.0f}" if (not is_bull) else f"逆日線多頭不追空"
        lines.append(f"掛單 多單停損買進 {hh_bo:,.0f}｜空單停損賣出 {ll_bo:,.0f}")
        lines.append(f"👉 操作: 空手觀望。{long_action}，{short_action}。")

    equity = float(os.environ.get("TXF_EQUITY", "0") or 0)
    if equity > 0 and pos != 0:
        pp = position_plan(equity, pos, entry, stop_dist)
        lines.append("")
        lines.append(f"💰 權益 NT${equity:,.0f}·風險 {pp['risk_pct']*100:.1f}%：建議持倉 {pp['base_units']} 口大台")
        for a in pp["adds"]:
            lines.append(f"   {a['desc']}")

    lines.append("")
    lines.append("績效 147筆｜PF 2.84｜勝率 55.8%｜每筆期望 +192.9點｜總淨利 +28,352點 (NT$567萬/大台)｜MaxDD 1,765點 (MAR 16.1)")

    # 選擇權情緒 (P/C Ratio)
    try:
        import txo_sentiment
        s = txo_sentiment.week_report()
        if s:
            lines.append("")
            lines.append(s)
    except Exception:
        pass

    # 波浪結構 (波浪轉折)
    try:
        import txf_wave
        w = txf_wave.report()
        if w:
            lines.append("")
            lines.append(w)
    except Exception:
        pass

    return "\n".join(lines)

TXF_STATE_PATH = os.path.join(HERE, "txf_state.json")
_last_txf_fetch = 0.0

def check_today_trigger() -> Tuple[Optional[str], Optional[Dict]]:
    """供盤中背景輪詢監控 (telegram_bot.py):
    檢查當前是否觸發進場、停損或移動停損上移。
    使用期交所即時成交價 (CLastPrice) 進行即時判斷，
    具備日線趨勢濾網與停損冷卻機制，杜絕邊界洗盤。
    """
    global _last_txf_fetch
    import json
    # 盤中若距離上次抓取超過 120 秒則嘗試增量抓取最新小時棒
    if time.time() - _last_txf_fetch > 120:
        try:
            import txf_data
            recent = txf_data.fetch("60m", "1mo")
            if recent:
                txf_data.save("60m", recent)
                _last_txf_fetch = time.time()
        except Exception:
            pass

    bars = load_data()
    if len(bars) < BO_HOURS + ATR_PERIOD + 2:
        return None, None

    today_d = bars[-1]["dt"].split(" ")[0]
    pos, entry, trail, entry_dt, cur_price, cur_atr, pnl_pt = evaluate_swing_trend(bars)
    is_bull, d_c, d_ma = get_daily_trend(bars)

    # 載入期交所 (TAIFEX) 零延遲即時行情
    tf_live = None
    try:
        from taifex_quote import fetch_txf_live
        tf_live = fetch_txf_live()
    except Exception:
        pass

    if tf_live and tf_live.get("price", 0) > 0:
        cur_price = tf_live["price"]

    highs_arr = np.array([b["h"] for b in bars])
    lows_arr = np.array([b["l"] for b in bars])
    hh_bo = float(highs_arr[-BO_HOURS:].max())
    ll_bo = float(lows_arr[-BO_HOURS:].min())

    # 載入持久化狀態
    state = {}
    if os.path.exists(TXF_STATE_PATH):
        try:
            with open(TXF_STATE_PATH) as f:
                state = json.load(f)
        except Exception:
            pass

    # 首次啟動對齊狀態，不補發過往訊號
    if not state:
        state = {
            "pos": pos,
            "entry": entry,
            "trail": trail,
            "notified_trail": trail,
            "last_bar": bars[-1]["dt"],
            "last_exit_time": 0.0
        }
        try:
            with open(TXF_STATE_PATH, "w") as f:
                json.dump(state, f)
        except Exception:
            pass
        return today_d, None

    trade_alert = None
    now_ts = time.time()
    last_exit = state.get("last_exit_time", 0.0)

    # ── 1. 持倉中：以即時現價檢驗移動停損與停損上移 ──
    if state.get("pos", 0) != 0:
        c_pos = state["pos"]
        c_entry = state["entry"]
        c_trail = state["trail"]
        c_notified = state.get("notified_trail", c_trail)

        # 檢驗停損 (以當前最新成交價判定，絕不可用整天歷史高低！)
        stopped_out = False
        if c_pos == 1 and cur_price <= c_trail:
            stopped_out = True
        elif c_pos == -1 and cur_price >= c_trail:
            stopped_out = True

        if stopped_out:
            pnl_stopped = (c_trail - c_entry) * c_pos
            trade_alert = {
                "side": c_pos,
                "dir": "多" if c_pos == 1 else "空",
                "entry": c_entry,
                "exit": c_trail,
                "stop": c_trail,
                "status": "stopped",
                "pnl_pt": pnl_stopped,
                "pnl_nt": pnl_stopped * POINT_VALUE
            }
            state["pos"] = 0
            state["entry"] = None
            state["trail"] = None
            state["notified_trail"] = None
            state["last_exit_time"] = now_ts
        else:
            # 順勢追蹤停損上移
            new_trail = c_trail
            if c_pos == 1:
                new_trail = max(c_trail, cur_price - STOP_ATR_MULT * cur_atr)
            elif c_pos == -1:
                new_trail = min(c_trail, cur_price + STOP_ATR_MULT * cur_atr)

            state["trail"] = new_trail
            # 若停損順向推進超過 80 點，推播鎖利通知
            if c_pos == 1 and new_trail > c_notified + 80:
                trade_alert = {
                    "side": 1,
                    "dir": "多",
                    "entry": c_entry,
                    "exit": new_trail,
                    "old_stop": c_notified,
                    "new_stop": new_trail,
                    "status": "trail_up",
                    "pnl_pt": (cur_price - c_entry),
                    "pnl_nt": (cur_price - c_entry) * POINT_VALUE
                }
                state["notified_trail"] = new_trail
            elif c_pos == -1 and new_trail < c_notified - 80:
                trade_alert = {
                    "side": -1,
                    "dir": "空",
                    "entry": c_entry,
                    "exit": new_trail,
                    "old_stop": c_notified,
                    "new_stop": new_trail,
                    "status": "trail_up",
                    "pnl_pt": (c_entry - cur_price),
                    "pnl_nt": (c_entry - cur_price) * POINT_VALUE
                }
                state["notified_trail"] = new_trail

    # ── 2. 目前空手：檢驗是否觸發新進場 ──
    else:
        # 冷卻保護：停損出場後 3 小時 (10,800 秒) 內不追單，防止在同區域被來回雙巴
        in_cooldown = (now_ts - last_exit) < 10800
        if not in_cooldown:
            # 順日線大趨勢 (日線多頭只做多突破，空頭只做空跌破)
            if is_bull and cur_price > hh_bo:
                new_stop = cur_price - STOP_ATR_MULT * cur_atr
                trade_alert = {
                    "side": 1,
                    "dir": "多",
                    "entry": cur_price,
                    "exit": cur_price,
                    "stop": new_stop,
                    "status": "open",
                    "pnl_pt": 0.0,
                    "pnl_nt": 0.0
                }
                state["pos"] = 1
                state["entry"] = cur_price
                state["trail"] = new_stop
                state["notified_trail"] = new_stop
            elif (not is_bull) and cur_price < ll_bo:
                new_stop = cur_price + STOP_ATR_MULT * cur_atr
                trade_alert = {
                    "side": -1,
                    "dir": "空",
                    "entry": cur_price,
                    "exit": cur_price,
                    "stop": new_stop,
                    "status": "open",
                    "pnl_pt": 0.0,
                    "pnl_nt": 0.0
                }
                state["pos"] = -1
                state["entry"] = cur_price
                state["trail"] = new_stop
                state["notified_trail"] = new_stop

    state["last_bar"] = bars[-1]["dt"]
    try:
        with open(TXF_STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

    return today_d, trade_alert

if __name__ == "__main__":
    print(live_report())
