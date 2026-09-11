"""
grain_signals.py — 大宗農產品期貨量化策略模組 (升級版 S1 + S2)
升級特色：
1. 雙季節性窗口：收割去庫存窗口 (秋冬) + 生長季天氣溢價窗口 (6-7月)。
2. 期限結構 (Term Structure / Backwardation) 濾網：依據近遠月價差動態調節部位。
3. 非對稱保護：2.0 ATR 初始停損 + 浮盈 2.0 ATR 後啟動 1.2 ATR 移動追蹤停利。
"""

import argparse
import csv
import os
import time
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "futures_data")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

GRAIN_CONFIG = {
    "ZS": {
        "name": "黃豆 (Soybeans)",
        "unit": "美分/蒲式耳",
        "tick_value": 50,
        "seasons": [
            {
                "name": "秋冬收割去庫存與南美博弈",
                "entry_month": 10,
                "entry_day": 15,
                "exit_month": 3,
                "exit_day": 31,
                "thesis": "秋季收割低點消化完畢，定價轉向冬季南美天氣與全球去庫存"
            },
            {
                "name": "夏季生長季天氣溢價",
                "entry_month": 6,
                "entry_day": 1,
                "exit_month": 7,
                "exit_day": 31,
                "thesis": "北美生長關鍵期天氣溢價炒作（授粉/灌漿期乾旱不確定性）"
            }
        ],
        "default_stop_atr": 2.0,
        "trailing_trigger_atr": 2.0,
        "trailing_stop_atr": 1.2
    },
    "ZC": {
        "name": "玉米 (Corn)",
        "unit": "美分/蒲式耳",
        "tick_value": 50,
        "seasons": [
            {
                "name": "收割後消費回升與春播爭地",
                "entry_month": 12,
                "entry_day": 1,
                "exit_month": 3,
                "exit_day": 15,
                "thesis": "收割低點回升，冬季飼料消費高峰與春季爭地預期"
            },
            {
                "name": "夏季授粉期天氣溢價",
                "entry_month": 6,
                "entry_day": 1,
                "exit_month": 7,
                "exit_day": 15,
                "thesis": "夏季授粉關鍵期乾旱天氣溢價爆發"
            }
        ],
        "default_stop_atr": 2.0,
        "trailing_trigger_atr": 2.0,
        "trailing_stop_atr": 1.2
    }
}

def _path(key):
    return os.path.join(DATA, f"{key}.csv")

def fetch_grains():
    """抓 ZS=F / ZC=F 日線 (2000 年至今), 覆寫 futures_data/{key}.csv。"""
    import requests
    os.makedirs(DATA, exist_ok=True)
    p1 = 946684800
    ok = True
    for key, sym in (("ZS", "ZS=F"), ("ZC", "ZC=F")):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"period1": p1, "period2": int(time.time()),
                                     "interval": "1d"}, headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]; q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
                if None in (o, h, l, c):
                    continue
                rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)),
                             round(o, 2), round(h, 2), round(l, 2), round(c, 2),
                             int(q["volume"][i] or 0)))
            if not rows:
                ok = False
                continue
            with open(_path(key), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                w.writerows(rows)
        except Exception:
            ok = False
    return ok

def _load(key):
    o, h, l, c, dt = [], [], [], [], []
    with open(_path(key)) as f:
        for r in csv.DictReader(f):
            try:
                o.append(float(r["open"])); h.append(float(r["high"]))
                l.append(float(r["low"])); c.append(float(r["close"]))
            except ValueError:
                continue
            dt.append(r["date"])
    return dt, np.array(o), np.array(h), np.array(l), np.array(c)

def _atr(h, l, c, n=20):
    a = np.full(len(c), np.nan)
    if len(c) < n:
        return a
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    a[n - 1] = tr[:n].mean()
    for i in range(n, len(c)):
        a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a

def evaluate_term_structure(front_price: float, deferred_price: float) -> Tuple[str, float, float]:
    """評估期限結構與建議部位縮放係數"""
    if front_price <= 0 or deferred_price <= 0:
        return ("無法計算 (缺乏遠月報價)", 0.0, 1.0)
        
    spread_pct = (front_price - deferred_price) / front_price
    
    if spread_pct > 0.015:
        return ("🔥 強烈反向市場 (Backwardation / 現貨極度緊缺)", spread_pct, 1.25)
    elif spread_pct > 0.002:
        return ("🟢 溫和反向市場 (Backwardation / 正展期收益)", spread_pct, 1.0)
    elif spread_pct >= -0.010:
        return ("⚪ 平水/均衡結構 (Balanced)", spread_pct, 0.8)
    elif spread_pct >= -0.025:
        return ("🟡 溫和正向市場 (Contango / 庫存充裕)", spread_pct, 0.5)
    else:
        return ("🔴 深度正向市場 (Super Contango / 庫存嚴重過剩)", spread_pct, 0.0)

def is_in_season_window(symbol: str, target_date: Optional[date] = None) -> Tuple[bool, Optional[Dict]]:
    """判斷指定日期是否落在季節性多頭窗口"""
    if target_date is None:
        target_date = datetime.now().date()
        
    cfg = GRAIN_CONFIG.get(symbol)
    if not cfg:
        return False, None
        
    m, d = target_date.month, target_date.day
    
    for season in cfg["seasons"]:
        em, ed = season["entry_month"], season["entry_day"]
        xm, xd = season["exit_month"], season["exit_day"]
        
        if em > xm:
            if (m > em or (m == em and d >= ed)) or (m < xm or (m == xm and d <= xd)):
                return True, season
        else:
            if (m > em or (m == em and d >= ed)) and (m < xm or (m == xm and d <= xd)):
                return True, season
                
    return False, None

def get_grain_signal_report(
    symbol: str,
    current_price: float,
    front_price: float,
    deferred_price: float,
    atr: float,
    highest_since_entry: Optional[float] = None,
    entry_price: Optional[float] = None,
    current_date: Optional[date] = None,
    stopped_out: bool = False,
    stopped_date: Optional[str] = None,
    stopped_price: Optional[float] = None,
) -> Dict:
    """生成農產品期貨即時信號報告"""
    if current_date is None:
        current_date = datetime.now().date()
        
    cfg = GRAIN_CONFIG[symbol]
    in_season, active_season = is_in_season_window(symbol, current_date)
    ts_desc, spread_pct, weight_factor = evaluate_term_structure(front_price, deferred_price)
    
    if stopped_out:
        stop_price = stopped_price or 0.0
        trailing_status = f"已於 {stopped_date} 觸發出場 (觸發價 {stop_price:.1f})"
        action = f"🛑 季節性多單已觸發停損/停利平倉 (於 {stopped_date} 觸及 {stop_price:.1f})，本季空手觀望"
    elif entry_price and entry_price > 0:
        base_stop = entry_price - cfg["default_stop_atr"] * atr
        high_p = max(highest_since_entry or current_price, current_price)
        profit_r = (high_p - entry_price) / (atr + 1e-6)
        
        if profit_r >= cfg["trailing_trigger_atr"]:
            trailing_stop = high_p - cfg["trailing_stop_atr"] * atr
            stop_price = max(base_stop, trailing_stop)
            trailing_status = f"已啟動 (最高價 {high_p:.1f} 追蹤停利於 {stop_price:.1f})"
        else:
            stop_price = base_stop
            trailing_status = f"初始停損防禦 ({stop_price:.1f})"
            
        if in_season:
            if weight_factor >= 0.8:
                action = "🟢 建議多單續抱 (季節性窗口 + 供需支持)"
            elif weight_factor > 0:
                action = "🟡 建議減碼續抱 (季節性窗口但處於 Contango，防範庫存壓制)"
            else:
                action = "🔴 建議減碼或退場 (處於深度 Contango 累庫期)"
        else:
            action = "⚪ 非季節性窗口 (建議空手觀望)"
    else:
        stop_price = current_price - cfg["default_stop_atr"] * atr
        trailing_status = f"建議參考停損價 ({stop_price:.1f})"
        if in_season:
            if weight_factor >= 0.8:
                action = "🟢 建議多單進場 (季節性窗口 + 供需支持)"
            elif weight_factor > 0:
                action = "🟡 建議小部位多單 (季節性窗口但處於 Contango，需防範庫存壓制)"
            else:
                action = "🔴 建議空手觀望 (處於深度 Contango 累庫期，暫緩季節性做多)"
        else:
            action = "⚪ 非季節性窗口 (建議空手觀望)"
        
    return {
        "symbol": symbol,
        "name": cfg["name"],
        "date": current_date.strftime("%Y-%m-%d"),
        "current_price": current_price,
        "in_season": in_season,
        "stopped_out": stopped_out,
        "active_season_name": active_season["name"] if active_season else "無",
        "active_thesis": active_season["thesis"] if active_season else "無",
        "term_structure_status": ts_desc,
        "spread_pct": round(spread_pct * 100, 2),
        "suggested_weight": f"{int(weight_factor * 100)}%" if not stopped_out else "0%",
        "suggested_action": action,
        "atr": round(atr, 2),
        "stop_price": round(stop_price, 2),
        "trailing_status": trailing_status
    }

def fetch_deferred_price(symbol: str, current_price: float) -> float:
    """嘗試從 Yahoo Finance 抓取次月（遠月）期貨合約價格，用以計算期限結構。"""
    import requests
    base = "ZS" if symbol == "ZS" else "ZC"
    months = ["F","H","K","N","Q","U","X","Z"] if base == "ZS" else ["H","K","N","U","Z"]
    
    year = datetime.now().year % 100
    tickers = []
    for y in range(year, year + 2):
        for m_idx, m in enumerate(months):
            tickers.append((y, m_idx, f"{base}{m}{y}.CBT"))
            
    valid_prices = []
    for y, m_idx, t in tickers:
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", 
                             headers=HEADERS, timeout=2)
            res = r.json()
            if res.get("chart", {}).get("result"):
                price = res["chart"]["result"][0]["meta"]["regularMarketPrice"]
                valid_prices.append((t, price))
        except Exception:
            pass
            
    if not valid_prices:
        return 0.0
        
    front_idx = -1
    for i, (t, p) in enumerate(valid_prices):
        if abs(p - current_price) < 1.0:
            front_idx = i
            break
            
    if front_idx != -1 and front_idx + 1 < len(valid_prices):
        return valid_prices[front_idx + 1][1]
    
    for t, p in valid_prices:
        if abs(p - current_price) > 0.1:
            return p
            
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch:
        fetch_grains()

    print("🌾 穀物期貨個別季節進出場（升級版 S1+S2）")
    for key in ("ZS", "ZC"):
        try:
            dt_strs, o, h, l, c = _load(key)
        except Exception:
            continue
        a = _atr(h, l, c, 20)
        if len(c) == 0:
            continue
            
        current_price = float(c[-1])
        current_atr = float(a[-1])
        date_obj = datetime.strptime(dt_strs[-1], "%Y-%m-%d").date()
        
        # 嘗試從 Yahoo 動態抓取對應的遠月期貨報價
        front_price = current_price
        deferred_price = fetch_deferred_price(key, current_price)
        
        # 模擬本季節多頭窗口的進場與持倉演化
        in_season, active_season = is_in_season_window(key, date_obj)
        entry_price = None
        highest_since_entry = None
        stopped_out = False
        stopped_date = None
        stopped_price = None
        
        if in_season and active_season:
            cfg = GRAIN_CONFIG[key]
            # 往回尋找本季節窗口的起點
            entry_idx = None
            for i in range(len(dt_strs) - 1, -1, -1):
                d_obj = datetime.strptime(dt_strs[i], "%Y-%m-%d").date()
                in_s, _ = is_in_season_window(key, d_obj)
                if not in_s:
                    entry_idx = i + 1
                    break
            
            if entry_idx is not None and entry_idx < len(c):
                entry_price = float(c[entry_idx])
                init_stop = entry_price - cfg["default_stop_atr"] * a[entry_idx]
                curr_stop = init_stop
                high_p = entry_price
                
                for k in range(entry_idx + 1, len(c)):
                    high_p = max(high_p, float(h[k]))
                    profit_r = (high_p - entry_price) / (a[k] + 1e-6)
                    if profit_r >= cfg["trailing_trigger_atr"]:
                        trail_stop = high_p - cfg["trailing_stop_atr"] * a[k]
                        curr_stop = max(curr_stop, trail_stop)
                    if float(l[k]) <= curr_stop:
                        stopped_out = True
                        stopped_date = dt_strs[k]
                        stopped_price = curr_stop
                        break
                highest_since_entry = high_p
        
        report = get_grain_signal_report(
            symbol=key,
            current_price=current_price,
            front_price=front_price,
            deferred_price=deferred_price,
            atr=current_atr,
            highest_since_entry=highest_since_entry,
            entry_price=entry_price,
            current_date=date_obj,
            stopped_out=stopped_out,
            stopped_date=stopped_date,
            stopped_price=stopped_price,
        )
        
        print(f"\n{report['name']}  現價 {report['current_price']:,.1f}｜ATR {report['atr']:.1f}")
        print(f"｜結構狀態: {report['term_structure_status']}")
        if report['in_season']:
            print(f"  🟢 目前為季節性多頭窗口: {report['active_season_name']}")
            print(f"     邏輯: {report['active_thesis']}")
            print(f"  👉 操作建議: {report['suggested_action']} (部位縮放 {report['suggested_weight']})")
            if entry_price:
                print(f"     停損狀態: {report['trailing_status']} (估計進場價 {entry_price:.1f})")
            else:
                print(f"     停損狀態: {report['trailing_status']}")
        else:
            print(f"  {report['suggested_action']}")

if __name__ == "__main__":
    main()
