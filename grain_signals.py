"""
grain_signals.py — 大宗農產品期貨量化策略模組 (升級版 S1 + S2)
升級特色：
1. 雙季節性窗口：收割去庫存窗口 (秋冬) + 生長季天氣溢價窗口 (6-7月)。
2. 期限結構 (Term Structure / Backwardation) 濾網：依據近遠月價差動態調節部位。
3. 非對稱保護：2.0 ATR 初始停損 + 浮盈 2.0 ATR 後啟動 1.2 ATR 移動追蹤停利。
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

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
    current_date: Optional[date] = None
) -> Dict:
    """生成農產品期貨即時信號報告"""
    if current_date is None:
        current_date = datetime.now().date()
        
    cfg = GRAIN_CONFIG[symbol]
    in_season, active_season = is_in_season_window(symbol, current_date)
    ts_desc, spread_pct, weight_factor = evaluate_term_structure(front_price, deferred_price)
    
    if entry_price and entry_price > 0:
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
    else:
        stop_price = current_price - cfg["default_stop_atr"] * atr
        trailing_status = f"建議參考停損價 ({stop_price:.1f})"
        
    if in_season:
        if weight_factor >= 0.8:
            action = "🟢 建議多單進場 / 續抱 (季節性窗口 + 供需支持)"
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
        "active_season_name": active_season["name"] if active_season else "無",
        "active_thesis": active_season["thesis"] if active_season else "無",
        "term_structure_status": ts_desc,
        "spread_pct": round(spread_pct * 100, 2),
        "suggested_weight": f"{int(weight_factor * 100)}%",
        "suggested_action": action,
        "atr": round(atr, 2),
        "stop_price": round(stop_price, 2),
        "trailing_status": trailing_status
    }
