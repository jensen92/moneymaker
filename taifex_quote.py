"""taifex_quote.py — 台灣期貨交易所 (TAIFEX) 零延遲即時行情模組。

直接連線期交所官方行情網 (mis.taifex.com.tw) 取得近月台指期 (TXF) 即時資訊：
- 成交價 (Last)
- 最高價 (High)
- 最低價 (Low)
- 開盤價 (Open)
- 昨結價 (Ref)
- 報價時間 (HH:MM:SS)
徹底解決 Yahoo Finance 延遲 15 分鐘與盤中斷線問題。
備援支援: 若 TAIFEX 查詢逾時，自動切換臺灣證券交易所 (TWSE MIS) 官方即時現貨報價。
"""

import datetime as dt
import time
from typing import Dict, Optional

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json"
}

TAIFEX_MIS_URL = "https://mis.taifex.com.tw/futures/api/getQuoteList"
TWSE_MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0"


def fetch_txf_live() -> Optional[Dict]:
    """從台灣期貨交易所抓取台指期 (TXF) 最新零延遲報價。

    回傳 dict:
      price: float (最新成交價)
      high: float (今日最高)
      low: float (今日最低)
      open: float (今日開盤)
      ref: float (昨結/參考價)
      time: str (HH:MM:SS)
      source: "TAIFEX" 或 "TWSE"
      symbol: str (例: 臺指期096)
      market: "日盤" 或 "夜盤"
    """
    now = dt.datetime.now()
    # 判斷盤別: 日盤 08:45-13:45 查 0; 其餘時間查夜盤 1
    is_day_session = (now.hour == 8 and now.minute >= 45) or (9 <= now.hour < 13) or (now.hour == 13 and now.minute <= 45)
    market_types = ["0", "1"] if is_day_session else ["1", "0"]

    for m_type in market_types:
        try:
            r = requests.post(
                TAIFEX_MIS_URL,
                json={"MarketType": m_type, "SymbolType": "F"},
                headers=HEADERS,
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                items = data.get("RtData", {}).get("QuoteList", [])
                # 篩選台指期 (TXF) 期貨合約 (排除現貨 -S 指數)
                txf_items = [
                    x for x in items
                    if "TXF" in x.get("SymbolID", "") and not x.get("SymbolID", "").endswith("-S")
                ]
                if txf_items:
                    # 第一檔通常為近月主力合約
                    target = txf_items[0]
                    last_px = float(target.get("CLastPrice") or target.get("CRefPrice") or 0.0)
                    if last_px > 0:
                        raw_time = str(target.get("CTime", "000000")).zfill(6)
                        fmt_time = f"{raw_time[:2]}:{raw_time[2:4]}:{raw_time[4:6]}"
                        return {
                            "price": last_px,
                            "high": float(target.get("CHighPrice") or last_px),
                            "low": float(target.get("CLowPrice") or last_px),
                            "open": float(target.get("COpenPrice") or last_px),
                            "ref": float(target.get("CRefPrice") or last_px),
                            "time": fmt_time,
                            "source": "TAIFEX",
                            "symbol": target.get("DispCName") or target.get("SymbolID"),
                            "market": "日盤" if m_type == "0" else "夜盤"
                        }
        except Exception:
            pass

    # 備援: 若 TAIFEX 暫時無法連線，切換至 TWSE MIS 官方現貨即時報價
    try:
        r2 = requests.get(TWSE_MIS_URL, headers=HEADERS, timeout=5)
        if r2.status_code == 200:
            msg = r2.json().get("msgArray", [])
            if msg:
                m0 = msg[0]
                px = float(m0.get("z") or m0.get("y") or 0.0)
                if px > 0:
                    return {
                        "price": px,
                        "high": float(m0.get("h") or px),
                        "low": float(m0.get("l") or px),
                        "open": float(m0.get("o") or px),
                        "ref": float(m0.get("y") or px),
                        "time": m0.get("t", dt.datetime.now().strftime("%H:%M:%S")),
                        "source": "TWSE",
                        "symbol": "加權指數現貨(備援)",
                        "market": "日盤"
                    }
    except Exception:
        pass

    return None


if __name__ == "__main__":
    t0 = time.time()
    info = fetch_txf_live()
    cost_ms = (time.time() - t0) * 1000
    if info:
        print(f"✅ 取得期交所即時報價 ({cost_ms:.0f}ms):")
        print(f"   標的: {info['symbol']} ({info['market']}) 來源: {info['source']}")
        print(f"   最新成交價: {info['price']:,.2f} (時間: {info['time']})")
        print(f"   今日區間: 最低 {info['low']:,.2f} ~ 最高 {info['high']:,.2f} (開盤 {info['open']:,.2f})")
    else:
        print("❌ 無法取得即時報價")
