"""下載穀物期貨連續合約日線 (黃豆 / 小麥 / 玉米).

資料來源: Yahoo Finance chart API (連續近月合約, 後復權).
  ZS=F 黃豆 Soybean   (CBOT, 5000 bushels, 報價美分/bu, 1 點 = US$50)
  ZW=F 小麥 Wheat     (CBOT SRW,  同上)
  ZC=F 玉米 Corn      (CBOT,      同上)

三者合約規格一致: 5000 蒲式耳, 最小跳動 1/4 美分 = US$12.5, 1 美分(1 點) = US$50.

輸出: futures_data/{key}.csv (date,open,high,low,close,volume)
"""
import csv
import os
import time

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "futures_data")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# 合約規格 (回測引擎共用). point_value = 報價每變動 1.0 的每口損益 (美元).
FUTURES = {
    "ZS": {"yahoo": "ZS=F", "name": "黃豆 Soybean", "point_value": 50.0, "tick": 0.25},
    "ZW": {"yahoo": "ZW=F", "name": "小麥 Wheat",   "point_value": 50.0, "tick": 0.25},
    "ZC": {"yahoo": "ZC=F", "name": "玉米 Corn",    "point_value": 50.0, "tick": 0.25},
}


def fetch_daily(yahoo_symbol):
    """抓取全期日線 (period1=0 → 上市至今)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    r = requests.get(url, params={"period1": 0, "period2": int(time.time()),
                                  "interval": "1d"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    ts = result.get("timestamp")
    if not ts:
        return None
    q = result["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        v = q["volume"][i]
        if None in (o, h, l, c):
            continue
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)),
                     round(o, 2), round(h, 2), round(l, 2), round(c, 2),
                     int(v) if v else 0))
    return rows


def save_csv(key, rows):
    with open(os.path.join(DATA_DIR, f"{key}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        w.writerows(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, spec in FUTURES.items():
        for attempt in range(3):
            try:
                rows = fetch_daily(spec["yahoo"])
                if rows:
                    save_csv(key, rows)
                    print(f"{key} {spec['name']:<14} {len(rows):>5} bars "
                          f"{rows[0][0]} ~ {rows[-1][0]}")
                    break
                print(f"{key}: 無資料")
            except Exception as e:
                if attempt == 2:
                    print(f"{key}: 失敗 {e}")
                time.sleep(2 ** attempt)


if __name__ == "__main__":
    main()
