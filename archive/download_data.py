"""下載台灣上市股票日線資料.

資料來源:
- 股票清單/流動性: TWSE OpenAPI STOCK_DAY_ALL
- 歷史日線: Yahoo Finance chart API ({code}.TW)

輸出: data/{code}.csv (date,open,high,low,close,volume)
"""
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UNIVERSE_FILE = os.path.join(DATA_DIR, "universe.json")
TOP_N = 500          # 取成交值前 N 檔, 確保流動性
YEARS = "3y"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def get_universe(top_n=TOP_N):
    """取上市普通股(4 碼數字代號), 依當日成交值排序取前 top_n 檔."""
    r = requests.get(
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = r.json()
    stocks = []
    for row in rows:
        code = row["Code"]
        if len(code) != 4 or not code.isdigit():
            continue  # 排除 ETF/權證/特別股
        try:
            value = float(row["TradeValue"])
        except (ValueError, KeyError):
            continue
        stocks.append({"code": code, "name": row["Name"], "trade_value": value})
    stocks.sort(key=lambda s: -s["trade_value"])
    return stocks[:top_n]


def fetch_daily(code, range_=YEARS):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.TW"
    r = requests.get(url, params={"range": range_, "interval": "1d"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    ts = result.get("timestamp")
    if not ts:
        return None
    q = result["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c, v = (q["open"][i], q["high"][i], q["low"][i],
                         q["close"][i], q["volume"][i])
        if None in (o, h, l, c) or not v:
            continue
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t + 8 * 3600)),
                     round(o, 2), round(h, 2), round(l, 2), round(c, 2), v))
    return rows


def save_csv(code, rows):
    with open(os.path.join(DATA_DIR, f"{code}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        w.writerows(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    universe = get_universe()
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(universe, f, ensure_ascii=False, indent=1)
    print(f"universe: {len(universe)} stocks")

    ok, fail = 0, []

    def task(stock):
        code = stock["code"]
        for attempt in range(3):
            try:
                rows = fetch_daily(code)
                if rows and len(rows) >= 250:
                    save_csv(code, rows)
                    return code, True
                return code, False
            except Exception:
                time.sleep(2 ** attempt)
        return code, False

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(task, s) for s in universe]
        for fut in as_completed(futures):
            code, success = fut.result()
            if success:
                ok += 1
            else:
                fail.append(code)
            if (ok + len(fail)) % 50 == 0:
                print(f"progress: {ok + len(fail)}/{len(universe)}")
    print(f"done. ok={ok}, fail={len(fail)} {fail[:20]}")


if __name__ == "__main__":
    main()
