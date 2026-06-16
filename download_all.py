"""下載全台股 (上市 + 上櫃) 除權息調整日線至 data_adj/.

- 清單: TWSE OpenAPI (上市) + TPEx OpenAPI (上櫃)
- 已存在於 data_adj/ 的檔案跳過
- 價格: Yahoo Finance chart API, 以 adjclose/close 比例調整 OHLC (除權息調整)
- 上市試 {code}.TW, 失敗再試 {code}.TWO
"""
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DATA_ADJ = os.path.join(os.path.dirname(__file__), "data_adj")
UNIVERSE_FILE = os.path.join(DATA_ADJ, "_universe_all.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
RANGE = "15y"


def get_json(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2 ** attempt)
    return None


def get_universe():
    stocks = {}
    rows = get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
    for row in rows or []:
        code = row["Code"]
        if len(code) == 4 and code.isdigit():
            stocks[code] = {"code": code, "name": row["Name"], "market": "TWSE"}
    rows = get_json("https://www.tpex.org.tw/openapi/v1/"
                    "tpex_mainboard_daily_close_quotes")
    for row in rows or []:
        code = row.get("SecuritiesCompanyCode", "")
        if len(code) == 4 and code.isdigit() and code not in stocks:
            stocks[code] = {"code": code, "name": row.get("CompanyName", ""),
                            "market": "TPEx"}
    return stocks


def fetch_adjusted(code, suffix):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}{suffix}"
    r = requests.get(url, params={"range": RANGE, "interval": "1d",
                                  "events": "div,split"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    ts = result.get("timestamp")
    if not ts:
        return None
    q = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c, v = (q["open"][i], q["high"][i], q["low"][i],
                         q["close"][i], q["volume"][i])
        if None in (o, h, l, c) or not v:
            continue
        ratio = (adj[i] / c) if adj and adj[i] is not None and c else 1.0
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t + 8 * 3600)),
                     round(o * ratio, 4), round(h * ratio, 4),
                     round(l * ratio, 4), round(c * ratio, 4), int(v)))
    return rows


def task(stock):
    code, market = stock["code"], stock["market"]
    suffixes = [".TW", ".TWO"] if market == "TWSE" else [".TWO", ".TW"]
    for suffix in suffixes:
        for attempt in range(3):
            try:
                rows = fetch_adjusted(code, suffix)
                if rows and len(rows) >= 60:  # 新上市股資料較少，降低門檻
                    with open(os.path.join(DATA_ADJ, f"{code}.csv"),
                              "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["date", "open", "high", "low",
                                    "close", "volume"])
                        w.writerows(rows)
                    return code, True
                break  # 有回應但資料不足, 換下一個 suffix
            except Exception:
                time.sleep(2 ** attempt)
    return code, False


def fetch_index():
    """下載加權指數 ^TWII 至 _TWII.csv (大盤濾網用)。指數無需除權息調整。"""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"range": RANGE, "interval": "1d"},
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            result = r.json()["chart"]["result"][0]
            ts = result.get("timestamp")
            if not ts:
                return False
            q = result["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c = (q["open"][i], q["high"][i],
                              q["low"][i], q["close"][i])
                if None in (o, h, l, c):
                    continue
                rows.append((time.strftime("%Y-%m-%d", time.gmtime(t + 8 * 3600)),
                             round(o, 2), round(h, 2), round(l, 2), round(c, 2), 0))
            if not rows:
                return False
            with open(os.path.join(DATA_ADJ, "_TWII.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                w.writerows(rows)
            return True
        except Exception:
            time.sleep(2 ** attempt)
    return False


def main():
    os.makedirs(DATA_ADJ, exist_ok=True)
    # 大盤指數每次都更新 (檔案小, 且選股當日一定要最新), 不走跳過邏輯
    print("下載加權指數 _TWII.csv ...", "OK" if fetch_index() else "失敗")
    universe = get_universe()
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(list(universe.values()), f, ensure_ascii=False, indent=1)
    print(f"universe: {len(universe)} 檔 "
          f"(上市 {sum(1 for s in universe.values() if s['market']=='TWSE')}, "
          f"上櫃 {sum(1 for s in universe.values() if s['market']=='TPEx')})")

    todo = [s for s in universe.values()
            if not os.path.exists(os.path.join(DATA_ADJ, f"{s['code']}.csv"))]
    print(f"已存在 {len(universe) - len(todo)} 檔, 需下載 {len(todo)} 檔")

    ok, fail = 0, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(task, s): s["code"] for s in todo}
        for fut in as_completed(futs):
            code, success = fut.result()
            if success:
                ok += 1
            else:
                fail.append(code)
            if (ok + len(fail)) % 100 == 0:
                print(f"progress: {ok + len(fail)}/{len(todo)} (ok={ok})")
    print(f"done. ok={ok}, fail={len(fail)}")
    if fail:
        print("失敗 (多為新上市未滿 250 交易日):", fail[:30])


if __name__ == "__main__":
    main()
