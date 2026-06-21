"""下載全台股 (上市 + 上櫃) 除權息調整日線至 data_adj/.

- 清單: TWSE OpenAPI (上市) + TPEx OpenAPI (上櫃)
- 新股: 全量下載 15 年歷史
- 已存在且日期 < 今天-2: 增量補齊 (只補缺少的新資料, 不覆蓋舊資料)
- 已存在且日期 >= 今天-2: 跳過 (資料已是最新)
"""
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests

DATA_ADJ = os.path.join(os.path.dirname(__file__), "data_adj")
UNIVERSE_FILE = os.path.join(DATA_ADJ, "_universe_all.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
FULL_RANGE = "15y"
INC_RANGE  = "3mo"   # 增量更新範圍 (足夠補 1-2 個月的落差)


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


def _last_date(path):
    """回傳 CSV 最後一行的日期字串, 若無法讀取則回傳 ''."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200))
            tail = f.read().decode(errors="ignore")
        last_line = [l for l in tail.splitlines() if l.strip()][-1]
        return last_line.split(",")[0]
    except Exception:
        return ""


def _fetch_yf(code, suffix, range_):
    """抓 Yahoo Finance, 回傳 list of (date,o,h,l,c,v) 或 None."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}{suffix}"
    r = requests.get(url, params={"range": range_, "interval": "1d",
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
    return rows or None


def task(stock):
    code, market = stock["code"], stock["market"]
    path = os.path.join(DATA_ADJ, f"{code}.csv")
    suffixes = [".TW", ".TWO"] if market == "TWSE" else [".TWO", ".TW"]

    # 判斷是否需要更新
    cutoff = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    last = _last_date(path) if os.path.exists(path) else ""
    if last >= cutoff:
        return code, "skip"   # 資料已是最新

    incremental = bool(last)  # 有舊檔 → 只補新資料
    range_ = INC_RANGE if incremental else FULL_RANGE

    for suffix in suffixes:
        for attempt in range(3):
            try:
                rows = _fetch_yf(code, suffix, range_)
                if not rows:
                    break
                if incremental:
                    new_rows = [r for r in rows if r[0] > last]
                    if new_rows:
                        with open(path, "a", newline="") as f:
                            csv.writer(f).writerows(new_rows)
                    return code, True
                else:
                    if len(rows) >= 60:
                        with open(path, "w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["date", "open", "high", "low",
                                        "close", "volume"])
                            w.writerows(rows)
                        return code, True
                    break  # 資料不足, 換 suffix
            except Exception:
                time.sleep(2 ** attempt)
    return code, False


def fetch_index():
    """下載加權指數 ^TWII 至 _TWII.csv (大盤濾網用)。指數無需除權息調整。"""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"range": FULL_RANGE, "interval": "1d"},
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
    # 大盤指數每次都更新 (檔案小, 且選股當日一定要最新)
    print("下載加權指數 _TWII.csv ...", "OK" if fetch_index() else "失敗")

    universe = get_universe()
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(list(universe.values()), f, ensure_ascii=False, indent=1)
    print(f"universe: {len(universe)} 檔 "
          f"(上市 {sum(1 for s in universe.values() if s['market']=='TWSE')}, "
          f"上櫃 {sum(1 for s in universe.values() if s['market']=='TPEx')})")

    ok, skip, fail = 0, 0, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(task, s): s["code"] for s in universe.values()}
        for fut in as_completed(futs):
            code, result = fut.result()
            if result == "skip":
                skip += 1
            elif result:
                ok += 1
            else:
                fail.append(code)
            total = ok + skip + len(fail)
            if total % 200 == 0:
                print(f"progress: {total}/{len(universe)} "
                      f"(新增/更新={ok}, 跳過={skip})")

    print(f"done. 新增/更新={ok}, 跳過={skip}, 失敗={len(fail)}")
    if fail:
        print("失敗 (多為新上市未滿 60 交易日):", fail[:30])


if __name__ == "__main__":
    main()
