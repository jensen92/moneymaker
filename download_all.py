"""下載全台股 (上市 + 上櫃) 除權息調整日線至 data_adj/.

- 清單: TWSE OpenAPI (上市) + TPEx OpenAPI (上櫃)
- 新股: 全量下載 15 年歷史
- 已存在且日期 < 今天-2: 增量補齊 (只補缺少的新資料, 不覆蓋舊資料)
- 已存在且日期 >= 今天-2: 跳過 (資料已是最新)

完整性比對 (verify_and_fill):
- 完整資料以台灣交易所 (TWSE) 官方 OpenAPI 為準 (https://openapi.twse.com.tw/,
  上櫃則用 TPEx OpenAPI), 下載結束後比對本地最後一筆與官方今日收盤:
  - 本地缺當日資料 (Yahoo 落後/失敗) → 用官方資料補齊, 確保完整性。
  - 本地已有但收盤價與官方差異 > 容忍度 → 視為除權息還原股價造成的正常差異,
    以 Yahoo (還原股價) 為準, 不覆蓋, 僅記錄供人工複核。
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


def _last_line(path):
    """回傳 CSV 最後一行 (字串), 若無法讀取則回傳 ''."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200))
            tail = f.read().decode(errors="ignore")
        return [l for l in tail.splitlines() if l.strip()][-1]
    except Exception:
        return ""


def _last_date(path):
    """回傳 CSV 最後一行的日期字串, 若無法讀取則回傳 ''."""
    line = _last_line(path)
    return line.split(",")[0] if line else ""


def _roc_to_iso(roc_date):
    """台灣官方民國日期 '1150618' -> ISO '2026-06-18'."""
    s = str(roc_date)
    y = int(s[:-4]) + 1911
    return f"{y}-{s[-4:-2]}-{s[-2:]}"


def get_twse_today():
    """官方今日收盤 (上市+上櫃, 未還原股價), 回傳 {code: (date_iso,o,h,l,c,v)}.

    用作「完整資料來源」的校驗基準: TWSE/TPEx 官方資料保證涵蓋所有交易日,
    可補齊 Yahoo 落後/抓取失敗造成的缺漏。
    """
    out = {}
    for row in get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL") or []:
        code = row.get("Code", "")
        if len(code) != 4 or not code.isdigit():
            continue
        try:
            out[code] = (
                _roc_to_iso(row["Date"]),
                float(row["OpeningPrice"]), float(row["HighestPrice"]),
                float(row["LowestPrice"]), float(row["ClosingPrice"]),
                int(float(row["TradeVolume"])))
        except (KeyError, ValueError):
            continue
    for row in get_json("https://www.tpex.org.tw/openapi/v1/"
                        "tpex_mainboard_daily_close_quotes") or []:
        code = row.get("SecuritiesCompanyCode", "")
        if len(code) != 4 or not code.isdigit() or code in out:
            continue
        try:
            out[code] = (
                _roc_to_iso(row["Date"]),
                float(row["Open"]), float(row["High"]),
                float(row["Low"]), float(row["Close"]),
                int(float(row["TradingShares"])))
        except (KeyError, ValueError):
            continue
    return out


def verify_and_fill(codes, twse_today, tol=0.005, max_gap_days=5):
    """以官方今日收盤比對本地 (Yahoo 還原股價) 資料, 補齊缺漏 + 標記差異.

    只補「剛好缺最後一天」的情況 (gap <= max_gap_days 個日曆日, 對應 Yahoo
    當天還沒發布或抓取失敗一次); gap 較大代表 Yahoo 連續多日抓取失敗, 直接
    插入官方單筆資料反而會在中間留下缺口 (破壞移動平均等指標計算), 此時只
    回報、不寫入, 留給下次 Yahoo 增量抓取自然補齊。

    回傳 (filled, mismatched, stale):
      filled=已用官方資料補上的代號清單
      mismatched=(代號, 本地收盤, 官方收盤) 清單, 僅記錄不覆蓋 (以 Yahoo 為準)
      stale=(代號, 本地最後日期, 官方最後日期) 缺口過大, 待下次增量抓取補齊
    """
    filled, mismatched, stale = [], [], []
    for code in codes:
        official = twse_today.get(code)
        if official is None:
            continue
        path = os.path.join(DATA_ADJ, f"{code}.csv")
        if not os.path.exists(path):
            continue
        d_official, o, h, l, c, v = official
        line = _last_line(path)
        if not line:
            continue
        parts = line.split(",")
        last_date = parts[0]
        if last_date == d_official:
            try:
                local_close = float(parts[4])
            except (IndexError, ValueError):
                continue
            if abs(local_close - c) / c > tol:
                mismatched.append((code, local_close, c))
        elif last_date < d_official:
            try:
                gap = (datetime.strptime(d_official, "%Y-%m-%d")
                       - datetime.strptime(last_date, "%Y-%m-%d")).days
            except ValueError:
                continue
            if gap > max_gap_days:
                stale.append((code, last_date, d_official))
                continue
            # 本地僅缺最後一天 (Yahoo 落後/失敗) -> 用官方完整資料補齊
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow([d_official, o, h, l, c, v])
            filled.append(code)
    return filled, mismatched, stale


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


def task(stock, cutoff):
    code, market = stock["code"], stock["market"]
    path = os.path.join(DATA_ADJ, f"{code}.csv")
    suffixes = [".TW", ".TWO"] if market == "TWSE" else [".TWO", ".TW"]

    # 判斷是否需要更新: cutoff = 該股官方最新交易日, 本地已到此日才跳過。
    # 改用「每股各自的官方最新日期」而非「今天-2」估計, 也避免上市/上櫃
    # 兩市場交易日不同步時, 用單一全市場日期誤判某一邊永遠落後。
    last = _last_date(path) if os.path.exists(path) else ""
    if cutoff and last >= cutoff:
        return code, "skip"   # 資料已到該股最新交易日

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

    # 先取官方今日資料, 由此得出「每股官方最新交易日」作為更新門檻 (cutoff)。
    # 本地最後一筆 < 該股最新交易日 → 需更新; 已到最新交易日才跳過。
    twse_today = get_twse_today()
    official_date = {code: v[0] for code, v in twse_today.items()}
    fallback = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    if twse_today:
        latest = max(official_date.values(), default="")
        print(f"官方最新交易日: 最新 {latest} (依各股官方日期判斷是否更新)")
    else:
        print(f"⚠️ 官方資料取得失敗, 改用估計門檻 {fallback}")

    ok, skip, fail = 0, 0, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        # 找不到官方日期的股票 (停牌等) → 用 fallback, 至少不漏抓
        futs = {ex.submit(task, s, official_date.get(s["code"], fallback)): s["code"]
                for s in universe.values()}
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

    print("以台灣交易所/櫃買中心官方資料比對完整性 ...")
    filled, mismatched, stale = verify_and_fill(list(universe.keys()), twse_today)
    print(f"官方資料補齊缺漏 (僅缺最後一天): {len(filled)} 檔"
          + (f" {filled[:20]}" if filled else ""))
    if mismatched:
        print(f"⚠️ 與官方收盤價差異 > 0.5% (還原股價正常現象, 已保留 Yahoo 還原值): "
              f"{len(mismatched)} 檔")
        for code, local_close, official_close in mismatched[:10]:
            print(f"   {code}: yahoo還原={local_close} 官方未還原={official_close}")
    if stale:
        print(f"ℹ️ 缺口過大未補 (留給下次 Yahoo 增量抓取): {len(stale)} 檔"
              f" {[s[0] for s in stale[:20]]}")


if __name__ == "__main__":
    main()
