"""補充 data_adj/ 中截止 2026-04-02 的檔案，從 Yahoo Finance 抓取後續資料並寫回 Drive.

只抓取 2026-04-03 ~ 今日，append 到現有 CSV，然後透過 Drive MCP 上傳。
"""
import csv
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_ADJ = os.path.join(os.path.dirname(__file__), "data_adj")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
START_DATE = "2026-04-03"


def fetch_recent(code):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.TW"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"period1": "1743638400",  # 2026-04-03 UTC
                                          "period2": "9999999999",
                                          "interval": "1d"},
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            result = r.json()["chart"]["result"][0]
            ts = result.get("timestamp")
            if not ts:
                return []
            q = result["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c, v = (q["open"][i], q["high"][i], q["low"][i],
                                  q["close"][i], q["volume"][i])
                if None in (o, h, l, c) or not v:
                    continue
                date = time.strftime("%Y-%m-%d", time.gmtime(t + 8 * 3600))
                if date <= "2026-04-02":
                    continue
                rows.append((date, round(o, 2), round(h, 2), round(l, 2),
                              round(c, 2), int(v)))
            return rows
        except Exception as e:
            time.sleep(2 ** attempt)
    return None


def update_file(code):
    path = os.path.join(DATA_ADJ, f"{code}.csv")
    rows = fetch_recent(code)
    if rows is None:
        return code, "fetch_fail"
    if not rows:
        return code, "no_new_data"
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    return code, f"+{len(rows)} rows"


def main():
    stale = []
    for fn in os.listdir(DATA_ADJ):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        path = os.path.join(DATA_ADJ, fn)
        with open(path) as f:
            lines = f.readlines()
        last_date = lines[-1].split(",")[0] if len(lines) > 1 else ""
        if last_date == "2026-04-02":
            stale.append(fn[:-4])

    print(f"需要更新: {len(stale)} 檔")
    ok, fail = 0, []

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(update_file, code): code for code in stale}
        for fut in as_completed(futs):
            code, result = fut.result()
            if "fail" in result:
                fail.append(code)
            else:
                ok += 1
            if (ok + len(fail)) % 50 == 0:
                print(f"  progress: {ok + len(fail)}/{len(stale)}")

    print(f"完成: ok={ok}, fail={len(fail)}")
    if fail:
        print("失敗:", fail[:20])


if __name__ == "__main__":
    main()
