"""下載台股加權指數 (^TWII) 日內 K 線, 作為台指期 (TXF) 策略開發的標的。

說明 (誠實揭露):
- Yahoo 沒有乾淨的台指期歷史日內資料, 故以 ^TWII (加權指數現貨) 為開發標的。
  台指期 TXF 與加權指數高度連動 (日內相關 > 0.99), 是業界標準的開發代理;
  實單在 TXF/MXF 執行, 需另計基差與夜盤, 但日內訊號邏輯一致。
- Yahoo 日內歷史長度限制: 1h 約 2 年、15m/5m 僅 ~60 天。故回測主時間框架
  為「小時 K」(唯一有足夠歷史者); 5m/15m 僅供即時進場細修, 歷史過短不做回測。

輸出: futures_data/TWII_<interval>.csv (date,open,high,low,close,volume)
其中 date 為台北時間 (UTC+8) 的 'YYYY-MM-DD HH:MM'。
"""
import csv
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "futures_data")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
SYMBOL = "%5ETWII"   # ^TWII

# interval -> Yahoo 允許的最長 range
SPECS = {"60m": "730d", "15m": "60d", "5m": "60d"}


def fetch(interval, range_):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}"
    for attempt in range(4):
        try:
            r = requests.get(url, params={"range": range_, "interval": interval},
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp")
            if not ts:
                return None
            q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c = (q["open"][i], q["high"][i],
                              q["low"][i], q["close"][i])
                v = q["volume"][i]
                if None in (o, h, l, c):
                    continue
                dt = time.strftime("%Y-%m-%d %H:%M", time.gmtime(t + 8 * 3600))
                rows.append((dt, round(o, 2), round(h, 2), round(l, 2),
                             round(c, 2), int(v or 0)))
            return rows or None
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return None


def save(interval, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"TWII_{interval}.csv")
    # 增量合併: 保留舊資料, 用新資料覆蓋/延伸 (以 date 為鍵)
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    existing[row[0]] = row
    for row in rows:
        existing[row[0]] = [str(x) for x in row]
    merged = [existing[k] for k in sorted(existing)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        w.writerows(merged)
    return path, len(merged)


def main():
    for interval, range_ in SPECS.items():
        rows = fetch(interval, range_)
        if not rows:
            print(f"{interval}: 下載失敗")
            continue
        path, n = save(interval, rows)
        print(f"{interval}: {len(rows)} 筆下載, 合併後 {n} 筆 -> {os.path.basename(path)} "
              f"({rows[0][0]} .. {rows[-1][0]})")


if __name__ == "__main__":
    main()
