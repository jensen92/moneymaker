"""黃金期貨 (GC=F) 小時線順勢突破 — 每日/即時訊號報告.

抓取最新 GC 小時線, 用 gold_strategies 的順勢突破邏輯判斷:
  - 最新一根是否「剛突破」過去 N 小時高點 → 新進場做多訊號
  - 或目前若已在突破區間上方, 報告參考停損 (ATR 移動停損)

用法:
    python3 gold_signals.py              # 抓最新資料並報告
    python3 gold_signals.py --no-fetch   # 用本地既有 GC_60m.csv (不連網)
"""
import argparse
import csv
import os
import time

import numpy as np

import gold_strategies as gs

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def fetch_gc_hourly(path=gs.CSV):
    """抓 Yahoo GC=F 近 730 日小時線, 覆寫 path。失敗則回傳 False。"""
    import requests
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"range": "730d", "interval": "1h"},
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]; q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
                v = q["volume"][i]
                if None in (o, h, l, c):
                    continue
                rows.append((time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)),
                             round(o, 2), round(h, 2), round(l, 2), round(c, 2),
                             int(v) if v else 0))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                w.writerows(rows)
            return True
        except Exception:
            if attempt == 2:
                return False
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    if not args.no_fetch:
        ok = fetch_gc_hourly()
        if not ok and not os.path.exists(gs.CSV):
            print("❌ 無法抓取 GC 資料且無本地檔。")
            return

    cfg = gs.CONFIG
    dt, o, h, l, c = gs.load_bars()
    a = gs.atr(h, l, c, cfg["atr_n"])
    i = len(c) - 1
    bo = cfg["breakout"]
    price = c[i]
    hh = h[i - bo:i].max()   # 過去 N 小時高點 (不含當根)
    ll = l[i - bo:i].min()
    atr_now = a[i]

    print(f"黃金 GC 順勢突破訊號  (資料截至 {dt[i]} UTC)")
    print(f"參數: 突破窗 {bo} 小時, 移動停損 {cfg['atr_stop']}×ATR{cfg['atr_n']}, "
          f"{'多空雙向' if cfg['allow_short'] else '僅做多'}\n")
    print(f"最新收盤 {price:,.2f}  |  過去{bo}H高 {hh:,.2f}  低 {ll:,.2f}  "
          f"|  ATR {atr_now:,.2f}")

    m = gs.metrics(gs.backtest())

    sig = gs.signal_breakout(h, l, c, i, cfg, a)
    if sig == 1:
        stop = price - cfg["atr_stop"] * atr_now
        print(f"\n🟢 進場訊號: 做多 (突破過去{bo}H高點)")
        print(f"   進場價 {price:,.2f}   停損價 {stop:,.2f}  "
              f"(風險 {price - stop:,.2f} 點 ≈ ${(price - stop) * gs.POINT_VALUE:,.0f}/口)")
        print("   停利: 無固定停利, 以 ATR 移動停損追蹤獲利 (價格回落跌破停損即出場)")
    elif sig == -1:
        stop = price + cfg["atr_stop"] * atr_now
        print(f"\n🔴 進場訊號: 做空 (跌破過去{bo}H低點)")
        print(f"   進場價 {price:,.2f}   停損價 {stop:,.2f}")
    else:
        gap = (hh - price) / price
        print(f"\n⚪ 無新進場訊號。距突破做多還差 {hh - price:,.2f} 點 ({gap:+.2%})。")
        print("   (持有中部位請依 ATR 移動停損規則管理出場)")

    print(f"\n策略歷史績效 ({m['n']}筆, 2024-2026): 勝率 {m['win']:.1%}  PF {m['pf']:.2f}  "
          f"最大回撤 ${m['dd']:,.0f}  MAR {m['mar']:.2f}")


if __name__ == "__main__":
    main()
