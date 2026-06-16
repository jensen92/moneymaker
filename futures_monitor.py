"""5 市場策略 — 即時進出場點位監控 (供每 2 小時回報).

抓 5 個標的 (GC/NQ/ZS/ZT/LE) 的最新價格, 比對 futures_portfolio.py 策略的
進出場點位, 回報「目前價格是否觸及進場/出場點」。

進出場點位定義 (與 v3 策略一致):
  進場點 = 前 55 日最高價 (唐奇安上緣); 收/現價突破 → 觸發做多
  出場點 = 前 55 日最低價 (唐奇安下緣); 跌破 → 觸發出場
  輔助條件: MA50 > MA100 (均線多頭)、現價 > 252 日前 (動能向上)
  三訊號投票 → 多單強度 0 / 0.33 / 0.67 / 1.0

用法:
    python3 futures_monitor.py          # 跑一次, 印出即時點位表
"""
import time
from datetime import datetime

import numpy as np
import requests

from futures_portfolio import SPECS

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def fetch(sym):
    """回傳 (現價, 日線 high/low/close 陣列). 現價取盤中即時報價。"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    r = requests.get(url, params={"range": "2y", "interval": "1d"},
                     headers=HEADERS, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    meta = res["meta"]
    q = res["indicators"]["quote"][0]
    hi = [x for x in q["high"] if x is not None]
    lo = [x for x in q["low"] if x is not None]
    cl = [x for x in q["close"] if x is not None]
    cur = meta.get("regularMarketPrice") or cl[-1]
    return cur, np.array(hi), np.array(lo), np.array(cl)


def analyze(sym):
    cur, hi, lo, cl = fetch(sym)
    prior_hi = hi[:-1]          # 不含今日(形成中)的前 55 日
    prior_lo = lo[:-1]
    prior_cl = cl[:-1]
    hi55 = prior_hi[-55:].max()
    lo55 = prior_lo[-55:].min()
    ma50 = prior_cl[-50:].mean()
    ma100 = prior_cl[-100:].mean()
    mom_ref = prior_cl[-252] if len(prior_cl) >= 252 else prior_cl[0]

    donch_long = cur > hi55
    donch_exit = cur < lo55
    ma_ok = ma50 > ma100
    mom_ok = cur > mom_ref
    strength = (int(donch_long) + int(ma_ok) + int(mom_ok)) / 3

    if donch_long:
        status = "進場/加碼 (突破55日高)"
    elif donch_exit:
        status = "出場 (跌破55日低)"
    else:
        status = "區間內 觀望"
    return {
        "name": SPECS[sym]["name"], "cur": cur, "hi55": hi55, "lo55": lo55,
        "to_entry": hi55 / cur - 1, "to_exit": lo55 / cur - 1,
        "ma_ok": ma_ok, "mom_ok": mom_ok, "strength": strength, "status": status,
    }


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"⏰ {now}  5 市場進出場點位監控")
    print(f"{'商品':<14}{'現價':>10}{'進場點(55高)':>13}{'出場點(55低)':>13}"
          f"{'距進場':>8}{'距出場':>8}{'均線':>5}{'動能':>5}{'強度':>6}  狀態")
    print("-" * 104)
    alerts = []
    for sym in SPECS:
        try:
            a = analyze(sym)
        except Exception as e:
            print(f"{SPECS[sym]['name']:<14}  抓取失敗: {e}")
            continue
        ma = "多" if a["ma_ok"] else "空"
        mom = "↑" if a["mom_ok"] else "↓"
        flag = ""
        if "進場" in a["status"] or "出場" in a["status"]:
            flag = "  ⚠️"
            alerts.append(a)
        print(f"{a['name']:<14}{a['cur']:>10.2f}{a['hi55']:>13.2f}"
              f"{a['lo55']:>13.2f}{a['to_entry']:>+8.1%}{a['to_exit']:>+8.1%}"
              f"{ma:>5}{mom:>5}{a['strength']:>6.0%}  {a['status']}{flag}")

    print("-" * 104)
    if alerts:
        print("⚠️ 觸及進出場點:")
        for a in alerts:
            print(f"   {a['name']}: {a['status']}  (現價 {a['cur']:.2f}, "
                  f"多單強度 {a['strength']:.0%})")
    else:
        print("✅ 目前無任何市場觸及進出場點 (皆在通道區間內)。")


if __name__ == "__main__":
    main()
