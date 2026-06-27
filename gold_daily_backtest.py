"""黃金 (GC=F) 日線長歷史回測 — 順勢突破策略的『跨regime穩健度』驗證.

生產策略 (gold_strategies) 跑的是『小時線』, 但 Yahoo 小時線只給最近約 730 天
(僅 2.4 年, 且剛好是 2024-2026 大多頭), 樣本太短、regime 太單一。

本腳本改抓『日線』(Yahoo 日線可回溯到 2000 年, 26 年), 用同一套順勢突破
邏輯 (Donchian N 日突破 + ATR 移動停損) 跨 26 年回測, 重點看策略在
**非多頭年份** (2013-2015 黃金大空頭、2021-2022 盤整) 會不會崩。

注意: 這是『日線』近似驗證, 突破窗單位是『日』而非『小時』, 不是生產小時策略
的精確重現; 目的是檢驗『黃金順勢突破』這個edge是否跨regime成立, 而非調參。

用法: python3 gold_daily_backtest.py [--no-fetch]
資料存 futures_data/GC_1d.csv (gitignored, 可重抓)。
"""
import argparse
import csv
import os
import time

import numpy as np

import gold_strategies as gs

CSV_1D = os.path.join(gs.HERE, "futures_data", "GC_1d.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# Donchian N 日突破 + ATR 倍數移動停損 (跨26年MAR最佳, 平台中心; 只做多)
DBO = 40      # 突破窗 (日)
DATR_STOP = 4.0
DATR_N = 20


def fetch_daily(path=CSV_1D):
    """抓 Yahoo GC=F 日線 (2000 年至今), 覆寫 path。失敗回傳 False。"""
    import requests
    p1 = 946684800  # 2000-01-01
    for attempt in range(3):
        try:
            r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F",
                             params={"period1": p1, "period2": int(time.time()),
                                     "interval": "1d"},
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
                rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)),
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


def load(path=CSV_1D):
    o, h, l, c, dt = [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            dt.append(r["date"]); o.append(float(r["open"])); h.append(float(r["high"]))
            l.append(float(r["low"])); c.append(float(r["close"]))
    return dt, np.array(o), np.array(h), np.array(l), np.array(c)


def backtest(dt, h, l, c, bo=DBO, st=DATR_STOP, an=DATR_N, short=False):
    a = gs.atr(h, l, c, an)
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; tr = []

    def close(px, j):
        nonlocal pos
        pnl = (px - entry) * pos * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2
        tr.append({"pnl": pnl, "entry_date": dt[ei], "exit_date": dt[j], "dir": pos})
        pos = 0

    for i in range(bo + an + 1, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        hh = h[i - bo:i].max(); ll = l[i - bo:i].min()
        if pos == 0:
            if px > hh:
                pos, entry, ei, trail = 1, px, i, px - st * a[i]
            elif short and px < ll:
                pos, entry, ei, trail = -1, px, i, px + st * a[i]
        elif pos == 1:
            trail = max(trail, px - st * a[i])
            if px <= trail:
                close(px, i)
        else:
            trail = min(trail, px + st * a[i])
            if px >= trail:
                close(px, i)
    if pos:
        close(c[-1], len(c) - 1)
    return tr


def metrics(tr):
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr])
    w = p[p > 0]; ls = p[p < 0]
    eq = np.cumsum(p)
    dd = (np.maximum.accumulate(eq) - eq).max()
    return dict(n=len(tr), win=len(w) / len(tr),
                pf=(w.sum() / -ls.sum()) if len(ls) else float("inf"),
                tot=p.sum(), dd=dd, mar=(eq[-1] / dd) if dd > 0 else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch:
        if not fetch_daily() and not os.path.exists(CSV_1D):
            print("❌ 無法抓取日線且無本地檔。")
            return
    dt, o, h, l, c = load()
    print(f"黃金日線 {len(c)} 根  {dt[0]} ~ {dt[-1]}  (26年跨regime驗證, 只做多)\n")

    print("參數網格 (Donchian突破窗日 × ATR停損, ATR20):")
    print(f"{'bo日':>5}{'atr':>5}{'交易':>6}{'勝率':>7}{'PF':>6}{'總$':>12}{'最大DD$':>11}{'MAR':>6}")
    for bo in (20, 30, 40, 55):
        for st in (2.5, 3.0, 4.0):
            m = metrics(backtest(dt, h, l, c, bo, st))
            print(f"{bo:>5}{st:>5}{m['n']:>6}{m['win']:>7.1%}{m['pf']:>6.2f}"
                  f"{m['tot']:>+12,.0f}{m['dd']:>11,.0f}{m['mar']:>6.2f}")

    tr = backtest(dt, h, l, c)
    m = metrics(tr)
    print(f"\n採用 bo{DBO}日/atr{DATR_STOP}/ATR{DATR_N}: 交易{m['n']} 勝率{m['win']:.1%} "
          f"PF{m['pf']:.2f} 總${m['tot']:+,.0f} 最大DD${m['dd']:,.0f} MAR{m['mar']:.2f}")
    years = sorted(set(t["exit_date"][:4] for t in tr))
    py = {y: 0.0 for y in years}; ny = {y: 0 for y in years}
    for t in tr:
        py[t["exit_date"][:4]] += t["pnl"]; ny[t["exit_date"][:4]] += 1
    print("逐年總損益 (跨regime穩健度):")
    for y in years:
        bar = "█" * int(abs(py[y]) / 3000)
        print(f"  {y}: {py[y]:>+10,.0f} ({ny[y]:>2}筆) {'+' if py[y] >= 0 else '-'}{bar}")


if __name__ == "__main__":
    main()
