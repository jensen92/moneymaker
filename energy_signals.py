"""能源期貨季節性策略 — 天然氣 (NG) / 原油 (CL) 做多, 即時訊號 + EV 績效.

策略思維 (非趨勢): 能源是均值回歸+強季節商品, 趨勢跟隨在其上為負期望
(26年實測 NG 趨勢 EV/筆 -0.07R 勝率25%「寡婦製造機」, CL ~0R), 但季節性
極強且穩健:
  NG  9 月初進場、持有 ~3 月 (夏末囤貨季+颶風季噴, 11月底前出避開冬季回吐)
      EV/筆 +1.09R 勝率64% 賺賠2.18  樣本前/後半 +1.09/+1.08 (幾乎一致, 極穩)
  CL 12 月初進場、持有 ~6 月 (冬季淡季進, Q2 駕駛季前漲)
      EV/筆 +0.90R 勝率46% 賺賠3.02  樣本前/後半 +0.68/+1.11
窗口為能源行事曆常識 + 樣本內外對切驗證, 非網格挑選 (避免實盤失真)。

每商品一年僅一次進場 → EV/年約 1R, 單獨報酬不高; 價值在與股票/黃金低相關的
分散。用法: python3 energy_signals.py [--no-fetch]
"""
import argparse
import csv
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "futures_data")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# 各能源季節窗 (進場月, 持有月數, 停損 ATR 倍數, 每點$, 顯示名)
CONFIG = {
    "NG": {"month": 9,  "hold_mo": 3, "stop_atr": 3.0, "pv": 10000.0, "name": "天然氣 NG"},
    "CL": {"month": 12, "hold_mo": 6, "stop_atr": 3.0, "pv": 1000.0,  "name": "原油 CL"},
}


def _path(key):
    return os.path.join(DATA, f"{key}_1d.csv")


def fetch_energy():
    """抓 NG=F / CL=F 日線 (2000 年至今), 覆寫 futures_data/{NG,CL}_1d.csv。"""
    import requests
    p1 = 946684800
    ok = True
    for key, sym in (("NG", "NG=F"), ("CL", "CL=F")):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"period1": p1, "period2": int(time.time()),
                                     "interval": "1d"}, headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]; q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
                if None in (o, h, l, c):
                    continue
                rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)),
                             round(o, 3), round(h, 3), round(l, 3), round(c, 3),
                             int(q["volume"][i] or 0)))
            with open(_path(key), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                w.writerows(rows)
        except Exception:
            ok = False
    return ok


def _load(key):
    o, h, l, c, dt = [], [], [], [], []
    with open(_path(key)) as f:
        for r in csv.DictReader(f):
            try:
                o.append(float(r["open"])); h.append(float(r["high"]))
                l.append(float(r["low"])); c.append(float(r["close"]))
            except ValueError:
                continue
            dt.append(r["date"])
    return dt, np.array(o), np.array(h), np.array(l), np.array(c)


def _atr(h, l, c, n=20):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    a = np.full(len(c), np.nan)
    a[n - 1] = tr[:n].mean()
    for i in range(n, len(c)):
        a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def backtest(key):
    """季節性做多回測, 回傳逐筆 R (損益/初始風險)。"""
    cfg = CONFIG[key]
    dt, o, h, l, c = _load(key)
    a = _atr(h, l, c, 20)
    n = len(c); hold = int(21 * cfg["hold_mo"]); st = cfg["stop_atr"]
    R = []; ydone = set()
    for i in range(21, n):
        m = int(dt[i][5:7]); pm = int(dt[i - 1][5:7]); yr = dt[i][:4]
        if m == cfg["month"] and pm != cfg["month"] and yr not in ydone and not np.isnan(a[i]):
            ydone.add(yr)
            entry = c[i]; risk = st * a[i]; stop = entry - risk
            j = min(i + hold, n - 1); ex = c[j]
            for k in range(i + 1, j + 1):
                if c[k] <= stop:
                    ex = c[k]; break
            R.append((ex - entry) / risk if risk > 0 else 0.0)
    return np.array(R)


def metrics(R):
    if len(R) == 0:
        return None
    w = R[R > 0]; ls = R[R < 0]
    eq = np.cumsum(R); dd = (np.maximum.accumulate(eq) - eq).max()
    return {"n": len(R), "win": float((R > 0).mean()), "ev": float(R.mean()),
            "payoff": float(w.mean() / -ls.mean()) if len(ls) and len(w) else 0,
            "mar": float(eq[-1] / dd) if dd > 0 else 0}


def _state(key):
    """目前季節狀態: 待進場 / 進場月 / 持有中。回傳 dict。"""
    cfg = CONFIG[key]
    dt, o, h, l, c = _load(key)
    a = _atr(h, l, c, 20)
    i = len(c) - 1
    cur_m = int(dt[i][5:7])
    price = float(c[i]); atr = float(a[i]); st = cfg["stop_atr"]
    em = cfg["month"]; hold_m = cfg["hold_mo"]
    # 持有窗: 進場月起算 hold_mo 個月 (跨年處理)
    months_since = (cur_m - em) % 12
    in_entry = (cur_m == em)
    holding = (0 <= months_since < hold_m) and not in_entry
    exit_m = ((em + hold_m - 1) % 12) + 1
    return {"price": price, "atr": atr, "stop": price - st * atr,
            "entry_m": em, "exit_m": exit_m, "cur_m": cur_m,
            "in_entry": in_entry, "holding": holding,
            "months_to_entry": (em - cur_m) % 12, "name": cfg["name"], "st": st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch:
        fetch_energy()

    print("🔥 能源期貨季節做多（NG/CL, 非趨勢）")
    for key in ("NG", "CL"):
        s = _state(key); m = metrics(backtest(key))
        print(f"\n{s['name']}  現價 {s['price']:,.2f}｜ATR {s['atr']:.2f}"
              f"｜季節窗 {s['entry_m']}月進→{s['exit_m']}月出")
        if s["in_entry"]:
            print(f"  🟢 進場月！買進 {s['price']:,.2f}｜停損 {s['stop']:,.2f}"
                  f"（{s['st']}×ATR）｜持有到 {s['exit_m']}月")
        elif s["holding"]:
            print(f"  🟡 持有中（季節窗內）｜參考停損 {s['stop']:,.2f}｜{s['exit_m']}月到期出場")
        else:
            print(f"  ⚪ 待 {s['entry_m']}月進場（還 {s['months_to_entry']} 個月）")
        if m:
            print(f"  EV/筆 {m['ev']:+.2f}R｜勝率 {m['win']:.0%}｜賺賠 {m['payoff']:.2f}｜"
                  f"MAR {m['mar']:.1f}（{m['n']}年）")
    print("\n(能源=季節非趨勢; 一年一次, 單獨報酬低, 價值在與股票/黃金低相關的分散)")


if __name__ == "__main__":
    main()
