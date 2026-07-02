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
            if not rows:      # API 回傳異常空資料 — 絕不可覆寫掉既有歷史
                ok = False
                continue
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
    a = np.full(len(c), np.nan)
    if len(c) < n:                       # 序列過短保護 (與 gold_strategies 一致)
        return a
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
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
    # 'mar' = 累計R ÷ 最大R回撤 (報酬/回撤比), 非年化 MAR; 顯示標『報酬/回撤比』。
    return {"n": len(R), "win": float((R > 0).mean()), "ev": float(R.mean()),
            "payoff": float(w.mean() / -ls.mean()) if len(ls) and len(w) else 0,
            "mar": float(eq[-1] / dd) if dd > 0 else 0}


def _state(key):
    """目前季節狀態 — 以回測同邏輯重放找『目前實際部位』(待進場/進場日/持有中),
    確保與 backtest 的 21×hold_mo 交易日持有窗完全一致 (不再用日曆月近似)。"""
    cfg = CONFIG[key]
    dt, o, h, l, c = _load(key)
    a = _atr(h, l, c, 20)
    n = len(c); i = n - 1
    hold = int(21 * cfg["hold_mo"]); st = cfg["stop_atr"]; em = cfg["month"]
    price = float(c[i]); atr = float(a[i]); cur_m = int(dt[i][5:7])
    # 找最近一次進場 (進場月第一個交易日) 與其實際出場索引 (停損或到期)
    ydone = set(); last_e = None
    for k in range(21, n):
        m = int(dt[k][5:7]); pm = int(dt[k - 1][5:7]); yr = dt[k][:4]
        if m == em and pm != em and yr not in ydone and not np.isnan(a[k]):
            ydone.add(yr); last_e = k
    pos = None
    if last_e is not None:
        entry = c[last_e]; stop = entry - st * a[last_e]; exit_i = min(last_e + hold, n - 1)
        close_i = exit_i
        for k in range(last_e + 1, exit_i + 1):
            if c[k] <= stop:
                close_i = k; break
        if last_e <= i < close_i:            # 目前仍持倉 (含進場日, 不含出場日)
            pos = {"entry": float(entry), "stop": float(stop),
                   "exit_i": exit_i, "exit_m": int(dt[exit_i][5:7]),
                   "days_left": exit_i - i, "just_entered": (last_e == i)}
    in_entry = pos is not None and pos["just_entered"]
    holding = pos is not None and not pos["just_entered"]
    exit_m = pos["exit_m"] if pos else ((em + cfg["hold_mo"] - 1) % 12) + 1
    return {"price": price, "atr": atr, "cur_m": cur_m, "entry_m": em, "exit_m": exit_m,
            "in_entry": in_entry, "holding": holding, "st": st, "name": cfg["name"],
            "pos": pos, "months_to_entry": (em - cur_m) % 12,
            "entry": pos["entry"] if pos else None,
            "stop": pos["stop"] if pos else price - st * atr,
            "days_left": pos["days_left"] if pos else 0}


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
            print(f"  🟢 進場日！買進 {s['entry']:,.2f}｜停損 {s['stop']:,.2f}"
                  f"（{s['st']}×ATR）｜持有到 {s['exit_m']}月（約{s['days_left']}交易日）")
            print(f"  👉 操作: 今日市價買進, 同時掛停損賣單 {s['stop']:,.2f}(GTC)。"
                  f"之後不動, 等到期或停損")
        elif s["holding"]:
            print(f"  🟡 持有中｜進場 {s['entry']:,.2f}｜停損 {s['stop']:,.2f}｜"
                  f"{s['exit_m']}月到期（剩約{s['days_left']}交易日）")
            print(f"  👉 操作: 續抱不動。確認停損單仍掛 {s['stop']:,.2f}(不移動); "
                  f"{s['exit_m']}月初到期日收盤平倉")
        else:
            print(f"  ⚪ 待 {s['entry_m']}月進場（還 {s['months_to_entry']} 個月）")
            print(f"  👉 操作: 目前無單、無需動作。{s['entry_m']}月第一個交易日再進場")
        if m:
            print(f"  EV/筆 {m['ev']:+.2f}R｜勝率 {m['win']:.0%}｜賺賠 {m['payoff']:.2f}｜"
                  f"報酬/回撤比 {m['mar']:.1f}（{m['n']}年）")
    print("\n💡 概念: NG=夏末囤貨+颶風季買進、入冬前平倉(避開冬季回吐);")
    print("   CL=冬季淡季買進、Q2駕駛旺季前平倉。寬停損只防黑天鵝, 期間震盪不出場;")
    print("   出場只有兩種: 到期月平倉 或 觸停損。(已驗證收緊/移動停損反而砍贏家)")


if __name__ == "__main__":
    main()
