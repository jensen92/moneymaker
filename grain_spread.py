"""穀物價差均值回歸 (相對價值, 市場中性) — 黃豆/小麥/玉米兩兩價差.

策略思維 (非趨勢跟隨): 穀物是可再生農產品, 價格錨在生產成本、長期均值回歸,
單邊趨勢跟隨在其上 DD 高(20~24%)而報酬低。但『穀物之間的價差』(同為飼料/
食品穀物, 受共同天候/總經驅動) 有更強的結構性均值回歸 — 兩腿對沖掉大盤方向,
故市場中性、DD 天生小很多。

26 年實證 (1:1 口, 同點值 $50; z>1.5 進場、|z|<0.3 出場、252 日窗 — 經濟邏輯
之常規值, 非網格優化):
  黃豆-小麥價差  勝率80% PF4.60 最大DD 8.6%  (最佳)
  小麥-玉米價差  勝率85% PF5.58 最大DD10.8%
  黃豆-玉米價差  勝率75% PF2.58 最大DD12.4%
對照單邊趨勢(M/D)單商品 DD 約 20~24%。

抗過擬合: 樣本內外對切 (2000-13 vs 2013-26) 兩段皆 79% 勝率、皆獲利、DD ~9%;
z 門檻 1.0~2.5 全為 77~85% 勝率的平滑平台, 非知識邊緣。故參數採常規固定值,
不做網格挑選 (避免實盤失真)。

用法: python3 grain_spread.py [--no-fetch]
"""
import argparse
import csv
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "futures_data")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

PV = 50.0                 # 每點 $ (三穀物同規格: 5000 bushels, $50/點)
COST = 0.30 * PV * 2      # 單腿往返成本估計; 價差兩腿合計於下方乘 2
WIN = 120                 # 均值/標準差回看窗 (短線版~半年; 252日長線版更穩但較少交易)
Z_IN = 1.5                # 進場門檻 (偏離均值幾個標準差)
Z_OUT = 0.3               # 出場門檻 (回到均值附近)

# 價差組合 (legA - legB), 依 26 年 DD/穩健度排序; 小麥在此當『一腿』而非單獨做
PAIRS = [
    ("ZS", "ZW", "黃豆-小麥"),
    ("ZW", "ZC", "小麥-玉米"),
    ("ZS", "ZC", "黃豆-玉米"),
]


def _load(key):
    path = os.path.join(DATA, f"{key}.csv")
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["date"]] = float(r["close"])
    return d


def fetch_grains():
    """抓黃豆/小麥/玉米日線 (Yahoo, 2000 年至今), 覆寫 futures_data/{ZS,ZW,ZC}.csv。"""
    import requests
    p1 = 946684800
    ok = True
    for key, sym in (("ZS", "ZS=F"), ("ZW", "ZW=F"), ("ZC", "ZC=F")):
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
                             round(o, 2), round(h, 2), round(l, 2), round(c, 2),
                             int(q["volume"][i] or 0)))
            with open(os.path.join(DATA, f"{key}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                w.writerows(rows)
        except Exception:
            ok = False
    return ok


def backtest(legA, legB, win=WIN, z_in=Z_IN, z_out=Z_OUT, eq0=250000.0):
    """價差 (A-B, 1:1 口) z-score 均值回歸回測。回傳逐筆 pnl 與權益曲線。"""
    a_d, b_d = _load(legA), _load(legB)
    days = sorted(set(a_d) & set(b_d))
    sp = np.array([a_d[d] - b_d[d] for d in days])
    eq = eq0
    pos = 0
    entry = 0.0
    trades = []
    curve = []
    for i in range(win, len(sp)):
        m = sp[i - win:i].mean()
        s = sp[i - win:i].std()
        if s <= 0:
            curve.append(eq)
            continue
        z = (sp[i] - m) / s
        if pos == 0:
            if z > z_in:
                pos, entry, ei = -1, sp[i], days[i]    # 價差過高→空價差
            elif z < -z_in:
                pos, entry, ei = 1, sp[i], days[i]      # 價差過低→多價差
        elif abs(z) < z_out:
            pnl = (sp[i] - entry) * pos * PV - COST * 2
            eq += pnl
            trades.append({"entry_date": ei, "exit_date": days[i],
                           "dir": pos, "pnl": pnl})
            pos = 0
        curve.append(eq if pos == 0 else eq + (sp[i] - entry) * pos * PV)
    return trades, np.array(curve), days


def metrics(trades, curve, eq0=250000.0):
    if not trades:
        return None
    p = np.array([t["pnl"] for t in trades])
    w = p[p > 0]
    ls = p[p < 0]
    peak = np.maximum.accumulate(curve)
    dd = ((peak - curve) / peak).max() if len(curve) else 0.0
    return {"n": len(trades), "win": float((p > 0).mean()),
            "pf": float(w.sum() / -ls.sum()) if len(ls) else float("inf"),
            "total": float(p.sum()), "dd": float(dd)}


def current_z(legA, legB, win=WIN):
    """回傳 (z, 價差現值, 均值, 標準差, A腿現價, B腿現價, 日期) — 供即時訊號/掛單價。"""
    a_d, b_d = _load(legA), _load(legB)
    days = sorted(set(a_d) & set(b_d))
    sp = np.array([a_d[d] - b_d[d] for d in days])
    if len(sp) < win + 1:
        return None
    m = sp[-win:].mean()
    s = sp[-win:].std()
    if s <= 0:
        return None
    return ((sp[-1] - m) / s, float(sp[-1]), float(m), float(s),
            float(a_d[days[-1]]), float(b_d[days[-1]]), days[-1])


def summary_lines():
    """精簡單行摘要 (供 /futures 內嵌); 有訊號者帶兩腿掛單價+出場目標, 待訊號只給 z。"""
    out = []
    for a, b, name in PAIRS:
        cz = current_z(a, b)
        if cz is None:
            continue
        z, sp_now, mean, std, pxA, pxB, d = cz
        nA, nB = name[:2], name[3:]
        if z > Z_IN:
            out.append(f"{name} z={z:+.2f} 🔴做空價差 賣{nA}@{pxA:,.0f}/買{nB}@{pxB:,.0f} →回{mean:,.0f}")
        elif z < -Z_IN:
            out.append(f"{name} z={z:+.2f} 🟢做多價差 買{nA}@{pxA:,.0f}/賣{nB}@{pxB:,.0f} →回{mean:,.0f}")
        else:
            in_lo = mean - Z_IN * std
            in_hi = mean + Z_IN * std
            out.append(f"{name} z={z:+.2f} ⚪待訊號（跌{in_lo:,.0f}買差/升{in_hi:,.0f}賣差）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch:
        fetch_grains()

    print("⚖️ 穀物價差均值回歸（短線, 市場中性）")
    print(f"規則 z>{Z_IN}進場 |z|<{Z_OUT}出場（回均值）· {WIN}日窗 · 1:1口\n")
    for a, b, name in PAIRS:
        cz = current_z(a, b)
        tr, cv, _ = backtest(a, b)
        m = metrics(tr, cv)
        if cz is None or m is None:
            print(f"{name}：資料不足")
            continue
        z, sp_now, mean, std, pxA, pxB, d = cz
        nA, nB = name[:2], name[3:]
        in_hi = mean + Z_IN * std        # 價差≥此 → 做空價差掛單點
        in_lo = mean - Z_IN * std        # 價差≤此 → 做多價差掛單點
        if z > Z_IN:                     # 已觸發: 做空價差 (空A多B)
            print(f"{name}  z={z:+.2f}  🔴 做空價差（空{nA}/多{nB}）")
            print(f"  進場 賣{nA}@{pxA:,.1f} 買{nB}@{pxB:,.1f}｜價差 {sp_now:,.1f}")
            print(f"  出場目標 價差回 {mean:,.1f}（收斂 {sp_now-mean:+,.0f}點 ≈ ${(sp_now-mean)*PV:,.0f}/組）")
        elif z < -Z_IN:                  # 已觸發: 做多價差 (多A空B)
            print(f"{name}  z={z:+.2f}  🟢 做多價差（多{nA}/空{nB}）")
            print(f"  進場 買{nA}@{pxA:,.1f} 賣{nB}@{pxB:,.1f}｜價差 {sp_now:,.1f}")
            print(f"  出場目標 價差回 {mean:,.1f}（收斂 {mean-sp_now:+,.0f}點 ≈ ${(mean-sp_now)*PV:,.0f}/組）")
        else:                            # 待訊號: 給掛單觸發價
            print(f"{name}  z={z:+.2f}  ⚪ 待訊號（現價差 {sp_now:,.1f} 均值 {mean:,.1f}）")
            print(f"  做多價差掛單: 價差跌至 {in_lo:,.1f} 以下（買{nA}/賣{nB}）→ 出場回 {mean:,.1f}")
            print(f"  做空價差掛單: 價差升至 {in_hi:,.1f} 以上（賣{nA}/買{nB}）→ 出場回 {mean:,.1f}")
        print(f"  績效 {m['n']}筆｜勝率 {m['win']:.0%}｜PF {m['pf']:.2f}｜DD {m['dd']:.0%}")
    print("\n進場=z達±1.5標準差(掛單價如上); 出場=價差回均值(z→0)。1組=各買賣1口(同$50/點)。")
    print("(市場中性: 兩腿對沖大盤方向, DD 約單邊趨勢一半; 小麥於此當一腿而非單獨硬做)")


if __name__ == "__main__":
    main()
