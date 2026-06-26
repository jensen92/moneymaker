"""黃金期貨 (GC=F COMEX) 小時線 — 順勢突破策略 (生產版).

沿革: 原始需求為「兩高兩低定區間, 上緣空/下緣多, 停損反手」(區間逆勢),
經 gold_range_backtest.py / gold_optimize.py 實測證實**結構性負期望**
(毛利近 0, 反手機制空轉吃光成本, 對抗黃金多年上行趨勢):

  V1 區間逆勢+停損反手          總損益 -$787,020   PF 0.83   MAR -0.98
  V3 順勢突破(只做多) 24/3ATR    總損益 +$180,330   PF 1.83   MAR  6.07  ← 採用

故改採方向相反的「順勢突破」: 收盤突破最近 N 小時高點進場做多, ATR 倍數
移動停損出場 (對稱跌破做空在黃金上經驗證為負期望, 預設關閉)。
參數 (breakout=24, atr_stop=3.0) 經整片網格 (breakout 12~96 × atr 2~6,
PF 全部 1.57~2.82) 與逐年 (2024/2025/2026 全正, 含黃金回落的 2026 年)
驗證非單一過擬合尖峰, 見 gold_optimize.py。

資料: futures_data/GC_60m.csv (Yahoo 連續近月合約 1 小時線)。
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "futures_data", "GC_60m.csv")

POINT_VALUE = 100.0   # US$ / 每 1.0 報價變動 / 口 (COMEX GC, 100 oz)
COST_PER_SIDE = 0.30  # 滑價+手續費估計 (報價單位), 進出各一次

# 經 gold_optimize.py 參數網格 + 逐年驗證選定 (見上方說明)
CONFIG = {
    "breakout":    24,    # 突破窗: 收盤 > 過去 N 小時高點 (不含當小時)
    "atr_n":       14,
    "atr_stop":    3.0,   # 移動停損 = 最高收盤後高水位 - atr_stop * ATR
    "allow_short": False, # 對稱做空在黃金上驗證為負期望 (拖累 PF 1.83→1.11), 預設關閉
}


def load_bars(path=CSV):
    o, h, l, c, dt = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            dt.append(row["date"])
            o.append(float(row["open"])); h.append(float(row["high"]))
            l.append(float(row["low"]));  c.append(float(row["close"]))
    return dt, np.array(o), np.array(h), np.array(l), np.array(c)


def atr(h, l, c, n=14):
    """Wilder ATR (簡化遞迴版, 與 gold_optimize.py 一致)。"""
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    a = np.full(len(c), np.nan)
    if len(c) >= n:
        a[n - 1] = tr[:n].mean()
        for i in range(n, len(c)):
            a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def signal_breakout(h, l, c, i, cfg=CONFIG, a=None):
    """順勢突破訊號 — 收盤突破過去 breakout 小時高點 (不含當小時) 即做多.

    對稱跌破做空 (allow_short=True 時) 同邏輯反向。回傳進場方向 (+1/-1/None);
    出場由呼叫端以 ATR 移動停損逐根追蹤 (見 backtest())。
    """
    bo = cfg["breakout"]
    if i < bo + cfg["atr_n"] + 1:
        return None
    if a is None:
        a = atr(h, l, c, cfg["atr_n"])
    if np.isnan(a[i]):
        return None
    hh = h[i - bo:i].max()
    ll = l[i - bo:i].min()
    if c[i] > hh:
        return 1
    if cfg["allow_short"] and c[i] < ll:
        return -1
    return None


def backtest(cfg=CONFIG, csv_path=CSV):
    """單口、無加碼的順勢突破回測 (ATR 移動停損), 回傳逐筆交易清單。"""
    dt, o, h, l, c = load_bars(csv_path)
    n = len(c)
    a = atr(h, l, c, cfg["atr_n"])
    pos = 0
    entry = 0.0
    trail = 0.0
    trades = []

    def close(exit_px, j):
        nonlocal pos
        pnl = (exit_px - entry) * pos * POINT_VALUE - COST_PER_SIDE * POINT_VALUE * 2
        trades.append({"dir": pos, "entry": entry, "exit": exit_px,
                       "pnl": pnl, "entry_date": dt[entry_i], "exit_date": dt[j]})
        pos = 0

    entry_i = 0
    for i in range(cfg["breakout"] + cfg["atr_n"] + 1, n):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            sig = signal_breakout(h, l, c, i, cfg, a)
            if sig == 1:
                pos, entry, entry_i = 1, px, i
                trail = px - cfg["atr_stop"] * a[i]
            elif sig == -1:
                pos, entry, entry_i = -1, px, i
                trail = px + cfg["atr_stop"] * a[i]
        elif pos == 1:
            trail = max(trail, px - cfg["atr_stop"] * a[i])
            if px <= trail:
                close(px, i)
        else:
            trail = min(trail, px + cfg["atr_stop"] * a[i])
            if px >= trail:
                close(px, i)
    if pos:
        close(c[-1], n - 1)
    return trades


def metrics(trades):
    if not trades:
        return None
    p = np.array([t["pnl"] for t in trades])
    w = p[p > 0]; ls = p[p < 0]
    gp = w.sum(); gl = -ls.sum()
    eq = np.cumsum(p)
    dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) else 0.0
    return {"n": len(trades), "win": len(w) / len(trades),
            "pf": gp / gl if gl > 0 else float("inf"),
            "total": p.sum(), "dd": dd,
            "mar": (eq[-1] / dd) if dd > 0 else float("inf")}


def main():
    trades = backtest()
    m = metrics(trades)
    print(f"黃金順勢突破 (breakout={CONFIG['breakout']}, atr_stop={CONFIG['atr_stop']}, "
          f"allow_short={CONFIG['allow_short']})")
    print(f"交易 {m['n']}  勝率 {m['win']:.1%}  PF {m['pf']:.2f}  "
          f"總損益 ${m['total']:+,.0f}  最大DD ${m['dd']:,.0f}  MAR {m['mar']:.2f}")


if __name__ == "__main__":
    main()
