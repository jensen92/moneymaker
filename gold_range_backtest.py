"""黃金期貨 (GC=F COMEX) 小時 K 線 — 區間逆勢「停損反手」策略回測.

策略 (依需求):
  1. 用最近的『兩個擺動高點』與『兩個擺動低點』界定一個水平區間 [support, resistance]。
  2. 價格觸及區間上緣 → 放空; 觸及下緣 → 做多 (區間逆勢)。
  3. 停損: 突破進場那一側的緣口 (緩衝 buffer) → 停損並『反手』(空翻多 / 多翻空)。
  4. 獲利了結: 觸及區間另一側 (對向緣口)。

合約規格: GC 1 口 = 100 金衡盎司, 報價每變動 US$1 = US$100 損益。
資料: Yahoo 連續近月合約 1 小時線 (約 2.4 年, 樣本受 Yahoo 1h 上限所限)。
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "futures_data", "GC_60m.csv")

POINT_VALUE = 100.0     # US$ / 每 1.0 報價變動 / 口
COST_PER_SIDE = 0.30    # 滑價+手續費 估計 (報價單位), 進出各一次

CFG = {
    "pivot_w":      6,     # 擺動點確認窗 (左右各 w 根)
    "range_tol":    0.004, # 兩高/兩低叢集容忍 (相對價), 確認為同一水平
    "min_width":    0.006, # 區間最小寬度 (相對價), 過窄不交易
    "max_width":    0.05,  # 區間最大寬度, 過寬視為趨勢非震盪
    "touch_band":   0.0,   # 觸緣判定緩衝 (0 = 收盤越過緣口)
    "stop_buf":     0.003, # 停損緩衝: 突破緣口超過此比例才停損反手
}


def load_bars(path):
    o, h, l, c = [], [], [], []
    dt = []
    with open(path) as f:
        for row in csv.DictReader(f):
            dt.append(row["date"])
            o.append(float(row["open"])); h.append(float(row["high"]))
            l.append(float(row["low"]));  c.append(float(row["close"]))
    return dt, np.array(o), np.array(h), np.array(l), np.array(c)


def find_range(h, l, i, cfg):
    """用 i 之前『已確認』的擺動高/低點界定區間。回傳 (support, resistance) 或 None。
    取最近兩個叢集在同一水平 (容忍 range_tol) 的擺動高 → 阻力; 同理擺動低 → 支撐。
    只用 j <= i-w 的點 (右窗已過), 不看未來。"""
    w = cfg["pivot_w"]
    lo_i = max(w, i - 200)
    hi_pts, lo_pts = [], []
    for j in range(i - w, lo_i - 1, -1):     # 由近到遠
        if h[j] == h[j - w:j + w + 1].max():
            hi_pts.append(h[j])
        if l[j] == l[j - w:j + w + 1].min():
            lo_pts.append(l[j])
        if len(hi_pts) >= 4 and len(lo_pts) >= 4:
            break
    if len(hi_pts) < 2 or len(lo_pts) < 2:
        return None
    # 最近兩個擺動高須相近 (同一阻力水平) → 阻力取其均值; 支撐同理
    res = (hi_pts[0] + hi_pts[1]) / 2
    if abs(hi_pts[0] - hi_pts[1]) / res > cfg["range_tol"] * 4:
        res = max(hi_pts[0], hi_pts[1])      # 不相近則取較高者 (保守阻力)
    sup = (lo_pts[0] + lo_pts[1]) / 2
    if abs(lo_pts[0] - lo_pts[1]) / sup > cfg["range_tol"] * 4:
        sup = min(lo_pts[0], lo_pts[1])
    if res <= sup:
        return None
    width = (res - sup) / sup
    if width < cfg["min_width"] or width > cfg["max_width"]:
        return None
    return sup, res


def backtest(allow_reverse=True):
    """allow_reverse=True: 停損後立即反手 (依需求原意, 沿用同一區間)。
    allow_reverse=False: 停損後出場、等待重新形成新區間才再進場 (突破即離場)。"""
    dt, o, h, l, c = load_bars(CSV)
    n = len(c)
    cfg = CFG
    pos = 0            # +1 long, -1 short, 0 flat
    entry = 0.0
    sup = res = 0.0
    trades = []

    def close_trade(exit_px, j, reason):
        nonlocal pos
        pnl = (exit_px - entry) * pos * POINT_VALUE - COST_PER_SIDE * POINT_VALUE * 2
        trades.append({"dir": pos, "entry": entry, "exit": exit_px,
                       "pnl": pnl, "exit_i": j, "reason": reason})
        pos = 0

    for i in range(30, n):
        price = c[i]
        if pos == 0:
            rng = find_range(h, l, i, cfg)
            if rng is None:
                continue
            sup, res = rng
            if price >= res * (1 + cfg["touch_band"]):
                pos, entry = -1, price            # 上緣放空
            elif price <= sup * (1 - cfg["touch_band"]):
                pos, entry = 1, price             # 下緣做多
            continue
        # 持倉中
        if pos == -1:                              # 空單: 上停損 / 下獲利
            if price >= res * (1 + cfg["stop_buf"]):
                close_trade(price, i, "stop")
                if allow_reverse:
                    pos, entry = 1, price          # 反手做多 (沿用同區間)
            elif price <= sup:
                close_trade(price, i, "target")
                if allow_reverse:
                    pos, entry = 1, price          # 到下緣翻多
        elif pos == 1:                             # 多單: 下停損 / 上獲利
            if price <= sup * (1 - cfg["stop_buf"]):
                close_trade(price, i, "stop")
                if allow_reverse:
                    pos, entry = -1, price         # 反手放空
            elif price >= res:
                close_trade(price, i, "target")
                if allow_reverse:
                    pos, entry = -1, price         # 到上緣翻空
    if pos != 0:
        close_trade(c[-1], n - 1, "eod")

    return trades, dt


def split_dir(trades):
    lp = sum(t["pnl"] for t in trades if t["dir"] == 1)
    sp = sum(t["pnl"] for t in trades if t["dir"] == -1)
    ln = sum(1 for t in trades if t["dir"] == 1)
    sn = sum(1 for t in trades if t["dir"] == -1)
    return (ln, lp), (sn, sp)


def metrics(trades):
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    gp = wins.sum(); gl = -losses.sum()
    eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq).max() if len(eq) else 0.0
    return {
        "n": len(trades),
        "win": len(wins) / len(trades),
        "pf": gp / gl if gl > 0 else float("inf"),
        "total": pnls.sum(),
        "avg": pnls.mean(),
        "maxdd": dd,
        "mar": (eq[-1] / dd) if dd > 0 else float("inf"),
    }


def report(label, trades):
    m = metrics(trades)
    if m is None:
        print(f"{label}: 無交易"); return
    by = {}
    for t in trades:
        by[t["reason"]] = by.get(t["reason"], 0) + 1
    print(f"{label:<22}{m['n']:>5}{m['win']:>8.1%}{m['pf']:>7.2f}"
          f"{m['total']:>+11,.0f}{m['avg']:>+9,.0f}{m['maxdd']:>10,.0f}{m['mar']:>7.2f}"
          f"   出場:{by}")


def main():
    dt, o, h, l, c = load_bars(CSV)
    print(f"GC 小時線: {len(c)} 根, {dt[0]} ~ {dt[-1]}")
    print(f"報價區間: {c.min():.0f} ~ {c.max():.0f}  (買進持有 1 口損益 "
          f"${(c[-1]-c[0])*POINT_VALUE:+,.0f})\n")
    print(f"{'策略':<22}{'交易':>5}{'勝率':>8}{'PF':>7}{'總損益$':>11}"
          f"{'均$':>9}{'最大DD$':>10}{'MAR':>7}")
    t_rev, _ = backtest(allow_reverse=True)
    report("區間+停損反手", t_rev)
    t_norev, _ = backtest(allow_reverse=False)
    report("區間(停損出場待新區間)", t_norev)

    print("\n多空拆解 (停損反手版):")
    (ln, lp), (sn, sp) = split_dir(t_rev)
    print(f"  做多: {ln:>6} 筆  損益 ${lp:>+12,.0f}")
    print(f"  做空: {sn:>6} 筆  損益 ${sp:>+12,.0f}")


if __name__ == "__main__":
    main()
