"""慢速趨勢閘 (regime gate) 的抗過擬合驗證 — 決定性的第4關: 日線26年跨regime。

gold_hd_study 顯示: 在進場條件外加『收盤 > 慢速MA』的趨勢閘, 一致提升 PF/勝率/Sharpe
並(在部分MA長度)降低DD。但 MA150 在小時線是單點尖峰, 鄰居不確認 —— 需驗證:
  (A) 小時線細掃 MA 長度, 找『平台』而非尖峰;
  (B) 日線26年 (含2013-15空頭、2021-22盤整) 套同一趨勢閘概念, 看是否
      跨regime一致降DD/提risk-reward —— 這是核心紀律的第4關。
若兩關皆過, 才算穩健正交增強。
"""
import numpy as np

import gold_strategies as gs
import gold_daily_backtest as gd


def sma(c, n):
    out = np.full(len(c), np.nan)
    if len(c) >= n:
        cs = np.cumsum(c)
        out[n - 1] = cs[n - 1] / n
        out[n:] = (cs[n:] - cs[:-n]) / n
    return out


def bt_hourly(dt, h, l, c, a, ma_n=None):
    """小時線核心策略 + 可選趨勢閘。回傳 (metrics dict, OOS_total)。"""
    cfg = gs.CONFIG; bo = cfg["breakout"]; st = cfg["atr_stop"]
    ma = sma(c, ma_n) if ma_n else None
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; tr = []
    for i in range(bo + cfg["atr_n"] + 1, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            if gs.signal_breakout(h, l, c, i, cfg, a) != 1:
                continue
            if ma is not None and not (px > ma[i]):
                continue
            pos, entry, ei, trail = 1, px, i, px - st * a[i]
        else:
            trail = max(trail, px - st * a[i])
            if px <= trail:
                tr.append({"pnl": (px - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                           "exit_date": dt[i]}); pos = 0
    if pos:
        tr.append({"pnl": (c[-1] - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                   "exit_date": dt[-1]})
    return tr


def bt_daily(dt, h, l, c, ma_n=None, bo=gd.DBO, st=gd.DATR_STOP, an=gd.DATR_N):
    a = gs.atr(h, l, c, an)
    ma = sma(c, ma_n) if ma_n else None
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; tr = []
    for i in range(bo + an + 1, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        hh = h[i - bo:i].max()
        if pos == 0:
            if px > hh and (ma is None or px > ma[i]):
                pos, entry, ei, trail = 1, px, i, px - st * a[i]
        else:
            trail = max(trail, px - st * a[i])
            if px <= trail:
                tr.append({"pnl": (px - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                           "exit_date": dt[i]}); pos = 0
    if pos:
        tr.append({"pnl": (c[-1] - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                   "exit_date": dt[-1]})
    return tr


def m(tr):
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr]); eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w = p[p > 0]; ls = p[p < 0]
    return dict(n=len(tr), win=float((p > 0).mean()),
                pf=float(w.sum() / -ls.sum()) if len(ls) else float("inf"),
                tot=float(p.sum()), dd=dd, mar=float(eq[-1] / dd) if dd > 0 else float("inf"),
                sharpe=float(p.mean() / p.std()) if p.std() > 0 else 0.0)


def oos(tr, y="2026"):
    return float(sum(t["pnl"] for t in tr if t["exit_date"][:4] >= y))


def bearyears(tr, years):
    return {yr: round(sum(t["pnl"] for t in tr if t["exit_date"][:4] == yr)) for yr in years}


def row(label, mm, extra=""):
    if not mm:
        print(f"{label:<20} 無交易"); return
    print(f"{label:<20}{mm['n']:>5}{mm['win']:>7.1%}{mm['pf']:>6.2f}{mm['tot']:>+11,.0f}"
          f"{mm['dd']:>10,.0f}{mm['mar']:>7.2f}{mm['sharpe']:>7.3f}{extra}")


def main():
    print("關A ── 小時線細掃 MA 趨勢閘 (找平台, 非尖峰)")
    dt, o, h, l, c = gs.load_bars()
    a = gs.atr(h, l, c, gs.CONFIG["atr_n"])
    print(f"{'MA長度':<20}{'筆':>5}{'勝率':>7}{'PF':>6}{'總損益$':>11}{'最大DD$':>10}{'報酬/DD':>7}{'Sharpe':>7}   OOS26")
    row("無閘(基準)", m(bt_hourly(dt, h, l, c, a)), f"  {oos(bt_hourly(dt,h,l,c,a))/1000:+.0f}k")
    for mn in (75, 100, 125, 150, 175, 200, 225, 250, 300):
        tr = bt_hourly(dt, h, l, c, a, mn)
        row(f"MA{mn}", m(tr), f"  {oos(tr)/1000:+.0f}k")

    print("\n關B ── 日線26年跨regime套趨勢閘 (決定性; MA長度換算成『日』)")
    ddt, do, dh, dl, dc = gd.load()
    bears = ["2013", "2014", "2015", "2021", "2022"]
    print(f"{'日線MA':<20}{'筆':>5}{'勝率':>7}{'PF':>6}{'總損益$':>11}{'最大DD$':>10}{'報酬/DD':>7}{'Sharpe':>7}")
    row("無閘(基準日線)", m(bt_daily(ddt, dh, dl, dc)))
    for mn in (50, 100, 150, 200):
        tr = bt_daily(ddt, dh, dl, dc, mn)
        row(f"日線MA{mn}", m(tr))
    print("\n  空頭/盤整年逐年損益 (趨勢閘應在此縮小虧損):")
    print(f"  {'年':<6}" + "".join(f"{y:>10}" for y in bears))
    base_by = bearyears(bt_daily(ddt, dh, dl, dc), bears)
    print(f"  {'無閘':<6}" + "".join(f"{base_by[y]:>+10,}" for y in bears))
    for mn in (100, 150, 200):
        by = bearyears(bt_daily(ddt, dh, dl, dc, mn), bears)
        print(f"  {'MA'+str(mn):<6}" + "".join(f"{by[y]:>+10,}" for y in bears))


if __name__ == "__main__":
    main()
