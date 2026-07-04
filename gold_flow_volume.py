"""資金流向研究 (1/3): 成交量確認 — 突破配量 vs 縮量假突破, 能否提升績效。

假設: 真突破由『資金進場』推動 → 突破當根成交量應放大; 縮量突破多為假突破/流動性陷阱。
做法: 在既有 24H 突破進場條件外, 多要求『突破當根 volume ≥ k × 近 N 根均量』才進場。
資料: GC_60m.csv / GC_1d.csv 皆已含 volume 欄 (無需外抓)。

四關紀律 (同 gold_hd_study/gold_ceiling_robust):
  - 小時線 IS(2024-25)/OOS(2026) 分離, OOS 不得惡化;
  - 日線 26 年跨 regime;
  - base 敏感度: 同時測【採用配置】與【弱base】, 增益不得隨 base 強度消失 (否則=過擬合)。
"""
import csv
import numpy as np

import gold_strategies as gs
import gold_daily_backtest as gd


def load_with_vol(path):
    dt, o, h, l, c, v = [], [], [], [], [], []
    for r in csv.DictReader(open(path)):
        ts = r["date"]
        if path.endswith("60m.csv") and not ts.endswith(":00"):
            continue
        dt.append(ts); o.append(float(r["open"])); h.append(float(r["high"]))
        l.append(float(r["low"])); c.append(float(r["close"])); v.append(float(r["volume"]))
    return dt, np.array(o), np.array(h), np.array(l), np.array(c), np.array(v)


def sma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(x); out[n - 1] = cs[n - 1] / n; out[n:] = (cs[n:] - cs[:-n]) / n
    return out


def bt(dt, h, l, c, v, *, vol_k=None, vol_n=20, bo=None, st=None, an=None, daily=False):
    cfg = gs.CONFIG
    bo = bo or cfg["breakout"]; st = st or cfg["atr_stop"]; an = an or cfg["atr_n"]
    a = gs.atr(h, l, c, an)
    vma = sma(v, vol_n) if vol_k else None
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; tr = []
    for i in range(bo + an + 1, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            sig = (1 if px > h[i - bo:i].max() else None) if daily \
                else gs.signal_breakout(h, l, c, i, cfg, a)
            if sig != 1:
                continue
            if vol_k and vma is not None and not np.isnan(vma[i]) and vma[i] > 0 \
               and v[i] < vol_k * vma[i]:        # 縮量突破: 不進場 (均量為0/缺值則放行)
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


def M(tr):
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr]); eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()); w = p[p > 0]; ls = p[p < 0]
    return dict(n=len(tr), win=float((p > 0).mean()),
                pf=float(w.sum() / -ls.sum()) if len(ls) else 9.99,
                tot=float(p.sum()), dd=dd, mar=float(eq[-1] / dd) if dd > 0 else 0,
                sharpe=float(p.mean() / p.std()) if p.std() > 0 else 0)


def oos(tr, y="2026"):
    return float(sum(t["pnl"] for t in tr if t["exit_date"][:4] >= y))


def row(lbl, m, extra=""):
    if not m:
        print(f"{lbl:<20} 無交易"); return
    print(f"{lbl:<20}{m['n']:>5}{m['win']:>7.1%}{m['pf']:>6.2f}{m['tot']:>+11,.0f}"
          f"{m['dd']:>10,.0f}{m['mar']:>7.2f}{m['sharpe']:>7.3f}{extra}")


HDR = f"{'':<20}{'筆':>5}{'勝率':>7}{'PF':>6}{'總損益$':>11}{'最大DD$':>10}{'報酬/DD':>7}{'Sharpe':>7}"


def main():
    print("═══ 成交量確認研究 (突破配量 ≥ k×近20均量 才進場) ═══\n")
    hdt, ho, hh, hl, hc, hv = load_with_vol(gs.CSV)
    print("關1+2 小時線 (IS=2024-25 OOS=2026):"); print(HDR + "   OOS26")
    base = bt(hdt, hh, hl, hc, hv)
    row("無濾網(基準)", M(base), f"  {oos(base)/1000:+.0f}k")
    for k in (0.8, 1.0, 1.2, 1.5, 2.0):
        tr = bt(hdt, hh, hl, hc, hv, vol_k=k)
        row(f"配量 ≥{k}×", M(tr), f"  {oos(tr)/1000:+.0f}k")

    ddt, do, dh, dl, dc, dv = load_with_vol(gd.CSV_1D)
    print("\n關3 日線26年 (base敏感度: 採用配置 vs 弱base):")
    for name, kw in [("採用 bo40/atr4/ATR20", dict(bo=40, st=4.0, an=20)),
                     ("弱base bo24/atr2.5/ATR14", dict(bo=24, st=2.5, an=14))]:
        print(f"  【{name}】"); print("  " + HDR)
        b = bt(ddt, dh, dl, dc, dv, daily=True, **kw)
        row("  無濾網", M(b))
        for k in (1.0, 1.2, 1.5):
            row(f"  配量 ≥{k}×", M(bt(ddt, dh, dl, dc, dv, vol_k=k, daily=True, **kw)))


if __name__ == "__main__":
    main()
