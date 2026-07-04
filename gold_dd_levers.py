"""兩個以『降DD』為目標的正交槓桿, 一開始就內建日線26年關卡 (避免只在近期regime過擬合)。

DD 在趨勢跟隨策略主要來自:『同一失敗突破區被連續洗損』與『高波動爆量棒進場後急殺』。
對應兩個不動核心進出場價的正交過濾:
  L1 停損後冷卻 (cooldown): 剛被停損出場後, 未來 k 根內不再進場 —— 避開同一區反覆洗損。
  L2 ATR 天花板 (vol-ceiling): ATR 相對其自身中位數過高 (爆量/新聞尖峰) 時不進場 ——
     這些棒進場後最易急殺, 是大$虧損來源。

每個槓桿同時看: 小時線 (含IS/OOS分離) + 日線26年跨regime。兩處皆需不惡化才採用。
"""
import numpy as np

import gold_strategies as gs
import gold_daily_backtest as gd


def med_atr(a, win=200):
    """ATR 的滾動中位數 (作為『正常波動』基準, 供 L2 相對判斷)。"""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        lo = max(0, i - win + 1)
        seg = a[lo:i + 1]; seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def engine(dt, h, l, c, a, *, cooldown=0, vol_ceiling=None, med=None,
           bo=None, st=None, an=None, daily=False):
    cfg = gs.CONFIG
    bo = bo or cfg["breakout"]; st = st or cfg["atr_stop"]; an = an or cfg["atr_n"]
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; cd = 0; tr = []
    start = bo + an + 1
    for i in range(start, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            if cd > 0:
                cd -= 1
                continue
            if daily:
                sig = 1 if px > h[i - bo:i].max() else None
            else:
                sig = gs.signal_breakout(h, l, c, i, cfg, a)
            if sig != 1:
                continue
            if vol_ceiling is not None and med is not None and not np.isnan(med[i]) \
               and a[i] > vol_ceiling * med[i]:
                continue
            pos, entry, ei, trail = 1, px, i, px - st * a[i]
        else:
            trail = max(trail, px - st * a[i])
            if px <= trail:
                tr.append({"pnl": (px - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                           "exit_date": dt[i]})
                pos = 0; cd = cooldown
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


def row(label, mm, extra=""):
    if not mm:
        print(f"{label:<22} 無交易"); return
    print(f"{label:<22}{mm['n']:>5}{mm['win']:>7.1%}{mm['pf']:>6.2f}{mm['tot']:>+11,.0f}"
          f"{mm['dd']:>10,.0f}{mm['mar']:>7.2f}{mm['sharpe']:>7.3f}{extra}")


HDR = f"{'':<22}{'筆':>5}{'勝率':>7}{'PF':>6}{'總損益$':>11}{'最大DD$':>10}{'報酬/DD':>7}{'Sharpe':>7}"


def main():
    dt, o, h, l, c = gs.load_bars()
    a = gs.atr(h, l, c, gs.CONFIG["atr_n"])
    med = med_atr(a)
    ddt, do, dh, dl, dc = gd.load()
    da = gs.atr(dh, dl, dc, gd.DATR_N)
    dmed = med_atr(da, win=250)

    print("═══ L1 停損後冷卻 (避開同區連續洗損) ═══")
    print("小時線:"); print(HDR + "   OOS26")
    row("無冷卻(基準)", m(engine(dt, h, l, c, a)), f"  {oos(engine(dt,h,l,c,a))/1000:+.0f}k")
    for k in (3, 6, 12, 24):
        tr = engine(dt, h, l, c, a, cooldown=k)
        row(f"冷卻{k}根", m(tr), f"  {oos(tr)/1000:+.0f}k")
    print("日線26年:"); print(HDR)
    row("無冷卻(基準日線)", m(engine(ddt, dh, dl, dc, da, daily=True)))
    for k in (2, 3, 5, 10):
        row(f"冷卻{k}日", m(engine(ddt, dh, dl, dc, da, cooldown=k, daily=True)))

    print("\n═══ L2 ATR 天花板 (爆量/新聞尖峰不進場) ═══")
    print("小時線 (ATR > 倍數×其200根中位數 則跳過):"); print(HDR + "   OOS26")
    row("無上限(基準)", m(engine(dt, h, l, c, a)), f"  {oos(engine(dt,h,l,c,a))/1000:+.0f}k")
    for v in (1.5, 2.0, 2.5, 3.0):
        tr = engine(dt, h, l, c, a, vol_ceiling=v, med=med)
        row(f"ATR上限 {v}×中位", m(tr), f"  {oos(tr)/1000:+.0f}k")
    print("日線26年:"); print(HDR)
    row("無上限(基準日線)", m(engine(ddt, dh, dl, dc, da, daily=True)))
    for v in (1.5, 2.0, 2.5, 3.0):
        row(f"ATR上限 {v}×中位", m(engine(ddt, dh, dl, dc, da, vol_ceiling=v, med=dmed, daily=True)))


if __name__ == "__main__":
    main()
