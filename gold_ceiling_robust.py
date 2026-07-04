"""ATR 天花板 (atr_ceiling) 的『過擬合』專項檢驗 —— 結論: 否決 (對弱基準的相對改善)。

問題: gold_dd_levers 顯示日線加 ATR 天花板降DD (1.5×)。這是真 edge 還是過擬合? 五個角度拆:
  (1) 2D 平台: ceiling 倍數 × 中位數窗 全網格。
  (2) 逐事件貢獻: 天花板擋掉哪幾筆? DD 改善集中在單一崩盤事件(脆弱)還是分散。
  (3) 前後半樣本分離: 2000-2012 vs 2013-2026 是否都改善。
  (4) 小時線再確認: 生產時間軸上是否無益。
  (5) *** base 配置敏感度 (決定性) ***: 前四關 (1)~(3) 都用【小時參數 bo24/atr2.5 硬套日線】
      的弱base量的; 一旦換到日線【實際採用配置 bo40/atr4/ATR20】(本就用寬停損+長突破避開洗損),
      天花板的『增益』整個蒸發、只剩砍利。→ 那個增益是『相對弱基準』的假象 = 過擬合。故否決。
"""
import numpy as np

import gold_strategies as gs
import gold_daily_backtest as gd


def med_atr(a, win):
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        seg = a[max(0, i - win + 1):i + 1]; seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def bt(dt, h, l, c, a, ceiling=None, med=None, bo=None, st=None, an=None, daily=False):
    cfg = gs.CONFIG
    bo = bo or cfg["breakout"]; st = st or cfg["atr_stop"]; an = an or cfg["atr_n"]
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; tr = []
    for i in range(bo + an + 1, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            if daily:
                sig = 1 if px > h[i - bo:i].max() else None
            else:
                sig = gs.signal_breakout(h, l, c, i, cfg, a)
            if sig != 1:
                continue
            filtered = (ceiling is not None and med is not None and not np.isnan(med[i])
                        and a[i] > ceiling * med[i])
            if filtered:
                continue
            pos, entry, ei, trail = 1, px, i, px - st * a[i]
        else:
            trail = max(trail, px - st * a[i])
            if px <= trail:
                tr.append({"pnl": (px - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                           "entry_date": dt[ei], "exit_date": dt[i]})
                pos = 0
    if pos:
        tr.append({"pnl": (c[-1] - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                   "entry_date": dt[ei], "exit_date": dt[-1]})
    return tr


def M(tr, lo=None, hi=None):
    if lo or hi:
        tr = [t for t in tr if (not lo or t["exit_date"] >= lo) and (not hi or t["exit_date"] < hi)]
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr]); eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w = p[p > 0]; ls = p[p < 0]
    return dict(n=len(tr), win=float((p > 0).mean()),
                pf=float(w.sum() / -ls.sum()) if len(ls) else float("inf"),
                tot=float(p.sum()), dd=dd, mar=float(eq[-1] / dd) if dd > 0 else float("inf"),
                sharpe=float(p.mean() / p.std()) if p.std() > 0 else 0.0)


def trade_keys(tr):
    return {(t["entry_date"], t["exit_date"]) for t in tr}


def main():
    ddt, do, dh, dl, dc = gd.load()
    da = gs.atr(dh, dl, dc, gd.DATR_N)
    base = bt(ddt, dh, dl, dc, da, daily=True)
    mb = M(base)
    print(f"日線26年基準 (無天花板): {mb['n']}筆 PF{mb['pf']:.2f} 總${mb['tot']:+,.0f} "
          f"DD${mb['dd']:,.0f} 報酬/DD{mb['mar']:.2f} Sharpe{mb['sharpe']:.3f}\n")

    # ── (1) 2D 平台掃描 ──────────────────────────────────────────
    print("(1) 2D 平台掃描 — 報酬/回撤比 (DD$) ; 找平台 vs 單點尖峰")
    wins = [100, 150, 200, 250, 300]
    ceils = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    hdr = "ceil\\win"
    print(f"{hdr:<9}" + "".join(f"{w:>13}" for w in wins))
    for cl in ceils:
        cells = []
        for w in wins:
            m = M(bt(ddt, dh, dl, dc, da, cl, med_atr(da, w), daily=True))
            cells.append(f"{m['mar']:.2f}({m['dd']/1000:.0f}k)")
        print(f"{cl:<9}" + "".join(f"{x:>13}" for x in cells))

    # ── (2) 逐事件貢獻: 天花板擋掉哪些筆, 改善集中度 ──────────────
    print("\n(2) 逐事件拆解 (win=200, ceil=1.5) — 天花板『擋掉』的進場與其若進場的損益")
    med200 = med_atr(da, 200)
    filt = bt(ddt, dh, dl, dc, da, 1.5, med200, daily=True)
    blocked = trade_keys(base) - trade_keys(filt)
    # 找出基準中被擋掉的那些筆 (近似: base 有、filt 無的 entry_date)
    base_by_entry = {t["entry_date"]: t for t in base}
    filt_entries = {t["entry_date"] for t in filt}
    blocked_trades = [t for e, t in base_by_entry.items() if e not in filt_entries]
    blocked_trades.sort(key=lambda t: t["pnl"])
    print(f"  基準{len(base)}筆 → 加天花板{len(filt)}筆; 受影響進場 {len(blocked_trades)} 筆")
    print(f"  {'進場日':<12}{'出場日':<12}{'該筆損益$':>12}")
    for t in blocked_trades:
        print(f"  {t['entry_date']:<12}{t['exit_date']:<12}{t['pnl']:>+12,.0f}")
    tot_blocked = sum(t["pnl"] for t in blocked_trades)
    print(f"  被擋筆合計損益 ${tot_blocked:+,.0f} (負值=天花板幫你避開的淨虧)")

    # ── (3) 前後半樣本分離 ──────────────────────────────────────
    print("\n(3) 前後半子期一致性 (跨子期都改善=穩健, 只一邊=靠單一事件)")
    for lo, hi, lbl in [(None, "2013-01-01", "前半 2000-2012"),
                        ("2013-01-01", None, "後半 2013-2026")]:
        b = M(base, lo, hi)
        f15 = M(bt(ddt, dh, dl, dc, da, 1.5, med200, daily=True), lo, hi)
        f20 = M(bt(ddt, dh, dl, dc, da, 2.0, med200, daily=True), lo, hi)
        print(f"  {lbl}:")
        for tag, m in [("無天花板", b), ("ceil1.5", f15), ("ceil2.0", f20)]:
            if m:
                print(f"    {tag:<10} {m['n']:>3}筆 PF{m['pf']:>5.2f} 總${m['tot']:>+10,.0f} "
                      f"DD${m['dd']:>9,.0f} 報酬/DD{m['mar']:>6.2f}")

    # ── (4) 小時線再確認 (誠實面對生產時間軸) ────────────────────
    print("\n(4) 小時線 (生產時間軸) 再確認 — 是否真的無益/略傷")
    hdt, ho, hh_, hl_, hc = gs.load_bars()
    ha = gs.atr(hh_, hl_, hc, gs.CONFIG["atr_n"])
    hbase = M(bt(hdt, hh_, hl_, hc, ha))
    print(f"  {'設定':<12}{'筆':>5}{'PF':>6}{'總$':>11}{'DD$':>9}{'報酬/DD':>8}   OOS26")
    def oos(tr):
        return sum(t["pnl"] for t in tr if t["exit_date"][:4] >= "2026")
    tb = bt(hdt, hh_, hl_, hc, ha)
    print(f"  {'無天花板':<12}{hbase['n']:>5}{hbase['pf']:>6.2f}{hbase['tot']:>+11,.0f}"
          f"{hbase['dd']:>9,.0f}{hbase['mar']:>8.2f}   {oos(tb)/1000:+.0f}k")
    for cl in (1.5, 2.0, 2.5):
        tr = bt(hdt, hh_, hl_, hc, ha, cl, med_atr(ha, 200))
        m = M(tr)
        print(f"  {'ceil'+str(cl):<12}{m['n']:>5}{m['pf']:>6.2f}{m['tot']:>+11,.0f}"
              f"{m['dd']:>9,.0f}{m['mar']:>8.2f}   {oos(tr)/1000:+.0f}k")

    # ── (5) base 配置敏感度 — 決定性: 增益是否只存在於弱base ─────────────
    print("\n(5) *** base 敏感度 (決定性) *** — 同一天花板在弱base vs 採用配置的差異")
    med200d = med_atr(da, 200)
    configs = [("弱base bo24/atr2.5/ATR14 (先前(1)~(3)用)", dict(bo=24, st=2.5, an=14)),
               ("採用配置 bo40/atr4.0/ATR20 (日線實際跑的)", dict(bo=40, st=4.0, an=20))]
    for name, kw in configs:
        da_c = gs.atr(dh, dl, dc, kw["an"])
        med_c = med_atr(da_c, 200)
        b0 = M(bt(ddt, dh, dl, dc, da_c, daily=True, **kw))
        c15 = M(bt(ddt, dh, dl, dc, da_c, ceiling=1.5, med=med_c, daily=True, **kw))
        print(f"  {name}")
        print(f"    無天花板 : PF{b0['pf']:.2f} 總${b0['tot']:+,.0f} DD${b0['dd']:,.0f} 報酬/DD{b0['mar']:.2f}")
        print(f"    +ceil1.5 : PF{c15['pf']:.2f} 總${c15['tot']:+,.0f} DD${c15['dd']:,.0f} 報酬/DD{c15['mar']:.2f}"
              f"  ΔDD{c15['dd']-b0['dd']:+,.0f} Δ總{c15['tot']-b0['tot']:+,.0f}")
    print("  → 弱base上天花板漂亮 (ΔDD大降); 採用配置上ΔDD≈0且Δ總為負(砍利)。")
    print("    增益隨base強度消失 = 對弱基準的相對改善 = 過擬合。結論: 否決, 生產維持 atr_ceiling=None。")


if __name__ == "__main__":
    main()
