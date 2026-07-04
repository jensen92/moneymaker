"""黃金順勢突破『高維度』增強研究 — 在不動核心進出場參數的前提下降低DD、提高風險報酬比。

背景: gold_strategies 的核心 (breakout=24 / atr_stop=2.5 / atr_n=14, 只做多) 已由
walk-forward + 樣本外 + 日線26年『四關』驗證為穩健最優, 且 MA同期過濾 / 保本停損 /
突破margin 三個增強假設皆已否決 (見 gold_strategies docstring)。故本研究不再碰這三個
核心參數, 改探索**正交的新維度** —— 這些維度改變的是『下多少單 / 何時允許進場 / 如何了結』,
而非『何價進、何價出』, 因此與既有結論不衝突:

  D1 波動率目標部位 (vol-targeting): 部位大小 ∝ 目標$風險 / (atr_stop·ATR·PV)。
     高波動棒縮口、低波動棒放口, 均衡每筆$風險 —— 直攻DD (大賠多發生在高波動段)。
  D2 高階時間框架趨勢閘 (regime gate): 只在慢速趨勢向上時允許進場 (close > MA(long))。
     與被否決的『同期MA過濾』不同 —— 那個近乎no-op因突破本身已含同期動能;
     這裡用遠慢的 MA (100~250H) 濾掉空頭/盤整regime裡的逆勢突破 (DD主要來源)。
  D3 分段了結 (scale-out): 觸及 +kR 先平半落袋, 其餘照原ATR移動停損。鎖利降DD。
  D4 ADX 趨勢閘: 只在 ADX≥門檻 (真有趨勢) 時進場, 濾掉盤整假突破。

方法學 (沿用 gold_strategies 的抗過擬合紀律):
  - 訓練 2024-2025 / 樣本外 2026 分離評估, 樣本外不得惡化。
  - 每個維度掃鄰近參數看是否為平台 (非單點僥倖)。
  - 主指標: 報酬/回撤比 (total/maxDD) 與 最大DD; 兼看 PF、勝率、Sharpe(逐筆)。

用法: python3 gold_hd_study.py   (需先有 futures_data/GC_60m.csv)
"""
import numpy as np

import gold_strategies as gs

PV = gs.POINT_VALUE
COST = gs.COST_PER_SIDE
CFG = gs.CONFIG


def load():
    dt, o, h, l, c = gs.load_bars()
    a = gs.atr(h, l, c, CFG["atr_n"])
    return dt, o, h, l, c, a


def adx(h, l, c, n=14):
    up = h[1:] - h[:-1]; dn = l[:-1] - l[1:]
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))

    def smooth(x):
        s = np.full(len(x), np.nan); s[n - 1] = x[:n].sum()
        for i in range(n, len(x)):
            s[i] = s[i - 1] - s[i - 1] / n + x[i]
        return s
    atr_ = smooth(tr); pdi = 100 * smooth(plus) / atr_; mdi = 100 * smooth(minus) / atr_
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    out = np.full(len(dx), np.nan)
    valid = ~np.isnan(dx); first = np.argmax(valid) + n
    if first < len(dx):
        out[first] = np.nanmean(dx[first - n + 1:first + 1])
        for i in range(first + 1, len(dx)):
            if not np.isnan(dx[i]):
                out[i] = (out[i - 1] * (n - 1) + dx[i]) / n
    res = np.full(len(c), np.nan); res[1:] = out
    return res


def sma(c, n):
    out = np.full(len(c), np.nan)
    if len(c) >= n:
        cs = np.cumsum(c)
        out[n - 1] = cs[n - 1] / n
        out[n:] = (cs[n:] - cs[:-n]) / n
    return out


def run(dt, o, h, l, c, a, *, vol_target=None, ma_gate=None, adx_gate=None,
        adx_arr=None, scale_out_r=None, ma_arr=None):
    """核心引擎: 完全沿用 gold_strategies 的24H突破/2.5ATR移動停損進出場;
    只疊加正交增強。回傳逐筆 {pnl, size, entry_date, exit_date}。

    vol_target: 每筆目標$風險; 部位 = round(target / (atr_stop·ATR·PV)), 最少1口。None=固定1口。
    ma_gate:    進場過濾, 需 close > ma_arr[i] (慢速趨勢閘)。
    adx_gate:   進場過濾, 需 adx_arr[i] >= adx_gate。
    scale_out_r: 觸及 entry + k·(初始風險) 時平掉半口落袋, 其餘續抱移動停損。
    """
    n = len(c); bo = CFG["breakout"]; st = CFG["atr_stop"]
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; size = 0.0
    init_risk = 0.0; scaled = False; realized_half = 0.0
    tr = []

    def close(px, j):
        nonlocal pos
        pnl = (px - entry) * pos * PV * size - COST * PV * 2 * size + realized_half
        tr.append({"pnl": pnl, "size": size, "entry_date": dt[ei], "exit_date": dt[j]})
        pos = 0

    for i in range(bo + CFG["atr_n"] + 1, n):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            sig = gs.signal_breakout(h, l, c, i, CFG, a)
            if sig != 1:
                continue
            if ma_gate is not None and not (px > ma_arr[i]):
                continue
            if adx_gate is not None and not (adx_arr[i] >= adx_gate):
                continue
            pos, entry, ei = 1, px, i
            trail = px - st * a[i]
            init_risk = st * a[i]
            scaled = False; realized_half = 0.0
            if vol_target is not None:
                size = max(1.0, round(vol_target / (init_risk * PV)))
            else:
                size = 1.0
        elif pos == 1:
            # 分段了結: 先鎖半口
            if scale_out_r is not None and not scaled and px >= entry + scale_out_r * init_risk:
                half = size / 2.0
                realized_half = (px - entry) * PV * half - COST * PV * half
                size -= half
                scaled = True
            trail = max(trail, px - st * a[i])
            if px <= trail:
                close(px, i)
    if pos:
        close(c[-1], n - 1)
    return tr


def metrics(tr, split_year=None):
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr])
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    w = p[p > 0]; ls = p[p < 0]
    sharpe = float(p.mean() / p.std()) if p.std() > 0 else 0.0
    m = {"n": len(tr), "win": float((p > 0).mean()),
         "pf": float(w.sum() / -ls.sum()) if len(ls) else float("inf"),
         "tot": float(p.sum()), "dd": dd,
         "mar": float(eq[-1] / dd) if dd > 0 else float("inf"),
         "sharpe": sharpe}
    if split_year:
        tr_in = [t for t in tr if t["exit_date"][:4] < split_year]
        tr_out = [t for t in tr if t["exit_date"][:4] >= split_year]
        m["in_tot"] = float(sum(t["pnl"] for t in tr_in))
        m["out_tot"] = float(sum(t["pnl"] for t in tr_out))
        m["out_n"] = len(tr_out)
    return m


def line(label, m):
    if not m:
        print(f"{label:<34} 無交易"); return
    extra = ""
    if "out_tot" in m:
        extra = f"  IS{m['in_tot']/1000:>+6.0f}k OOS{m['out_tot']/1000:>+5.0f}k"
    print(f"{label:<34}{m['n']:>5}{m['win']:>7.1%}{m['pf']:>6.2f}"
          f"{m['tot']:>+11,.0f}{m['dd']:>10,.0f}{m['mar']:>7.2f}{m['sharpe']:>7.3f}{extra}")


def header():
    print(f"{'策略/維度':<34}{'筆':>5}{'勝率':>7}{'PF':>6}{'總損益$':>11}"
          f"{'最大DD$':>10}{'報酬/DD':>7}{'Sharpe':>7}  (IS=2024-25 OOS=2026)")


def main():
    dt, o, h, l, c, a = load()
    ad = adx(h, l, c)
    print(f"GC 小時線 {len(c)} 根  {dt[0]} ~ {dt[-1]}\n")

    base = run(dt, o, h, l, c, a)
    print("═══ 基準 (核心不動: 24H突破 / 2.5ATR移動停損 / 只做多 / 固定1口) ═══")
    header(); line("BASE 固定1口", metrics(base, "2026")); print()

    # ── D1 波動率目標部位 ────────────────────────────────────────
    print("═══ D1 波動率目標部位 (均衡每筆$風險; 高波動縮口/低波動放口) ═══")
    print("    掃目標$風險 (=每筆願承受的停損金額); 報酬/DD 與 Sharpe 是重點")
    header()
    for vt in (1500, 2000, 2500, 3000, 4000):
        line(f"D1 vol-target ${vt}", metrics(run(dt, o, h, l, c, a, vol_target=vt), "2026"))
    print()

    # ── D2 高階慢速趨勢閘 ────────────────────────────────────────
    print("═══ D2 慢速趨勢閘 (只在 close>MA(long) 時允許進場; 濾空頭/盤整regime) ═══")
    header()
    for mn in (100, 150, 200, 250):
        line(f"D2 MA{mn} 趨勢閘", metrics(run(dt, o, h, l, c, a, ma_gate=True, ma_arr=sma(c, mn)), "2026"))
    print()

    # ── D3 分段了結 ──────────────────────────────────────────────
    print("═══ D3 分段了結 (+kR 平半口落袋, 其餘照移動停損) ═══")
    header()
    for k in (1.0, 1.5, 2.0, 3.0):
        line(f"D3 scale-out +{k}R", metrics(run(dt, o, h, l, c, a, scale_out_r=k), "2026"))
    print()

    # ── D4 ADX 趨勢閘 ────────────────────────────────────────────
    print("═══ D4 ADX 趨勢閘 (只在 ADX≥門檻 有真趨勢時進場) ═══")
    header()
    for th in (15, 20, 25, 30):
        line(f"D4 ADX≥{th}", metrics(run(dt, o, h, l, c, a, adx_gate=th, adx_arr=ad), "2026"))
    print()

    # ── 組合: 最佳正交層疊加 (由上方結果挑, 這裡先示範 D1+D3) ──
    print("═══ 組合示範 (正交層疊加) ═══")
    header()
    line("D1$2500 + D3+2R", metrics(run(dt, o, h, l, c, a, vol_target=2500, scale_out_r=2.0), "2026"))
    line("D1$2500 + D2 MA200", metrics(run(dt, o, h, l, c, a, vol_target=2500, ma_gate=True, ma_arr=sma(c, 200)), "2026"))


if __name__ == "__main__":
    main()
