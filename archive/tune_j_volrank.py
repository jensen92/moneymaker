"""J 排名實驗: 把相對成交量/成交值納入排名分數, 是否能提升每日選 2 檔的品質
(進場精準度/勝率)? 不改變進場條件本身 (訊號集合不變), 只改變「同一天候選
> PICKS_PER_DAY=2 時優先選誰」的排序公式, 並用相對量值反推出/未出場後的勝率
驗證: 相對量越大, 之後勝率/精準度是否真的更高。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
TRADE_START = pd.Timestamp("2023-06-01")
WINDOW, THRESH = 60, 0.80


def find_rocket_events(df):
    close, high, n = df["close"].values, df["high"].values, len(df)
    ev, i = [], 0
    while i < n - WINDOW:
        if close[i] <= 0:
            i += 1; continue
        fut = high[i + 1:i + 1 + WINDOW]
        if len(fut) == 0:
            i += 1; continue
        rel = fut.argmax(); gain = fut[rel] / close[i] - 1.0
        if gain >= THRESH:
            peak = i + 1 + rel; ev.append((i, peak, gain)); i = peak + 1
        else:
            i += 1
    return ev


def metrics(trades, curve, since=None):
    eq = curve.set_index("date")["equity"].sort_index()
    if since is not None:
        eq = eq[eq.index >= since]
    if len(eq) < 2:
        return None
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, .01)) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    td = pd.DataFrame(trades)
    if since is not None and not td.empty:
        td = td[pd.to_datetime(td["exit_date"]) >= since]
    n = len(td)
    if n:
        w = td["pnl"] > 0
        gl = -td.loc[~w, "pnl"].sum()
        pf = td.loc[w, "pnl"].sum() / gl if gl > 0 else np.inf
        wr = w.mean()
    else:
        pf = wr = 0
    return dict(n=n, wr=wr, pf=pf, cagr=cagr, dd=dd, mar=cagr / dd if dd > 0 else np.nan)


def run_with_rank(data, sigs, regime, regime_tiers, vol_scalars, all_dates, date_idx, rank_fn):
    from collections import defaultdict
    em = defaultdict(list)
    for sd, lst in sigs.items():
        for score, code, i, s in lst:
            df = data[code]
            if i + 1 < len(df):
                new_score = rank_fn(s)
                em[df["date"].iloc[i + 1]].append((new_score, code, i + 1, s))
    return bt.run_sub(data, em, "J", 1.0, 2_000_000,
                       all_dates=all_dates, date_idx=date_idx,
                       regime_tiers=regime_tiers, vol_scalars=vol_scalars)


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    print("收集 J 訊號 ...")
    sigs_raw = bt.collect_signals(data, "J")
    sigs = {d: lst for d, lst in sigs_raw.items() if d in regime}

    # 候選擁擠程度: 有多少天 J 候選數 > 每日上限 2 (排序才有意義)
    n_days = len(sigs)
    n_crowded = sum(1 for lst in sigs.values() if len(lst) > 2)
    n_cand_total = sum(len(lst) for lst in sigs.values())
    print(f"J 候選: 共 {n_days} 個有訊號的交易日, 候選總數 {n_cand_total}, "
          f"候選數>2(排序才有差)的天數 {n_crowded} ({n_crowded/max(n_days,1)*100:.0f}%)\n")

    # 相對量與後續結果的關聯: 用 vol_surge/turnover 分組看 30 日後續漲幅
    rows = []
    for sd, lst in sigs.items():
        for score, code, i, s in lst:
            df = data[code]
            if i + 31 >= len(df):
                continue
            entry_px = df["close"].iloc[i + 1] if i + 1 < len(df) else None
            fut_high = df["high"].iloc[i + 1:i + 31].max()
            ret30 = fut_high / df["close"].iloc[i] - 1.0
            rows.append((s["vol_surge"], s["turnover"], ret30))
    rdf = pd.DataFrame(rows, columns=["vol_surge", "turnover", "ret30"])
    print("相對量 vs 後續30日最高漲幅 (相關性分析):")
    print(f"  vol_surge 與 ret30 相關係數: {rdf['vol_surge'].corr(rdf['ret30']):.3f}")
    print(f"  turnover  與 ret30 相關係數: {rdf['turnover'].corr(rdf['ret30']):.3f}")
    for q in [0.0, 0.25, 0.5, 0.75]:
        lo, hi = rdf["vol_surge"].quantile(q), rdf["vol_surge"].quantile(q + 0.25)
        g = rdf[(rdf["vol_surge"] >= lo) & (rdf["vol_surge"] < hi)]
        print(f"  vol_surge [{lo:.1f}~{hi:.1f}) 分位: n={len(g):>4}  均ret30={g['ret30'].mean()*100:>6.1f}%")
    print()

    # 用不同排名公式重跑引擎, 比較選股品質
    rockets = []
    for code, df in data.items():
        if df["date"].iloc[-1] < TRADE_START:
            continue
        for s_i, p_i, gain in find_rocket_events(df):
            sd, pd_ = df["date"].iloc[s_i], df["date"].iloc[p_i]
            if pd_ >= TRADE_START:
                rockets.append((code, sd, pd_, gain))
    n_rk = len(rockets)
    rk_by_code = {}
    for code, sd, pd_, gain in rockets:
        rk_by_code.setdefault(code, []).append((sd, pd_))

    rank_variants = {
        "R0 現行(rs+vol_surge/100)": lambda s: s["score"],
        "R1 純相對量(vol_surge)":    lambda s: s["vol_surge"],
        "R2 純成交值(turnover)":     lambda s: s["turnover"],
        "R3 量x值(log turnover x vol_surge)": lambda s: s["vol_surge"] * np.log1p(s["turnover"]),
        "R4 rs+量重權(vol_surge/5)": lambda s: s["score"] - min(s["vol_surge"], 20) / 100.0 + min(s["vol_surge"], 20) / 5.0,
    }
    hdr = f"  {'排名公式':<30}{'捕捉率':>8}{'精準':>7}{'交易':>6}{'勝率':>6}{'PF':>6}{'CAGR':>7}{'DD':>6}{'MAR':>6}"
    print(hdr)
    for label, fn in rank_variants.items():
        trades, curve = run_with_rank(data, sigs, regime, regime_tiers, vol_scalars,
                                       all_dates, date_idx, fn)
        tdf = pd.DataFrame(trades)
        tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
        tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])
        caught = 0
        for code, sd, pd_, gain in rockets:
            ct = tdf[(tdf["code"] == code)
                     & (tdf["entry_date"] >= sd - pd.Timedelta(days=30))
                     & (tdf["entry_date"] <= pd_)]
            if len(ct):
                caught += 1
        tdf3 = tdf[tdf["exit_date"] >= TRADE_START]
        hits = 0
        for _, t in tdf3.iterrows():
            for sd, pd_ in rk_by_code.get(t["code"], []):
                if sd - pd.Timedelta(days=30) <= t["entry_date"] <= pd_:
                    hits += 1; break
        prec = hits / len(tdf3) if len(tdf3) else 0
        m = metrics(trades, curve)
        print(f"  {label:<30}{caught/n_rk*100:>6.1f}% {prec*100:>5.0f}%"
              f"{m['n']:>6}{m['wr']*100:>5.0f}%{m['pf']:>6.2f}"
              f"{m['cagr']*100:>6.1f}%{m['dd']*100:>5.1f}%{m['mar']:>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
