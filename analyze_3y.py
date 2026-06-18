"""近三年 (2023-06 ~ 2026-06) 交易紀錄分析 + 優化路線診斷.

針對使用者長期痛點「現金滿手卻沒部位」, 量化:
  1. 報酬/勝率/PF/均R (逐年)
  2. 資金利用率: 每日平均持倉數、有部位天數佔比、最長空手連續天數
  3. 持有天數分布 (太短=被洗, 太長=資金卡住)
  4. R 分布 (右尾肥不肥, 大贏家有沒有被砍)
跑 C 與 D 兩個現行核心策略, 對齊真實引擎 (含市況濾網/分級/波動標量).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")

TRADE_START = pd.Timestamp("2023-06-01")


def analyze(strategy, data, rs, regime, regime_tiers, vol_scalars,
            all_dates, date_idx):
    sigs = bt.collect_signals(data, strategy)
    sigs = {d: lst for d, lst in sigs.items() if d in regime}
    entry_map = bt.build_entry_map(sigs, data)
    trades, curve = bt.run_sub(data, entry_map, strategy, 1.0, 2_000_000,
                               all_dates=all_dates, date_idx=date_idx,
                               regime_tiers=regime_tiers, vol_scalars=vol_scalars)
    tdf = pd.DataFrame(trades)
    tdf = tdf[tdf["exit_date"] >= TRADE_START].copy()
    tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
    tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])
    tdf["hold"] = (tdf["exit_date"] - tdf["entry_date"]).dt.days
    tdf["year"] = tdf["exit_date"].dt.year

    print(f"\n{'='*64}\n策略 {strategy} — 近三年交易分析 ({TRADE_START.date()} ~ {all_dates[-1].date()})\n{'='*64}")
    n = len(tdf)
    if n == 0:
        print("無交易"); return
    wins = tdf["pnl"] > 0
    gw = tdf.loc[wins, "pnl"].sum()
    gl = -tdf.loc[~wins, "pnl"].sum()
    pf = gw / gl if gl > 0 else np.inf
    print(f"總交易: {n}  勝率: {wins.mean()*100:.1f}%  PF: {pf:.2f}  "
          f"均R: {tdf['r'].mean():.2f}  總損益: {tdf['pnl'].sum():,.0f}")

    print("\n逐年:")
    print(f"  {'年':<6}{'交易':>5}{'勝率':>8}{'PF':>7}{'均R':>7}{'損益':>14}")
    for y, g in tdf.groupby("year"):
        w = g["pnl"] > 0
        gwl = -g.loc[~w, "pnl"].sum()
        ypf = g.loc[w, "pnl"].sum() / gwl if gwl > 0 else np.inf
        print(f"  {y:<6}{len(g):>5}{w.mean()*100:>7.1f}%{ypf:>7.2f}"
              f"{g['r'].mean():>7.2f}{g['pnl'].sum():>14,.0f}")

    # 持有天數分布
    print("\n持有天數分布 (日曆日):")
    bins = [0, 5, 10, 20, 40, 90, 9999]
    labels = ["0-5", "6-10", "11-20", "21-40", "41-90", "90+"]
    tdf["hbin"] = pd.cut(tdf["hold"], bins=bins, labels=labels)
    for lb in labels:
        g = tdf[tdf["hbin"] == lb]
        if len(g) == 0:
            continue
        w = (g["pnl"] > 0).mean()
        print(f"  {lb:>6}日: {len(g):>4}筆 ({len(g)/n*100:>4.0f}%)  "
              f"勝率{w*100:>5.0f}%  均R{g['r'].mean():>6.2f}")

    # R 分布 (右尾)
    print("\nR 分布 (檢視右尾肥瘦):")
    rbins = [-99, -1, 0, 1, 2, 3, 5, 99]
    rlabels = ["<-1R", "-1~0", "0~1R", "1~2R", "2~3R", "3~5R", "5R+"]
    tdf["rbin"] = pd.cut(tdf["r"], bins=rbins, labels=rlabels)
    for lb in rlabels:
        g = tdf[tdf["rbin"] == lb]
        bar = "█" * int(len(g) / max(n, 1) * 40)
        print(f"  {lb:>6}: {len(g):>4}筆 {bar}")
    big = tdf[tdf["r"] >= 3]
    print(f"  3R+ 大贏家: {len(big)}筆, 貢獻損益 {big['pnl'].sum():,.0f} "
          f"({big['pnl'].sum()/max(tdf['pnl'].sum(),1)*100:.0f}% of 總損益)")

    # 資金利用率: 用每日持倉數重建
    period_dates = [d for d in all_dates if d >= TRADE_START]
    held = {d: 0 for d in period_dates}
    for _, t in tdf.iterrows():
        for d in period_dates:
            if t["entry_date"] <= d <= t["exit_date"]:
                held[d] += 1
    held_series = pd.Series(held)
    days_with_pos = (held_series > 0).sum()
    print(f"\n資金利用率 (近三年 {len(period_dates)} 交易日):")
    print(f"  有部位天數: {days_with_pos} ({days_with_pos/len(period_dates)*100:.0f}%)")
    print(f"  完全空手天數: {len(period_dates)-days_with_pos} "
          f"({(len(period_dates)-days_with_pos)/len(period_dates)*100:.0f}%)")
    print(f"  平均持倉檔數: {held_series.mean():.2f} (MAX_POS={bt.MAX_POS[strategy]})")
    # 最長空手連續
    flat = (held_series.values == 0).astype(int)
    longest = cur = 0
    for v in flat:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    print(f"  最長連續空手: {longest} 交易日")


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)
    print(f"{len(data)} 檔, 全期 {all_dates[0].date()} ~ {all_dates[-1].date()}")
    for s in ["C", "D"]:
        analyze(s, data, rs, regime, regime_tiers, vol_scalars, all_dates, date_idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
