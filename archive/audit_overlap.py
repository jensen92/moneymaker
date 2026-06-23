"""現有策略「單一文化 (monoculture)」實證: 衡量 C/D/J/E 之間的進場重疊度與
報酬流相關性, 檢驗它們是否其實在買同一批股票、暴露同一個因子。

對每個策略用 collect_signals 取得 (code, signal_date) 進場集合:
  1. Jaccard 重疊 (兩策略買到同一檔同一天的比例)
  2. 同檔重疊 (忽略日期, 兩策略生涯買過的股票交集)
  3. 每日新進場檔數時間序列的相關係數 (報酬流同步程度的代理)
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
STRATS = ["C", "D", "J", "E"]


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    bt.compute_rs_rank(data)
    regime = bt.load_regime()

    entries = {}        # strat -> set of (code, date)
    by_code = {}        # strat -> set of code
    daily_cnt = {}      # strat -> {date: n}
    for s in STRATS:
        print(f"掃描策略 {s} 訊號 ...")
        sigs = bt.collect_signals(data, s)
        sigs = {d: lst for d, lst in sigs.items() if d in regime}
        ent, codes, cnt = set(), set(), defaultdict(int)
        for d, lst in sigs.items():
            for score, code, i, sg in lst:
                ent.add((code, d))
                codes.add(code)
                cnt[d] += 1
        entries[s] = ent
        by_code[s] = codes
        daily_cnt[s] = cnt
        print(f"  {s}: {len(ent)} 個 (code,date) 進場訊號, 涉及 {len(codes)} 檔股票")

    print("\n=== (1) 進場 (code,date) Jaccard 重疊 ===")
    print("    (兩策略在同一天對同一檔都發出訊號的程度; 1.0=完全相同)")
    print("        " + "".join(f"{s:>8}" for s in STRATS))
    for a in STRATS:
        row = f"  {a:<6}"
        for b in STRATS:
            inter = len(entries[a] & entries[b])
            union = len(entries[a] | entries[b])
            row += f"{inter/union if union else 0:>8.2f}"
        print(row)

    print("\n=== (2) 生涯買過的「股票池」交集比例 (相對 A 的占比) ===")
    print("    (a 買過的股票中, 有多少 % 也被 b 買過; 揭示是否在同一票池打轉)")
    print("        " + "".join(f"{s:>8}" for s in STRATS))
    for a in STRATS:
        row = f"  {a:<6}"
        for b in STRATS:
            ov = len(by_code[a] & by_code[b]) / len(by_code[a]) if by_code[a] else 0
            row += f"{ov*100:>7.0f}%"
        print(row)

    print("\n=== (3) 每日新進場檔數時間序列相關係數 ===")
    print("    (進場時機是否同步; 高=同漲同跌的報酬流, 分散效果差)")
    all_dates = sorted({d for s in STRATS for d in daily_cnt[s]})
    cnt_df = pd.DataFrame({s: pd.Series(daily_cnt[s]) for s in STRATS},
                          index=all_dates).fillna(0)
    # 以週彙總降噪後算相關
    weekly = cnt_df.resample("W", level=0).sum() if isinstance(cnt_df.index, pd.DatetimeIndex) else cnt_df
    weekly.index = pd.to_datetime(weekly.index)
    weekly = cnt_df.copy()
    weekly.index = pd.to_datetime(weekly.index)
    weekly = weekly.resample("W").sum()
    corr = weekly.corr()
    print("        " + "".join(f"{s:>8}" for s in STRATS))
    for a in STRATS:
        print(f"  {a:<6}" + "".join(f"{corr.loc[a,b]:>8.2f}" for b in STRATS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
