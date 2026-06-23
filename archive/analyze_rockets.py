"""飆股捕捉率分析: 近三年「真飆股」(短期內爆量上漲) 策略 C/D 抓到了多少?

定義飆股事件: 任一交易日起算未來 60 個交易日內, 最高價相對該日收盤上漲 >= 80%
(即 3 個月內接近翻倍), 同一檔股票同一波段只取一次 (避免同一波重複計入)。

對每個飆股事件, 檢查 C/D 的實際交易紀錄是否在「事件起點 ~ 波段高點」期間
對該股有任何進場 (entry_date 落在 [event_start, event_peak] 之內,
或更寬鬆: 進場日在事件期間 [event_start - 20, event_peak] 之內 — 容許略早於
官方訊號定義的飆漲起點布局)。

回報: 抓到數 / 總飆股數 (捕捉率), 抓到時的實際報酬率 vs 飆股理論漲幅 (效率)。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
TRADE_START = pd.Timestamp("2023-06-01")
WINDOW = 60        # 未來 60 交易日 (~3 個月)
THRESH = 0.80      # 上漲門檻 80%


def find_rocket_events(df):
    """回傳 [(start_idx, peak_idx, gain)], 不重疊."""
    close = df["close"].values
    high = df["high"].values
    n = len(df)
    events = []
    i = 0
    while i < n - WINDOW:
        if close[i] <= 0:
            i += 1
            continue
        fut_high = high[i + 1:i + 1 + WINDOW]
        if len(fut_high) == 0:
            i += 1
            continue
        peak_rel = fut_high.argmax()
        peak_val = fut_high[peak_rel]
        gain = peak_val / close[i] - 1.0
        if gain >= THRESH:
            peak_idx = i + 1 + peak_rel
            events.append((i, peak_idx, gain))
            i = peak_idx + 1   # 跳過此波段, 避免重複計入
        else:
            i += 1
    return events


def main():
    print("載入資料 ...")
    data, names = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    print("掃描飆股事件 (未來60日內漲幅>=80%) ...")
    rockets = []  # (code, start_date, peak_date, gain)
    for code, df in data.items():
        d = df["date"]
        if d.iloc[-1] < TRADE_START:
            continue
        for s_i, p_i, gain in find_rocket_events(df):
            sd = df["date"].iloc[s_i]
            pd_ = df["date"].iloc[p_i]
            if pd_ < TRADE_START:
                continue
            rockets.append((code, sd, pd_, gain))
    print(f"近三年飆股事件數: {len(rockets)} (跨 {len(set(r[0] for r in rockets))} 檔股票)\n")

    results = {}
    for strat in ["C", "D"]:
        sigs = bt.collect_signals(data, strat)
        sigs = {dt: lst for dt, lst in sigs.items() if dt in regime}
        em = bt.build_entry_map(sigs, data)
        trades, curve = bt.run_sub(data, em, strat, 1.0, 2_000_000,
                                   all_dates=all_dates, date_idx=date_idx,
                                   regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        tdf = pd.DataFrame(trades)
        tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
        tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])

        caught, missed, captured_pcts = [], [], []
        for code, sd, pd_, gain in rockets:
            ct = tdf[(tdf["code"] == code)
                     & (tdf["entry_date"] >= sd - pd.Timedelta(days=30))
                     & (tdf["entry_date"] <= pd_)]
            if len(ct) == 0:
                missed.append((code, sd, gain))
                continue
            # 抓到了: 取該波段內所有相關交易的累積報酬率 (以第一筆進場價為基準粗估)
            entry0 = ct["entry"].iloc[0]
            total_ret = 0.0
            for _, t in ct.iterrows():
                total_ret += t["pnl"]
            caught.append((code, sd, gain, len(ct), total_ret))

        results[strat] = (caught, missed)
        n_total = len(rockets)
        n_caught = len(caught)
        print(f"=== 策略 {strat} 飆股捕捉率 ===")
        print(f"  總飆股事件: {n_total}  抓到: {n_caught} ({n_caught/n_total*100:.1f}%)  "
              f"完全錯過: {len(missed)} ({len(missed)/n_total*100:.1f}%)")
        if caught:
            avg_gain_caught = np.mean([g for _, _, g, _, _ in caught])
            print(f"  抓到的飆股, 平均理論漲幅: {avg_gain_caught*100:.0f}%")
        if missed:
            avg_gain_missed = np.mean([g for _, _, g in missed])
            print(f"  錯過的飆股, 平均理論漲幅: {avg_gain_missed*100:.0f}% (錯過更猛的部位嗎?)")
        print()

    # 列出前10大飆股 (依漲幅排序) 與 C/D 是否抓到
    print("=== 近三年漲幅最猛的 15 檔飆股事件, C/D 是否抓到 ===")
    top = sorted(rockets, key=lambda r: -r[3])[:15]
    cd_caught_codes = {s: set(c for c, *_ in results[s][0]) for s in ["C", "D"]}
    for code, sd, pd_, gain in top:
        name = names.get(code, "")
        flag_c = "✓" if code in cd_caught_codes["C"] else "✗"
        flag_d = "✓" if code in cd_caught_codes["D"] else "✗"
        print(f"  {code} {name:<10} {sd:%Y-%m-%d}~{pd_:%Y-%m-%d}  "
              f"漲幅{gain*100:>6.0f}%   C:{flag_c}  D:{flag_d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
