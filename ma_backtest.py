"""均線交叉趨勢跟隨策略 (P) vs 現行生產策略 head-to-head 回測對照.

策略 P: 經典黃金交叉 (MA50 上穿 MA200) 純均線交易訊號 — 不疊加任何型態/base 條件,
用於量化「均線交易」本身的績效, 並與現行每日輪動 PA/PB/K/L/D 對照。
同一引擎 (run_sub)、同期 (2011-2026 data_adj 全市場)、各策略獨立等額子帳戶 (40 萬)。
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt
bt.DATA_DIR = os.path.join(HERE, "data_adj")
from backtest import (load_all, load_regime, load_regime_tiers, load_vol_scalars,
                      compute_rs_rank, build_entry_map, run_sub,
                      build_date_index, INIT_CAPITAL)
import strategies as st
from fund_backtest import build_signals as build_d_signals, metrics

PROD = ["PA", "PB", "K", "L", "D"]
NEW = ["P"]


def build_new_signals(data, rs, risk_on, keys):
    fns = {k: st.STRATEGIES[k] for k in keys}
    sigs = {k: defaultdict(list) for k in keys}
    for code, df in data.items():
        n = len(df)
        for i in range(210, n - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if np.isnan(rank):
                continue
            for k, fn in fns.items():
                s = fn(df, i, rs_rank=rank)
                if s:
                    sigs[k][d].append((s["score"], code, i, s))
    for k in sigs:
        sigs[k] = {d: lst for d, lst in sigs[k].items() if d in risk_on}
    return sigs


def run_one(data, bydate, all_dates, date_idx, regime_tiers, vol_scalars,
            engine_key, init_eq):
    em = build_entry_map(bydate, data)
    trades, curve = run_sub(data, em, engine_key, 1.0, init_eq,
                            all_dates=all_dates, date_idx=date_idx,
                            regime_tiers=regime_tiers, vol_scalars=vol_scalars)
    m = metrics(trades, curve, init_eq)
    yR = defaultdict(float)
    for t in trades:
        yR[t["exit_date"].year] += t["r"]
    return m, yR


def main():
    print("載入 data_adj...")
    data, names = load_all()
    print(f"{len(data)} 檔, RS rank...")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    all_dates, date_idx = build_date_index(data)
    init_eq = INIT_CAPITAL / 5

    print("收集現行策略訊號 (PA/PB/K/L/D, 共用 _d_features)...")
    d_sigs = build_d_signals(data, rs, risk_on)
    print("收集均線交叉訊號 (P)...")
    new_sigs = build_new_signals(data, rs, risk_on, NEW)
    for k in NEW:
        nsig = sum(len(v) for v in new_sigs[k].values())
        print(f"  {k}: 原始訊號 {nsig}")

    print("回測中 (各策略獨立子帳戶)...")
    rows, yRs = {}, {}
    for k in PROD:
        rows[k], yRs[k] = run_one(data, d_sigs[k], all_dates, date_idx,
                                  regime_tiers, vol_scalars, k, init_eq)
    for k in NEW:
        rows[k], yRs[k] = run_one(data, new_sigs[k], all_dates, date_idx,
                                  regime_tiers, vol_scalars, k, init_eq)

    LABEL = {"PA": "PA 現行", "PB": "PB 現行", "K": "K 現行", "L": "L 現行",
             "D": "D 現行", "P": "P 均線黃金交叉(MA50/200)"}
    order = PROD + NEW

    print(f"\n{'='*88}")
    print("均線交叉策略 vs 現行生產策略 (2011-2026, data_adj 全市場, 各策略等額 40 萬子帳戶)")
    print('=' * 88)
    print(f"\n{'策略':<28}{'交易':>5}{'勝率':>7}{'PF':>6}{'年化':>7}"
          f"{'回撤':>7}{'MAR':>6}{'總R':>8}")
    for k in order:
        m = rows[k]
        print(f"{LABEL[k]:<28}{m['n']:>5}{m['win']:>7.1%}{m['pf']:>6.2f}"
              f"{m['cagr']:>7.1%}{m['dd']:>7.1%}{m['mar']:>6.2f}{m['totR']:>+8.0f}")

    years = sorted({y for k in order for y in yRs[k]})
    print("\n逐年總R:")
    print(f"{'年':>6}" + "".join(f"{k:>7}" for k in order))
    for y in years:
        print(f"{y:>6}" + "".join(f"{yRs[k].get(y, 0):>+7.0f}" for k in order))


if __name__ == "__main__":
    main()
