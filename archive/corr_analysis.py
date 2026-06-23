"""策略相關性分析: 量化新策略 E 與既有 C / D 的相關係數與分散效益.

目標: 找到能「打敗 / 互補」C+D 的低相關策略. 評估三件事:
  1. 各策略單獨績效 (CAGR / MaxDD / MAR / PF)
  2. 策略間「月報酬」相關係數矩陣 (越低越好, 代表報酬來源不同)
  3. 不同組合的投組績效 (C+D vs C+D+E), 看加入 E 是否提升 MAR / 降低回撤

訊號 (慢) 各算一次, 再以共用引擎回測. 用法: python3 corr_analysis.py
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")

from backtest import (load_all, collect_signals, build_entry_map, run_sub,
                      load_regime, load_regime_tiers, load_vol_scalars,
                      build_date_index, INIT_CAPITAL)

STRATS = ("C", "D", "E")


def metrics(eq):
    """由權益序列算 (總報酬, CAGR, MaxDD, MAR)."""
    end = eq.iloc[-1]
    base = eq.iloc[0]
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    years = len(eq) / 244
    cagr = (end / base) ** (1 / max(years, 0.01)) - 1
    mar = cagr / dd if dd > 0 else float("inf")
    return end / base - 1, cagr, dd, mar


def main():
    print("載入資料...")
    data, _ = load_all()
    print(f"{len(data)} 檔股票")
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    all_dates, date_idx = build_date_index(data)
    init_eq = INIT_CAPITAL / 2   # 每子帳統一以 100 萬基準, 方便比較

    curves = {}
    trades_by = {}
    for k in STRATS:
        print(f"  計算策略 {k} 訊號...")
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        em = build_entry_map(sigs, data)
        trades, curve = run_sub(data, em, k, 1.0, init_eq,
                                all_dates=all_dates, date_idx=date_idx,
                                regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        curves[k] = curve.set_index("date")["equity"]
        trades_by[k] = trades

    # ── 1. 各策略單獨績效 ────────────────────────────────────────────────
    print(f"\n{'策略':<5} {'交易':>5} {'勝率':>6} {'PF':>6} "
          f"{'總報酬':>9} {'年化':>8} {'回撤':>7} {'MAR':>6}")
    print("-" * 56)
    for k in STRATS:
        eq = curves[k]
        ret, cagr, dd, mar = metrics(eq)
        td = pd.DataFrame(trades_by[k])
        if len(td):
            win = (td["pnl"] > 0).mean()
            pf = (td.loc[td.pnl > 0, "pnl"].sum()
                  / max(1, -td.loc[td.pnl < 0, "pnl"].sum()))
        else:
            win = pf = 0
        print(f"  {k:<3} {len(td):>5} {win:>6.1%} {pf:>6.2f} "
              f"{ret:>+8.0%} {cagr:>+7.1%} {dd:>6.1%} {mar:>6.2f}")

    # ── 2. 月報酬相關係數矩陣 ────────────────────────────────────────────
    monthly = {}
    for k in STRATS:
        m = curves[k].resample("ME").last().pct_change().dropna()
        monthly[k] = m
    mdf = pd.DataFrame(monthly).dropna()
    corr = mdf.corr()
    print("\n月報酬相關係數矩陣 (越低越互補):")
    print(corr.round(2).to_string())

    # ── 3. 組合績效比較 ─────────────────────────────────────────────────
    print("\n投組比較 (各子帳 100 萬, 等權):")
    print(f"{'組合':<10} {'總報酬':>9} {'年化':>8} {'回撤':>7} {'MAR':>6}")
    print("-" * 44)
    combos = [("C+D", ("C", "D")), ("C+D+E", ("C", "D", "E")),
              ("C+E", ("C", "E")), ("D+E", ("D", "E"))]
    for label, ks in combos:
        eq = sum(curves[k] for k in ks).dropna()
        ret, cagr, dd, mar = metrics(eq)
        print(f"  {label:<8} {ret:>+8.0%} {cagr:>+7.1%} {dd:>6.1%} {mar:>6.2f}")

    print("\n判讀: E 與 C/D 相關係數越低、且 C+D+E 的 MAR 高於 C+D, "
          "即代表 E 提供有效分散 (降風險或提報酬).")


if __name__ == "__main__":
    main()
