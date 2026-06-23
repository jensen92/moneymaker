"""組合部位規模 (RISK_PCT) × 槓桿敏感度: 量化 A+C+D 投組的成長/風險取捨.

發現: 單純調 leverage 無效 — 本引擎以「風險」決定部位大小 (每筆風險 = 權益 ×
RISK_PCT ÷ 停損幅度), leverage 只是曝險上限, 風險式部位很少觸及. 真正的成長
槓桿是 RISK_PCT (每筆風險). Minervini 原則容許單筆組合風險 1.25%~2.5%, 現行
1% 偏保守. 提高 RISK_PCT 需同步放寬曝險上限 (leverage) 部位才放得下.

訊號 (慢, 含 A 的 VCP 掃描) 只算一次, 再以不同 (RISK_PCT, leverage) 重跑.
用法: python3 leverage_sweep.py
"""
import os

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")

from backtest import (load_all, collect_signals, build_entry_map, run_sub,
                      load_regime, build_date_index, INIT_CAPITAL)

STRATS = ("A", "C", "D")
# (每筆組合風險 RISK_PCT, 對應曝險上限 leverage)
COMBOS = [(0.010, 1.0), (0.015, 1.5), (0.020, 2.0), (0.025, 2.5)]


def main():
    print("載入資料...")
    data, _ = load_all()
    print(f"{len(data)} 檔股票")
    risk_on = load_regime()
    all_dates, date_idx = build_date_index(data)
    init_eq = INIT_CAPITAL / len(STRATS)

    entry_maps = {}
    for k in STRATS:
        print(f"  計算策略 {k} 訊號 (一次性)...")
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        entry_maps[k] = build_entry_map(sigs, data)

    orig_risk = bt.RISK_PCT
    print(f"\n{'風險/筆':>7} {'槓桿':>5} {'總報酬':>10} {'年化':>8} "
          f"{'最大回撤':>9} {'MAR':>6} {'期末權益':>14}")
    print("-" * 66)
    rows = []
    for risk_pct, lev in COMBOS:
        bt.RISK_PCT = risk_pct
        curves = {}
        for k in STRATS:
            bt.DD_PAUSE[k] = 1.0
            bt.DD_RESUME[k] = 1.0
            _, curve = run_sub(data, entry_maps[k], k, lev, init_eq,
                               all_dates=all_dates, date_idx=date_idx)
            curves[k] = curve
        combined = None
        for k in STRATS:
            c = curves[k][["date", "equity"]].rename(columns={"equity": f"eq_{k}"})
            combined = c if combined is None else combined.merge(c, on="date")
        eq = sum(combined[f"eq_{k}"] for k in STRATS)
        end = eq.iloc[-1]
        dd = ((eq.cummax() - eq) / eq.cummax()).max()
        years = len(eq) / 244
        cagr = (end / INIT_CAPITAL) ** (1 / max(years, 0.01)) - 1
        mar = cagr / dd if dd > 0 else float("inf")
        rows.append({"risk_pct": risk_pct, "lev": lev,
                     "ret": end / INIT_CAPITAL - 1, "cagr": cagr,
                     "dd": dd, "mar": mar, "end": end})
        print(f"{risk_pct:>6.1%} {lev:>5.1f} {end/INIT_CAPITAL-1:>+9.0%} "
              f"{cagr:>+7.1%} {dd:>8.1%} {mar:>6.2f} {end:>14,.0f}")
    bt.RISK_PCT = orig_risk

    pd.DataFrame(rows).to_csv("leverage_sweep_results.csv", index=False)
    print("\n結果已存至 leverage_sweep_results.csv")
    print("註: RISK_PCT 放大部位 (受單檔 25% 上限與 8 檔槽位約束); 報酬與回撤同步放大.")
    print("    Minervini 容許單筆組合風險至 2.5%; 實務建議視風險胃納選 1.5~2.0%.")


if __name__ == "__main__":
    main()
