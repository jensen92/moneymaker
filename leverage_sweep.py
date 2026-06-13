"""組合槓桿敏感度: 量化 A+C+D 投組的成長/風險取捨.

訊號 (慢, 含 A 的 VCP 掃描) 只算一次, 再以不同槓桿重跑 run_sub 合併.
組合回撤遠低於個別策略 (分散效果), 故有空間用槓桿放大報酬 — 本表量化之.
用法: python3 leverage_sweep.py
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")

from backtest import (load_all, collect_signals, build_entry_map, run_sub,
                      load_regime, build_date_index, INIT_CAPITAL)

STRATS = ("A", "C", "D")
LEVERAGES = [1.0, 1.3, 1.5, 2.0]


def main():
    print("載入資料...")
    data, _ = load_all()
    print(f"{len(data)} 檔股票")
    risk_on = load_regime()
    all_dates, date_idx = build_date_index(data)
    init_eq = INIT_CAPITAL / len(STRATS)

    # 訊號只算一次 (collect_signals 內含 RS rank + 各策略訊號)
    entry_maps = {}
    for k in STRATS:
        print(f"  計算策略 {k} 訊號 (一次性)...")
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        entry_maps[k] = build_entry_map(sigs, data)

    print(f"\n{'槓桿':>5} {'總報酬':>10} {'年化':>8} {'最大回撤':>9} {'MAR':>6} {'期末權益':>14}")
    print("-" * 60)
    rows = []
    for lev in LEVERAGES:
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
        rows.append({"lev": lev, "ret": end / INIT_CAPITAL - 1, "cagr": cagr,
                     "dd": dd, "mar": mar, "end": end})
        print(f"{lev:>5.1f} {end/INIT_CAPITAL-1:>+9.0%} {cagr:>+7.1%} "
              f"{dd:>8.1%} {mar:>6.2f} {end:>14,.0f}")

    pd.DataFrame(rows).to_csv("leverage_sweep_results.csv", index=False)
    print("\n結果已存至 leverage_sweep_results.csv")
    print("註: 槓桿放大報酬與回撤; 組合分散使 1x 回撤僅 ~7%, 故有放大空間,")
    print("    但槓桿亦放大尾端風險與融資成本, 實務建議 1.3~1.5x 為穩健上限.")


if __name__ == "__main__":
    main()
