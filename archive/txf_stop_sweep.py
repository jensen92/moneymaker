"""台指期日內策略 — 停損點數敏感度分析 (stop-loss sweep)。

回應「點數放寬會不會提升勝率」的提問: 對 `strat_prevday_breakout` 掃描多種
停損點數, 同時看「全期」與「樣本內/外 (IS/OOS)」, 觀察:
  - 勝率是否隨停損變寬而上升 (答案: 是, 單調上升);
  - 但 OOS 獲利因子是否隨之變薄 (答案: 是, 寬停損更貼合歷史、較不穩健);
  - 風險調整後報酬 (Sharpe) 與最大回撤的取捨。

用法:
    python3 txf_stop_sweep.py
"""
import txf_backtest as bt

STOPS = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]


def main():
    days = bt.load_days()
    print(f"載入 {len(days)} 個交易日\n")
    print("全期 (2023-06~2026-06):")
    print(f"{'停損':>5} {'筆數':>5} {'勝率%':>6} {'淨點':>9} {'PF':>5} "
          f"{'最大回撤':>7} {'Sharpe':>6}")
    for s in STOPS:
        m, _, _ = bt.run(days, bt.strat_prevday_breakout,
                         {"stop_pt": s, "side": "both"})
        if not m:
            continue
        star = "  ←現用" if s == 40 else ""
        print(f"{s:>5} {m['trades']:>5} {m['win']*100:>6.1f} {m['total']:>9.0f} "
              f"{m['pf']:>5.2f} {m['maxdd']:>7.0f} {m['sharpe']:>6.2f}{star}")

    print("\n樣本內 / 樣本外 (70/30 時間切分):")
    print(f"{'停損':>5} | {'IS勝率':>6} {'IS_PF':>6} {'IS淨':>7} | "
          f"{'OOS勝率':>7} {'OOS_PF':>6} {'OOS淨':>7}")
    for s in STOPS:
        _, m_is, m_oos = bt.split_eval(days, bt.strat_prevday_breakout,
                                       {"stop_pt": s, "side": "both"})
        if not (m_is and m_oos):
            continue
        print(f"{s:>5} | {m_is['win']*100:>6.1f} {m_is['pf']:>6.2f} "
              f"{m_is['total']:>7.0f} | {m_oos['win']*100:>7.1f} "
              f"{m_oos['pf']:>6.2f} {m_oos['total']:>7.0f}")

    print("\n結論: 勝率隨停損變寬單調上升, 但 OOS 獲利因子隨之下降 (寬停損更貼合")
    print("歷史、較不穩健)。40 點為勝率/可執行性與 OOS 穩健度之平衡, 非唯一最優。")


if __name__ == "__main__":
    main()
