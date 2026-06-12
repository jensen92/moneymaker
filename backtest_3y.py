"""Strategy C 近三年回測 (2023-06-01 ~ 2026-06-12)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import backtest as _bt
_bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
from backtest import (load_all, collect_signals, build_entry_map,
                      run_sub, load_regime, INIT_CAPITAL)
from strategies import add_indicators

TRADE_START = pd.Timestamp("2023-06-01")
WARMUP_ROWS = 215   # signal_c 需要 210 列歷史

def main():
    print("載入資料 (with warmup)...")
    data, names = load_all()

    # 只保留到 TRADE_START 之前足夠 warmup 的列
    trimmed = {}
    for code, df in data.items():
        # 找到 TRADE_START 之後的第一個 index
        mask = df["date"] >= TRADE_START
        if mask.sum() == 0:
            continue
        first_idx = df.index[mask][0]
        # 保留 warmup + 交易期
        start_idx = max(0, first_idx - WARMUP_ROWS)
        trimmed[code] = df.iloc[start_idx:].reset_index(drop=True)

    print(f"{len(trimmed)} 檔股票 (有 {TRADE_START.date()} 後資料)")

    risk_on = load_regime()

    init_eq = INIT_CAPITAL / 3   # 獨立 C 子帳戶

    print("計算 RS Rank...")
    from backtest import compute_rs_rank
    # RS rank 用 trimmed data
    sigs_all = collect_signals(trimmed, "C")

    # 只取 TRADE_START 之後的訊號
    sigs = {d: lst for d, lst in sigs_all.items()
            if d >= TRADE_START and d in risk_on}

    entry_map = build_entry_map(sigs, trimmed)
    print(f"訊號日數: {len(sigs)}, 進場候選筆數: {sum(len(v) for v in entry_map.values())}")

    trades, curve = run_sub(trimmed, entry_map, "C", leverage=1.0, init_eq=init_eq)

    # 只顯示 TRADE_START 之後的曲線
    curve = curve[curve["date"] >= TRADE_START].reset_index(drop=True)
    if curve.empty:
        print("無交易資料")
        return

    start_eq = curve["equity"].iloc[0]
    end_eq = curve["equity"].iloc[-1]
    start_date = curve["date"].iloc[0]
    end_date = curve["date"].iloc[-1]
    dd_max = ((curve["equity"].cummax() - curve["equity"]) / curve["equity"].cummax()).max()
    years = len(curve) / 244
    cagr = (end_eq / init_eq) ** (1 / max(years, 0.01)) - 1

    tdf = pd.DataFrame(trades)
    tdf_period = tdf[tdf["exit_date"] >= TRADE_START] if not tdf.empty else tdf

    print(f"\n{'='*55}")
    print(f"策略 C 近三年回測  ({start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d})")
    print(f"{'='*55}")
    print(f"初始資金: {init_eq:,.0f}")
    print(f"期末權益: {end_eq:,.0f}")
    print(f"總報酬:   {(end_eq/init_eq - 1):+.1%}")
    print(f"年化報酬: {cagr:+.1%}")
    print(f"最大回撤: {dd_max:.1%}")
    if not tdf_period.empty:
        wins = (tdf_period["pnl"] > 0).sum()
        print(f"交易筆數: {len(tdf_period)}  勝率: {wins/len(tdf_period):.1%}")
    curve.to_csv("equity_c_3y.csv", index=False)
    print("\n權益曲線已存至 equity_c_3y.csv")


if __name__ == "__main__":
    main()
