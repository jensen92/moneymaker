"""每日選股: 以最新資料對全部股票跑策略 PA/PB/K/L/D/C (現行輪動), 各輸出最多 2 檔.

使用方式: 先跑 download_data.py 更新資料, 再跑本腳本.
"""
import json
import os

import numpy as np
import pandas as pd

from strategies import STRATEGIES, add_indicators
from backtest import DATA_DIR, load_regime, INIT_CAPITAL, RISK_PCT, PICKS_PER_DAY
from backtest import compute_rs_rank

# 與每日報告 (github_scan.py) / /scan 一致的策略輪動
PICK_KEYS = ["PA", "PB", "K", "L", "D", "C"]


def main():
    uni = os.path.join(DATA_DIR, "universe.json")
    if os.path.exists(uni):
        with open(uni) as f:
            names = {s["code"]: s["name"] for s in json.load(f)}
    else:
        names = {}

    # 載入所有股票 (需要 RS rank)
    all_data = {}
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(DATA_DIR, fn), parse_dates=["date"])
        df = add_indicators(df).reset_index(drop=True)
        if len(df) >= 210:
            all_data[code] = df

    rs = compute_rs_rank(all_data)
    risk_on = load_regime()

    candidates = {k: [] for k in PICK_KEYS}
    latest_date = None

    for code, df in all_data.items():
        i = len(df) - 1
        d = df["date"].iloc[i]
        if latest_date is None or d > latest_date:
            latest_date = d
        try:
            rank = rs.at[d, code]
        except KeyError:
            rank = None
        if rank is not None and not (isinstance(rank, float) and np.isnan(rank)):
            for key in PICK_KEYS:
                s = STRATEGIES[key](df, i, rs_rank=rank)
                if s:
                    candidates[key].append((s["score"], code, df, s))

    regime_ok = latest_date in risk_on
    print(f"資料日期: {latest_date:%Y-%m-%d}  大盤濾網: {'通過 (可進場)' if regime_ok else '未通過 (今日不進場)'}")
    if not regime_ok:
        return

    for key in PICK_KEYS:
        print(f"\n=== 策略 {key} 明日進場候選 (取前 {PICKS_PER_DAY}) ===")
        picks = sorted(candidates[key], reverse=True)[:PICKS_PER_DAY]
        if not picks:
            print("(無訊號)")
            continue
        for score, code, df, s in picks:
            row = df.iloc[-1]
            ref_entry = row["close"]
            stop_pct = s.get("stop_pct", 0.08)
            stop = ref_entry * (1 - stop_pct)
            risk_amt = INIT_CAPITAL * RISK_PCT
            shares = int(risk_amt / (ref_entry - stop) / 1000) * 1000
            if shares <= 0:
                shares = int(risk_amt / (ref_entry - stop))
            print(f"{code} {names.get(code, '')}: 參考價 {ref_entry:.1f}, "
                  f"停損 {stop:.1f} (-{stop_pct:.0%}), "
                  f"建議 {shares} 股, 最長持有 {s['max_hold']} 日, "
                  f"RS {score:.2f}")


if __name__ == "__main__":
    main()
