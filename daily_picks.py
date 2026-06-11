"""每日選股: 以最新資料對全部股票跑兩個策略, 各輸出最多 2 檔.

使用方式: 先跑 download_data.py 更新資料, 再跑本腳本.
輸出含進場參考價(次日開盤)、停損價、停利/移動停利規則與建議部位
(以 200 萬權益、單筆風險 1%、總曝險上限 2 倍計).
"""
import json
import os

import pandas as pd

from strategies import STRATEGIES, add_indicators
from backtest import DATA_DIR, load_regime, INIT_CAPITAL, RISK_PCT, PICKS_PER_DAY


def main():
    with open(os.path.join(DATA_DIR, "universe.json")) as f:
        names = {s["code"]: s["name"] for s in json.load(f)}

    candidates = {"A": [], "B": []}
    latest_date = None
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(DATA_DIR, fn), parse_dates=["date"])
        df = add_indicators(df).reset_index(drop=True)
        if len(df) < 130:
            continue
        i = len(df) - 1
        if latest_date is None or df["date"].iloc[i] > latest_date:
            latest_date = df["date"].iloc[i]
        for key, fn_sig in STRATEGIES.items():
            s = fn_sig(df, i)
            if s:
                candidates[key].append((s["score"], code, df, s))

    risk_on = load_regime()
    regime_ok = latest_date in risk_on
    print(f"資料日期: {latest_date:%Y-%m-%d}  大盤濾網: {'通過 (可進場)' if regime_ok else '未通過 (今日不進場)'}")
    if not regime_ok:
        return

    for key in ("A", "B"):
        print(f"\n=== 策略 {key} 明日進場候選 (取前 {PICKS_PER_DAY}) ===")
        picks = sorted(candidates[key], reverse=True)[:PICKS_PER_DAY]
        if not picks:
            print("(無訊號)")
        for score, code, df, s in picks:
            row = df.iloc[-1]
            ref_entry = row["close"]  # 實際以次日開盤為準
            atr = row["atr14"]
            stop = (s["stop_price"] if "stop_price" in s
                    else ref_entry - s["stop_atr"] * atr)
            risk_amt = INIT_CAPITAL * RISK_PCT
            shares = int(risk_amt / (ref_entry - stop) / 1000) * 1000
            if shares <= 0:  # 高價股改用零股
                shares = int(risk_amt / (ref_entry - stop))
            exit_rule = (f"停利 {ref_entry + s['target_r'] * (ref_entry - stop):.1f}"
                         if "target_r" in s
                         else f"移動停利: 最高收盤 - {s['trail_atr']} ATR")
            print(f"{code} {names.get(code, '')}: 參考價 {ref_entry:.1f}, "
                  f"停損 {stop:.1f}, {exit_rule}, "
                  f"建議 {shares} 股, 最長持有 {s['max_hold']} 日")


if __name__ == "__main__":
    main()
