import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 加入當前路徑以 import 現有模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backtest import load_all, build_date_index, collect_signals, build_entry_map, _climax_run, _is_limit_day
from strategies import STRATEGIES

def single_stock_simulation(data, strategy_key, init_eq=2_000_000, start_date='2021-08-01'):
    """全押單一股票回測 (200萬一次, 賣掉再買第一名)"""
    # 擷取訊號並建立 entry_map
    sigs = collect_signals(data, strategy_key)
    entry_map = build_entry_map(sigs, data)
    
    all_dates, date_idx = build_date_index(data)
    # 過濾最近 5 年
    start_date_ts = pd.Timestamp(start_date)
    all_dates = [d for d in all_dates if d >= start_date_ts]
    
    equity = init_eq
    peak_eq = init_eq
    open_pos = None
    trades = []
    
    for d in all_dates:
        # 1. 檢查出場
        if open_pos is not None:
            p = open_pos
            di = date_idx[p["code"]].get(d)
            if di is None or di <= p["entry_idx"]:
                continue
            
            row = data[p["code"]].iloc[di]
            df_stock = data[p["code"]]
            exit_price = None
            
            stop_triggered = row["low"] <= p["stop"]
            if stop_triggered and _is_limit_day(df_stock, di) and row["close"] < row["open"]:
                # 跌停延遲
                p["expire_idx"] = max(p["expire_idx"], di + 1)
                if di + 1 >= len(df_stock):
                    exit_price = row["close"]
                else:
                    continue
            elif stop_triggered:
                exit_price = min(p["stop"], row["open"])
            elif p["target"] is not None and row["high"] >= p["target"]:
                exit_price = max(p["target"], row["open"])
            elif p.get("climax_exit") and _climax_run(df_stock, di, p):
                exit_price = row["close"]
            elif p.get("minervini") and row["close"] < row["ma50"]:
                exit_price = row["close"]
            elif di >= p["expire_idx"]:
                exit_price = row["close"]
                
            if exit_price is None:
                # 更新移動停損
                if p.get("trail_atr"):
                    p["high_close"] = max(p["high_close"], row["close"])
                    atr = row["atr14"] if not np.isnan(row["atr14"]) else p["init_atr"]
                    p["stop"] = max(p["stop"], p["high_close"] - p["trail_atr"] * atr)
                if p.get("minervini"):
                    risk_per_sh = p["entry"] - p["init_stop"]
                    if risk_per_sh > 0 and row["close"] >= p["entry"] + 3 * risk_per_sh:
                        p["stop"] = max(p["stop"], p["entry"])
                    if row["ma50"] >= p["entry"]:
                        p["stop"] = max(p["stop"], row["ma50"])
                continue # 繼續持有
                
            # 結算
            exit_price *= 0.999 # slip
            proceeds = p["shares"] * exit_price * (1 - 0.001425 - 0.003)
            cost = p["shares"] * p["entry"] * (1 + 0.001425)
            pnl = proceeds - cost
            equity += pnl
            ret_pct = pnl / cost
            trades.append({
                "code": p["code"],
                "entry_date": p["entry_date"],
                "exit_date": d,
                "entry": p["entry"],
                "exit": exit_price,
                "pnl": pnl,
                "ret_pct": ret_pct,
                "hold_days": di - p["entry_idx"]
            })
            open_pos = None
            peak_eq = max(peak_eq, equity)
            
        # 2. 尋找進場
        if open_pos is None:
            candidates = sorted(entry_map.get(d, []), reverse=True)
            for score, code, ei, s in candidates:
                df = data[code]
                if _is_limit_day(df, ei):
                    prev_close = df["close"].iloc[ei-1] if ei > 0 else df["open"].iloc[ei]
                    if df["close"].iloc[ei] > prev_close * 1.08:
                        continue # 漲停不追
                
                entry = df["open"].iloc[ei] * 1.001 # slip
                atr = df["atr14"].iloc[ei-1]
                if np.isnan(atr) or atr <= 0:
                    continue
                    
                if "stop_pct" in s:
                    stop = entry * (1 - s["stop_pct"])
                else:
                    stop = entry - s["stop_atr"] * atr
                
                if stop <= 0 or entry <= stop:
                    continue
                    
                # 全押！
                shares = int(equity / entry / 1000) * 1000
                if shares <= 0:
                    shares = max(1, int(equity / entry))
                    
                open_pos = {
                    "code": code,
                    "shares": shares,
                    "entry": entry,
                    "init_entry": entry,
                    "stop": stop,
                    "init_stop": stop,
                    "target": s.get("target_r") * (entry - stop) + entry if "target_r" in s else None,
                    "trail_atr": s.get("trail_atr"),
                    "minervini": s.get("minervini", False),
                    "climax_exit": s.get("climax_exit", False),
                    "climax_sprint_pct": s.get("climax_sprint_pct", 0.25),
                    "climax_min_gain": s.get("climax_min_gain", 0.30),
                    "extended_updays": s.get("extended_updays", 7),
                    "extended_min_gain": s.get("extended_min_gain", 0.20),
                    "high_close": entry,
                    "init_atr": atr,
                    "entry_idx": ei,
                    "expire_idx": ei + s["max_hold"],
                    "entry_date": d
                }
                break # 只買第一檔
                
    return trades, equity

if __name__ == "__main__":
    from backtest import DATA_DIR
    # 強制指派 DATA_DIR 為調整後資料
    import backtest
    backtest.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
    
    print("載入全市場還原日線資料中 (約需幾秒)...")
    data, _ = load_all()
    
    # 計算五年前的日期
    start_dt = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    print(f"回測區間: {start_dt} 至今 (近五年)")
    print("初始本金: 2,000,000")
    print("部位控制: 每次交易 100% 全押, 單一持倉, 賣出後再換股\n")
    
    print(f"{'策略':<4} {'交易數':>5} {'勝率':>6} {'總報酬':>9} {'平均獲利':>8} {'中位數獲利':>9} {'期末權益'}")
    print("-" * 70)
    
    # 只測試 A, C, D (現有活躍策略)
    for k in ["A", "C", "D"]:
        trades, final_eq = single_stock_simulation(data, k, start_date=start_dt)
        if not trades:
            print(f"{k:<4} 無交易")
            continue
            
        ret_pcts = [t["ret_pct"] for t in trades]
        win_rate = sum(1 for r in ret_pcts if r > 0) / len(ret_pcts)
        total_ret = final_eq / 2_000_000 - 1
        avg_ret = np.mean(ret_pcts)
        med_ret = np.median(ret_pcts)
        
        print(f"{k:<4} {len(trades):>6} {win_rate:>7.1%} {total_ret:>10.1%} {avg_ret:>9.2%} {med_ret:>10.2%}  {final_eq:,.0f}")
