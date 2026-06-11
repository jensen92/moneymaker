"""組合回測: 200 萬本金, 每日每策略最多選 2 檔, 可 2 倍槓桿.

交易成本: 手續費 0.1425% 買賣各一次, 證交稅 0.3% (賣出), 滑價 0.1% 單邊.
進場: 訊號日次日開盤價. 出場: 觸停損/停利 (以當日 high/low 判斷), 或時間出場收盤價.
部位: 每筆風險 = 權益 1%, 名目曝險上限 = 權益 x 槓桿.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from strategies import STRATEGIES, add_indicators

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEE = 0.001425
TAX = 0.003
SLIP = 0.001
INIT_CAPITAL = 2_000_000
RISK_PCT = 0.01
PICKS_PER_DAY = 2


def load_all():
    data = {}
    with open(os.path.join(DATA_DIR, "universe.json")) as f:
        names = {s["code"]: s["name"] for s in json.load(f)}
    for fn in os.listdir(DATA_DIR):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(DATA_DIR, fn), parse_dates=["date"])
        df = add_indicators(df).reset_index(drop=True)
        data[code] = df
    return data, names


def collect_signals(data, strategy_key):
    """回傳 {訊號日: [(score, code, row_index), ...]}"""
    sig_fn = STRATEGIES[strategy_key]
    signals = defaultdict(list)
    for code, df in data.items():
        for i in range(120, len(df) - 1):  # 需留次日進場
            s = sig_fn(df, i)
            if s:
                signals[df["date"].iloc[i]].append((s["score"], code, i, s))
    return signals


def load_regime():
    """大盤濾網: 加權指數收盤 > MA60 的日期集合."""
    df = pd.read_csv(os.path.join(DATA_DIR, "_TWII.csv"), parse_dates=["date"])
    df["ma60"] = df["close"].rolling(60).mean()
    return set(df.loc[df["close"] > df["ma60"], "date"])


def run(data, names, strategy_key, leverage=2.0, verbose=False):
    signals = collect_signals(data, strategy_key)
    risk_on = load_regime()
    # 只保留訊號日為 risk-on 的訊號
    signals = {d: lst for d, lst in signals.items() if d in risk_on}
    equity = INIT_CAPITAL
    open_pos = []   # dict: code, shares, entry, stop, target, expire_idx, df
    trades = []
    curve = []
    all_dates = sorted({d for df in data.values() for d in df["date"]})
    date_idx = {code: {d: i for i, d in enumerate(df["date"])}
                for code, df in data.items()}
    entry_map = defaultdict(list)  # 進場日 -> [(score, code, entry_idx, s)]
    for sd, lst in signals.items():
        for score, code, i, s in lst:
            df = data[code]
            if i + 1 < len(df):
                entry_map[df["date"].iloc[i + 1]].append((score, code, i + 1, s))

    for d in all_dates:
        # 1. 處理出場
        still_open = []
        for p in open_pos:
            di = date_idx[p["code"]].get(d)
            if di is None or di <= p["entry_idx"]:
                still_open.append(p)
                continue
            row = data[p["code"]].iloc[di]
            exit_price = None
            if row["low"] <= p["stop"]:
                exit_price = min(p["stop"], row["open"])
            elif p["target"] is not None and row["high"] >= p["target"]:
                exit_price = max(p["target"], row["open"]) \
                    if row["open"] >= p["target"] else p["target"]
            elif di >= p["expire_idx"]:
                exit_price = row["close"]
            if exit_price is None and p.get("trail_atr"):
                # Chandelier: 停損上移至 進場後最高收盤 - trail_atr * ATR
                p["high_close"] = max(p["high_close"], row["close"])
                new_stop = p["high_close"] - p["trail_atr"] * row["atr14"]
                p["stop"] = max(p["stop"], new_stop)
            if exit_price is None:
                still_open.append(p)
                continue
            exit_price *= (1 - SLIP)
            proceeds = p["shares"] * exit_price * (1 - FEE - TAX)
            cost = p["shares"] * p["entry"] * (1 + FEE)
            pnl = proceeds - cost
            equity += pnl
            trades.append({"code": p["code"], "entry_date": p["entry_date"],
                           "exit_date": d, "entry": p["entry"],
                           "exit": exit_price, "pnl": pnl,
                           "r": pnl / p["risk_amt"]})
        open_pos = still_open

        # 2. 新訊號: 訊號日的次一交易日 == 今日 -> 今日開盤進場
        todays_entries = sorted(entry_map.get(d, []), reverse=True)
        held = {p["code"] for p in open_pos}
        exposure = sum(p["shares"] * p["entry"] for p in open_pos)
        taken = 0
        for score, code, ei, s in todays_entries:
            if taken >= PICKS_PER_DAY or code in held:
                continue
            df = data[code]
            entry = df["open"].iloc[ei] * (1 + SLIP)
            atr = df["atr14"].iloc[ei - 1]
            if "stop_price" in s:
                stop = s["stop_price"]
            else:
                stop = entry - s["stop_atr"] * atr
            if stop <= 0 or entry <= stop:
                continue
            risk_amt = equity * RISK_PCT
            shares = int(risk_amt / (entry - stop) / 1000) * 1000
            if shares <= 0:
                shares = int(risk_amt / (entry - stop))  # 零股
            if shares <= 0:
                continue
            notional = shares * entry
            if exposure + notional > equity * leverage:
                max_notional = equity * leverage - exposure
                shares = int(max_notional / entry / 1000) * 1000
                notional = shares * entry
                if shares <= 0:
                    continue
            target = (entry + s["target_r"] * (entry - stop)
                      if "target_r" in s else None)
            open_pos.append({"code": code, "shares": shares, "entry": entry,
                             "stop": stop, "target": target,
                             "trail_atr": s.get("trail_atr"),
                             "high_close": entry,
                             "entry_idx": ei,
                             "expire_idx": ei + s["max_hold"],
                             "entry_date": d,
                             "risk_amt": shares * (entry - stop)})
            exposure += notional
            held.add(code)
            taken += 1
        curve.append((d, equity))

    return trades, pd.DataFrame(curve, columns=["date", "equity"])


def report(trades, curve, label):
    tdf = pd.DataFrame(trades)
    eq = curve["equity"]
    ret = eq.iloc[-1] / INIT_CAPITAL - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    years = len(curve) / 244
    cagr = (eq.iloc[-1] / INIT_CAPITAL) ** (1 / years) - 1
    if len(tdf):
        win = (tdf["pnl"] > 0).mean()
        avg_r = tdf["r"].mean()
        pf = tdf.loc[tdf.pnl > 0, "pnl"].sum() / max(1, -tdf.loc[tdf.pnl < 0, "pnl"].sum())
    else:
        win = avg_r = pf = 0
    print(f"\n=== 策略 {label} ===")
    print(f"交易次數: {len(tdf)}  勝率: {win:.1%}  平均R: {avg_r:.2f}  獲利因子: {pf:.2f}")
    print(f"總報酬: {ret:+.1%}  年化: {cagr:+.1%}  最大回撤: {dd:.1%}")
    print(f"期末權益: {eq.iloc[-1]:,.0f}")
    return tdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--strategy", default="all", choices=["A", "B", "all"])
    args = ap.parse_args()
    print("載入資料...")
    data, names = load_all()
    print(f"{len(data)} 檔股票")
    keys = ["A", "B"] if args.strategy == "all" else [args.strategy]
    for k in keys:
        trades, curve = run(data, names, k, leverage=args.leverage)
        tdf = report(trades, curve, k)
        tdf.to_csv(f"trades_{k}.csv", index=False)
        curve.to_csv(f"equity_{k}.csv", index=False)


if __name__ == "__main__":
    main()
