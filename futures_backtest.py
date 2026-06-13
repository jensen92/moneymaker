"""穀物期貨回測引擎 (黃豆 / 小麥 / 玉米) — 多空雙向, 單一帳戶.

部位控制:
  - 單筆風險 = 當下權益(含未實現) × RISK_PCT, 以「停損距離 × 點值」換算口數
  - 口數至少 1 口; 名目曝險上限 = 權益 × MAX_NOTIONAL
  - 每商品每策略同時最多 1 個部位 (always-in 策略以反手換邊)

成本: 每口每邊手續費 + 滑價 (跳動點), 於平倉時一次計入往返成本.
進場: 訊號日次一交易日開盤價. 出場: 停損 / 通道 / 中軌 / 時間.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from futures_data import FUTURES
from futures_strategies import STRATEGIES, STRATEGY_NAMES, add_indicators

DATA_DIR = os.path.join(os.path.dirname(__file__), "futures_data")
INIT_CAPITAL = 250_000.0
RISK_PCT = 0.01          # 單筆風險佔權益比例
MAX_NOTIONAL = 8.0       # 單商品名目曝險 / 權益 上限
COMMISSION = 2.5         # 每口每邊手續費 (美元)
SLIP_TICKS = 1.0         # 每邊滑價 (跳動點數)
TRADING_DAYS = 252


def load_all():
    data = {}
    for key in FUTURES:
        path = os.path.join(DATA_DIR, f"{key}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        data[key] = add_indicators(df).reset_index(drop=True)
    return data


def rt_cost(market):
    """每口往返成本 (兩邊手續費 + 兩邊滑價)."""
    spec = FUTURES[market]
    return 2 * COMMISSION + 2 * SLIP_TICKS * spec["tick"] * spec["point_value"]


def collect_signals(data, key):
    """各商品各 bar 算訊號, 回傳 entry_map[執行日] = [(market, sig, entry_idx)]."""
    sig_fn = STRATEGIES[key]
    entry_map = defaultdict(list)
    for market, df in data.items():
        for i in range(1, len(df) - 1):
            s = sig_fn(df, i)
            if s:
                entry_map[df["date"].iloc[i + 1]].append((market, s, i + 1))
    return entry_map


def _build_index(data):
    all_dates = sorted({d for df in data.values() for d in df["date"]})
    date_idx = {m: {d: i for i, d in enumerate(df["date"])}
                for m, df in data.items()}
    return all_dates, date_idx


def _close_position(p, exit_price, exit_date, trades):
    pv = FUTURES[p["market"]]["point_value"]
    gross = p["dir"] * (exit_price - p["entry"]) * pv * p["contracts"]
    pnl = gross - rt_cost(p["market"]) * p["contracts"]
    trades.append({
        "market": p["market"], "dir": p["dir"],
        "entry_date": p["entry_date"], "exit_date": exit_date,
        "entry": p["entry"], "exit": exit_price, "contracts": p["contracts"],
        "pnl": pnl, "r": pnl / p["risk_amt"] if p["risk_amt"] > 0 else 0.0,
    })
    return pnl


def _check_exit(p, row, di):
    """回傳 (exit_price 或 None). 順序: 停損 → 通道/中軌 → 時間."""
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    if p["dir"] > 0:                                   # 多單
        if l <= p["stop"]:
            return min(o, p["stop"])
        if p["exit_channel"]:
            lvl = row[f"dc_lo{p['exit_channel']}"]
            if not np.isnan(lvl) and l <= lvl:
                return min(o, lvl)
        if p["exit_mid"] and not np.isnan(row["ma20"]) and h >= row["ma20"]:
            return max(o, row["ma20"])
        if p["exit_ma"]:
            mf, ms = row[f"ma{p['exit_ma'][0]}"], row[f"ma{p['exit_ma'][1]}"]
            if not np.isnan(ms) and mf < ms:
                return c                               # 趨勢翻空, 收盤出場
    else:                                              # 空單
        if h >= p["stop"]:
            return max(o, p["stop"])
        if p["exit_channel"]:
            lvl = row[f"dc_hi{p['exit_channel']}"]
            if not np.isnan(lvl) and h >= lvl:
                return max(o, lvl)
        if p["exit_mid"] and not np.isnan(row["ma20"]) and l <= row["ma20"]:
            return min(o, row["ma20"])
        if p["exit_ma"]:
            mf, ms = row[f"ma{p['exit_ma'][0]}"], row[f"ma{p['exit_ma'][1]}"]
            if not np.isnan(ms) and mf > ms:
                return c
    if di >= p["expire_idx"]:
        return c
    return None


def _update_trail(p, row):
    if not p["trail_atr"]:
        return
    atr = row["atr20"] if not np.isnan(row["atr20"]) else p["init_atr"]
    if p["dir"] > 0:
        p["hi_ext"] = max(p["hi_ext"], row["close"])
        p["stop"] = max(p["stop"], p["hi_ext"] - p["trail_atr"] * atr)
    else:
        p["hi_ext"] = min(p["hi_ext"], row["close"])
        p["stop"] = min(p["stop"], p["hi_ext"] + p["trail_atr"] * atr)


def _open_position(market, s, di, df, equity):
    row = df.iloc[di]
    entry = row["open"]
    atr = df["atr20"].iloc[di - 1]
    if np.isnan(entry) or np.isnan(atr) or atr <= 0:
        return None
    stop = entry - s["dir"] * s["stop_atr"] * atr
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return None
    pv = FUTURES[market]["point_value"]
    risk_amt = equity * RISK_PCT
    contracts = int(risk_amt / (stop_dist * pv))
    contracts = max(1, contracts)
    # 名目曝險上限
    notional = contracts * entry * pv
    if notional > equity * MAX_NOTIONAL:
        contracts = max(1, int(equity * MAX_NOTIONAL / (entry * pv)))
    return {
        "market": market, "dir": s["dir"], "entry": entry, "entry_idx": di,
        "entry_date": row["date"], "contracts": contracts,
        "stop": stop, "init_stop": stop,
        "exit_channel": s.get("exit_channel", 0),
        "exit_mid": s.get("exit_mid", False),
        "exit_ma": s.get("exit_ma", None),
        "trail_atr": s.get("trail_atr", 0.0), "reverse": s.get("reverse", False),
        "expire_idx": di + s["max_hold"],
        "hi_ext": entry, "init_atr": atr,
        "risk_amt": contracts * stop_dist * pv,
    }


def run_strategy(data, entry_map, key, init_eq, all_dates, date_idx,
                 date_from=None, date_to=None, return_open=False):
    """date_from/date_to (pd.Timestamp) 限制回測區間, 供樣本內/外驗證。

    return_open=True 時額外回傳結束時仍持有的部位 (market -> position),
    供 futures_positions.py 檢視目前持倉狀態。"""
    cash = init_eq
    open_pos = {}        # market -> position
    trades, curve = [], []
    if date_from is not None or date_to is not None:
        all_dates = [d for d in all_dates
                     if (date_from is None or d >= date_from)
                     and (date_to is None or d <= date_to)]

    for d in all_dates:
        # 1. 出場 (持倉中、且非進場當根)
        for market in list(open_pos):
            p = open_pos[market]
            di = date_idx[market].get(d)
            if di is None or di <= p["entry_idx"]:
                continue
            row = data[market].iloc[di]
            ex = _check_exit(p, row, di)
            if ex is not None:
                cash += _close_position(p, ex, d, trades)
                del open_pos[market]
            else:
                _update_trail(p, row)

        # 2. 計算未實現權益 (mark-to-market)
        unreal = 0.0
        for market, p in open_pos.items():
            di = date_idx[market].get(d)
            if di is None:
                continue
            c = data[market].iloc[di]["close"]
            pv = FUTURES[market]["point_value"]
            unreal += p["dir"] * (c - p["entry"]) * pv * p["contracts"]
        equity = cash + unreal

        # 3. 進場 / 反手 (前一日訊號於今日開盤執行)
        for market, s, ei in entry_map.get(d, []):
            held = open_pos.get(market)
            if held is not None:
                if held["dir"] == s["dir"]:
                    continue                       # 同向, 已在場
                if not (held["reverse"] and s.get("reverse")):
                    continue                       # 非反手策略: 等出場
                # 反手: 先以今日開盤平舊倉, 再開反向
                row = data[market].iloc[ei]
                cash += _close_position(held, row["open"], d, trades)
                del open_pos[market]
                equity = cash + sum(
                    pp["dir"] * (data[m].iloc[date_idx[m][d]]["close"] - pp["entry"])
                    * FUTURES[m]["point_value"] * pp["contracts"]
                    for m, pp in open_pos.items() if date_idx[m].get(d) is not None)
            pos = _open_position(market, s, ei, data[market], equity)
            if pos is not None:
                open_pos[market] = pos

        curve.append({"date": d, "equity": equity})

    if return_open:
        return trades, pd.DataFrame(curve), open_pos
    return trades, pd.DataFrame(curve)


def metrics(trades, curve, init_eq):
    eq = curve["equity"]
    ret = eq.iloc[-1] / init_eq - 1
    years = len(curve) / TRADING_DAYS
    if eq.iloc[-1] <= 0:                       # 爆倉: 年化以 -100% 計
        cagr = -1.0
    else:
        cagr = (eq.iloc[-1] / init_eq) ** (1 / max(years, 0.01)) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    daily = eq.pct_change().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)
              if daily.std() > 0 else 0.0)
    tdf = pd.DataFrame(trades)
    if len(tdf):
        win = (tdf["pnl"] > 0).mean()
        pf = (tdf.loc[tdf.pnl > 0, "pnl"].sum()
              / max(1e-9, -tdf.loc[tdf.pnl < 0, "pnl"].sum()))
        avg_r = tdf["r"].mean()
    else:
        win = pf = avg_r = 0.0
    return {"n": len(tdf), "ret": ret, "cagr": cagr, "dd": dd,
            "sharpe": sharpe, "win": win, "pf": pf, "avg_r": avg_r,
            "mar": cagr / dd if dd > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="T,D,M,B,S")
    ap.add_argument("--risk", type=float, default=None,
                    help="覆寫單筆風險比例 (例如 0.02)")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--allow-short", action="store_true",
                    help="開啟做空 (預設僅做多)")
    args = ap.parse_args()
    keys = args.strategies.split(",")

    global RISK_PCT, INIT_CAPITAL
    if args.risk:
        RISK_PCT = args.risk
    if args.capital:
        INIT_CAPITAL = args.capital
    if args.allow_short:
        import futures_strategies
        futures_strategies.ALLOW_SHORT = True

    print("載入穀物期貨資料...")
    data = load_all()
    for m, df in data.items():
        print(f"  {m} {FUTURES[m]['name']:<14} {len(df)} bars "
              f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    all_dates, date_idx = _build_index(data)
    init_eq = INIT_CAPITAL          # 每策略獨立以全額本金衡量

    print(f"\n單策略表現 (各以 {init_eq:,.0f} 本金, 風險 {RISK_PCT:.0%}/筆)\n")
    hdr = (f"{'策略':<16} {'交易':>5} {'勝率':>6} {'均R':>6} {'PF':>5} "
           f"{'總報酬':>8} {'年化':>7} {'MaxDD':>7} {'Sharpe':>7} {'MAR':>5}")
    print(hdr)
    print("-" * len(hdr))

    curves = {}
    all_trades = []
    for k in keys:
        emap = collect_signals(data, k)
        trades, curve = run_strategy(data, emap, k, init_eq, all_dates, date_idx)
        curves[k] = curve
        for t in trades:
            t["strategy"] = k
        all_trades.extend(trades)
        m = metrics(trades, curve, init_eq)
        print(f"{k} {STRATEGY_NAMES[k]:<13} {m['n']:>5} {m['win']:>6.1%} "
              f"{m['avg_r']:>6.2f} {m['pf']:>5.2f} {m['ret']:>8.1%} "
              f"{m['cagr']:>7.1%} {m['dd']:>7.1%} {m['sharpe']:>7.2f} "
              f"{m['mar']:>5.2f}")

    # 組合: 等權配置, 各策略正規化權益 (equity/本金) 取平均後再乘回本金
    # (相當於每日再平衡至等權, 用以呈現策略間的分散效果)
    combined = None
    for k in keys:
        c = curves[k][["date", "equity"]].copy()
        c[k] = c["equity"] / init_eq
        combined = c[["date", k]] if combined is None \
            else combined.merge(c[["date", k]], on="date", how="outer")
    combined = combined.sort_values("date").ffill().fillna(1.0)
    combined["equity"] = combined[keys].mean(axis=1) * INIT_CAPITAL
    cm = metrics(all_trades, combined[["equity"]].reset_index(drop=True),
                 INIT_CAPITAL)
    print("-" * len(hdr))
    print(f"{'組合 (等權)':<14} {cm['n']:>5} {cm['win']:>6.1%} "
          f"{cm['avg_r']:>6.2f} {cm['pf']:>5.2f} {cm['ret']:>8.1%} "
          f"{cm['cagr']:>7.1%} {cm['dd']:>7.1%} {cm['sharpe']:>7.2f} "
          f"{cm['mar']:>5.2f}")

    pd.DataFrame(all_trades).to_csv("futures_trades.csv", index=False)
    combined[["date", "equity"]].to_csv("futures_equity.csv", index=False)
    print("\n→ futures_trades.csv, futures_equity.csv")


if __name__ == "__main__":
    main()
