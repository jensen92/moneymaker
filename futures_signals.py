"""穀物期貨每日訊號 — 跑最新一根 K 棒, 列出明日開盤可進場的部位.

用法:
    python3 futures_data.py              # 先更新資料
    python3 futures_signals.py           # 預設組合 M,D,S
    python3 futures_signals.py --strategies T,D,M,B,S --capital 250000 --risk 0.01

進場價以「次一交易日開盤」為準, 此處以最新收盤價為參考估算停損與口數。
"""
import argparse

from futures_backtest import (load_all, RISK_PCT, INIT_CAPITAL, MAX_NOTIONAL,
                              collect_signals, run_strategy, metrics, _build_index)
from futures_data import FUTURES
from futures_strategies import STRATEGIES, STRATEGY_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="M,D,S")
    ap.add_argument("--capital", type=float, default=INIT_CAPITAL)
    ap.add_argument("--risk", type=float, default=RISK_PCT)
    args = ap.parse_args()
    keys = args.strategies.split(",")

    data = load_all()
    last_date = max(df["date"].iloc[-1] for df in data.values())
    print(f"資料截至 {last_date.date()}  |  本金 {args.capital:,.0f}  "
          f"風險 {args.risk:.1%}/筆\n")

    all_dates, date_idx = _build_index(data)
    print("策略歷史績效 (全期回測):")
    for key in keys:
        emap = collect_signals(data, key)
        trades, curve = run_strategy(data, emap, key, args.capital, all_dates, date_idx)
        m = metrics(trades, curve, args.capital)
        print(f"  {key} {STRATEGY_NAMES[key]:<11} 交易{m['n']:>4}  勝率{m['win']:>6.1%}  "
              f"PF{m['pf']:>5.2f}  最大回撤{m['dd']:>6.1%}  MAR{m['mar']:>5.2f}")
    print()

    hits = []
    for key in keys:
        fn = STRATEGIES[key]
        for market, df in data.items():
            i = len(df) - 1
            s = fn(df, i)
            if not s:
                continue
            row = df.iloc[i]
            ref = row["close"]                          # 次日開盤的參考價
            atr = row["atr20"]
            stop = ref - s["dir"] * s["stop_atr"] * atr
            stop_dist = abs(ref - stop)
            pv = FUTURES[market]["point_value"]
            contracts = max(1, int(args.capital * args.risk / (stop_dist * pv)))
            risk_usd = contracts * stop_dist * pv
            hits.append((key, market, s["dir"], ref, stop, contracts, risk_usd))

    if not hits:
        print("今日無新進場訊號 (持有中部位請依各策略移動停損/通道出場規則管理)。")
        return

    print(f"{'策略':<14} {'商品':<14} {'方向':<4} {'進場價':>9} "
          f"{'停損價':>9} {'口數':>4} {'風險$':>9}")
    print("-" * 70)
    for key, m, d, ref, stop, n, risk in hits:
        side = "做多" if d > 0 else "做空"
        print(f"{key} {STRATEGY_NAMES[key]:<11} {FUTURES[m]['name']:<12} "
              f"{side:<4} {ref:>9.2f} {stop:>9.2f} {n:>4} {risk:>9,.0f}")
    print("\n進場 = 次一交易日開盤; 停損為初始值, 部分策略出場後續以反向通道/MA交叉動態調整。")
    print("停利: 無固定停利, 依各策略出場規則 (反向通道/MA交叉/季節持有到期) 浮動鎖利。")


if __name__ == "__main__":
    main()
