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
    print(f"🌽 穀物期貨訊號 · 本金 {args.capital:,.0f} · 風險 {args.risk:.0%}/筆 · 截至 {last_date.strftime('%m-%d')}\n")

    all_dates, date_idx = _build_index(data)
    print("績效(全期)")
    for key in keys:
        emap = collect_signals(data, key)
        trades, curve = run_strategy(data, emap, key, args.capital, all_dates, date_idx)
        m = metrics(trades, curve, args.capital)
        print(f"{key} {STRATEGY_NAMES[key]}｜{m['n']}筆｜PF {m['pf']:.2f}｜"
              f"勝率 {m['win']:.0%}｜DD {m['dd']:.0%}｜MAR {m['mar']:.2f}")

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
        print("\n⚪ 今日無新進場訊號")
        print("(持倉依各策略移動停損/通道出場規則管理)")
        return

    print("\n🟢 今日進場訊號（次一交易日開盤進場）")
    for key, mk, d, ref, stop, n, risk in hits:
        side = "做多" if d > 0 else "做空"
        print(f"{key} {STRATEGY_NAMES[key]}·{FUTURES[mk]['name']} {side}")
        print(f"  進場 {ref:,.1f}｜停損 {stop:,.1f}｜{n}口（風險 ${risk:,.0f}）")
    print("停利 無 · 依各策略反向通道/MA交叉/季節到期浮動鎖利")


if __name__ == "__main__":
    main()
