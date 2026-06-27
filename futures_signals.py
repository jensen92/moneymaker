"""穀物期貨每日訊號 — 跑最新一根 K 棒, 列出明日開盤可進場的部位.

用法:
    python3 futures_data.py              # 先更新資料
    python3 futures_signals.py           # 預設組合 M,D,S
    python3 futures_signals.py --strategies T,D,M,B,S --capital 250000 --risk 0.01

進場價以「次一交易日開盤」為準, 此處以最新收盤價為參考估算停損與口數。
"""
import argparse

import numpy as np

from futures_backtest import (load_all, RISK_PCT, INIT_CAPITAL, MAX_NOTIONAL,
                              collect_signals, run_strategy, metrics, _build_index)
from futures_data import FUTURES
from futures_strategies import STRATEGIES, STRATEGY_NAMES, CONFIG


def prospective(key, market, df, capital, risk_pct):
    """該策略/商品『待觸發』的預備進場計劃 (供盯盤/掛單參考)。
    突破型(T/D)給明確突破價+預估停損+風險金額; 均線(M)/季節(S)無固定價, 給狀態。"""
    row = df.iloc[-1]
    close, atr = row["close"], row["atr20"]
    pv = FUTURES[market]["point_value"]
    cfg = CONFIG[key]
    if key in ("T", "D"):
        lvl = row[f"dc_hi{cfg['entry']}"]
        if np.isnan(lvl) or np.isnan(atr) or atr <= 0:
            return None
        stop = lvl - cfg["stop"] * atr
        dist = abs(lvl - stop)
        n = max(1, int(capital * risk_pct / (dist * pv)))
        return (f"突破 {lvl:,.1f} 做多｜停損 {stop:,.1f}｜{n}口"
                f"（風險 ${n * dist * pv:,.0f}）｜距 {(lvl - close) / close:+.1%}")
    if key == "M":
        fma, sma = row[f"ma{cfg['fast']}"], row[f"ma{cfg['slow']}"]
        if np.isnan(sma):
            return None
        st = "多頭排列(等下次交叉)" if fma > sma else "待黃金交叉做多"
        return f"MA{cfg['fast']}/{cfg['slow']} {st}（差 {fma / sma - 1:+.1%}）"
    if key == "S":
        return f"季節做多: 每年 {cfg['month']}月初進場（現 {row['date'].month}月）"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="S")
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

    triggered = {(key, mk) for key, mk, *_ in hits}

    if hits:
        print("\n🟢 今日進場訊號（次一交易日開盤進場）")
        for key, mk, d, ref, stop, n, risk in hits:
            side = "做多" if d > 0 else "做空"
            print(f"{key} {STRATEGY_NAMES[key]}·{FUTURES[mk]['name']} {side}")
            print(f"  進場 {ref:,.1f}｜停損 {stop:,.1f}｜{n}口（風險 ${risk:,.0f}）")
        print("停利 無 · 依各策略反向通道/MA交叉/季節到期浮動鎖利")
    else:
        print("\n⚪ 今日無新進場訊號")

    # 預備計劃 — 尚未觸發者的進場價/停損/風險金額 (盯盤掛單參考)
    # 突破型(D/T)有明確價格, 排前面; 季節(S)各商品相同, 收成一行
    print("\n📋 預備計劃（待觸發, 進場/停損/風險金額）")
    for key in sorted(keys, key=lambda k: {"D": 0, "T": 0, "M": 1, "S": 2}.get(k, 3)):
        if key == "S":
            from futures_strategies import S_SEASON
            mth = next(iter(data.values())).iloc[-1]["date"].month
            wins = "／".join(f"{FUTURES[mk]['name'][:2]}{S_SEASON[mk]['month']}月"
                             for mk in data if mk in S_SEASON)
            print(f"S 季節做多（各穀物專屬窗, 現 {mth}月）：{wins}進場")
            continue
        for market, df in data.items():
            if (key, market) in triggered:
                continue
            plan = prospective(key, market, df, args.capital, args.risk)
            if plan:
                print(f"{key}·{FUTURES[market]['name']}：{plan}")
    print("(持倉依各策略移動停損/通道出場規則管理)")

    # 短線首選: 直接內嵌穀物價差訊號 (市場中性, 勝率72-81% DD9-10%)
    try:
        import grain_spread as gx
        lines = gx.summary_lines()
        if lines:
            print("\n⚖️ 穀物價差短線（市場中性, 詳見 /spread）")
            for ln in lines:
                print(ln)
    except Exception:
        pass
    print("\n(單邊趨勢 D/M DD高30-40%且短線無穩定edge, 已不列預設; 需要可 /futures D,M)")


if __name__ == "__main__":
    main()
