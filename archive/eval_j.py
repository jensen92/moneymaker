"""策略 J (動能點火) 驗證: 同時檢驗使用者的兩個要求 —
   (1) 抓得到飆股嗎?  (捕捉率, 對比 C/D 的 2.4%)
   (2) 有沒有抓一堆非飆股?  (精準度: J 的訊號裡多少落在飆股波段內)
並走真實引擎算 J 的獨立績效 (報酬/勝率/PF/MaxDD/MAR)。

飆股事件定義同 analyze_rockets.py: 任一日起算未來 60 交易日內最高價漲幅 >= 80%。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
TRADE_START = pd.Timestamp("2023-06-01")
WINDOW = 60
THRESH = 0.80


def find_rocket_events(df):
    close = df["close"].values
    high = df["high"].values
    n = len(df)
    events, i = [], 0
    while i < n - WINDOW:
        if close[i] <= 0:
            i += 1
            continue
        fut = high[i + 1:i + 1 + WINDOW]
        if len(fut) == 0:
            i += 1
            continue
        rel = fut.argmax()
        gain = fut[rel] / close[i] - 1.0
        if gain >= THRESH:
            peak = i + 1 + rel
            events.append((i, peak, gain))
            i = peak + 1
        else:
            i += 1
    return events


def metrics(trades, curve, since=None):
    eq = curve.set_index("date")["equity"].sort_index()
    if since is not None:
        eq = eq[eq.index >= since]
    if len(eq) < 2:
        return None
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, .01)) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    td = pd.DataFrame(trades)
    if since is not None and not td.empty:
        td = td[pd.to_datetime(td["exit_date"]) >= since]
    n = len(td)
    if n:
        w = td["pnl"] > 0
        gl = -td.loc[~w, "pnl"].sum()
        pf = td.loc[w, "pnl"].sum() / gl if gl > 0 else np.inf
        wr = w.mean()
    else:
        pf = wr = 0
    return dict(n=n, wr=wr, pf=pf, cagr=cagr, dd=dd, mar=cagr / dd if dd > 0 else np.nan)


def main():
    print("載入資料 ...")
    data, names = bt.load_all()
    bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    print("掃描飆股事件 ...")
    rockets = []
    for code, df in data.items():
        if df["date"].iloc[-1] < TRADE_START:
            continue
        for s_i, p_i, gain in find_rocket_events(df):
            sd, pd_ = df["date"].iloc[s_i], df["date"].iloc[p_i]
            if pd_ < TRADE_START:
                continue
            rockets.append((code, sd, pd_, gain))
    n_rk = len(rockets)
    print(f"近三年飆股事件: {n_rk} (跨 {len(set(r[0] for r in rockets))} 檔)\n")

    for strat in ["J", "C", "D"]:
        sigs = bt.collect_signals(data, strat)
        sigs = {dt: lst for dt, lst in sigs.items() if dt in regime}
        em = bt.build_entry_map(sigs, data)
        trades, curve = bt.run_sub(data, em, strat, 1.0, 2_000_000,
                                   all_dates=all_dates, date_idx=date_idx,
                                   regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        tdf = pd.DataFrame(trades)
        tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
        tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])
        tdf3 = tdf[tdf["exit_date"] >= TRADE_START]

        # 捕捉率: 每個飆股事件, 是否有進場落在 [start-30d, peak]
        caught = 0
        for code, sd, pd_, gain in rockets:
            ct = tdf[(tdf["code"] == code)
                     & (tdf["entry_date"] >= sd - pd.Timedelta(days=30))
                     & (tdf["entry_date"] <= pd_)]
            if len(ct):
                caught += 1

        # 精準度: 近三年每筆進場, 是否落在「某飆股事件波段內」
        rk_by_code = {}
        for code, sd, pd_, gain in rockets:
            rk_by_code.setdefault(code, []).append((sd, pd_))
        hits = 0
        for _, t in tdf3.iterrows():
            for sd, pd_ in rk_by_code.get(t["code"], []):
                if sd - pd.Timedelta(days=30) <= t["entry_date"] <= pd_:
                    hits += 1
                    break
        n_tr3 = len(tdf3)

        print(f"=== 策略 {strat} ===")
        print(f"  飆股捕捉率: {caught}/{n_rk} = {caught/n_rk*100:.1f}%")
        if n_tr3:
            print(f"  進場精準度 (近三年 {n_tr3} 筆進場落在飆股波段內): "
                  f"{hits}/{n_tr3} = {hits/n_tr3*100:.1f}%")
        for label, since in [("全期", None), ("近三年", TRADE_START)]:
            m = metrics(trades, curve, since)
            if m:
                print(f"  [{label}] 交易{m['n']:>4} 勝率{m['wr']*100:>4.0f}% "
                      f"PF{m['pf']:>5.2f} CAGR{m['cagr']*100:>5.1f}% "
                      f"MaxDD{m['dd']*100:>4.1f}% MAR{m['mar']:>5.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
