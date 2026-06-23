"""策略 J 調參掃描 (單次載入資料, 多組 config), 目標: 拉高精準度與 PF/MAR,
同時保持飆股捕捉率明顯高於 C/D 的 2.4%。每組印 捕捉率/精準度/PF/MaxDD/MAR。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402
import strategies as stg  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
TRADE_START = pd.Timestamp("2023-06-01")
WINDOW, THRESH = 60, 0.80


def find_rocket_events(df):
    close, high, n = df["close"].values, df["high"].values, len(df)
    ev, i = [], 0
    while i < n - WINDOW:
        if close[i] <= 0:
            i += 1; continue
        fut = high[i + 1:i + 1 + WINDOW]
        if len(fut) == 0:
            i += 1; continue
        rel = fut.argmax(); gain = fut[rel] / close[i] - 1.0
        if gain >= THRESH:
            peak = i + 1 + rel; ev.append((i, peak, gain)); i = peak + 1
        else:
            i += 1
    return ev


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
    data, _ = bt.load_all()
    bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    rockets = []
    for code, df in data.items():
        if df["date"].iloc[-1] < TRADE_START:
            continue
        for s_i, p_i, gain in find_rocket_events(df):
            sd, pd_ = df["date"].iloc[s_i], df["date"].iloc[p_i]
            if pd_ >= TRADE_START:
                rockets.append((code, sd, pd_, gain))
    n_rk = len(rockets)
    rk_by_code = {}
    for code, sd, pd_, gain in rockets:
        rk_by_code.setdefault(code, []).append((sd, pd_))
    print(f"飆股事件: {n_rk}\n")

    base = dict(stg.J_CONFIG)
    variants = {
        "V0 baseline":       base,
        "V1 嚴進(rs.92,vs4,g.07)": {**base, "rs_min": 0.92, "vol_surge": 4.0, "min_gain": 0.07},
        "V2 緊跟(trail2.0)":  {**base, "trail_atr": 2.0},
        "V3 緊停(stop.10,tr2.5)": {**base, "stop_pct": 0.10, "trail_atr": 2.5},
        "V4 起漲(ext1.25,vs4)": {**base, "max_ext_ma50": 1.25, "vol_surge": 4.0},
        "V5 強點火(g.08,th.20)": {**base, "min_gain": 0.08, "thrust_gain": 0.20},
        "V6 綜合(rs.92,vs4,ext1.25,tr2.5)": {**base, "rs_min": 0.92, "vol_surge": 4.0,
                                            "max_ext_ma50": 1.25, "trail_atr": 2.5},
    }

    hdr = (f"  {'版本':<28}{'捕捉率':>8}{'精準':>7}"
           f"{'交易':>6}{'勝率':>6}{'PF':>6}{'CAGR':>7}{'DD':>6}{'MAR':>6}")
    print(hdr)
    for label, cfg in variants.items():
        stg.J_CONFIG = cfg
        try:
            sigs = bt.collect_signals(data, "J")
            sigs = {dt: lst for dt, lst in sigs.items() if dt in regime}
            em = bt.build_entry_map(sigs, data)
            trades, curve = bt.run_sub(data, em, "J", 1.0, 2_000_000,
                                       all_dates=all_dates, date_idx=date_idx,
                                       regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        finally:
            stg.J_CONFIG = base
        tdf = pd.DataFrame(trades)
        tdf["entry_date"] = pd.to_datetime(tdf["entry_date"])
        tdf["exit_date"] = pd.to_datetime(tdf["exit_date"])
        caught = 0
        for code, sd, pd_, gain in rockets:
            ct = tdf[(tdf["code"] == code)
                     & (tdf["entry_date"] >= sd - pd.Timedelta(days=30))
                     & (tdf["entry_date"] <= pd_)]
            if len(ct):
                caught += 1
        tdf3 = tdf[tdf["exit_date"] >= TRADE_START]
        hits = 0
        for _, t in tdf3.iterrows():
            for sd, pd_ in rk_by_code.get(t["code"], []):
                if sd - pd.Timedelta(days=30) <= t["entry_date"] <= pd_:
                    hits += 1; break
        prec = hits / len(tdf3) if len(tdf3) else 0
        m = metrics(trades, curve)
        print(f"  {label:<28}{caught/n_rk*100:>6.1f}% {prec*100:>5.0f}%"
              f"{m['n']:>6}{m['wr']*100:>5.0f}%{m['pf']:>6.2f}"
              f"{m['cagr']*100:>6.1f}%{m['dd']*100:>5.1f}%{m['mar']:>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
