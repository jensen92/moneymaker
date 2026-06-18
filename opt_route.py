"""優化路線驗證: 近三年交易分析顯示 3R+ 大贏家貢獻 ~108-120% 總損益,
而 C/D 目前 use_three_week=False (一律 +20% 賣半), 可能砍掉最關鍵的右尾。

測試: 對 C 與 D 開啟三週法則 (快速 +25% 不賣半, 讓主升段奔跑), 比較
全期與近三年的 報酬/PF/MaxDD/MAR。其餘參數完全不動 (非調參, 只切一個旗標)。
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


def metrics(trades, curve, init, all_dates, since=None):
    eq = curve.set_index("date")["equity"].sort_index()
    if since is not None:
        eq = eq[eq.index >= since]
    if len(eq) == 0:
        return None
    base = eq.iloc[0]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / base) ** (1 / max(years, .01)) - 1 if eq.iloc[-1] > 0 else -1
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
    return dict(n=n, wr=wr, pf=pf, cagr=cagr, dd=dd,
                mar=cagr / dd if dd > 0 else np.nan)


def run(strategy, cfg, data, regime, regime_tiers, vol_scalars, all_dates, date_idx):
    # 暫時覆寫該策略 config
    cfg_attr = {"C": "C_CONFIG", "D": "D_CONFIG"}[strategy]
    orig = getattr(stg, cfg_attr)
    setattr(stg, cfg_attr, cfg)
    try:
        sigs = bt.collect_signals(data, strategy)
        sigs = {d: lst for d, lst in sigs.items() if d in regime}
        em = bt.build_entry_map(sigs, data)
        trades, curve = bt.run_sub(data, em, strategy, 1.0, 2_000_000,
                                   all_dates=all_dates, date_idx=date_idx,
                                   regime_tiers=regime_tiers, vol_scalars=vol_scalars)
    finally:
        setattr(stg, cfg_attr, orig)
    return trades, curve


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    variants = {
        "C 現行 (賣半)":   ("C", dict(stg.C_CONFIG)),
        "C 三週法則(奔跑)": ("C", {**stg.C_CONFIG, "use_three_week": True, "three_week_gain": 0.25}),
        "D 現行 (賣半)":   ("D", dict(stg.D_CONFIG)),
        "D 三週法則(奔跑)": ("D", {**stg.D_CONFIG, "use_three_week": True, "three_week_gain": 0.25}),
    }

    hdr = f"  {'版本':<20}{'交易':>5}{'勝率':>7}{'PF':>6}{'CAGR':>7}{'MaxDD':>7}{'MAR':>6}"
    full_rows, y3_rows = [], []
    for label, (strat, cfg) in variants.items():
        tr, cv = run(strat, cfg, data, regime, regime_tiers, vol_scalars, all_dates, date_idx)
        full_rows.append((label, metrics(tr, cv, 2_000_000, all_dates)))
        y3_rows.append((label, metrics(tr, cv, 2_000_000, all_dates, since=TRADE_START)))

    def show(rows):
        print(hdr)
        for label, m in rows:
            print(f"  {label:<20}{m['n']:>5}{m['wr']*100:>6.1f}%{m['pf']:>6.2f}"
                  f"{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%{m['mar']:>6.2f}")

    print("\n=== 全期 (2011-2026) ===")
    show(full_rows)
    print("\n=== 近三年 (2023-06 ~ 2026-06) ===")
    show(y3_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
