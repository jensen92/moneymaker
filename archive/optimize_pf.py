"""高 PF 選股策略搜尋: 以已驗證的 D 引擎掃描進場品質, 目標 PF > 2.

思路: 不再找新書策略 (E/F/G/H/I 的 PF 均輸 D). 改以 D 的趨勢模板 + VCP 突破
為基底, 加嚴「進場品質」(更高 RS、更緊收縮、更強爆量) 並恢復「讓贏家奔跑」
(use_three_week=True), 換取更高的獲利因子 (PF). 進場越嚴 → 交易越少但越精準。

_d_features 對全市場每根只算一次 (重運算一次), 再以不同 cfg 快速套 _d_signal
→ 回測, 報告 PF/勝率/CAGR/DD/MAR, 篩出 PF>2 且 CAGR 仍佳者.

用法: python3 optimize_pf.py
"""
import os
import itertools

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")

from backtest import (load_all, build_entry_map, run_sub, load_regime,
                      load_regime_tiers, load_vol_scalars, build_date_index,
                      compute_rs_rank, INIT_CAPITAL)
from strategies import _d_features, _d_signal


def precompute_d(data, rs):
    """對每檔每根算 _d_features 一次. 回傳 {signal_date: [(code, i, feat)]}."""
    feats = {}
    for code, df in data.items():
        for i in range(210, len(df) - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if np.isnan(rank):
                continue
            f = _d_features(df, i, rank)
            if f is not None:
                feats.setdefault(d, []).append((code, i, f))
    return feats


def build_entry_from_feats(feats, data, cfg, risk_on):
    entry_map = {}
    for d, lst in feats.items():
        if d not in risk_on:
            continue
        for code, i, f in lst:
            s = _d_signal(f, cfg)
            if not s:
                continue
            df = data[code]
            if i + 1 < len(df):
                nd = df["date"].iloc[i + 1]
                entry_map.setdefault(nd, []).append((s["score"], code, i + 1, s))
    return entry_map


def metrics(eq):
    end, base = eq.iloc[-1], eq.iloc[0]
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    years = len(eq) / 244
    cagr = (end / base) ** (1 / max(years, 0.01)) - 1
    mar = cagr / dd if dd > 0 else float("inf")
    return cagr, dd, mar


def main():
    print("載入資料...")
    data, _ = load_all()
    print(f"{len(data)} 檔股票")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    all_dates, date_idx = build_date_index(data)
    init_eq = INIT_CAPITAL / 2

    print("預先計算 D 特徵 (一次性)...")
    feats = precompute_d(data, rs)

    # 掃描網格: 聚焦進場品質 + 出場 (讓贏家奔跑提升 PF)
    grid = {
        "rs_min":         [0.85, 0.90, 0.93],   # Minervini 偏好 ~90
        "contraction":    [0.65, 0.70, 0.75],   # 越小越嚴 (波動收縮越緊)
        "vol_mult":       [1.5, 2.0, 2.5],       # 突破爆量倍數
        "gain_cap":       [0.06, 0.08],          # 突破日漲幅上限 (不追高)
        "stop_pct":       [0.08, 0.10],
        "use_three_week": [True],                # 讓贏家奔跑 (高 PF 關鍵)
    }
    base = {"three_week_gain": 0.20, "max_hold": 9999}
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"掃描 {len(combos)} 組...\n")

    print(f"{'rs':>4} {'cont':>5} {'vol':>4} {'gain':>5} {'stop':>5} | "
          f"{'交易':>5} {'勝率':>6} {'PF':>5} {'年化':>7} {'回撤':>6} {'MAR':>6}")
    print("-" * 78)

    rows = []
    for combo in combos:
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        em = build_entry_from_feats(feats, data, cfg, risk_on)
        trades, curve = run_sub(data, em, "D", 1.0, init_eq, all_dates=all_dates,
                                date_idx=date_idx, regime_tiers=regime_tiers,
                                vol_scalars=vol_scalars)
        eq = curve.set_index("date")["equity"]
        cagr, dd, mar = metrics(eq)
        td = pd.DataFrame(trades)
        if len(td):
            win = (td["pnl"] > 0).mean()
            pf = (td.loc[td.pnl > 0, "pnl"].sum()
                  / max(1, -td.loc[td.pnl < 0, "pnl"].sum()))
        else:
            win = pf = 0
        rows.append({**dict(zip(keys, combo)), "n": len(td), "win": win,
                     "pf": pf, "cagr": cagr, "dd": dd, "mar": mar})
        print(f"{cfg['rs_min']:>4.2f} {cfg['contraction']:>5.2f} "
              f"{cfg['vol_mult']:>4.1f} {cfg['gain_cap']:>5.2f} "
              f"{cfg['stop_pct']:>5.2f} | {len(td):>5} {win:>6.1%} {pf:>5.2f} "
              f"{cagr:>+6.1%} {dd:>6.1%} {mar:>6.2f}")

    res = pd.DataFrame(rows).sort_values("pf", ascending=False)
    res.to_csv("optimize_pf_results.csv", index=False)

    good = res[(res.pf >= 2.0) & (res.n >= 100)]
    print("\n=== PF >= 2.0 且交易數 >= 100 (依 PF 排序) ===")
    if len(good):
        print(good.head(10).to_string(index=False))
    else:
        print("無組合達 PF>=2. 顯示 PF 前 5 名:")
        print(res.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
