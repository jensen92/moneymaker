"""策略 E (強勢股回檔買進) 優化: 找出正期望值且與 C/D 低相關的參數.

E 的價值在「分散」: 與 C/D 報酬流低相關. 但前一版單獨賠錢 (PF 0.73), 無效.
本腳本先把 _e_features 對全市場每根算一次 (重運算只做一次), 再以不同 cfg
快速套 _e_signal → 回測, 選出 MAR 最佳且 CAGR>0 的組合, 並印出與 C/D 相關係數.

用法: python3 optimize_e.py
"""
import os
import itertools

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")

from backtest import (load_all, collect_signals, build_entry_map, run_sub,
                      load_regime, load_regime_tiers, load_vol_scalars,
                      build_date_index, compute_rs_rank, INIT_CAPITAL)
from strategies import _e_features, _e_signal


def precompute_e(data, rs):
    """對每檔每根算 _e_features 一次. 回傳 {signal_date: [(code, i, feat)]}."""
    feats = {}
    for code, df in data.items():
        for i in range(200, len(df) - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if np.isnan(rank):
                continue
            f = _e_features(df, i, rank)
            if f is not None:
                feats.setdefault(d, []).append((code, i, f))
    return feats


def build_entry_from_feats(feats, data, cfg, risk_on):
    """套 cfg 把特徵轉成訊號, 再轉成次日進場 entry_map."""
    entry_map = {}
    for d, lst in feats.items():
        if d not in risk_on:
            continue
        for code, i, f in lst:
            s = _e_signal(f, cfg)
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

    print("預先計算 E 特徵 (一次性)...")
    feats = precompute_e(data, rs)

    # C / D 月報酬 (基準, 算一次) 供相關係數比較
    print("計算 C / D 基準曲線...")
    cd_curves = {}
    for k in ("C", "D"):
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        em = build_entry_map(sigs, data)
        _, curve = run_sub(data, em, k, 1.0, init_eq, all_dates=all_dates,
                           date_idx=date_idx, regime_tiers=regime_tiers,
                           vol_scalars=vol_scalars)
        cd_curves[k] = curve.set_index("date")["equity"]
    cd_month = {k: c.resample("ME").last().pct_change() for k, c in cd_curves.items()}

    # 掃描網格 (聚焦進場品質與出場節奏)
    grid = {
        "rs_min":         [0.80, 0.88],
        "pullback_max":   [0.15, 0.20],
        "near_ma_pct":    [0.03, 0.05],
        "rsi_os":         [50.0, 60.0],
        "require_dryup":  [True, False],
        "target_r":       [2.0, 3.0],
        "max_hold":       [25, 40],
    }
    base = {"pullback_min": 0.05, "require_ma20_up": True, "stop_pct": 0.07,
            "trail_atr": 0.0, "support_ma": "ma20"}
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"掃描 {len(combos)} 組...\n")

    print(f"{'rs':>4} {'pbMx':>5} {'maPc':>5} {'rsi':>4} {'dry':>4} "
          f"{'tR':>4} {'hold':>4} | {'交易':>5} {'勝率':>6} {'PF':>5} "
          f"{'年化':>7} {'回撤':>6} {'MAR':>6} {'ρC':>6} {'ρD':>6}")
    print("-" * 92)

    rows = []
    for combo in combos:
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        em = build_entry_from_feats(feats, data, cfg, risk_on)
        trades, curve = run_sub(data, em, "E", 1.0, init_eq, all_dates=all_dates,
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
        em_month = eq.resample("ME").last().pct_change()
        rho_c = em_month.corr(cd_month["C"])
        rho_d = em_month.corr(cd_month["D"])
        rows.append({**dict(zip(keys, combo)), "n": len(td), "win": win,
                     "pf": pf, "cagr": cagr, "dd": dd, "mar": mar,
                     "rho_c": rho_c, "rho_d": rho_d})
        print(f"{cfg['rs_min']:>4.2f} {cfg['pullback_max']:>5.2f} "
              f"{cfg['near_ma_pct']:>5.2f} {cfg['rsi_os']:>4.0f} "
              f"{str(cfg['require_dryup'])[0]:>4} {cfg['target_r']:>4.1f} "
              f"{cfg['max_hold']:>4} | {len(td):>5} {win:>6.1%} {pf:>5.2f} "
              f"{cagr:>+6.1%} {dd:>6.1%} {mar:>6.2f} {rho_c:>+6.2f} {rho_d:>+6.2f}")

    res = pd.DataFrame(rows).sort_values("mar", ascending=False)
    res.to_csv("optimize_e_results.csv", index=False)

    # 推薦: CAGR>0 且 與 C/D 相關均 < 0.4, 取 MAR 最高
    good = res[(res.cagr > 0) & (res.rho_c < 0.4) & (res.rho_d < 0.4)]
    print("\n=== 正期望值 + 低相關 (ρ<0.4) 前 5 名 (依 MAR) ===")
    if len(good):
        print(good.head(5).to_string(index=False))
    else:
        print("無組合同時滿足 CAGR>0 與低相關 — E 在此宇宙難有正優勢, 應誠實回報.")


if __name__ == "__main__":
    main()
