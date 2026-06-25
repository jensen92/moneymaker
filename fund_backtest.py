"""基本面門檻回測驗證: 有/無 Minervini 基本面篩選 (EPS 年增 & ROE) 的對照.

問題: 在現行技術面策略 (PA/PB/K/L/D) 的第一階段篩選後, 再加上
  「EPS 年增 >= X 且 ROE >= 15%」(point-in-time, 杜絕前視) 是否提升風險報酬?

做法 (與 production 同一引擎, 公平對照):
  1. 一次算 RS rank + 一次 _d_features, 套五組 config 收集各策略訊號。
  2. 對每個訊號以「訊號日」做 PIT 基本面查詢 (fundamentals.FundamentalDB), 依
     門檻過濾, 產生 baseline / gated 兩套訊號。
  3. 各策略獨立子帳戶回測 (相同初始資金/市況濾網/波動標量), 比較
     交易數 / 勝率 / PF / 年化 / 最大回撤 / MAR / 逐年總R。

可調: --eps 0 (年增門檻) --roe 0.15 --missing fail|pass (無基本面資料的處置)。
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt
bt.DATA_DIR = os.path.join(HERE, "data_adj")
from backtest import (load_all, load_regime, load_regime_tiers, load_vol_scalars,
                      compute_rs_rank, build_entry_map, run_sub,
                      build_date_index, INIT_CAPITAL)
import strategies as st
from fundamentals import FundamentalDB

CONFIGS = {"D": st.D_CONFIG, "K": st.K_CONFIG, "L": st.L_CONFIG,
           "PA": st.PA_CONFIG, "PB": st.PB_CONFIG}
ORDER = ["PA", "PB", "K", "L", "D"]


def build_signals(data, rs, risk_on):
    sigs = {k: defaultdict(list) for k in CONFIGS}
    for code, df in data.items():
        n = len(df)
        for i in range(210, n - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if np.isnan(rank):
                continue
            feat = st._d_features(df, i, rank)
            if feat is None:
                continue
            for k, cfg in CONFIGS.items():
                s = st._d_signal(feat, cfg)
                if s:
                    sigs[k][d].append((s["score"], code, i, s))
    for k in sigs:
        sigs[k] = {d: lst for d, lst in sigs[k].items() if d in risk_on}
    return sigs


def gate_signals(sigs, db, eps_min, roe_min, missing):
    """以 PIT 基本面門檻過濾訊號; 回傳 (gated_sigs, 通過率)。"""
    out = {k: {} for k in sigs}
    kept = total = 0
    for k, bydate in sigs.items():
        for d, lst in bydate.items():
            keep = [t for t in lst
                    if db.passes(t[1], d, eps_min, roe_min, missing)]
            total += len(lst)
            kept += len(keep)
            if keep:
                out[k][d] = keep
    return out, (kept / total if total else 0.0)


def metrics(trades, curve, init_eq):
    eq = curve["equity"]
    years = len(curve) / 244
    cagr = (eq.iloc[-1] / init_eq) ** (1 / max(years, .01)) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    wins = [t for t in trades if t["pnl"] > 0]
    pf = (sum(t["pnl"] for t in wins)
          / max(1, -sum(t["pnl"] for t in trades if t["pnl"] < 0)))
    win = len(wins) / max(1, len(trades))
    return {"n": len(trades), "win": win, "pf": pf, "cagr": cagr, "dd": dd,
            "mar": cagr / dd if dd > 0 else 0,
            "totR": sum(t["r"] for t in trades)}


def run_set(data, sigs, all_dates, date_idx, regime_tiers, vol_scalars):
    init_eq = INIT_CAPITAL / len(ORDER)
    rows = {}
    yR = {}
    for k in ORDER:
        em = build_entry_map(sigs[k], data)
        trades, curve = run_sub(data, em, k, 1.0, init_eq,
                                all_dates=all_dates, date_idx=date_idx,
                                regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        rows[k] = metrics(trades, curve, init_eq)
        by = defaultdict(float)
        for t in trades:
            by[t["exit_date"].year] += t["r"]
        yR[k] = by
    return rows, yR


def ptable(title, rows):
    print(f"\n{title}")
    print(f"{'策略':>4} {'交易':>5} {'勝率':>6} {'PF':>6} {'年化':>7} "
          f"{'回撤':>7} {'MAR':>6} {'總R':>7}")
    for k in ORDER:
        m = rows[k]
        print(f"{k:>4} {m['n']:>5} {m['win']:>6.1%} {m['pf']:>6.2f} "
              f"{m['cagr']:>7.1%} {m['dd']:>7.1%} {m['mar']:>6.2f} {m['totR']:>+7.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--roe", type=float, default=0.15)
    ap.add_argument("--missing", choices=["fail", "pass"], default="fail")
    args = ap.parse_args()

    print("載入 data_adj...")
    data, names = load_all()
    print(f"{len(data)} 檔, RS rank...")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    all_dates, date_idx = build_date_index(data)
    print("收集訊號 (五策略共用 _d_features)...")
    sigs = build_signals(data, rs, risk_on)

    print("載入基本面 PIT 資料庫...")
    db = FundamentalDB()
    print(f"  快取標的數: {len(db.q)}")

    print(f"套用門檻 EPS_YoY>={args.eps:.0%} ROE>={args.roe:.0%} "
          f"(無資料={args.missing})...")
    gated, keep_rate = gate_signals(sigs, db, args.eps, args.roe, args.missing)
    print(f"  訊號通過率: {keep_rate:.1%}")

    base_rows, base_yR = run_set(data, sigs, all_dates, date_idx,
                                 regime_tiers, vol_scalars)
    gate_rows, gate_yR = run_set(data, gated, all_dates, date_idx,
                                 regime_tiers, vol_scalars)

    print(f"\n{'='*70}\n基本面門檻回測對照 (2011-2026, data_adj 全市場)\n{'='*70}")
    ptable("【baseline 無基本面門檻】", base_rows)
    ptable(f"【gated EPS>={args.eps:.0%} ROE>={args.roe:.0%}】", gate_rows)

    print(f"\n{'差異 (gated - baseline)':>4}")
    print(f"{'策略':>4} {'交易Δ':>7} {'PFΔ':>7} {'年化Δ':>8} {'回撤Δ':>8} {'MARΔ':>7}")
    for k in ORDER:
        b, g = base_rows[k], gate_rows[k]
        print(f"{k:>4} {g['n']-b['n']:>+7} {g['pf']-b['pf']:>+7.2f} "
              f"{(g['cagr']-b['cagr']):>+8.1%} {(g['dd']-b['dd']):>+8.1%} "
              f"{g['mar']-b['mar']:>+7.2f}")

    # 投組層級 (五策略合併總R / 加權)
    years = sorted({y for k in ORDER for y in base_yR[k]})
    print(f"\n逐年總R 對照 (五策略合計):")
    print(f"{'年':>6} {'baseline':>9} {'gated':>9} {'差異':>8}")
    for y in years:
        b = sum(base_yR[k].get(y, 0) for k in ORDER)
        g = sum(gate_yR[k].get(y, 0) for k in ORDER)
        print(f"{y:>6} {b:>+9.1f} {g:>+9.1f} {g-b:>+8.1f}")


if __name__ == "__main__":
    main()
