"""三大交易哲學策略 vs 現行生產策略 head-to-head 回測對照.

新增策略 (本次):
  M  道氏理論主趨勢延續 — HH/HL 結構 + 擺動高點突破 + 量能確認
  N  李佛摩最小阻力線   — 關鍵樞紐突破 + 順勢加碼 (pyramid) + 寬 ATR 坐穩
  O  歐尼爾 CAN SLIM   — 領導股 base 突破 (N/S/L) + 基本面 EPS年增>=25% & ROE>=17%
對照組 (現行每日輪動): PA / PB / K / L / D

同一引擎 (run_sub)、同期 (2011-2026 data_adj 全市場)、各策略獨立等額子帳戶 (40 萬),
公平對照。指標: 交易數 / 勝率 / PF / 年化 / 最大回撤 / MAR / 總R + 逐年總R。
O 另出純技術版 (O) vs 完整 CAN SLIM (O+F, 基本面 PIT 疊加) 以量化基本面 C/A 的貢獻。
"""
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
from fund_backtest import build_signals as build_d_signals, metrics

PROD = ["PA", "PB", "K", "L", "D"]   # 現行生產對照組 (共用 _d_features)
NEW = ["M", "N", "O"]                # 新策略 (各自獨立 signal)


def build_new_signals(data, rs, risk_on, keys):
    """單次掃描收集新策略訊號 (rank 算一次, 各 signal 套用), 再以市況濾網過濾。"""
    fns = {k: st.STRATEGIES[k] for k in keys}
    sigs = {k: defaultdict(list) for k in keys}
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
            for k, fn in fns.items():
                s = fn(df, i, rs_rank=rank)
                if s:
                    sigs[k][d].append((s["score"], code, i, s))
    for k in sigs:
        sigs[k] = {d: lst for d, lst in sigs[k].items() if d in risk_on}
    return sigs


def gate_fundamental(bydate, db, c_min, a_min, roe_min, missing):
    """O 的 CAN SLIM 基本面疊加 (PIT): C 當季EPS年增>=c_min 且 A 年度EPS成長>=a_min
    且 ROE>=roe_min。回傳 (gated, 通過率)。"""
    out = {}
    kept = total = 0
    for d, lst in bydate.items():
        keep = [t for t in lst
                if db.canslim(t[1], d, c_min, a_min, roe_min, missing)]
        total += len(lst)
        kept += len(keep)
        if keep:
            out[d] = keep
    return out, (kept / total if total else 0.0)


def run_one(data, bydate, all_dates, date_idx, regime_tiers, vol_scalars,
            engine_key, init_eq):
    em = build_entry_map(bydate, data)
    trades, curve = run_sub(data, em, engine_key, 1.0, init_eq,
                            all_dates=all_dates, date_idx=date_idx,
                            regime_tiers=regime_tiers, vol_scalars=vol_scalars)
    m = metrics(trades, curve, init_eq)
    yR = defaultdict(float)
    for t in trades:
        yR[t["exit_date"].year] += t["r"]
    return m, yR


def main():
    print("載入 data_adj...")
    data, names = load_all()
    print(f"{len(data)} 檔, RS rank...")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    all_dates, date_idx = build_date_index(data)
    init_eq = INIT_CAPITAL / 5

    print("收集現行策略訊號 (PA/PB/K/L/D, 共用 _d_features)...")
    d_sigs = build_d_signals(data, rs, risk_on)
    print("收集新策略訊號 (M 道氏 / N 李佛摩 / O 歐尼爾)...")
    new_sigs = build_new_signals(data, rs, risk_on, NEW)
    for k in NEW:
        nsig = sum(len(v) for v in new_sigs[k].values())
        print(f"  {k}: 原始訊號 {nsig}")

    print("載入基本面 PIT 資料庫 (CAN SLIM 的 C/A)...")
    db = FundamentalDB()
    print(f"  快取標的數: {len(db.q)}")
    c_min = st.O_CONFIG["eps_yoy_min"]
    a_min = st.O_CONFIG["ann_growth_min"]
    roe_min = st.O_CONFIG["roe_min"]
    o_full, keep_full = gate_fundamental(new_sigs["O"], db, c_min, a_min, roe_min, "fail")
    o_pass, keep_pass = gate_fundamental(new_sigs["O"], db, c_min, a_min, roe_min, "pass")
    # 寬鬆 CAN SLIM (C>=20% A>=10% ROE>=15%): 觀察門檻鬆緊敏感度 (台股嚴版樣本少)
    o_relax, keep_relax = gate_fundamental(new_sigs["O"], db, 0.20, 0.10, 0.15, "fail")

    print("回測中 (各策略獨立子帳戶)...")
    rows, yRs = {}, {}
    for k in PROD:
        rows[k], yRs[k] = run_one(data, d_sigs[k], all_dates, date_idx,
                                  regime_tiers, vol_scalars, k, init_eq)
    for k in NEW:
        rows[k], yRs[k] = run_one(data, new_sigs[k], all_dates, date_idx,
                                  regime_tiers, vol_scalars, k, init_eq)
    rows["O+F"], yRs["O+F"] = run_one(data, o_full, all_dates, date_idx,
                                      regime_tiers, vol_scalars, "O", init_eq)
    rows["O+Fr"], yRs["O+Fr"] = run_one(data, o_relax, all_dates, date_idx,
                                        regime_tiers, vol_scalars, "O", init_eq)
    rows["O+Fp"], yRs["O+Fp"] = run_one(data, o_pass, all_dates, date_idx,
                                        regime_tiers, vol_scalars, "O", init_eq)

    LABEL = {"PA": "PA 現行", "PB": "PB 現行", "K": "K 現行", "L": "L 現行",
             "D": "D 現行", "M": "M 道氏理論", "N": "N 李佛摩",
             "O": "O 歐尼爾(純技術)", "O+F": "O CANSLIM(嚴 C25/A25/ROE17)",
             "O+Fr": "O CANSLIM(寬 C20/A10/ROE15)", "O+Fp": "O CANSLIM(無資料放行)"}
    order = PROD + NEW + ["O+F", "O+Fr", "O+Fp"]

    print(f"\n{'='*88}")
    print("三大交易哲學 vs 現行生產策略 (2011-2026, data_adj 全市場, 各策略等額 40 萬子帳戶)")
    print('=' * 88)
    print(f"\n{'策略':<24}{'交易':>5}{'勝率':>7}{'PF':>6}{'年化':>7}"
          f"{'回撤':>7}{'MAR':>6}{'總R':>8}")
    for k in order:
        m = rows[k]
        print(f"{LABEL[k]:<24}{m['n']:>5}{m['win']:>7.1%}{m['pf']:>6.2f}"
              f"{m['cagr']:>7.1%}{m['dd']:>7.1%}{m['mar']:>6.2f}{m['totR']:>+8.0f}")

    print(f"\nO CAN SLIM 基本面通過率 (O 原始技術訊號為母體):")
    print(f"  嚴版 C>={c_min:.0%} A>={a_min:.0%} ROE>={roe_min:.0%}: {keep_full:.1%}"
          f"  |  寬版 C>=20% A>=10% ROE>=15%: {keep_relax:.1%}"
          f"  |  放行無資料: {keep_pass:.1%}")

    years = sorted({y for k in order for y in yRs[k]})
    cols = PROD + NEW + ["O+F"]
    print("\n逐年總R:")
    print(f"{'年':>6}" + "".join(f"{k:>7}" for k in cols))
    for y in years:
        print(f"{y:>6}" + "".join(f"{yRs[k].get(y, 0):>+7.0f}" for k in cols))


if __name__ == "__main__":
    main()
