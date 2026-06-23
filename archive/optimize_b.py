"""策略 B 消融優化: 在 D 已驗證的事件引擎進場上, 逐一/組合疊加實證增強門檻,
用真實引擎 (run_sub, 含停損/集中/市況/波動風控) 比較對 MAR/CAGR/MaxDD/PF 的影響,
只保留能提升 MAR 的增強。增強來自 35因子+台股習性驗證:
  prox_min     : 收盤 >= 52週高 * prox_min (52週高點接近度, 全場最佳選股因子 M/X2)
  max_day_range: 突破日振幅 <= 上限 (量增價穩 Z1, 大單吃貨不衝高 → 籌碼安定)
  skip_months  : 不在 9/10 月進場 (季節習性 E, 個股報酬實測轉弱)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402
from strategies import _d_features, _d_signal, D_CONFIG  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
TRADE_START = pd.Timestamp("2011-01-01")


def precompute_features(data, rs):
    """對每檔每根算一次 _d_features (與參數無關的重運算), 存通過模板者."""
    feats = {}
    for code, df in data.items():
        rows = []
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
                rows.append((i, d, f))
        if rows:
            feats[code] = rows
    return feats


def signals_from_feats(feats, data, cfg):
    from collections import defaultdict
    sigs = defaultdict(list)
    for code, rows in feats.items():
        for i, d, f in rows:
            s = _d_signal(f, cfg)
            if s:
                sigs[d].append((s["score"], code, i, s))
    return sigs


def metrics(trades, curve):
    eq = curve.set_index("date")["equity"].sort_index()
    if len(eq) < 2:
        return None
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, .01)) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    td = pd.DataFrame(trades)
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
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    print("預算 _d_features ...")
    feats = precompute_features(data, rs)

    def run(cfg, label):
        sigs = signals_from_feats(feats, data, cfg)
        sigs = {d: lst for d, lst in sigs.items() if d in regime}
        em = bt.build_entry_map(sigs, data)
        trades, curve = bt.run_sub(data, em, "B", 1.0, 2_000_000,
                                   all_dates=all_dates, date_idx=date_idx,
                                   regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        m = metrics(trades, curve)
        print(f"  {label:<34}{m['n']:>5}{m['wr']*100:>6.0f}%{m['pf']:>6.2f}"
              f"{m['cagr']*100:>7.1f}%{m['dd']*100:>6.1f}%{m['mar']:>6.2f}")
        return m

    print(f"\n  {'設定':<34}{'交易':>5}{'勝率':>6}{'PF':>6}{'CAGR':>7}{'MaxDD':>6}{'MAR':>6}")
    print("  " + "-" * 70)
    base = run(dict(D_CONFIG), "D 基準 (=B無增強)")

    # Stage 1: 單一增強消融
    print("  --- 單一增強 ---")
    single = {
        "+prox 0.85":      {"prox_min": 0.85},
        "+prox 0.90":      {"prox_min": 0.90},
        "+prox 0.95":      {"prox_min": 0.95},
        "+range<=0.06":    {"max_day_range": 0.06},
        "+range<=0.08":    {"max_day_range": 0.08},
        "+skip 9-10月":    {"skip_months": {9, 10}},
    }
    res = {}
    for lab, extra in single.items():
        res[lab] = run({**D_CONFIG, **extra}, lab)

    # Stage 2: 組合最有效者
    print("  --- 組合 ---")
    combos = {
        "prox0.90+skip9-10":          {"prox_min": 0.90, "skip_months": {9, 10}},
        "prox0.90+range0.08":         {"prox_min": 0.90, "max_day_range": 0.08},
        "prox0.90+range0.08+skip":    {"prox_min": 0.90, "max_day_range": 0.08, "skip_months": {9, 10}},
        "prox0.85+skip9-10":          {"prox_min": 0.85, "skip_months": {9, 10}},
    }
    for lab, extra in combos.items():
        res[lab] = run({**D_CONFIG, **extra}, lab)

    print("\n基準 D MAR = %.2f" % base["mar"])
    best_lab = max(res, key=lambda k: (res[k]["mar"] if res[k]["mar"] == res[k]["mar"] else -9))
    print(f"最佳增強: {best_lab}  MAR={res[best_lab]['mar']:.2f}  "
          f"(vs D {base['mar']:.2f}, {'提升' if res[best_lab]['mar']>base['mar'] else '未提升'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
