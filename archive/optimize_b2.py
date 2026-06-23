"""在 B (D引擎+52週高點接近度 prox_min=0.85) 之上, 繼續找 2 個能再超越 B 的增強。
延用 optimize_b.py 的消融框架, 這次基準改成 B_CONFIG (不再是 D), 只有「同時優於
B 的 PF/CAGR/MaxDD/勝率」才算數。掃描方向: stop_pct / gain_cap / vol_mult /
contraction 在 prox_min=0.85 基礎上的再調優, 以及與既有特徵 (vol_dryup 已強制,
contraction 已有) 的交叉組合。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402
from strategies import _d_features, _d_signal, D_CONFIG, B_CONFIG  # noqa: E402

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")


def precompute_features(data, rs):
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


def signals_from_feats(feats, cfg):
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


def dominates(m, base):
    """m 同時優於 base 的 PF/CAGR/MaxDD/勝率 (嚴格佔優)."""
    return (m["pf"] >= base["pf"] and m["cagr"] >= base["cagr"]
            and m["dd"] <= base["dd"] and m["wr"] >= base["wr"]
            and (m["pf"] > base["pf"] or m["cagr"] > base["cagr"] or m["dd"] < base["dd"]))


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    print("預算 _d_features ...")
    feats = precompute_features(data, rs)

    def run(cfg, label):
        sigs = signals_from_feats(feats, cfg)
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
    base = run(B_CONFIG, "B 基準 (D+prox0.85)")

    print("  --- 在 B 上單一再調優 ---")
    single = {
        "stop_pct 0.06":       {"stop_pct": 0.06},
        "stop_pct 0.10":       {"stop_pct": 0.10},
        "stop_pct 0.12":       {"stop_pct": 0.12},
        "gain_cap 0.05":       {"gain_cap": 0.05},
        "gain_cap 0.06":       {"gain_cap": 0.06},
        "gain_cap 0.10":       {"gain_cap": 0.10},
        "vol_mult 1.2":        {"vol_mult": 1.2},
        "vol_mult 2.0":        {"vol_mult": 2.0},
        "vol_mult 2.5":        {"vol_mult": 2.5},
        "contraction 0.80":    {"contraction": 0.80},
        "contraction 0.90":    {"contraction": 0.90},
        "prox 0.80":           {"prox_min": 0.80},
        "prox 0.88":           {"prox_min": 0.88},
    }
    res = {}
    for lab, extra in single.items():
        res[lab] = run({**B_CONFIG, **extra}, lab)

    print("  --- 組合再調優 ---")
    combos = {
        "vol_mult2.0+gain0.06":         {"vol_mult": 2.0, "gain_cap": 0.06},
        "vol_mult2.0+stop0.10":         {"vol_mult": 2.0, "stop_pct": 0.10},
        "vol_mult2.0+gain0.06+stop0.10": {"vol_mult": 2.0, "gain_cap": 0.06, "stop_pct": 0.10},
        "contraction0.80+vol_mult2.0":  {"contraction": 0.80, "vol_mult": 2.0},
        "prox0.88+vol_mult2.0":         {"prox_min": 0.88, "vol_mult": 2.0},
        "prox0.88+gain0.06":            {"prox_min": 0.88, "gain_cap": 0.06},
        "prox0.88+vol_mult2.0+gain0.06": {"prox_min": 0.88, "vol_mult": 2.0, "gain_cap": 0.06},
    }
    for lab, extra in combos.items():
        res[lab] = run({**B_CONFIG, **extra}, lab)

    print(f"\n基準 B: PF={base['pf']:.2f} CAGR={base['cagr']*100:.1f}% "
          f"MaxDD={base['dd']*100:.1f}% WR={base['wr']*100:.0f}% MAR={base['mar']:.2f}")
    winners = [(lab, m) for lab, m in res.items() if dominates(m, base)]
    if winners:
        winners.sort(key=lambda x: x[1]["mar"], reverse=True)
        print(f"\n找到 {len(winners)} 個嚴格優於 B 的設定:")
        for lab, m in winners:
            print(f"  {lab:<34} PF={m['pf']:.2f} CAGR={m['cagr']*100:.1f}% "
                  f"MaxDD={m['dd']*100:.1f}% WR={m['wr']*100:.0f}% MAR={m['mar']:.2f}")
    else:
        print("\n本批沒有設定同時優於 B 的 PF/CAGR/MaxDD/勝率。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
