"""策略 C / D 成長性優化: 探索尚未開發的成長槓桿.

C 與 D 共用 _d_features/_d_signal, 故只需 precompute 一次特徵即可同時掃兩者.
本腳本聚焦「把報酬做大」的新槓桿:
  - pyramid   : 漸進式曝險 (起始 1/4 倉, +2%/+4% 順勢加碼) — Minervini 招牌放大器
  - max_hold  : 拉長持有讓主升段奔跑 (120/150/180)
  - gain_cap  : 突破日漲幅上限 (放寬可納入更多突破)
  - rs/stop   : 核心門檻微調
C 為首次正式參數掃描; D 在 d2 進場品質基礎上再找成長.

選優 (成長導向): 在 DD_14y <= DD_CEIL 前提下取 CAGR_14y 最高; 並列 MAR 最高者.
用法: python3 optimize_cd.py
"""
import os
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")

from backtest import (load_all, build_entry_map, run_sub, load_regime,
                      compute_rs_rank, build_date_index, INIT_CAPITAL)
import strategies
from strategies import _d_features, _d_signal

TRADE_START = pd.Timestamp("2023-06-01")
INIT_EQ = INIT_CAPITAL / 3
DD_CEIL = 0.22             # 成長導向: 回撤天花板 (超過視為不可接受)


def precompute_candidates(data, rs):
    cands = []
    for code, df in data.items():
        for i in range(210, len(df) - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if rank is None or (isinstance(rank, float) and np.isnan(rank)):
                continue
            feat = _d_features(df, i, rank)
            if feat is not None:
                cands.append((d, code, i, feat))
    return cands


def _metrics(curve, tdf, base_eq, period_start=None):
    if curve.empty:
        return None
    c = curve if period_start is None else \
        curve[curve["date"] >= period_start].reset_index(drop=True)
    if c.empty:
        return None
    eq = c["equity"].reset_index(drop=True)
    base = base_eq if period_start is None else eq.iloc[0]
    end = eq.iloc[-1]
    years = len(c) / 244
    cagr = (end / base) ** (1 / max(years, 0.01)) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    mar = cagr / dd if dd > 0 else float("inf")
    t = tdf[tdf["exit_date"] >= period_start] if (period_start is not None and not tdf.empty) else tdf
    n = len(t)
    wr = (t["pnl"] > 0).mean() if n else 0.0
    wins = t.loc[t["pnl"] > 0, "pnl"].sum() if n else 0.0
    losses = -t.loc[t["pnl"] < 0, "pnl"].sum() if n else 0.0
    pf = wins / losses if losses > 0 else float("inf")
    return {"ret": end / base - 1, "cagr": cagr, "dd": dd, "mar": mar,
            "n": n, "wr": wr, "pf": pf}


def eval_config(data, cands, risk_on, cfg, key, all_dates, date_idx, leverage=1.0):
    signals = defaultdict(list)
    for d, code, i, feat in cands:
        if d not in risk_on:
            continue
        s = _d_signal(feat, cfg)
        if s:
            signals[d].append((s["score"], code, i, s))
    entry_map = build_entry_map(signals, data)
    bt.DD_PAUSE[key] = 1.0
    bt.DD_RESUME[key] = 1.0
    trades, curve = run_sub(data, entry_map, key, leverage, INIT_EQ,
                            all_dates=all_dates, date_idx=date_idx)
    tdf = pd.DataFrame(trades)
    full = _metrics(curve, tdf, INIT_EQ)
    three = _metrics(curve, tdf, INIT_EQ, period_start=TRADE_START)
    return full, three


def _row(strat, tag, cfg, full, three):
    return {
        "strat": strat, "tag": tag,
        "rs": cfg["rs_min"], "stop": cfg["stop_pct"], "gain_cap": cfg["gain_cap"],
        "max_hold": cfg["max_hold"], "pyramid": cfg.get("pyramid", False),
        "tw": cfg["three_week_gain"],
        "ret_14y": full["ret"], "cagr_14y": full["cagr"],
        "dd_14y": full["dd"], "mar_14y": full["mar"],
        "pf_14y": full["pf"], "wr_14y": full["wr"], "n_14y": full["n"],
        "cagr_3y": three["cagr"] if three else float("nan"),
        "dd_3y": three["dd"] if three else float("nan"),
        "mar_3y": three["mar"] if three else float("nan"),
    }


def run_grid(name, base, grid_dims, key, data, cands, risk_on, all_dates, date_idx):
    print("\n" + "=" * 100)
    print(f"{name}  ({len(list(itertools.product(*grid_dims.values())))} 組)")
    print("=" * 100)
    rows = []
    keys = list(grid_dims.keys())
    combos = list(itertools.product(*grid_dims.values()))
    for n, vals in enumerate(combos, 1):
        cfg = dict(base)
        cfg.update(dict(zip(keys, vals)))
        full, three = eval_config(data, cands, risk_on, cfg, key,
                                  all_dates, date_idx)
        if full is None or full["n"] < 50:
            continue
        r = _row(key, f"{key}{n:03d}", cfg, full, three); rows.append(r)
        print(f"[{n:>3}] " + " ".join(f"{k}={cfg[k]}" for k in keys) +
              f" | 14y {full['ret']:+.0%} CAGR {full['cagr']:+.1%} "
              f"DD {full['dd']:.1%} MAR {full['mar']:.2f} PF {full['pf']:.2f} "
              f"n={full['n']} | 3y CAGR {three['cagr'] if three else 0:+.1%}")
    return rows


def report(name, rows, baseline):
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n{name}: 無有效組合"); return
    print(f"\n{'─'*100}\n{name} 選優  (baseline: CAGR {baseline['cagr']:+.1%} "
          f"DD {baseline['dd']:.1%} MAR {baseline['mar']:.2f} PF {baseline['pf']:.2f})")
    cols = ["tag", "rs", "stop", "gain_cap", "max_hold", "pyramid", "tw",
            "ret_14y", "cagr_14y", "dd_14y", "mar_14y", "pf_14y", "n_14y", "cagr_3y"]
    fmts = {c: (lambda x: f"{x:+.1%}") for c in
            ["ret_14y", "cagr_14y", "dd_14y", "cagr_3y"]}
    elig = df[df["dd_14y"] <= DD_CEIL]
    print(f"\n[成長] DD_14y <= {DD_CEIL:.0%} 內 CAGR 最高 前 5:")
    print((elig if not elig.empty else df).sort_values("cagr_14y", ascending=False)
          .head(5)[cols].to_string(index=False, formatters=fmts))
    print(f"\n[風險調整] MAR 最高 前 5:")
    print(df.sort_values("mar_14y", ascending=False).head(5)[cols]
          .to_string(index=False, formatters=fmts))


def main():
    print("載入資料...")
    data, _ = load_all()
    print(f"{len(data)} 檔股票")
    print("計算 RS Rank...")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    print("抽取候選特徵 (一次性, C/D 共用)...")
    cands = precompute_candidates(data, rs)
    print(f"模板通過候選: {len(cands)} 筆")
    all_dates, date_idx = build_date_index(data)

    # baseline 績效 (現行 C_CONFIG / D_CONFIG)
    cb_full, _ = eval_config(data, cands, risk_on, dict(strategies.C_CONFIG),
                             "C", all_dates, date_idx)
    db_full, _ = eval_config(data, cands, risk_on, dict(strategies.D_CONFIG),
                             "D", all_dates, date_idx)

    all_rows = []

    # ── 策略 C 成長網格 (首次正式掃描) ──────────────────────────────────
    c_rows = run_grid(
        "策略 C 成長網格", dict(strategies.C_CONFIG),
        {"rs_min": [0.70, 0.75, 0.80],
         "pyramid": [False, True],
         "max_hold": [90, 120, 150],
         "gain_cap": [0.08, 9.99],
         "stop_pct": [0.08, 0.10]},
        "C", data, cands, risk_on, all_dates, date_idx)
    all_rows += c_rows
    report("策略 C", c_rows, cb_full)

    # ── 策略 D 成長網格 (進場品質之上再找成長) ─────────────────────────
    d_rows = run_grid(
        "策略 D 成長網格", dict(strategies.D_CONFIG),
        {"rs_min": [0.80, 0.85],
         "pyramid": [False, True],
         "max_hold": [120, 150, 180],
         "gain_cap": [0.08, 0.10],
         "stop_pct": [0.08, 0.10]},
        "D", data, cands, risk_on, all_dates, date_idx)
    all_rows += d_rows
    report("策略 D", d_rows, db_full)

    pd.DataFrame(all_rows).to_csv("optimize_cd_results.csv", index=False)
    print("\n結果已存至 optimize_cd_results.csv")


if __name__ == "__main__":
    main()
