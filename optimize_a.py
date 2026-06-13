"""策略 A 深度優化: 消融分析 (ablation) + 參數網格掃描.

精髓: 重的部分 (載入資料、加指標、RS Rank、_a_features 模板+VCP swing 掃描)
只算一次; 各參數組合再以輕量 _a_signal 過濾 + run_sub 評估.

兩階段:
  Stage 1 消融: 由 baseline 單一切換各旗標, 看哪個結構性規則拖累績效.
  Stage 2 網格: 對最有效的槓桿做完整掃描.

選優: 平衡 — 先以 CAGR_14y >= CAGR_FLOOR 過濾, 再取 MAR_14y 最高且近三年穩健者.
用法: python3 optimize_a.py
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
from strategies import _a_features, _a_signal

TRADE_START = pd.Timestamp("2023-06-01")
INIT_EQ = INIT_CAPITAL / 3
CAGR_FLOOR = 0.08          # 平衡選優: 年化下限 8%


def precompute_candidates(data, rs):
    """對每檔每根 K 線, 抽取 _a_features (模板+VCP swing 原始量). 只跑一次."""
    cands = []   # (date, code, i, feat)
    for code, df in data.items():
        for i in range(210, len(df) - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if rank is None or (isinstance(rank, float) and np.isnan(rank)):
                continue
            feat = _a_features(df, i, rank)
            if feat is not None:
                cands.append((d, code, i, feat))
    return cands


def _metrics(curve, tdf, base_eq, period_start=None):
    """計算績效字典 (報酬基準 base_eq). period_start 給定時只算該段."""
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
    if period_start is not None and not tdf.empty:
        t = tdf[tdf["exit_date"] >= period_start]
    else:
        t = tdf
    n = len(t)
    wr = (t["pnl"] > 0).mean() if n else 0.0
    wins = t.loc[t["pnl"] > 0, "pnl"].sum() if n else 0.0
    losses = -t.loc[t["pnl"] < 0, "pnl"].sum() if n else 0.0
    pf = wins / losses if losses > 0 else float("inf")
    return {"ret": end / base - 1, "cagr": cagr, "dd": dd, "mar": mar,
            "n": n, "wr": wr, "pf": pf}


def eval_config(data, cands, risk_on, cfg, all_dates, date_idx, leverage=1.0):
    """以參數組合 cfg 評估, 回傳 (全期, 近三年) 績效字典."""
    signals = defaultdict(list)
    for d, code, i, feat in cands:
        if d not in risk_on:
            continue
        s = _a_signal(feat, cfg)
        if s:
            signals[d].append((s["score"], code, i, s))

    entry_map = build_entry_map(signals, data)
    bt.DD_PAUSE["A"] = 1.0
    bt.DD_RESUME["A"] = 1.0
    trades, curve = run_sub(data, entry_map, "A", leverage, INIT_EQ,
                            all_dates=all_dates, date_idx=date_idx)
    tdf = pd.DataFrame(trades)
    full = _metrics(curve, tdf, INIT_EQ)
    three = _metrics(curve, tdf, INIT_EQ, period_start=TRADE_START)
    return full, three


def _row(tag, cfg, full, three):
    return {
        "tag": tag,
        "rs": cfg["rs_min"], "stop": cfg["stop_pct"],
        "tw": cfg["three_week_gain"],
        "climax": cfg["use_climax"], "mvp": cfg["use_mvp"],
        "l1": cfg["use_l1"], "rsup": cfg["use_rs_uptrend"],
        "ret_14y": full["ret"], "cagr_14y": full["cagr"],
        "dd_14y": full["dd"], "mar_14y": full["mar"],
        "pf_14y": full["pf"], "wr_14y": full["wr"], "n_14y": full["n"],
        "cagr_3y": three["cagr"] if three else float("nan"),
        "dd_3y": three["dd"] if three else float("nan"),
        "mar_3y": three["mar"] if three else float("nan"),
    }


def _fmt(r):
    return (f"{r['tag']:<14} | 14y: {r['ret_14y']:+7.0%} CAGR {r['cagr_14y']:+5.1%} "
            f"DD {r['dd_14y']:4.1%} MAR {r['mar_14y']:5.2f} PF {r['pf_14y']:4.2f} "
            f"WR {r['wr_14y']:4.0%} n={r['n_14y']:>4} | "
            f"3y CAGR {r['cagr_3y']:+5.1%} DD {r['dd_3y']:4.1%}")


def main():
    print("載入資料...")
    data, _ = load_all()
    print(f"{len(data)} 檔股票")
    print("計算 RS Rank...")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    print("抽取候選特徵 (一次性, 含 VCP swing 掃描)...")
    cands = precompute_candidates(data, rs)
    print(f"模板+VCP 通過候選: {len(cands)} 筆")
    print("建立日期索引 (一次性)...")
    all_dates, date_idx = build_date_index(data)
    print()

    base = dict(strategies.A_CONFIG)
    rows = []

    # ───── Stage 1: 消融分析 ─────────────────────────────────────────────
    print("=" * 96)
    print("Stage 1  消融分析 (由 baseline 單一切換)")
    print("=" * 96)
    ablations = {
        "baseline":      {},
        "climax_OFF":    {"use_climax": False},
        "mvp_OFF":       {"use_mvp": False},
        "l1_OFF":        {"use_l1": False},
        "rsup_OFF":      {"use_rs_uptrend": False},
        "stop_0.08":     {"stop_pct": 0.08},
        "stop_0.10":     {"stop_pct": 0.10},
        "rs_0.80":       {"rs_min": 0.80},
        "rs_0.90":       {"rs_min": 0.90},
        "tw_0.25":       {"three_week_gain": 0.25},
        "tw_0.30":       {"three_week_gain": 0.30},
    }
    for tag, override in ablations.items():
        cfg = dict(base); cfg.update(override)
        full, three = eval_config(data, cands, risk_on, cfg, all_dates, date_idx)
        if full is None:
            print(f"{tag:<14} | (無交易)"); continue
        r = _row(tag, cfg, full, three); rows.append(r)
        print(_fmt(r))

    # ───── Stage 2: 參數網格 ─────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("Stage 2  參數網格掃描")
    print("=" * 96)
    grid = list(itertools.product(
        [True, False],          # use_climax
        [True, False],          # use_mvp
        [0.07, 0.08, 0.10],     # stop_pct
        [0.70, 0.80, 0.90],     # rs_min
        [0.20, 0.25],           # three_week_gain
    ))
    print(f"掃描 {len(grid)} 組...\n")
    for n, (clx, mvp, stop, rsm, tw) in enumerate(grid, 1):
        cfg = dict(base)
        cfg.update({"use_climax": clx, "use_mvp": mvp, "stop_pct": stop,
                    "rs_min": rsm, "three_week_gain": tw})
        full, three = eval_config(data, cands, risk_on, cfg, all_dates, date_idx)
        if full is None or full["n"] < 30:
            continue
        tag = f"g{n:03d}"
        r = _row(tag, cfg, full, three); rows.append(r)
        print(f"[{n:>3}/{len(grid)}] clx={int(clx)} mvp={int(mvp)} "
              f"stop={stop:.2f} rs={rsm:.2f} tw={tw:.2f} | "
              f"14y {full['ret']:+.0%} CAGR {full['cagr']:+.1%} "
              f"DD {full['dd']:.1%} MAR {full['mar']:.2f} PF {full['pf']:.2f} "
              f"n={full['n']}")

    res = pd.DataFrame(rows)
    res.to_csv("optimize_a_results.csv", index=False)

    # ───── 選優: 平衡 (CAGR_FLOOR 過濾後取 MAR 最高) ────────────────────
    print("\n" + "=" * 96)
    grid_rows = res[res["tag"].str.startswith("g")]
    elig = grid_rows[grid_rows["cagr_14y"] >= CAGR_FLOOR]
    print(f"通過 CAGR_14y >= {CAGR_FLOOR:.0%} 的組合: {len(elig)} / {len(grid_rows)}")
    if elig.empty:
        print("無組合達 CAGR 下限, 改取全網格 MAR 最高:")
        elig = grid_rows
    top = elig.sort_values("mar_14y", ascending=False).head(10)
    cols = ["tag", "rs", "stop", "tw", "climax", "mvp",
            "ret_14y", "cagr_14y", "dd_14y", "mar_14y", "pf_14y", "wr_14y",
            "n_14y", "cagr_3y", "dd_3y"]
    print("\n依 MAR_14y 排序 (CAGR 下限過濾後) 前 10:")
    print(top[cols].to_string(index=False,
          formatters={c: (lambda x: f"{x:+.1%}") for c in
                      ["ret_14y", "cagr_14y", "dd_14y", "wr_14y",
                       "cagr_3y", "dd_3y"]}))

    if not top.empty:
        b = top.iloc[0]
        print("\n>>> 建議最佳組合:")
        print(f"    use_climax={b['climax']}  use_mvp={b['mvp']}  "
              f"stop_pct={b['stop']}  rs_min={b['rs']}  three_week_gain={b['tw']}")
        print(f"    14y: {b['ret_14y']:+.0%} / CAGR {b['cagr_14y']:+.1%} / "
              f"DD {b['dd_14y']:.1%} / MAR {b['mar_14y']:.2f} / PF {b['pf_14y']:.2f}")

    print("\n結果已存至 optimize_a_results.csv")


if __name__ == "__main__":
    main()
