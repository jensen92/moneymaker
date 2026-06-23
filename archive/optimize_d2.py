"""策略 D 深度再優化: 掃描先前未探索的進場品質槓桿.

optimize_d.py 已掃過 rs / stop / dd_pause / three_week_gain, 定出 rs0.85/stop0.08/tw0.25.
本腳本固定該核心, 改掃 _d_signal 中先前固定的進場品質參數:
  - contraction : ATR5/ATR14 收縮門檻 (越小越嚴, 要求更明顯的波動收縮)
  - vol_mult    : 突破日放量倍數 (越大越嚴, 要求更強的買盤確認)
  - gain_cap    : 突破日漲幅上限 (越小越不追高)
  - max_hold    : 最長持有日 (讓利潤奔跑 vs 資金周轉)
並延伸 rs/stop 至更嚴區間驗證.

選優: 平衡 — CAGR_14y >= 0.10 過濾後取 MAR_14y 最高且近三年穩健者.
用法: python3 optimize_d2.py
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
CAGR_FLOOR = 0.10          # D 已是最強策略, 年化下限拉高到 10%


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


def eval_config(data, cands, risk_on, cfg, all_dates, date_idx, leverage=1.0):
    signals = defaultdict(list)
    for d, code, i, feat in cands:
        if d not in risk_on:
            continue
        s = _d_signal(feat, cfg)
        if s:
            signals[d].append((s["score"], code, i, s))
    entry_map = build_entry_map(signals, data)
    bt.DD_PAUSE["D"] = 1.0
    bt.DD_RESUME["D"] = 1.0
    trades, curve = run_sub(data, entry_map, "D", leverage, INIT_EQ,
                            all_dates=all_dates, date_idx=date_idx)
    tdf = pd.DataFrame(trades)
    full = _metrics(curve, tdf, INIT_EQ)
    three = _metrics(curve, tdf, INIT_EQ, period_start=TRADE_START)
    return full, three


def _row(tag, cfg, full, three):
    return {
        "tag": tag, "rs": cfg["rs_min"], "stop": cfg["stop_pct"],
        "contraction": cfg["contraction"], "vol_mult": cfg["vol_mult"],
        "gain_cap": cfg["gain_cap"], "max_hold": cfg["max_hold"],
        "tw": cfg["three_week_gain"],
        "ret_14y": full["ret"], "cagr_14y": full["cagr"],
        "dd_14y": full["dd"], "mar_14y": full["mar"],
        "pf_14y": full["pf"], "wr_14y": full["wr"], "n_14y": full["n"],
        "cagr_3y": three["cagr"] if three else float("nan"),
        "dd_3y": three["dd"] if three else float("nan"),
        "mar_3y": three["mar"] if three else float("nan"),
    }


def _fmt(r):
    return (f"{r['tag']:<16} | 14y: {r['ret_14y']:+7.0%} CAGR {r['cagr_14y']:+5.1%} "
            f"DD {r['dd_14y']:4.1%} MAR {r['mar_14y']:5.2f} PF {r['pf_14y']:4.2f} "
            f"WR {r['wr_14y']:4.0%} n={r['n_14y']:>4} | 3y MAR {r['mar_3y']:5.2f}")


def main():
    print("載入資料...")
    data, _ = load_all()
    print(f"{len(data)} 檔股票")
    print("計算 RS Rank...")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    print("抽取候選特徵 (一次性)...")
    cands = precompute_candidates(data, rs)
    print(f"模板通過候選: {len(cands)} 筆")
    print("建立日期索引 (一次性)...")
    all_dates, date_idx = build_date_index(data)
    print()

    base = dict(strategies.D_CONFIG)
    rows = []

    # ───── Stage 1: 消融分析 ─────────────────────────────────────────────
    print("=" * 92)
    print("Stage 1  消融分析 (由現行 D_CONFIG 單一切換)")
    print("=" * 92)
    ablations = {
        "baseline":         {},
        "contraction_0.70": {"contraction": 0.70},
        "contraction_0.90": {"contraction": 0.90},
        "vol_mult_1.25":    {"vol_mult": 1.25},
        "vol_mult_2.0":     {"vol_mult": 2.0},
        "gain_cap_0.07":    {"gain_cap": 0.07},
        "gain_cap_0.15":    {"gain_cap": 0.15},
        "max_hold_60":      {"max_hold": 60},
        "max_hold_120":     {"max_hold": 120},
        "rs_0.90":          {"rs_min": 0.90},
        "stop_0.10":        {"stop_pct": 0.10},
        "tw_0.30":          {"three_week_gain": 0.30},
    }
    for tag, override in ablations.items():
        cfg = dict(base); cfg.update(override)
        full, three = eval_config(data, cands, risk_on, cfg, all_dates, date_idx)
        if full is None:
            print(f"{tag:<16} | (無交易)"); continue
        r = _row(tag, cfg, full, three); rows.append(r)
        print(_fmt(r))

    # ───── Stage 2: 進場品質網格 (固定核心 rs0.85/stop0.08/tw0.25) ───────
    print("\n" + "=" * 92)
    print("Stage 2  進場品質網格 (contraction × vol_mult × gain_cap × max_hold)")
    print("=" * 92)
    grid = list(itertools.product(
        [0.75, 0.80, 0.85],     # contraction
        [1.25, 1.5, 2.0],       # vol_mult
        [0.08, 0.10, 0.12],     # gain_cap
        [90, 120],              # max_hold
    ))
    print(f"掃描 {len(grid)} 組...\n")
    for n, (con, vm, gc, mh) in enumerate(grid, 1):
        cfg = dict(base)
        cfg.update({"contraction": con, "vol_mult": vm, "gain_cap": gc,
                    "max_hold": mh})
        full, three = eval_config(data, cands, risk_on, cfg, all_dates, date_idx)
        if full is None or full["n"] < 50:
            continue
        tag = f"g{n:03d}"
        r = _row(tag, cfg, full, three); rows.append(r)
        print(f"[{n:>3}/{len(grid)}] con={con:.2f} vm={vm:.2f} gc={gc:.2f} "
              f"mh={mh} | 14y {full['ret']:+.0%} CAGR {full['cagr']:+.1%} "
              f"DD {full['dd']:.1%} MAR {full['mar']:.2f} PF {full['pf']:.2f} "
              f"n={full['n']}")

    res = pd.DataFrame(rows)
    res.to_csv("optimize_d2_results.csv", index=False)

    # ───── 選優 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    grid_rows = res[res["tag"].str.startswith("g")]
    elig = grid_rows[grid_rows["cagr_14y"] >= CAGR_FLOOR]
    print(f"通過 CAGR_14y >= {CAGR_FLOOR:.0%} 的組合: {len(elig)} / {len(grid_rows)}")
    if elig.empty:
        elig = grid_rows
    top = elig.sort_values("mar_14y", ascending=False).head(10)
    cols = ["tag", "contraction", "vol_mult", "gain_cap", "max_hold",
            "ret_14y", "cagr_14y", "dd_14y", "mar_14y", "pf_14y", "wr_14y",
            "n_14y", "mar_3y"]
    print("\n依 MAR_14y 排序 (CAGR 下限過濾後) 前 10:")
    print(top[cols].to_string(index=False,
          formatters={c: (lambda x: f"{x:+.1%}") for c in
                      ["ret_14y", "cagr_14y", "dd_14y", "wr_14y"]}))

    # 與 baseline 對照
    bl = res[res["tag"] == "baseline"]
    if not bl.empty:
        b = bl.iloc[0]
        print(f"\nbaseline (現行 D): CAGR {b['cagr_14y']:+.1%} DD {b['dd_14y']:.1%} "
              f"MAR {b['mar_14y']:.2f} PF {b['pf_14y']:.2f}")
    if not top.empty:
        t = top.iloc[0]
        print(f">>> 最佳組合: contraction={t['contraction']} vol_mult={t['vol_mult']} "
              f"gain_cap={t['gain_cap']} max_hold={t['max_hold']}")
        print(f"    14y: {t['ret_14y']:+.0%} / CAGR {t['cagr_14y']:+.1%} / "
              f"DD {t['dd_14y']:.1%} / MAR {t['mar_14y']:.2f} / PF {t['pf_14y']:.2f}")

    print("\n結果已存至 optimize_d2_results.csv")


if __name__ == "__main__":
    main()
