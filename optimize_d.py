"""策略 D 參數優化掃描.

精髓: 重的部分 (載入資料、加指標、RS Rank、模板特徵抽取) 只算一次,
各參數組合再以輕量過濾 + run_sub 評估, 報告風險調整後績效.

評估指標:
  - CAGR (年化), MaxDD (最大回撤), MAR = CAGR / MaxDD (越高越好)
  - 全期 (~14年) 與近三年並列, 取兩者都穩健者
用法: python3 optimize_d.py
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


def precompute_candidates(data, rs):
    """對每檔每根 K 線, 抽取模板通過的原始特徵 (與參數無關). 只跑一次."""
    cands = []   # (date, code, i, feat)
    for code, df in data.items():
        dates = df["date"].values
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


def eval_config(data, cands, risk_on, cfg, dd_pause, three_week_gain,
                all_dates, date_idx, leverage=1.0):
    """以給定參數組合評估, 回傳全期與近三年的績效字典."""
    # 套用參數: 訊號過濾 (three_week_gain 由訊號字典帶給引擎)
    cfg = dict(cfg)
    cfg["three_week_gain"] = three_week_gain
    signals = defaultdict(list)
    for d, code, i, feat in cands:
        if d not in risk_on:
            continue
        s = _d_signal(feat, cfg)
        if s:
            signals[d].append((s["score"], code, i, s))

    entry_map = build_entry_map(signals, data)

    # 設定引擎旁路參數
    bt.DD_PAUSE["D"] = dd_pause
    bt.DD_RESUME["D"] = max(0.0, dd_pause - 0.08) if dd_pause < 1.0 else 1.0

    trades, curve = run_sub(data, entry_map, "D", leverage, INIT_EQ,
                            all_dates=all_dates, date_idx=date_idx)

    def metrics(c, t):
        if c.empty:
            return None
        eq = c["equity"].reset_index(drop=True)
        end = eq.iloc[-1]
        start_eq = INIT_EQ  # 報酬基準固定初始資金
        years = len(c) / 244
        cagr = (end / start_eq) ** (1 / max(years, 0.01)) - 1
        dd = ((eq.cummax() - eq) / eq.cummax()).max()
        mar = cagr / dd if dd > 0 else float("inf")
        n = len(t)
        wr = (t["pnl"] > 0).mean() if n else 0
        return {"end": end, "ret": end / start_eq - 1, "cagr": cagr,
                "dd": dd, "mar": mar, "n": n, "wr": wr}

    tdf = pd.DataFrame(trades)
    full = metrics(curve, tdf)

    c3 = curve[curve["date"] >= TRADE_START].reset_index(drop=True)
    if not c3.empty:
        # 近三年以該段期初權益為基準
        base = c3["equity"].iloc[0]
        eq = c3["equity"].reset_index(drop=True)
        years = len(c3) / 244
        cagr3 = (eq.iloc[-1] / base) ** (1 / max(years, 0.01)) - 1
        dd3 = ((eq.cummax() - eq) / eq.cummax()).max()
        t3 = tdf[tdf["exit_date"] >= TRADE_START] if not tdf.empty else tdf
        three = {"ret": eq.iloc[-1] / base - 1, "cagr": cagr3, "dd": dd3,
                 "mar": cagr3 / dd3 if dd3 > 0 else float("inf"),
                 "n": len(t3), "wr": (t3["pnl"] > 0).mean() if len(t3) else 0}
    else:
        three = None
    return full, three


def main():
    print("載入資料...")
    data, names = load_all()
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

    # 參數網格
    rs_grid    = [0.75, 0.80, 0.85]
    stop_grid  = [0.06, 0.07, 0.08]
    ddp_grid   = [1.00, 0.20, 0.16]     # 1.00 = 不暫停
    tw_grid    = [0.20, 0.25]           # 三週法則漲幅門檻

    base_cfg = dict(strategies.D_CONFIG)

    rows = []
    combos = list(itertools.product(rs_grid, stop_grid, ddp_grid, tw_grid))
    print(f"掃描 {len(combos)} 組參數...\n")
    for n, (rs_min, stop, ddp, tw) in enumerate(combos, 1):
        cfg = dict(base_cfg)
        cfg["rs_min"] = rs_min
        cfg["stop_pct"] = stop
        full, three = eval_config(data, cands, risk_on, cfg, ddp, tw,
                                  all_dates, date_idx)
        if full is None:
            continue
        rows.append({
            "rs": rs_min, "stop": stop, "ddp": ddp, "tw": tw,
            "ret_14y": full["ret"], "cagr_14y": full["cagr"],
            "dd_14y": full["dd"], "mar_14y": full["mar"],
            "n_14y": full["n"], "wr_14y": full["wr"],
            "ret_3y": three["ret"] if three else float("nan"),
            "cagr_3y": three["cagr"] if three else float("nan"),
            "dd_3y": three["dd"] if three else float("nan"),
            "mar_3y": three["mar"] if three else float("nan"),
        })
        print(f"[{n:>2}/{len(combos)}] rs={rs_min} stop={stop:.2f} "
              f"ddp={ddp:.2f} tw={tw:.2f} | "
              f"14y: {full['ret']:+.0%} CAGR {full['cagr']:+.1%} "
              f"DD {full['dd']:.1%} MAR {full['mar']:.2f} | "
              f"3y: {three['ret']:+.0%} DD {three['dd']:.1%}"
              if three else "")

    res = pd.DataFrame(rows)
    res.to_csv("optimize_d_results.csv", index=False)

    print("\n" + "=" * 70)
    print("依 14年 MAR (CAGR/MaxDD) 排序 前 10:")
    top = res.sort_values("mar_14y", ascending=False).head(10)
    print(top.to_string(index=False,
          formatters={c: (lambda x: f"{x:+.1%}") for c in
                      ["ret_14y", "cagr_14y", "dd_14y", "wr_14y",
                       "ret_3y", "cagr_3y", "dd_3y"]}))

    print("\n依 14年 CAGR 排序 前 5:")
    print(res.sort_values("cagr_14y", ascending=False).head(5)
          .to_string(index=False))

    print("\n結果已存至 optimize_d_results.csv")


if __name__ == "__main__":
    main()
