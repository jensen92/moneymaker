"""穀物期貨策略 樣本內/外 (IS/OOS) 參數掃描 — 上線前防過擬合驗證.

方法:
  IS  樣本內 2000–2015 : 用來挑參數
  OOS 樣本外 2016–2026 : 鎖定參數後驗證 (策略從沒看過這段)

判定: 一組參數要「上線可用」, 必須 OOS 仍獲利 (PF>1.1, Sharpe>0), 且整片
參數網格大多獲利 (穩健, 非單一尖峰)。輸出依 IS Sharpe 排序, 並標示 OOS 表現。

用法:
    python3 futures_optimize.py                 # 掃 M,D,T,S
    python3 futures_optimize.py --strategy M
"""
import argparse
import copy
import itertools

import pandas as pd

import futures_strategies as fs
from futures_backtest import (INIT_CAPITAL, _build_index, collect_signals,
                              load_all, metrics, run_strategy)

IS_END = pd.Timestamp("2015-12-31")
OOS_START = pd.Timestamp("2016-01-01")

GRIDS = {
    "M": {"fast": [20, 30, 50], "slow": [80, 100, 150], "stop": [2.5, 3.0, 4.0]},
    "D": {"entry": [40, 55, 80], "exit": [20, 40], "stop": [2.0, 2.5, 3.0],
          "trend": [150, 200]},
    "T": {"entry": [15, 20, 40], "exit": [10, 15, 20], "stop": [2.0, 2.5],
          "trend": [80, 100]},
    "S": {"month": [10, 11, 12], "hold": [100, 150, 200], "stop": [3.0, 4.0],
          "trend": [0, 200]},
}


def _combos(grid):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        d = dict(zip(keys, vals))
        if "fast" in d and "slow" in d and d["fast"] >= d["slow"]:
            continue
        yield d


def optimize(data, all_dates, date_idx, key):
    base = copy.deepcopy(fs.CONFIG[key])
    rows = []
    for params in _combos(GRIDS[key]):
        fs.CONFIG[key] = {**base, **params}
        emap = collect_signals(data, key)
        tr_i, cv_i = run_strategy(data, emap, key, INIT_CAPITAL,
                                  all_dates, date_idx, date_to=IS_END)
        tr_o, cv_o = run_strategy(data, emap, key, INIT_CAPITAL,
                                  all_dates, date_idx, date_from=OOS_START)
        mi, mo = metrics(tr_i, cv_i, INIT_CAPITAL), metrics(tr_o, cv_o, INIT_CAPITAL)
        rows.append((params, mi, mo))
    fs.CONFIG[key] = base
    rows.sort(key=lambda r: r[1]["sharpe"], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None, help="單一策略, 預設掃 M,D,T,S")
    args = ap.parse_args()
    keys = [args.strategy] if args.strategy else ["M", "D", "T", "S"]

    data = load_all()
    all_dates, date_idx = _build_index(data)

    for key in keys:
        rows = optimize(data, all_dates, date_idx, key)
        oos_pos = sum(1 for _, _, mo in rows if mo["pf"] > 1.0)
        print(f"\n{'='*92}")
        print(f"策略 {key} {fs.STRATEGY_NAMES[key]}  "
              f"({len(rows)} 組;  OOS 獲利 {oos_pos}/{len(rows)} = "
              f"{oos_pos/len(rows):.0%})   依 IS Sharpe 排序")
        print('='*92)
        print(f"{'參數':<46} | {'IS Sh':>6} {'IS PF':>6} {'IS DD':>6} "
              f"| {'OOS Sh':>6} {'OOS PF':>6} {'OOS DD':>6}")
        print("-" * 92)
        for params, mi, mo in rows[:12]:
            ps = ",".join(f"{k}={v}" for k, v in params.items())
            flag = "  <= OOS佳" if (mo["pf"] > 1.15 and mo["sharpe"] > 0.15) else ""
            print(f"{ps:<46} | {mi['sharpe']:>6.2f} {mi['pf']:>6.2f} "
                  f"{mi['dd']:>6.1%} | {mo['sharpe']:>6.2f} {mo['pf']:>6.2f} "
                  f"{mo['dd']:>6.1%}{flag}")


if __name__ == "__main__":
    main()
