"""真實引擎驗測 strategies_lab 的 8 個真空地帶策略, 與現行冠軍 K/L/PA/PB 同台比較。

每個策略獨立子帳戶 (2,000,000 本金, 1x 槓桿), 走 backtest.run_sub 完整事件風控
(停損/MA50移動停損/3R保本/+20%賣半或trail_atr/市況分級/波動標量/集中度/回撤熔斷),
與正式策略完全相同的成本 (手續費0.1425%×2 + 證交稅0.3% + 滑價0.1%)。

輸出: 排序表 (依 MAR) + 封存 lab_results.csv。
判定「比現行好」: 標準 = 不被現行冠軍 K(MAR1.49)/L(MAR1.44) 嚴格佔優,
即至少在 MAR 上 >= L 的 1.44 (與最佳現行同級或更優) 才值得納入輪動。
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.environ.get("MM_DATA_DIR",
                             os.path.join(os.path.dirname(__file__), "data_adj"))

from strategies import STRATEGIES
from strategies_lab import LAB_STRATEGIES

INIT = bt.INIT_CAPITAL


def collect_all_signals(data, fns, rs, min_i=210):
    """單一掃描 pass: 對每檔每根計算一次 RS, 同時餵給所有策略 (省 8× 查表/迴圈)。

    回傳 {key: {date: [(score,code,i,sig)]}}。RS 以 numpy 陣列對齊各檔列, O(1) 存取。
    """
    out = {k: defaultdict(list) for k in fns}
    items = list(data.items())
    total = len(items)
    for n_done, (code, df) in enumerate(items, 1):
        dates = df["date"]
        try:
            rank_arr = rs[code].reindex(dates).values
        except KeyError:
            rank_arr = np.full(len(df), np.nan)
        ts_list = list(dates)            # pandas Timestamp, 與 risk_on 同型別
        n = len(df)
        for i in range(min_i, n - 1):
            rk = rank_arr[i]
            rk = None if (rk is None or np.isnan(rk)) else float(rk)
            for k, fn in fns.items():
                s = fn(df, i, rs_rank=rk)
                if s:
                    out[k][ts_list[i]].append((s["score"], code, i, s))
        if n_done % 300 == 0:
            print(f"    scan {n_done}/{total}", flush=True)
    return out


def summarize(key, trades, curve, years):
    eq = curve["equity"]
    ret = eq.iloc[-1] / INIT - 1
    dd = curve["drawdown"].max()
    cagr = (eq.iloc[-1] / INIT) ** (1 / max(years, 0.01)) - 1
    tdf = pd.DataFrame([t for t in trades if t["strategy"] == key])
    if len(tdf):
        win = (tdf["pnl"] > 0).mean()
        avg_r = tdf["r"].mean()
        gains = tdf.loc[tdf.pnl > 0, "pnl"].sum()
        losses = -tdf.loc[tdf.pnl < 0, "pnl"].sum()
        pf = gains / max(1.0, losses)
    else:
        win = avg_r = pf = 0.0
    mar = cagr / dd if dd > 0 else 0.0
    return {"strategy": key, "trades": len(tdf), "win": win, "avg_r": avg_r,
            "pf": pf, "ret": ret, "cagr": cagr, "maxdd": dd, "mar": mar}


def main():
    print("載入資料 ...")
    data, names = bt.load_all()
    rs = bt.compute_rs_rank(data)
    risk_on = bt.load_regime()
    tiers = bt.load_regime_tiers()
    vsca = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)
    years = len(all_dates) / 244
    print(f"{len(data)} 檔股票, {len(all_dates)} 交易日, 跨 {years:.1f} 年\n")

    # 現行冠軍基準 + lab 真空地帶策略, 同一引擎同台比較
    baseline = {k: STRATEGIES[k] for k in ("K", "L", "PA", "PB")}
    run_set = {**baseline, **LAB_STRATEGIES}

    print("  單一 pass 掃描全部策略訊號 ...", flush=True)
    all_sigs = collect_all_signals(data, run_set, rs)

    rows = []
    for key in run_set:
        print(f"  回測 {key} ...", flush=True)
        sigs = {d: lst for d, lst in all_sigs[key].items() if d in risk_on}
        entry_map = bt.build_entry_map(sigs, data)
        # 確保引擎全域字典有該 key (沿用預設風控閾值)
        bt.MAX_POS.setdefault(key, 8)
        bt.DD_PAUSE.setdefault(key, 0.20)
        bt.DD_RESUME.setdefault(key, 0.12)
        trades, curve = bt.run_sub(data, entry_map, key, 1.0, INIT,
                                   all_dates, date_idx,
                                   regime_tiers=tiers, vol_scalars=vsca)
        rows.append(summarize(key, trades, curve, years))

    res = pd.DataFrame(rows).sort_values("mar", ascending=False)
    res.to_csv("lab_results.csv", index=False)

    print("\n" + "=" * 78)
    print("真實引擎驗測 — lab 真空地帶策略 vs 現行冠軍 (1x, 2M 本金, 同成本/風控)")
    print("=" * 78)
    print(f"{'策略':<5}{'交易':>6}{'勝率':>7}{'均R':>7}{'PF':>7}"
          f"{'總報酬':>9}{'年化':>8}{'最大回撤':>9}{'MAR':>7}  類型")
    print("-" * 78)
    L_MAR = 1.44
    tag_map = {"K": "現行冠軍", "L": "現行冠軍", "PA": "現行", "PB": "現行"}
    for _, r in res.iterrows():
        is_base = r["strategy"] in baseline
        if is_base:
            tag = tag_map[r["strategy"]]
        else:
            tag = "✅勝過現行" if r["mar"] >= L_MAR else "✗ 不如現行"
        print(f"{r['strategy']:<5}{int(r['trades']):>6}{r['win']*100:>6.1f}%"
              f"{r['avg_r']:>7.2f}{r['pf']:>7.2f}{r['ret']*100:>8.1f}%"
              f"{r['cagr']*100:>7.1f}%{r['maxdd']*100:>8.1f}%{r['mar']:>7.2f}  {tag}")

    print("\n判定基準: MAR >= L(1.44) 才算「比現行最佳更好/同級」, 否則封存。")
    winners = [r["strategy"] for _, r in res.iterrows()
               if r["strategy"] not in baseline and r["mar"] >= L_MAR]
    print("勝出 (值得納入):", winners or "(無 — 全數封存)")
    print("封存: lab_results.csv")


if __name__ == "__main__":
    main()
