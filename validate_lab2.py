"""快速 (向量化) 真實引擎驗測: strategies_lab 的 8 個真空地帶策略 vs 現行冠軍 K/L/PA/PB。

訊號以 lab_vectorized (numpy 遮罩) 產生, 並以原版 strategies_lab 函式抽樣交叉驗證一致;
回測走 backtest.run_sub 完整事件風控 (與正式策略同成本/同風控), 各策略獨立 2M 子帳戶 1x。
輸出排序表 + 封存 lab_results.csv。
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.environ.get("MM_DATA_DIR",
                             os.path.join(os.path.dirname(__file__), "data_adj"))

from strategies import STRATEGIES
import strategies_lab as lab
import lab_vectorized as vec

INIT = bt.INIT_CAPITAL


def crosscheck(data, rs, n_sample=120):
    """抽樣比對向量化遮罩 vs 原版 strategies_lab 函式, 不一致則中止 (保護正確性)。"""
    import random
    random.seed(7)
    codes = random.sample(list(data), min(n_sample, len(data)))
    mism = defaultdict(int)
    vtot = defaultdict(int)
    ftot = defaultdict(int)
    for code in codes:
        df = data[code]
        rank_arr = rs[code].reindex(df["date"]).values
        d = vec.precompute(df)
        for k in vec.VEC:
            vidx = set(vec.VEC[k](d, rank_arr).tolist())
            vtot[k] += len(vidx)
            fn = lab.LAB_STRATEGIES[k]
            fidx = set()
            for i in range(210, len(df) - 1):
                rk = rank_arr[i]
                s = fn(df, i, rs_rank=(None if np.isnan(rk) else float(rk)))
                if s:
                    fidx.add(i)
            ftot[k] += len(fidx)
            mism[k] += len(vidx ^ fidx)
    print("  交叉驗證 (向量化 vs 原版函式, %d 檔抽樣):" % len(codes))
    ok = True
    for k in vec.VEC:
        flag = "OK" if mism[k] == 0 else f"⚠ 不一致 {mism[k]}"
        if mism[k]:
            ok = False
        print(f"    {k}: vec={vtot[k]:>4} fn={ftot[k]:>4} 差異={mism[k]:>3}  {flag}")
    return ok


def summarize(key, trades, curve, years):
    eq = curve["equity"]
    ret = eq.iloc[-1] / INIT - 1
    dd = curve["drawdown"].max()
    cagr = (eq.iloc[-1] / INIT) ** (1 / max(years, 0.01)) - 1
    tdf = pd.DataFrame([t for t in trades if t["strategy"] == key])
    if len(tdf):
        win = (tdf["pnl"] > 0).mean()
        avg_r = tdf["r"].mean()
        pf = tdf.loc[tdf.pnl > 0, "pnl"].sum() / max(1.0, -tdf.loc[tdf.pnl < 0, "pnl"].sum())
    else:
        win = avg_r = pf = 0.0
    return {"strategy": key, "trades": len(tdf), "win": win, "avg_r": avg_r,
            "pf": pf, "ret": ret, "cagr": cagr, "maxdd": dd,
            "mar": cagr / dd if dd > 0 else 0.0}


def run_one(key, signals, data, all_dates, date_idx, tiers, vsca, years):
    sigs = {d: lst for d, lst in signals.items() if d in RISK_ON}
    entry_map = bt.build_entry_map(sigs, data)
    bt.MAX_POS.setdefault(key, 8)
    bt.DD_PAUSE.setdefault(key, 0.20)
    bt.DD_RESUME.setdefault(key, 0.12)
    trades, curve = bt.run_sub(data, entry_map, key, 1.0, INIT, all_dates, date_idx,
                               regime_tiers=tiers, vol_scalars=vsca)
    return summarize(key, trades, curve, years)


def main():
    print("載入資料 ...", flush=True)
    data, names = bt.load_all()
    rs = bt.compute_rs_rank(data)
    global RISK_ON
    RISK_ON = bt.load_regime()
    tiers = bt.load_regime_tiers()
    vsca = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)
    years = len(all_dates) / 244
    print(f"{len(data)} 檔股票, {len(all_dates)} 交易日, 跨 {years:.1f} 年\n", flush=True)

    if not crosscheck(data, rs):
        print("\n✗ 向量化與原版函式不一致, 中止 (請修正 lab_vectorized)。")
        sys.exit(1)

    # 1) 向量化產生 lab 8 策略訊號 (快)
    print("\n  向量化掃描 lab 8 策略 ...", flush=True)
    lab_sigs = {k: defaultdict(list) for k in vec.VEC}
    for code, df in data.items():
        rank_arr = rs[code].reindex(df["date"]).values
        ts_list = list(df["date"])
        d = vec.precompute(df)
        for k in vec.VEC:
            for i in vec.VEC[k](d, rank_arr):
                rk = rank_arr[i]
                if np.isnan(rk):
                    score = 0.5 if k in vec.RS_OPTIONAL else None
                    if score is None:
                        continue
                else:
                    score = float(rk)
                lab_sigs[k][ts_list[i]].append((score, code, int(i), vec.SIG[k]))

    # 2) 現行冠軍基準 (走 backtest.collect_signals, 正確處理 RS)
    print("  產生現行冠軍 K/L/PA/PB 訊號 ...", flush=True)
    base_sigs = {k: bt.collect_signals(data, k) for k in ("K", "L", "PA", "PB")}

    rows = []
    for key in ("K", "L", "PA", "PB"):
        print(f"  回測 {key} (現行) ...", flush=True)
        rows.append(run_one(key, base_sigs[key], data, all_dates, date_idx, tiers, vsca, years))
    for key in vec.VEC:
        print(f"  回測 {key} (lab) ...", flush=True)
        rows.append(run_one(key, lab_sigs[key], data, all_dates, date_idx, tiers, vsca, years))

    res = pd.DataFrame(rows).sort_values("mar", ascending=False)
    res.to_csv("lab_results.csv", index=False)

    base = {"K", "L", "PA", "PB"}
    L_MAR = 1.44
    print("\n" + "=" * 80)
    print("真實引擎驗測 — lab 真空地帶 8 策略 vs 現行冠軍 (1x, 2M, 同成本/風控)")
    print("=" * 80)
    print(f"{'策略':<5}{'交易':>6}{'勝率':>7}{'均R':>7}{'PF':>7}"
          f"{'總報酬':>9}{'年化':>8}{'回撤':>8}{'MAR':>7}  判定")
    print("-" * 80)
    for _, r in res.iterrows():
        if r["strategy"] in base:
            tag = "現行冠軍" if r["strategy"] in ("K", "L") else "現行"
        else:
            tag = "✅勝過現行" if r["mar"] >= L_MAR else "✗ 不如現行→封存"
        print(f"{r['strategy']:<5}{int(r['trades']):>6}{r['win']*100:>6.1f}%"
              f"{r['avg_r']:>7.2f}{r['pf']:>7.2f}{r['ret']*100:>8.1f}%"
              f"{r['cagr']*100:>7.1f}%{r['maxdd']*100:>7.1f}%{r['mar']:>7.2f}  {tag}")
    winners = [r["strategy"] for _, r in res.iterrows()
               if r["strategy"] not in base and r["mar"] >= L_MAR]
    print(f"\n判定基準 MAR>=L(1.44)。勝出: {winners or '(無 — 全數封存)'}")
    print("封存: lab_results.csv")


if __name__ == "__main__":
    main()
