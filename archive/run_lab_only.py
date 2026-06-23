"""只跑 lab 8 策略 (向量化訊號 → run_sub 真實引擎), 與「現行冠軍文件基準」比較。

現行冠軍基準採用 optimize_b2.py 已驗證之標準帳 (1x, 2M) 標稱值:
  K MAR1.49 (PF1.85 CAGR7.6% MaxDD5.1% 勝54%) / L MAR1.44 (PF1.86 CAGR6.6% MaxDD4.6% 勝55%)
  PA MAR0.77 / PB MAR0.76。
判定「比現行好」= MAR >= L(1.44)。lab 訊號以 lab_vectorized 產生 (已對原版函式交叉驗證 差異=0)。
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import backtest as bt
bt.DATA_DIR = os.environ.get("MM_DATA_DIR",
                             os.path.join(os.path.dirname(__file__), "data_adj"))
import lab_vectorized as vec

INIT = bt.INIT_CAPITAL
BASELINE = {  # 現行冠軍標稱值 (optimize_b2.py 驗證)
    "K": dict(trades=581, win=0.54, pf=1.85, cagr=0.076, maxdd=0.051, mar=1.49),
    "L": dict(trades=561, win=0.55, pf=1.86, cagr=0.066, maxdd=0.046, mar=1.44),
    "PA": dict(trades=132, win=0.57, pf=3.53, cagr=0.042, maxdd=0.055, mar=0.77),
    "PB": dict(trades=208, win=0.51, pf=2.46, cagr=0.050, maxdd=0.065, mar=0.76),
}


def main():
    print("載入資料 ...", flush=True)
    data, names = bt.load_all()
    rs = bt.compute_rs_rank(data)
    risk_on = bt.load_regime()
    tiers = bt.load_regime_tiers()
    vsca = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)
    years = len(all_dates) / 244
    print(f"{len(data)} 檔, {len(all_dates)} 日, {years:.1f} 年", flush=True)

    print("向量化掃描 lab 8 策略 ...", flush=True)
    lab_sigs = {k: defaultdict(list) for k in vec.VEC}
    for code, df in data.items():
        rank_arr = rs[code].reindex(df["date"]).values
        ts_list = list(df["date"])
        d = vec.precompute(df)
        for k in vec.VEC:
            for i in vec.VEC[k](d, rank_arr):
                rk = rank_arr[i]
                if np.isnan(rk):
                    if k not in vec.RS_OPTIONAL:
                        continue
                    score = 0.5
                else:
                    score = float(rk)
                lab_sigs[k][ts_list[i]].append((score, code, int(i), vec.SIG[k]))

    rows = []
    for key in vec.VEC:
        print(f"  回測 {key} ...", flush=True)
        sigs = {dd: lst for dd, lst in lab_sigs[key].items() if dd in risk_on}
        entry_map = bt.build_entry_map(sigs, data)
        bt.MAX_POS.setdefault(key, 8)
        bt.DD_PAUSE.setdefault(key, 0.20)
        bt.DD_RESUME.setdefault(key, 0.12)
        trades, curve = bt.run_sub(data, entry_map, key, 1.0, INIT, all_dates, date_idx,
                                   regime_tiers=tiers, vol_scalars=vsca)
        eq = curve["equity"]
        ret = eq.iloc[-1] / INIT - 1
        dd = curve["drawdown"].max()
        cagr = (eq.iloc[-1] / INIT) ** (1 / max(years, 0.01)) - 1
        tdf = pd.DataFrame([t for t in trades if t["strategy"] == key])
        if len(tdf):
            win = (tdf["pnl"] > 0).mean(); avg_r = tdf["r"].mean()
            pf = tdf.loc[tdf.pnl > 0, "pnl"].sum() / max(1.0, -tdf.loc[tdf.pnl < 0, "pnl"].sum())
        else:
            win = avg_r = pf = 0.0
        rows.append(dict(strategy=key, trades=len(tdf), win=win, avg_r=avg_r, pf=pf,
                         ret=ret, cagr=cagr, maxdd=dd, mar=cagr / dd if dd > 0 else 0.0))

    res = pd.DataFrame(rows).sort_values("mar", ascending=False)
    res.to_csv("lab_results.csv", index=False)

    print("\n" + "=" * 78)
    print("lab 真空地帶 8 策略 — 真實引擎 (1x, 2M, 同成本/風控) · 排序 by MAR")
    print("=" * 78)
    print(f"{'策略':<5}{'交易':>6}{'勝率':>7}{'均R':>7}{'PF':>7}{'年化':>8}{'回撤':>8}{'MAR':>7}  判定")
    print("-" * 78)
    L_MAR = 1.44
    for _, r in res.iterrows():
        tag = "✅勝過現行" if r["mar"] >= L_MAR else "✗ 不如現行→封存"
        print(f"{r['strategy']:<5}{int(r['trades']):>6}{r['win']*100:>6.1f}%{r['avg_r']:>7.2f}"
              f"{r['pf']:>7.2f}{r['cagr']*100:>7.1f}%{r['maxdd']*100:>7.1f}%{r['mar']:>7.2f}  {tag}")
    print("\n現行冠軍基準: K MAR1.49 / L MAR1.44 / PA 0.77 / PB 0.76")
    winners = [r["strategy"] for _, r in res.iterrows() if r["mar"] >= L_MAR]
    print("勝出 (MAR>=1.44):", winners or "(無 — 全數封存)")


if __name__ == "__main__":
    main()
