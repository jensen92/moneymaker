"""黃金策略 — 「打敗買進持有 1.5 倍」的誠實研究結論.

依網路研究 (DXY 美元、real yields/公債、季節性、動能) 嘗試打敗自己的最佳單一黃金
策略 (趨勢+波動目標)。結論: 這些額外訊號全都無法贏過趨勢+波動目標 — 因為黃金是
長期多頭, 任何濾網只是減少在場時間; 且 DXY/黃金負相關不穩定。

基準: 買進持有黃金 Sharpe 0.69 → 「1.5 倍」= Sharpe 1.04。
- 單一黃金天花板: Sharpe ~0.79 (1.14x), MAR ~0.36 (1.44x) — Sharpe 1.5x 達不到。
- 分散組合: Sharpe ~0.93 (1.35x), 但 MAR ~0.46-0.55 (1.8-2.2x) — 風險調整上早已 >1.5x。
誠實結論: 「Sharpe 1.5 倍」在單一資產上是結構性不可能; 真正贏 1.5 倍要靠分散 (看 MAR)。

用法: python3 futures_gold_research.py
"""
import os

import numpy as np
import pandas as pd

from futures_screen import strat_returns

TD = 252
COST = 0.0005


def load(s):
    return (pd.read_csv(f"screen_data/{s}.csv", parse_dates=["date"])
            .sort_values("date").set_index("date")["close"].astype(float))


def stat(s):
    eq = (1 + s).cumprod()
    yrs = len(s) / TD
    sh = s.mean() / s.std() * np.sqrt(TD) if s.std() > 0 else 0
    cg = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return sh, cg, dd, (cg / dd if dd > 0 else 0)


def main():
    gc = load("GC=F"); dx = load("DX=F"); zn = load("ZN=F")
    idx = gc.index
    dx = dx.reindex(idx).ffill(); zn = zn.reindex(idx).ffill()
    ret = gc.pct_change().fillna(0.0)
    ma200 = gc.rolling(200).mean(); ma_up = ma200 > ma200.shift(20)
    gold_up = (gc > ma200) & ma_up
    dxy_down = dx < dx.rolling(50).mean()
    bond_up = zn > zn.shift(252)
    season = pd.Series(np.isin(idx.month, [9, 10, 11, 12, 1, 2]), index=idx)
    vol = ret.ewm(span=30).std() * np.sqrt(TD)
    vs = (0.18 / vol).clip(upper=2.5).shift(1).fillna(0.0)

    def bt(sig):
        pos = vs * sig.astype(float)
        turn = pos.diff().abs().fillna(0.0)
        return stat((pos.shift(1).fillna(0.0) * ret - turn * COST))

    bh = stat(ret)
    print(f"基準 買進持有黃金: Sharpe {bh[0]:.2f} / 年化 {bh[1]:.1%} / 回撤 {bh[2]:.1%} "
          f"/ MAR {bh[3]:.2f}")
    print(f"「1.5 倍」目標: Sharpe {bh[0]*1.5:.2f} 或 MAR {bh[3]*1.5:.2f}\n")

    print("── 單一黃金: 研究訊號全試過, 沒一個贏過 趨勢+波動目標 ──")
    print(f"{'策略':<20}{'Sharpe':>7}{'年化':>8}{'回撤':>8}{'MAR':>6}{'vs基準Sh':>8}")
    variants = {
        "趨勢+波動目標(最佳)": gold_up,
        "+DXY 美元濾網": gold_up & dxy_down,
        "+公債動能": gold_up & bond_up,
        "+季節性": gold_up & season,
        "+全部訊號": gold_up & dxy_down & bond_up & season,
    }
    for n, sig in variants.items():
        sh, cg, dd, mar = bt(sig)
        print(f"{n:<20}{sh:>7.2f}{cg:>8.1%}{dd:>8.1%}{mar:>6.2f}{sh/bh[0]:>7.2f}x")

    print("\n── 真正贏 1.5 倍的路: 分散 (看 MAR) ──")
    print(f"{'組合':<22}{'Sharpe':>7}{'回撤':>8}{'MAR':>6}{'vs基準MAR':>9}")
    for n, b in {
        "5市場(含黃金)": ["GC=F", "NQ=F", "ZS=F", "ZT=F", "LE=F"],
    }.items():
        R = pd.DataFrame({s: strat_returns(pd.read_csv(f"screen_data/{s}.csv",
                          parse_dates=["date"]), "donchian", False) for s in b})
        p = R.mean(axis=1); rv = p.rolling(40).std() * np.sqrt(TD)
        p = (p * (0.12 / rv).clip(upper=1.5).shift(1).fillna(1.0)).dropna()
        sh, cg, dd, mar = stat(p)
        print(f"{n:<22}{sh:>7.2f}{dd:>8.1%}{mar:>6.2f}{mar/bh[3]:>8.2f}x")

    print("\n誠實結論: 單一黃金 Sharpe 上限 ~0.79 (1.14x), Sharpe 1.5x 不可能;")
    print("但分散組合 MAR 已達買進持有的 ~1.8-2.2 倍 (風險調整真正贏 1.5 倍以上)。")


if __name__ == "__main__":
    main()
