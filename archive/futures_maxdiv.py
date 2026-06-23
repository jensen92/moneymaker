"""追求高 MAR (年化/回撤) — 廣度分散 + 多訊號集成的可行邊界分析.

目標檢驗: 能否達到「年化 15% + 最大回撤 <10%」(即 MAR≈1.5)?
關鍵事實: 槓桿不改變 MAR (報酬與回撤等比放大)。在 DD=10% 下,
          可達年化 = MAR × 10%。故要 15%/10% 必須 MAR≥1.5。

方法: 對 N 個市場各跑「集成趨勢」(唐奇安狀態 + 雙均線 + 12月動能, 僅多) 的
波動率標準化 %報酬, 等權組合, 加組合層波動風控, 比較 5 / 12 / 20 市場的 MAR。

用法: python3 futures_maxdiv.py
"""
import os

import numpy as np
import pandas as pd

from futures_screen import UNIVERSE

DATA_DIR = os.path.join(os.path.dirname(__file__), "screen_data")
TD = 252
META = {s: (n, sec, liq) for s, n, sec, liq in UNIVERSE}


def ens_ret(df, allow_short=False, target=0.10, cost_bps=2.0):
    df = df.sort_values("date").set_index("date")
    c = df["close"].astype(float)
    ret = c.pct_change()
    hi = df["high"].rolling(55).max().shift(1)
    lo = df["low"].rolling(55).min().shift(1)
    donch = pd.Series(np.where(c > hi, 1.0, np.where(c < lo, -1.0, np.nan)),
                      index=c.index).ffill().fillna(0.0)
    ma = np.sign(c.rolling(50).mean() - c.rolling(100).mean()).fillna(0.0)
    mom = np.sign(c - c.shift(252)).fillna(0.0)
    sig = (donch + ma + mom) / 3.0
    if not allow_short:
        sig = sig.clip(lower=0.0)
    vol = ret.rolling(20).std()
    size = (target / np.sqrt(TD) / vol).clip(upper=3.0).fillna(0.0)
    pos = sig * size
    turn = pos.diff().abs().fillna(0.0)
    return (pos.shift(1) * ret - turn.shift(1) * cost_bps / 1e4).fillna(0.0)


def load_rets(syms):
    cols = {}
    for s in syms:
        p = os.path.join(DATA_DIR, f"{s}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            if len(df) >= 500:
                cols[s] = ens_ret(df)
    return pd.DataFrame(cols)


def port_stats(R, vol_control=True):
    port = R.mean(axis=1)                       # 等權 (skipna)
    if vol_control:
        rv = port.rolling(40).std() * np.sqrt(TD)
        scale = (0.12 / rv).clip(upper=1.5).shift(1).fillna(1.0)
        port = port * scale
    port = port.dropna()
    eq = (1 + port).cumprod()
    yrs = len(port) / TD
    sharpe = port.mean() / port.std() * np.sqrt(TD) if port.std() > 0 else 0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    mar = cagr / dd if dd > 0 else 0
    return sharpe, cagr, dd, mar


# 各板塊長史高流動標的
BASKETS = {
    "5 市場 (上限5)": ["GC=F", "NQ=F", "ZS=F", "ZT=F", "LE=F"],
    "12 市場": ["ES=F", "NQ=F", "ZN=F", "ZT=F", "CL=F", "GC=F", "HG=F",
               "6E=F", "6J=F", "ZS=F", "ZC=F", "LE=F"],
    "20 市場 (全分散)": ["ES=F", "NQ=F", "YM=F", "ZN=F", "ZB=F", "ZF=F", "ZT=F",
                    "CL=F", "NG=F", "HO=F", "GC=F", "SI=F", "HG=F", "PL=F",
                    "6E=F", "6J=F", "6A=F", "ZS=F", "ZC=F", "LE=F"],
}


def main():
    print("高 MAR 可行邊界 (集成趨勢, 僅多, 波動標準化 + 組合風控)")
    print("目標: 年化15% / 回撤<10% = MAR 1.5\n")
    hdr = f"{'組合':<18}{'市場數':>5}{'Sharpe':>8}{'年化':>8}{'回撤':>8}{'MAR':>6}{'→10%DD年化':>11}"
    print(hdr); print("-" * len(hdr))
    for name, syms in BASKETS.items():
        R = load_rets(syms)
        sh, cagr, dd, mar = port_stats(R)
        at10 = mar * 0.10
        print(f"{name:<18}{R.shape[1]:>5}{sh:>8.2f}{cagr:>8.1%}{dd:>8.1%}"
              f"{mar:>6.2f}{at10:>11.1%}")
    print("\n說明: 「→10%DD年化」= 把部位縮放到最大回撤剛好 10% 時的年化報酬 (=MAR×10%)。")

    # 不同期間的 MAR (誠實檢驗 — 近期窗口會高估)
    b5 = BASKETS["5 市場 (上限5)"]
    cols = {s: ens_ret(pd.read_csv(os.path.join(DATA_DIR, f"{s}.csv"),
            parse_dates=["date"])) for s in b5
            if os.path.exists(os.path.join(DATA_DIR, f"{s}.csv"))}
    R = pd.DataFrame(cols)
    port0 = R.mean(axis=1)
    rv = port0.rolling(40).std() * np.sqrt(TD)
    port0 = port0 * (0.12 / rv).clip(upper=1.5).shift(1).fillna(1.0)
    print("\n5 市場組合在不同期間的 MAR (近期窗口會高估, 全期才是穩健基準):")
    for lbl, st in [("全期 2000-26", None), ("近10年 2016-26", "2016-01-01"),
                    ("近5年 2021-26", "2021-01-01")]:
        p = port0[port0.index >= pd.Timestamp(st)] if st else port0
        p = p.dropna(); eq = (1 + p).cumprod(); yrs = len(p) / TD
        mar = (eq.iloc[-1] ** (1 / yrs) - 1) / ((eq.cummax() - eq) / eq.cummax()).max()
        print(f"  {lbl:<14} MAR {mar:>5.2f}  → 10%DD年化 {mar*0.10:>5.1%}")
    print("\n結論: 全期 MAR ~0.57 → 10%DD 下年化僅 ~6%。15%/<10% (MAR1.5) 只有近5年")
    print("好行情達得到, 非穩健可期。誠實前瞻: ~6-9% 年化 @ 10% 回撤。")


if __name__ == "__main__":
    main()
