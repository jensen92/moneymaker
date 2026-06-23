"""大膽假設 + 真實實證: 把「拉回深度 (切割率)」當連續因子放進真實月再平衡引擎,
到底哪個方向能賺錢? 直接用 xs_backtest 的真實成本 + 大盤濾網引擎做公平比較。

retrace = (swing_high - close) / (swing_high - swing_low), swing_high=過去HIGH_WIN日高,
swing_low=swing_high之前LOOKBACK日低。三組對打:
  H1 shallow  : 分數 = -retrace            (拉回越淺越高 — 我事件研究的方向)
  H2 golden   : 分數 = -|retrace - 0.6425| (越接近 0.618-0.667 中心越高 — 網友主張)
  H3 deep     : 分數 = +retrace            (拉回越深越高 — 反向對照)
全部都 gate 在 close>MA200 (上升結構) 且只在動能股池 (近6月報酬為正) 內排名,
避免買到下跌中的弱股。再加一個動能基準 (純 12-1 動能) 與 EW 全市場對照。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402

LOOKBACK = 120
HIGH_WIN = 60
GOLDEN_C = (0.618 + 0.667) / 2


def retrace_panel(P):
    swing_high = P.rolling(HIGH_WIN).max()
    swing_low = P.rolling(LOOKBACK).min().shift(HIGH_WIN)   # high 之前的低點
    rng = (swing_high - swing_low)
    rt = (swing_high - P) / rng.where(rng > 0)
    return rt.clip(0, 1.5)


def main():
    print("載入價量面板 ...")
    P, V = xb.load_panels()
    TO = P * V
    R = P.pct_change()
    ma200 = P.rolling(200).mean()
    turnover20 = TO.rolling(20).mean()
    tw = pd.read_csv(os.path.join(xb.DATA, "_TWII.csv"), parse_dates=["date"]).set_index("date")["close"]
    tw = tw.reindex(P.index).ffill()
    regime_on = (tw > tw.rolling(60).mean())

    mom6 = P / P.shift(126) - 1.0           # 動能池 gate
    uptrend = (P > ma200) & (mom6 > 0)
    rt = retrace_panel(P)
    rt_g = rt.where(uptrend)                 # 只在上升+動能股取拉回分數

    panels = {
        "H1 淺拉回(-retrace)":   {"s": (-rt_g)},
        "H2 黃金帶(近0.6425)":   {"s": (-(rt_g - GOLDEN_C).abs())},
        "H3 深拉回(+retrace)":   {"s": (rt_g)},
        "MOM 12-1動能(基準)":    {"s": (P.shift(21) / P.shift(252) - 1.0).where(uptrend)},
    }

    per = P.index.to_period("M")
    rebal_all = pd.Series(P.index, index=per).groupby(level=0).max().tolist()
    rebal = [d for d in rebal_all if P.index.get_loc(d) >= xb.WARMUP]
    print(f"再平衡月數: {len(rebal)}  區間 {rebal[0].date()} ~ {rebal[-1].date()}\n")

    ew = xb.make_ew_select_fn(P, turnover20)
    print(f"  {'策略':<24}{'N':>4}{'CAGR':>7}{'MaxDD':>7}{'MAR':>6}{'Sharpe':>7}{'月勝率':>7}")
    bar = "  " + "-" * 64
    print(bar)
    eq, rr = xb.run_backtest(ew, P, rebal, regime_on, use_regime=True)
    m = xb.metrics(eq, rr)
    print(f"  {'EW 全市場+濾網':<24}{'all':>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
          f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['wr']*100:>6.0f}%")
    print(bar)
    results = {}
    for label, comp in panels.items():
        for N in [20, 50]:
            sel = xb.make_select_fn(comp, P, turnover20, N)
            eq, rr = xb.run_backtest(sel, P, rebal, regime_on, use_regime=True)
            m = xb.metrics(eq, rr)
            results[(label, N)] = rr
            tag = label if N == 20 else ""
            print(f"  {tag:<24}{N:>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
                  f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['wr']*100:>6.0f}%")
        print(bar)
    print("\n  對照 D (事件引擎): CAGR 8.3% / MaxDD 8.4% / MAR 0.99")
    return 0


if __name__ == "__main__":
    sys.exit(main())
