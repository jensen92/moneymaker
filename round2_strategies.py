"""第三批驗證 (Round 2/3): 再 5 個新因子, 與 K/L/M/N/O 及 Q1-Q5 皆不重複。
目標不變: MAR >= D(0.99) 的 1.3 倍 = 1.287。

  W1 動能加速度        — 短窗動能(63d) - 長窗動能(12-1) 越正代表動能正在加速
  W2 強勢股逢回(短期反轉) — 在 close>MA200 的多頭結構中, 5日跌幅越大分越高 (買強股回檔)
  W3 OBV 趨勢(量價背離/確認) — 簽號量(漲量正/跌量負)的60日累積斜率, 度量資金流方向
  W4 低回撤動能(平滑上升) — 12-1動能窗內的最大回檔越小分越高 (爬升越平穩)
  W5 波動率壓縮(VCP式)   — 近20日波動相對252日波動的壓縮比, 越壓縮分越高 (蓄勢待發)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402
import xs_strategies as xs  # noqa: E402

FORM, SKIP = xs.FORM, xs.SKIP


def score_W1(P):
    mom_long = P.shift(SKIP) / P.shift(FORM) - 1.0
    mom_short = P / P.shift(63) - 1.0
    return {"accel": mom_short - mom_long}


def score_W2(P, R, ma200):
    ret5 = P / P.shift(5) - 1.0
    score = (-ret5).where(P > ma200)
    return {"pullback_in_uptrend": score}


def score_W3(R, V):
    signed_vol = (np.sign(R) * V)
    obv = signed_vol.rolling(60).sum()
    return {"obv_trend": obv}


def score_W4(P):
    win = P.rolling(FORM)
    roll_max = P.rolling(FORM).max()
    dd = (roll_max - P) / roll_max
    max_dd = dd.rolling(FORM).max().shift(SKIP)
    mom = P.shift(SKIP) / P.shift(FORM) - 1.0
    score = mom.where(mom > 0) / max_dd.replace(0, np.nan)
    return {"low_dd_mom": score}


def score_W5(R):
    vol20 = R.rolling(20).std()
    vol252 = R.rolling(252).std()
    return {"vol_compress": -(vol20 / vol252.replace(0, np.nan))}


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

    panels = {
        "W1 動能加速度": score_W1(P),
        "W2 強勢股逢回(短反轉)": score_W2(P, R, ma200),
        "W3 OBV趨勢(量價確認)": score_W3(R, V),
        "W4 低回撤動能(平滑)": score_W4(P),
        "W5 波動率壓縮(VCP式)": score_W5(R),
    }

    per = P.index.to_period("M")
    rebal_all = pd.Series(P.index, index=per).groupby(level=0).max().tolist()
    rebal = [d for d in rebal_all if P.index.get_loc(d) >= xb.WARMUP]
    print(f"再平衡月數: {len(rebal)}  區間 {rebal[0].date()} ~ {rebal[-1].date()}\n")

    THRESH = 0.99 * 1.3
    print(f"目標門檻: MAR >= {THRESH:.3f} (D的1.3倍)\n")
    print(f"  {'策略':<24}{'N':>4}{'CAGR':>7}{'MaxDD':>7}{'MAR':>6}{'Sharpe':>7}{'月勝率':>7}")
    bar = "  " + "-" * 64
    print(bar)
    best = (None, -999)
    for label, comp in panels.items():
        for N in [20, 50]:
            sel = xb.make_select_fn(comp, P, turnover20, N)
            eq, rr = xb.run_backtest(sel, P, rebal, regime_on, use_regime=True)
            m = xb.metrics(eq, rr)
            tag = label if N == 20 else ""
            flag = " <== 達標!" if m["mar"] >= THRESH else ""
            print(f"  {tag:<24}{N:>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
                  f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['wr']*100:>6.0f}%{flag}")
            if m["mar"] > best[1]:
                best = (f"{label} N={N}", m["mar"])
        print(bar)
    print(f"\n本輪最佳: {best[0]}  MAR={best[1]:.2f}  (門檻 {THRESH:.2f})")
    print("達標" if best[1] >= THRESH else "未達標 -> 需進行下一輪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
