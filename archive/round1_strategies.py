"""第二批驗證 (Round 1/3): 5 個新的學術因子, 與既有 5 個 (K/L/M/N/O) 不重複,
目標: 是否有任何一個的 MAR 能達到現行冠軍 D (MAR 0.99) 的 1.3 倍 (>=1.287)?

文獻對照:
  Q1 Ang-Hodrick-Xing-Zhang (2006)  — 特質波動率異常: idio vol 越低分越高
  Q2 Amihud (2002)                  — 流動性溢酬: 越不流動(ILLIQ越高)分越高
  Q3 風險調整動能 (Sharpe-momentum, Du 2018 等) — 動能/波動, 減少動能崩盤
  Q4 Bali-Cakici-Whitelaw (2011)    — MAX effect: 避開近月有極端單日漲幅的「樂透股」
  Q5 新高頻率 (Persistent-strength)  — 過去60日內創20日新高的次數, 度量強勢持續性
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402
import xs_strategies as xs  # noqa: E402

BETA_W = xs.BETA_W
FORM, SKIP = xs.FORM, xs.SKIP


def score_Q1(P, R, Rm, mret, ma200):
    var_m = mret.rolling(BETA_W).var()
    mean_m = mret.rolling(BETA_W).mean()
    mean_r = R.rolling(BETA_W).mean()
    mean_rm = Rm.rolling(BETA_W).mean()
    cov_rm = mean_rm.sub(mean_r.mul(mean_m, axis=0))
    beta = cov_rm.div(var_m, axis=0)
    var_r = R.rolling(BETA_W).var()
    idio_var = (var_r - (beta ** 2).mul(var_m, axis=0)).clip(lower=1e-12)
    return {"neg_idiovol": -np.sqrt(idio_var)}


def score_Q2(P, R, turnover):
    illiq = (R.abs() / turnover.replace(0, np.nan)) * 1e9
    return {"illiq": illiq.rolling(21).mean()}


def score_Q3(P, R):
    mom = P.shift(SKIP) / P.shift(FORM) - 1.0
    vol = R.rolling(60).std()
    return {"sharpe_mom": mom / vol.replace(0, np.nan)}


def score_Q4(R):
    return {"neg_max": -R.rolling(21).max()}


def score_Q5(P):
    new_high20 = (P >= P.rolling(20).max())
    return {"newhigh_freq": new_high20.rolling(60).sum()}


def main():
    print("載入價量面板 ...")
    P, V = xb.load_panels()
    TO = P * V
    R = P.pct_change()
    ma200 = P.rolling(200).mean()
    turnover20 = TO.rolling(20).mean()
    tw = pd.read_csv(os.path.join(xb.DATA, "_TWII.csv"), parse_dates=["date"]).set_index("date")["close"]
    tw = tw.reindex(P.index).ffill()
    mret = tw.pct_change()
    Rm = R.mul(mret, axis=0)
    regime_on = (tw > tw.rolling(60).mean())

    panels = {
        "Q1 低特質波動(AHXZ06)": score_Q1(P, R, Rm, mret, ma200),
        "Q2 流動性溢酬(Amihud02)": score_Q2(P, R, TO),
        "Q3 風險調整動能": score_Q3(P, R),
        "Q4 避樂透股(MAX,BCW11)": score_Q4(R),
        "Q5 新高頻率(持續強勢)": score_Q5(P),
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
