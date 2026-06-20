"""第三輪驗證 (Round 3/3, 最終輪): 再 5 個新因子, 與前 15 個皆不重複。
目標不變: MAR >= D(0.99) 的 1.3 倍 = 1.287。若仍未達標, 即可下結論:
單純橫斷面排名投組無法在風險調整後打敗 D 的事件引擎風控設計。

  X1 報酬穩定度(品質代理)   — 252日日報酬 mean/std (長窗風險調整動能, 品質味)
  X2 52週區間位置(StochK)  — (close-low52)/(high52-low52), 與 M(George-Hwang) 不同算法
  X3 量能加權動能          — 用每日成交量占窗內總量的權重, 加權累積報酬
  X4 跳空動能              — 形成窗內隔夜跳空報酬(open/prev_close-1)的累積總和
  X5 均線排列強度          — (MA50-MA200)/MA200, 中期/長期均線乖離度當趨勢強度分數
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402
import xs_strategies as xs  # noqa: E402

FORM, SKIP = xs.FORM, xs.SKIP


def score_X1(R):
    m = R.rolling(252).mean()
    s = R.rolling(252).std()
    return {"quality_sharpe": (m / s.replace(0, np.nan)).shift(SKIP)}


def score_X2(P):
    hi = P.rolling(FORM).max()
    lo = P.rolling(FORM).min()
    return {"stoch_k": (P - lo) / (hi - lo).replace(0, np.nan)}


def score_X3(P, R, V):
    vol_sum = V.rolling(FORM).sum()
    w = V.div(vol_sum.replace(0, np.nan))
    weighted = (R * w).rolling(FORM).sum().shift(SKIP)
    return {"vol_weighted_mom": weighted}


def score_X4(P):
    gap = P.shift(1) / P.shift(1).shift(1) - 1.0  # 占位, 實際下方用 open
    return None  # 由 main 直接處理 (需 open 面板)


def score_X5(P):
    ma50 = P.rolling(50).mean()
    ma200 = P.rolling(200).mean()
    return {"ma_spread": (ma50 - ma200) / ma200}


def load_open_panel():
    closes = {}
    for fn in os.listdir(xb.DATA):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(xb.DATA, fn), parse_dates=["date"]).set_index("date")
        closes[code] = df["open"]
    return pd.DataFrame(closes).sort_index()


def main():
    print("載入價量面板 ...")
    P, V = xb.load_panels()
    O = load_open_panel().reindex(index=P.index, columns=P.columns)
    TO = P * V
    R = P.pct_change()
    Rclose_prev = P.shift(1)
    gap_ret = (O / Rclose_prev - 1.0)
    gap_mom = gap_ret.rolling(FORM - SKIP).sum().shift(SKIP)

    ma200 = P.rolling(200).mean()
    turnover20 = TO.rolling(20).mean()
    tw = pd.read_csv(os.path.join(xb.DATA, "_TWII.csv"), parse_dates=["date"]).set_index("date")["close"]
    tw = tw.reindex(P.index).ffill()
    regime_on = (tw > tw.rolling(60).mean())

    panels = {
        "X1 報酬穩定度(品質)": score_X1(R),
        "X2 52週區間位置(StochK)": score_X2(P),
        "X3 量能加權動能": score_X3(P, R, V),
        "X4 跳空動能": {"gap_mom": gap_mom},
        "X5 均線排列強度": score_X5(P),
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
    print("達標" if best[1] >= THRESH else "未達標 (三輪皆未達標, 結束)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
