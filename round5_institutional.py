"""主力/法人/投信操作方法代理因子驗證 (第五批)。
重要前提: data_adj/data 皆只有日線OHLCV, 無三大法人買賣超等真實籌碼資料 (與前面
控盤者結論相同), 以下用價量「代理」逼近這幾種法人行為特徵, 非真實籌碼數據。

  Z1 主力吸籌(量增價穩)  — 成交量異常放大但當日振幅低 (大單敲不動價, 吃貨不出貨)
  Z2 投信偏好中小型動能  — 動能排名限定在「中等成交值」帶 (排除最大權值股與最小型股)
  Z3 外資權值動能       — 動能排名限定在「最大成交值」帶 (外資集中大型股調倉代理)
  Z4 月初效應動能       — 用歷史「每月前3個交易日」報酬加總, 找對月初效應敏感的股票
  Z5 季底作帳慣性       — 用歷史「每季底前10個交易日」報酬加總, 找對作帳行情敏感的股票
門檻: MAR >= D(0.99)*1.3 = 1.287。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402

THRESH = 0.99 * 1.3


def score_Z1(P, R, V):
    vol_surge = V.rolling(5).mean() / V.rolling(60).mean().replace(0, np.nan)
    amp = (P.rolling(5).max() - P.rolling(5).min()) / P
    stability = 1.0 / amp.replace(0, np.nan)
    return {"accum": vol_surge * stability}


def score_Z3(P, turnover20):
    elig_rank = turnover20.rank(axis=1, pct=True)
    mom = P / P.shift(126) - 1.0
    return {"megacap_mom": mom.where(elig_rank >= 0.85)}


def score_Z2(P, turnover20):
    elig_rank = turnover20.rank(axis=1, pct=True)
    mom = P / P.shift(126) - 1.0
    return {"midcap_mom": mom.where((elig_rank >= 0.40) & (elig_rank <= 0.75))}


def month_start_score(P, R):
    is_start = pd.Series(np.arange(len(P.index)), index=P.index).groupby(P.index.to_period("M")).rank() <= 3
    mask = pd.DataFrame(np.tile(is_start.values[:, None], (1, R.shape[1])), index=R.index, columns=R.columns)
    masked = R.where(mask, 0.0)
    return masked.rolling(252).sum()


def quarter_end_score(P, R):
    qper = P.index.to_period("Q")
    rank_from_end = pd.Series(np.arange(len(P.index)), index=P.index).groupby(qper).rank(ascending=False)
    is_end = rank_from_end <= 10
    mask = pd.DataFrame(np.tile(is_end.values[:, None], (1, R.shape[1])), index=R.index, columns=R.columns)
    masked = R.where(mask, 0.0)
    return masked.rolling(252).sum()


def main():
    print("載入價量面板 ...")
    P, V = xb.load_panels()
    TO = P * V
    R = P.pct_change()
    turnover20 = TO.rolling(20).mean()
    tw = pd.read_csv(os.path.join(xb.DATA, "_TWII.csv"), parse_dates=["date"]).set_index("date")["close"]
    tw = tw.reindex(P.index).ffill()
    regime_on = (tw > tw.rolling(60).mean())

    panels = {
        "Z1 主力吸籌(量增價穩)": score_Z1(P, R, V),
        "Z2 投信偏好中小型動能": score_Z2(P, turnover20),
        "Z3 外資權值動能": score_Z3(P, turnover20),
        "Z4 月初效應動能": {"month_start": month_start_score(P, R)},
        "Z5 季底作帳慣性": {"quarter_end": quarter_end_score(P, R)},
    }

    per = P.index.to_period("M")
    rebal_all = pd.Series(P.index, index=per).groupby(level=0).max().tolist()
    rebal = [d for d in rebal_all if P.index.get_loc(d) >= xb.WARMUP]
    print(f"再平衡月數: {len(rebal)}  區間 {rebal[0].date()} ~ {rebal[-1].date()}\n")
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
    print(f"\n本批最佳: {best[0]}  MAR={best[1]:.2f}  (門檻 {THRESH:.2f})")
    print("達標" if best[1] >= THRESH else "未達標")
    return 0


if __name__ == "__main__":
    sys.exit(main())
