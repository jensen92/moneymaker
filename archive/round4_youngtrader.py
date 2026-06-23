"""「少年股神」熱錢短線思維 5 因子驗證。與前 20 個偏學術/中長期因子完全不同路:
追強勢、爆量突破、漲停續攻、強者恆強、高週轉熱錢。短線本色 -> 同時跑「週再平衡」
與「月再平衡」, 看在真實成本 (來回0.785%) 下高週轉打法能否存活、能否打贏 D 的1.3倍。

  S1 漲停續攻      — 近20日「接近漲停(日漲>=9%)」次數最多 (主力強拉, 散戶最愛追)
  S2 短線最噴(1月) — 近21日報酬最高 (純追最熱, 在 close>MA200 多頭結構內)
  S3 爆量突破龍頭   — 5日均量/60日均量 爆量比最高 + 創60日新高 (量價齊揚突破)
  S4 強者恆強(RS)  — 126日相對強度最高的少數領頭羊 (強勢股集中抱)
  S5 高週轉熱錢     — 週轉代理(成交值/價格)最活躍 (熱錢聚集, 籌碼換手快)
門檻: MAR >= D(0.99)*1.3 = 1.287。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402

THRESH = 0.99 * 1.3


def score_S1(R):
    return {"limitup_freq": (R >= 0.09).rolling(20).sum()}


def score_S2(P, ma200):
    mom1m = P / P.shift(21) - 1.0
    return {"hot_1m": mom1m.where(P > ma200)}


def score_S3(P, V):
    surge = V.rolling(5).mean() / V.rolling(60).mean().replace(0, np.nan)
    at_high = (P >= P.rolling(60).max() * 0.98)
    return {"vol_breakout": surge.where(at_high)}


def score_S4(P):
    return {"rs_126": P / P.shift(126) - 1.0}


def score_S5(P, V):
    return {"churn": (V / P.replace(0, np.nan)).rolling(20).mean() * P}  # ~成交值活躍度代理


def metrics_pp(eq, rets, ppy):
    eq = eq.dropna()
    n = len(rets)
    if n < 12 or eq.iloc[0] <= 0:
        return None
    years = n / ppy
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    sharpe = rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else 0
    return dict(cagr=cagr, dd=dd, mar=cagr / dd if dd > 0 else np.nan,
                sharpe=sharpe, wr=(rets > 0).mean())


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
        "S1 漲停續攻": score_S1(R),
        "S2 短線最噴(1月)": score_S2(P, ma200),
        "S3 爆量突破龍頭": score_S3(P, V),
        "S4 強者恆強(RS126)": score_S4(P),
        "S5 高週轉熱錢": score_S5(P, V),
    }

    per_m = P.index.to_period("M")
    rebal_m_all = pd.Series(P.index, index=per_m).groupby(level=0).max().tolist()
    rebal_m = [d for d in rebal_m_all if P.index.get_loc(d) >= xb.WARMUP]
    per_w = P.index.to_period("W")
    rebal_w_all = pd.Series(P.index, index=per_w).groupby(level=0).max().tolist()
    rebal_w = [d for d in rebal_w_all if P.index.get_loc(d) >= xb.WARMUP]

    print(f"目標門檻: MAR >= {THRESH:.3f} (D的1.3倍)")
    print(f"月再平衡 {len(rebal_m)} 期 / 週再平衡 {len(rebal_w)} 期\n")

    best = (None, -999)
    for freq, rebal, ppy in [("月", rebal_m, 12), ("週", rebal_w, 52)]:
        print(f"=== {freq}再平衡 (含真實成本+大盤濾網) ===")
        print(f"  {'策略':<22}{'N':>4}{'CAGR':>7}{'MaxDD':>7}{'MAR':>6}{'Sharpe':>7}{'勝率':>6}")
        bar = "  " + "-" * 60
        print(bar)
        for label, comp in panels.items():
            for N in [10, 30]:
                sel = xb.make_select_fn(comp, P, turnover20, N)
                eq, rr = xb.run_backtest(sel, P, rebal, regime_on, use_regime=True)
                m = metrics_pp(eq, rr, ppy)
                tag = label if N == 10 else ""
                flag = " <== 達標!" if m and m["mar"] >= THRESH else ""
                if m:
                    print(f"  {tag:<22}{N:>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
                          f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['wr']*100:>5.0f}%{flag}")
                    if m["mar"] > best[1]:
                        best = (f"{freq}/{label} N={N}", m["mar"])
            print(bar)
        print()
    print(f"本批最佳: {best[0]}  MAR={best[1]:.2f}  (門檻 {THRESH:.2f})")
    print("達標" if best[1] >= THRESH else "未達標")
    return 0


if __name__ == "__main__":
    sys.exit(main())
