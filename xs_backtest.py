"""橫斷面投組回測引擎: 月再平衡, 等權持有分數前 N 名, 真實成本 + 大盤濾網.

對 xs_strategies.py 的 5 個學術策略 (K/L/M/N/O) 做公平測試, 與現有冠軍 D
(全期 CAGR 8.3% / MaxDD 8.4% / MAR 0.99) 比較風險調整後報酬 (MAR / Sharpe)。

公平設計:
  - 共同暖身 756 日 (殘差動能需 36 個月), 所有策略同起點 (~2014.5–2026)。
  - 同一套交易成本 (手續費/稅/滑價) 與大盤濾網 (TAIEX<MA60 → 全現金, 同 D 的擇時)。
  - 等權前 N 名, 月再平衡; 換手成本按實際權重變動計。
  - 對照組: 等權全市場 (有/無濾網) = 市場 beta 基準。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_strategies as xs  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data_adj")
FEE, TAX, SLIP = 0.001425, 0.003, 0.001
TWO_WAY = FEE * 2 + TAX + SLIP * 2        # 進出一次的總成本率 ≈ 0.785%
TURNOVER_FLOOR = 20_000_000               # 20 日均成交額門檻 (可交易性)
WARMUP = xs.BETA_W                         # 共同暖身 (756)
MONTHS_PER_YEAR = 12


def load_panels():
    closes, vols = {}, {}
    for fn in os.listdir(DATA):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(DATA, fn), parse_dates=["date"]).set_index("date")
        closes[code] = df["close"]
        vols[code] = df["volume"]
    P = pd.DataFrame(closes).sort_index()
    V = pd.DataFrame(vols).reindex(P.index)
    return P, V


def metrics(eq, rets):
    eq = eq.dropna()
    n = len(rets)
    if n < 12 or eq.iloc[0] <= 0:
        return None
    years = n / MONTHS_PER_YEAR
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    vol = rets.std() * np.sqrt(MONTHS_PER_YEAR)
    sharpe = rets.mean() / rets.std() * np.sqrt(MONTHS_PER_YEAR) if rets.std() > 0 else 0
    return dict(cagr=cagr, dd=dd, mar=cagr / dd if dd > 0 else np.nan,
                sharpe=sharpe, vol=vol, wr=(rets > 0).mean(),
                worst=rets.min(), n=n)


def weights_of(names):
    if not names:
        return {}
    w = 1.0 / len(names)
    return {c: w for c in names}


def eligible_at(P, turnover20, d):
    elig = ((P.loc[d] >= 10) & (turnover20.loc[d] >= TURNOVER_FLOOR))
    return elig[elig].index


def make_select_fn(comp_panels, P, turnover20, N):
    """回傳 select_fn(d)->names: 取分數前 N 名 (多 component 先橫斷面 z-score 再加總)."""
    def select(d):
        elig = eligible_at(P, turnover20, d)
        comps = {name: panel.loc[d].reindex(elig) for name, panel in comp_panels.items()}
        cdf = pd.DataFrame(comps).dropna()
        if cdf.empty:
            return []
        z = (cdf - cdf.mean()) / cdf.std(ddof=0).replace(0, np.nan)
        composite = z.sum(axis=1).dropna()
        if composite.empty:                       # 退化 (單一 component 零變異) → 用原始分數
            composite = cdf.sum(axis=1).dropna()
        return composite.nlargest(N).index.tolist()
    return select


def make_ew_select_fn(P, turnover20):
    """等權全市場基準: 取全部可交易名單."""
    def select(d):
        return eligible_at(P, turnover20, d).tolist()
    return select


def run_backtest(select_fn, P, rebal, regime_on, use_regime=True):
    """以 select_fn 逐月選股, 月再平衡等權持有, 回傳 (equity, monthly returns)."""
    eq, rets, dates, prev_w = [1.0], [], [rebal[0]], {}
    for k in range(len(rebal) - 1):
        d, d1 = rebal[k], rebal[k + 1]
        if use_regime and not bool(regime_on.get(d, False)):
            names = []                            # 防禦期全現金
        else:
            names = select_fn(d)
        if names:
            seg = P.loc[d:d1, names]
            entry = P.loc[d, names]
            last_valid = seg.ffill().iloc[-1]     # 含下市: 以最後有效價了結
            gross = (last_valid / entry - 1.0).mean()
        else:
            gross = 0.0
        new_w = weights_of(names)
        allc = set(new_w) | set(prev_w)
        turnover = 0.5 * sum(abs(new_w.get(c, 0) - prev_w.get(c, 0)) for c in allc)
        net = gross - turnover * TWO_WAY
        eq.append(eq[-1] * (1 + net))
        rets.append(net)
        dates.append(d1)
        prev_w = new_w
    return pd.Series(eq, index=dates), pd.Series(rets, index=dates[1:])


def main():
    print("載入價量面板 ...")
    P, V = load_panels()
    TO = P * V
    R = P.pct_change()
    ma200 = P.rolling(200).mean()
    turnover20 = TO.rolling(20).mean()

    tw = pd.read_csv(os.path.join(DATA, "_TWII.csv"), parse_dates=["date"]).set_index("date")["close"]
    tw = tw.reindex(P.index).ffill()
    mret = tw.pct_change()
    Rm = R.mul(mret, axis=0)
    regime_on = (tw > tw.rolling(60).mean())

    # 月再平衡日 = 每月最後一個交易日 (暖身後)
    per = P.index.to_period("M")
    rebal_all = pd.Series(P.index, index=per).groupby(level=0).max().tolist()
    rebal = [d for d in rebal_all if P.index.get_loc(d) >= WARMUP]
    print(f"再平衡月數: {len(rebal)}  區間 {rebal[0].date()} ~ {rebal[-1].date()}\n")

    # 預先算各策略 component panels (重運算只做一次)
    print("計算各策略分數面板 ...")
    panels = {}
    for key, fn in xs.SCORERS.items():
        panels[key] = fn(P, R, Rm, mret, ma200)
        print(f"  {key} done")

    ew_sel = make_ew_select_fn(P, turnover20)

    print("\n回測中 ...")
    print(f"  {'策略':<26}{'N':>4}{'CAGR':>7}{'MaxDD':>7}{'MAR':>6}{'Sharpe':>7}"
          f"{'年波動':>7}{'月勝率':>7}{'最差月':>7}")
    bar = "  " + "-" * 76
    print(bar)
    # 基準
    for label, ureg in [("EW 全市場+濾網", True), ("EW 全市場(無濾網)", False)]:
        eq, rr = run_backtest(ew_sel, P, rebal, regime_on, use_regime=ureg)
        m = metrics(eq, rr)
        print(f"  {label:<26}{'all':>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
              f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['vol']*100:>6.1f}%"
              f"{m['wr']*100:>6.0f}%{m['worst']*100:>6.1f}%")
    print(bar)
    results = {}
    for key in ["K", "L", "M", "N", "O"]:
        for N in [10, 20, 50]:
            sel = make_select_fn(panels[key], P, turnover20, N)
            eq, rr = run_backtest(sel, P, rebal, regime_on, use_regime=True)
            m = metrics(eq, rr)
            results[(key, N)] = (m, eq, rr)
            tag = xs.LABELS[key] if N == 10 else ""
            print(f"  {tag:<26}{N:>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
                  f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['vol']*100:>6.1f}%"
                  f"{m['wr']*100:>6.0f}%{m['worst']*100:>6.1f}%")
        print(bar)

    print("\n  對照基準 D (事件引擎, 全期): CAGR 8.3% / MaxDD 8.4% / MAR 0.99")
    print("  (註: D 為 8 檔事件停損制; 此處為月再平衡投組, 比較聚焦 MAR / Sharpe)")

    # 與 D 之外, 也看彼此相關性 (是否真的正交) — 用 N=20 的月報酬
    print("\n=== 5 策略月報酬相關矩陣 (N=20, 驗證彼此正交) ===")
    rdf = pd.DataFrame({key: results[(key, 20)][2] for key in ["K", "L", "M", "N", "O"]})
    corr = rdf.corr()
    print("        " + "".join(f"{k:>7}" for k in ["K", "L", "M", "N", "O"]))
    for a in ["K", "L", "M", "N", "O"]:
        print(f"  {a:<6}" + "".join(f"{corr.loc[a,b]:>7.2f}" for b in ["K", "L", "M", "N", "O"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
