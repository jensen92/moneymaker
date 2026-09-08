"""台指期 兩策略合併書分析 — 既有『前日高低突破』(當沖) + 新『多時間框架順勢回檔』(波段)。

動機: 兩者每日淨損益近乎零相關 (實測 −0.014), 屬真分散。單獨看新策略 bootstrap 不顯著
(t=1.61); 本檔驗證『各 1 口合併』的資金曲線是否比任一單獨更穩 (Sharpe↑、報酬/回撤↑),
甚至是否讓合併書達到統計顯著。

單一程式碼真相: 突破用 `txf_backtest.strat_prevday_breakout` + 動態停損 (同 /txf 生產);
MTF 用 `txf_mtf_backtest.backtest` 的交易紀錄。皆以『每日曆日淨點』對齊成資金曲線。
所有數字本次跑出。用法: python3 txf_mtf_combo.py
"""
import os
from collections import defaultdict

import numpy as np

import txf_mtf_backtest as B
from txf_backtest import (load_days, cost_pts, strat_prevday_breakout,
                          POINT_VALUE)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_60M = os.path.join(HERE, "futures_data", "TWII_60m.csv")


def breakout_daily():
    """既有前日高低突破 (動態停損 max(40,0.5×前日區間), 收盤平倉) 的每日淨點。"""
    days = load_days()
    daily = defaultdict(float)
    prev = None
    for d, bars in days:
        if prev is not None:
            ph = max(b["h"] for b in prev[1]); pl = min(b["l"] for b in prev[1])
            stop = max(40, round((ph - pl) * 0.5))
            res = strat_prevday_breakout(
                bars, {"h": ph, "l": pl, "c": prev[1][-1]["c"]},
                {"stop_pt": stop, "side": "both"})
            if res:
                dirn, entry, ex = res
                daily[d] += dirn * (ex - entry) - (cost_pts(entry) + cost_pts(ex))
        prev = (d, bars)
    return daily


def mtf_daily():
    """新 MTF 順勢回檔波段的每日淨點 (依出場日彙整)。"""
    rows = B.load_rows(DATA_60M)
    _, nets, trades = B.backtest(rows)
    daily = defaultdict(float)
    for t, net in zip(trades, nets):
        daily[t["exit_dt"].split(" ")[0]] += net
    return daily


def curve_metrics(daily_series, calendar=None):
    """給定 {date: net_pt}, 回傳資金曲線指標。

    Sharpe 一律以『全交易日曆』(calendar, 未交易日補 0) 年化, 與 txf_mtf_backtest 主報告
    同慣例, 避免只算有損益日而高估 Sharpe。calendar=None 時退回用序列自身日期。
    """
    if calendar is None:
        calendar = sorted(daily_series)
    x = np.array([daily_series.get(d, 0.0) for d in calendar])
    if len(x) == 0:
        return None
    eq = np.cumsum(x)
    maxdd = (np.maximum.accumulate(eq) - eq).max()
    wins = x[x > 0].sum(); losses = -x[x < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    sharpe = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    return {"days": len(x), "active": int((x != 0).sum()), "total": x.sum(),
            "maxdd": maxdd, "pf": pf, "sharpe": sharpe,
            "mar": x.sum() / maxdd if maxdd > 0 else float("inf")}


def bootstrap_daily(daily_series, n_boot=5000, seed=42):
    """對『有損益日』的每日淨點 bootstrap, 回傳 (mean, ci_lo, ci_hi)。"""
    x = np.array([v for v in daily_series.values() if v != 0.0])
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return x.mean(), lo, hi


def align_sum(a, b):
    """兩每日序列相加 (各 1 口), 回傳合併每日淨點 dict (union of dates)。"""
    out = defaultdict(float)
    for d, v in a.items():
        out[d] += v
    for d, v in b.items():
        out[d] += v
    return out


def split_curve(daily_series, calendar, frac=0.7):
    """以全交易日曆 70/30 切 IS/OOS (未交易日仍計入曆日, Sharpe 一致)。"""
    cut = int(len(calendar) * frac)
    return (curve_metrics(daily_series, calendar[:cut]),
            curve_metrics(daily_series, calendar[cut:]))


def _row(name, m):
    if not m:
        return f"{name:<10} 無交易"
    return (f"{name:<10}{m['total']:>9.0f}{m['sharpe']:>8.2f}{m['maxdd']:>9.0f}"
            f"{m['mar']:>8.2f}{m['pf']:>7.2f}{m['active']:>7}")


def main():
    bo = breakout_daily()
    mt = mtf_daily()
    combo = align_sum(bo, mt)

    # 全交易日曆 = 60m 資料的所有交易日 (未交易日補 0, Sharpe 一致年化)
    cal = sorted({r[0].split(" ")[0] for r in B.load_rows(DATA_60M)})

    # 相關性 (全曆日, 空日補 0)
    xb = np.array([bo.get(d, 0.0) for d in cal])
    xm = np.array([mt.get(d, 0.0) for d in cal])
    corr = np.corrcoef(xb, xm)[0, 1]
    alld = cal

    print("台指期 兩策略合併書分析 (^TWII 小時K, 各 1 口, 扣成本淨點)")
    print("=" * 64)
    print(f"共同期間交易日 {len(alld)} ({alld[0]} .. {alld[-1]})")
    print(f"每日淨損益相關係數 = {corr:+.3f} "
          f"({'低相關, 真分散' if abs(corr) < 0.3 else '相關偏高'})\n")

    print(f"{'策略':<10}{'淨點':>9}{'Sharpe':>8}{'MaxDD':>9}{'報酬/DD':>8}{'PF':>7}{'交易日':>7}")
    print("-" * 64)
    m_bo = curve_metrics(bo, cal)
    m_mt = curve_metrics(mt, cal)
    m_co = curve_metrics(combo, cal)
    print(_row("突破(當沖)", m_bo))
    print(_row("MTF(波段)", m_mt))
    print(_row("合併(1+1)", m_co))

    print("\n--- 合併書 IS/OOS (70/30 曆日切) ---")
    ci, co = split_curve(combo, cal)
    print(_row("合併IS", ci))
    print(_row("合併OOS", co))
    ok = co and co["total"] > 0 and co["pf"] > 1.1
    print(f"  → {'✅ 合併書樣本外仍獲利' if ok else '❌ 合併書樣本外未過關'}")

    print("\n--- Bootstrap 顯著性 (每日淨點, 5000 次) ---")
    for name, s in (("突破", bo), ("MTF", mt), ("合併", combo)):
        r = bootstrap_daily(s)
        if not r:
            continue
        mean, lo, hi = r
        print(f"  {name:<6} 平均/日 {mean:+.1f}, 95% CI [{lo:+.1f}, {hi:+.1f}] → "
              f"{'✅ 顯著' if lo > 0 else '❌ 下界≤0'}")

    print("\n【結論方向 (推論)】兩策略近乎零相關;比較上表 Sharpe / 報酬-回撤比,")
    print("合併書若優於任一單獨, 佐證『分散價值』——即使新策略單獨統計未顯著,")
    print("在既有生產策略上『加一口 MTF』仍可能改善整體風險調整後報酬。以合併書數字為準。")


if __name__ == "__main__":
    main()
