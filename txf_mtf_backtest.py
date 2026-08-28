"""台指期 多時間框架順勢回檔波段 (txf_mtf) 回測 + 抗過擬合驗證。

依 strategy-pipeline 精神:
  ‧ 統計主錨 = 小時 K (^TWII 60m, 3 年) — IS/OOS 對切、參數平台、逐年、bootstrap。
  ‧ 執行頻率確認 = 15 分 K (~60 天) — 相同 txf_mtf.run() 邏輯, 小樣本結果誠實標註。
  ‧ 日線趨勢濾網由盤中資料 resample 出的『每日收盤』計算 (同一資料源)。

成本沿用 txf_backtest 的模型 (滑價1點 + 手續費 + 期交稅), 以淨點數評估。
所有數字皆由本檔即時跑出 (禁止抄舊文件)。

用法:
  python3 txf_mtf_backtest.py            # 主報告 (小時K IS/OOS + 15分確認 + 基準)
  python3 txf_mtf_backtest.py --grid     # 參數平台掃描 (G3.3)
  python3 txf_mtf_backtest.py --years    # 逐年分解 (G3.4)
  python3 txf_mtf_backtest.py --boot     # bootstrap 顯著性 (G4)
  python3 txf_mtf_backtest.py --wf       # walk-forward (G3.2)
"""
import argparse
import csv
import os
import sys

import numpy as np

import txf_mtf
from txf_backtest import cost_pts, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_60M = os.path.join(HERE, "futures_data", "TWII_60m.csv")
DATA_15M = os.path.join(HERE, "futures_data", "TWII_15m.csv")


def load_rows(path):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for dt, o, h, l, c, v in r:
            rows.append((dt, float(o), float(h), float(l), float(c)))
    return rows


def _net(t):
    """交易淨點 = 毛點 − 進出兩邊成本。"""
    return t["gross_pt"] - (cost_pts(t["entry"]) + cost_pts(t["exit"]))


def backtest(rows, fast=txf_mtf.EMA_FAST, slow=txf_mtf.EMA_SLOW,
             ema_daily=txf_mtf.EMA_DAILY, stop_pt=txf_mtf.STOP_PT,
             row_slice=None):
    """回傳 (metrics_dict, net_trades_list, trade_records)。

    row_slice: (lo, hi) 以『K 棒索引』切段做 IS/OOS/walk-forward; None = 全期。
    切段時 EMA/日線濾網只用該段資料 (段內獨立), 貼近實務逐段重算。
    """
    if row_slice is not None:
        rows = rows[row_slice[0]:row_slice[1]]
    bars, ddates, dcloses = txf_mtf.prepare(rows, fast, slow)
    bias = txf_mtf.daily_bias_map(ddates, dcloses, ema_daily)
    trades, _open = txf_mtf.run(bars, bias, stop_pt)
    nets = [_net(t) for t in trades]
    # 依出場日彙整每日損益 (Sharpe 用曆日)
    daily = {d: 0.0 for d in ddates}
    for t, net in zip(trades, nets):
        d = t["exit_dt"].split(" ")[0]
        daily[d] = daily.get(d, 0.0) + net
    return metrics(nets, daily), nets, trades


def split_report(rows, label, frac=0.7, **kw):
    n = len(rows)
    cut = int(n * frac)
    m_all, *_ = backtest(rows, **kw)
    m_is, *_ = backtest(rows, row_slice=(0, cut), **kw)
    m_oos, *_ = backtest(rows, row_slice=(cut, n), **kw)
    d0, d1 = rows[0][0].split(" ")[0], rows[-1][0].split(" ")[0]
    print(f"\n===== {label} =====")
    print(f"K棒 {n} ({d0} .. {d1}), 參數 fast={kw.get('fast',txf_mtf.EMA_FAST)} "
          f"slow={kw.get('slow',txf_mtf.EMA_SLOW)} emaD={kw.get('ema_daily',txf_mtf.EMA_DAILY)} "
          f"stop={kw.get('stop_pt',txf_mtf.STOP_PT):.0f}")
    if not m_all:
        print("  無交易。")
        return None
    print(f"  全期: {m_all['trades']} 筆, 勝率 {m_all['win']:.0%}, 淨 {m_all['total']:.0f} 點 "
          f"(NT${m_all['nt']:,.0f}/大台), PF {m_all['pf']:.2f}, 每筆 {m_all['avg']:+.1f}, "
          f"MaxDD {m_all['maxdd']:.0f} 點, Sharpe {m_all['sharpe']:.2f}")
    if m_is:
        print(f"  IS(前{int(frac*100)}%): {m_is['trades']} 筆, 淨 {m_is['total']:+.0f}, PF {m_is['pf']:.2f}")
    if m_oos:
        ok = m_oos["total"] > 0 and m_oos["pf"] > 1.1
        print(f"  OOS(後{100-int(frac*100)}%): {m_oos['trades']} 筆, 淨 {m_oos['total']:+.0f}, "
              f"PF {m_oos['pf']:.2f}  → {'✅ 樣本外仍獲利' if ok else '❌ 樣本外未過關'}")
    return m_all, m_is, m_oos


def grid_scan(rows):
    print("\n===== 參數平台掃描 (小時K 3年, G3.3) =====")
    print("取平台中心非最佳格; 鄰格應成片正期望且 OOS 不塌")
    print(f"{'fast':>5}{'slow':>5}{'stop':>6}{'交易':>6}{'勝率':>6}{'淨點':>9}{'PF':>6}"
          f"{'OOS淨':>8}{'OOS_PF':>7}")
    print("-" * 62)
    n = len(rows); cut = int(n * 0.7)
    for fast in (16, 20, 24):
        for slow in (40, 50, 60):
            for stop in (80.0, 100.0, 120.0):
                m, *_ = backtest(rows, fast, slow, txf_mtf.EMA_DAILY, stop)
                mo, *_ = backtest(rows, fast, slow, txf_mtf.EMA_DAILY, stop, (cut, n))
                if not m:
                    continue
                star = "  ⭐" if (fast == txf_mtf.EMA_FAST and slow == txf_mtf.EMA_SLOW
                                and stop == txf_mtf.STOP_PT) else ""
                print(f"{fast:>5}{slow:>5}{stop:>6.0f}{m['trades']:>6}{m['win']*100:>5.0f}%"
                      f"{m['total']:>9.0f}{m['pf']:>6.2f}{(mo['total'] if mo else 0):>8.0f}"
                      f"{(mo['pf'] if mo else 0):>7.2f}{star}")


def years_report(rows):
    print("\n===== 逐年分解 (小時K 3年, G3.4) =====")
    _, nets, trades = backtest(rows)
    by_year = {}
    for t, net in zip(trades, nets):
        by_year.setdefault(t["exit_dt"][:4], []).append(net)
    print(f"{'年度':>6}{'交易':>6}{'勝率':>6}{'淨點':>9}{'每筆':>7}")
    print("-" * 40)
    totals = {}
    for y in sorted(by_year):
        a = np.array(by_year[y]); totals[y] = a.sum()
        print(f"{y:>6}{len(a):>6}{(a>0).mean()*100:>5.0f}%{a.sum():>9.0f}{a.mean():>7.1f}")
    grand = sum(totals.values()); best = max(totals, key=totals.get)
    print("-" * 40)
    print(f"  全期淨 {grand:.0f} 點; 最佳年 {best} 貢獻 {totals[best]:.0f} 點")
    print(f"  去掉最佳年後淨 {grand-totals[best]:.0f} 點 → "
          f"{'✅ 不靠單一離群年' if grand-totals[best] > 0 else '❌ 獲利集中於單一年'}")


def walk_forward(rows, k=4):
    """G3.2: 切 k 段, 逐段當 OOS (段內獨立重算 EMA/濾網), 串接是否整體獲利且不集中。"""
    print(f"\n===== Walk-forward ({k} 段, 小時K, G3.2) =====")
    n = len(rows); step = n // k
    seg_tot = []
    for s in range(k):
        lo = s * step; hi = n if s == k - 1 else (s + 1) * step
        m, *_ = backtest(rows, row_slice=(lo, hi))
        d0 = rows[lo][0].split(" ")[0]; d1 = rows[hi-1][0].split(" ")[0]
        tot = m["total"] if m else 0.0; seg_tot.append(tot)
        print(f"  段{s+1} {d0}..{d1}: {m['trades'] if m else 0} 筆, 淨 {tot:+.0f}, "
              f"PF {m['pf'] if m else 0:.2f}")
    pos = sum(1 for x in seg_tot if x > 0)
    print(f"  {pos}/{k} 段獲利, 串接淨 {sum(seg_tot):+.0f} → "
          f"{'✅ 各段皆或多數獲利、不集中' if pos >= k-1 and sum(seg_tot) > 0 else '⚠️ 獲利集中或有虧損段'}")


def bootstrap_report(rows, n_boot=5000, seed=42, n_scan=27):
    print("\n===== Bootstrap 顯著性 (小時K 3年, G4) =====")
    _, nets, _ = backtest(rows)
    a = np.array(nets)
    if len(a) < 2:
        print("  樣本不足。"); return
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    t = a.mean() / a.std(ddof=1) * np.sqrt(len(a)) if a.std(ddof=1) > 0 else 0.0
    print(f"  樣本 {len(a)} 筆, 平均單筆 {a.mean():+.2f} 點, 標準差 {a.std(ddof=1):.1f}")
    print(f"  Bootstrap({n_boot}) 平均 95% CI = [{lo:+.2f}, {hi:+.2f}] → "
          f"{'✅ 下界>0 顯著' if lo > 0 else '❌ 下界≤0 不顯著'}")
    print(f"  掃描 N={n_scan} 組, 多重測試門檻 t>3;實際 t={t:.2f} → "
          f"{'✅ 通過' if t > 3 else ('⚠️ 介於 2~3 邊際' if t > 2 else '❌ <2')}")


def buy_hold(rows, label):
    first, last = rows[0][4], rows[-1][4]
    print(f"\n[基準] {label} 買進持有 ^TWII: {first:.0f} → {last:.0f} = {last-first:+.0f} 點 "
          f"(強多頭期; 本策略須在扣成本後提供正報酬且風險調整後有價值)")


def main():
    ap = argparse.ArgumentParser()
    for f in ("grid", "years", "boot", "wf"):
        ap.add_argument(f"--{f}", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DATA_60M):
        print("缺 futures_data/TWII_60m.csv, 先跑 python3 txf_data.py"); sys.exit(1)
    rows60 = load_rows(DATA_60M)
    rows15 = load_rows(DATA_15M) if os.path.exists(DATA_15M) else None

    if args.grid:  grid_scan(rows60); return
    if args.years: years_report(rows60); return
    if args.boot:  bootstrap_report(rows60); return
    if args.wf:    walk_forward(rows60); return

    print("台指期 多時間框架順勢回檔波段 (txf_mtf) — 回測主報告")
    print("=" * 64)
    split_report(rows60, "統計主錨 · 小時K (^TWII 60m, 3年)")
    buy_hold(rows60, "小時K區間")
    if rows15:
        split_report(rows15, "執行頻率確認 · 15分K (^TWII 15m, ~60天, 小樣本)")
        print("  ⚠️ 15分僅 ~60 交易日, 樣本過小 → 判定上限『觀察』, 不足以做統計檢定 (G4)。")
    print("\n" + "=" * 64)
    print("完整六檢: --grid (平台) / --wf (walk-forward) / --years (逐年) / --boot (顯著性)")


if __name__ == "__main__":
    main()
