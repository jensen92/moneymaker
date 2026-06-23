"""組合層級策略優化 (整體戰略思維, 非個別參數 curve-fit).

戰略三層:
  1. 相關性分析  — 5 個勝出法皆動能/突破型, 檢驗彼此 (及與防守核心 D) 的相關性,
                  決定「集中 vs 分散」。高度相關者全上 ≠ 真分散, 只是加槓桿。
  2. 品質疊加    — 5 法回撤普遍 20%+。把 C/D 低回撤的成因 (Minervini Stage-2 趨勢
                  模板: 收盤>MA150>MA200 且 MA200 上彎, 再加 RS>=0.80) 疊加到動能法上,
                  屬「移植已驗證濾網」而非調參。觀察是否大幅壓低回撤 → 提升 MAR。
  3. 組合建構    — 核心(防守: D) + 衛星(進攻: 最佳動能法), 等權合併, 比較效率前緣
                  (CAGR / MaxDD / MAR) 是否同時打贏現行 C/D。

合併方法: 每個 sleeve 各以 2M 獨立資金回測, 等權組合權益 = 各 sleeve 權益曲線之
平均 (與 backtest.py 合併多帳戶之邏輯一致, 僅縮放)。MAR = CAGR / MaxDD。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402
import research_expert10 as ex  # noqa: E402

bt.DATA_DIR = os.environ.get("MM_DATA_DIR",
                             os.path.join(os.path.dirname(__file__), "data_adj"))

RS_OVERLAY = 0.80   # 品質疊加之 RS 門檻 (與策略 C 的 rs_min 一致, 非本研究調參)


def collect_winners(data, rs, regime):
    """單趟掃描: 同時產生 5 法的 raw 與 overlay(品質疊加) entry_map."""
    raw = {name: {} for name, _, _ in ex.METHODS}
    ovl = {name: {} for name, _, _ in ex.METHODS}
    for code, df in data.items():
        n = len(df)
        close = df["close"].values
        ma150 = df["ma150"].values
        ma200 = df["ma200"].values
        dates = df["date"].values
        for i in range(210, n - 1):
            d = df["date"].iloc[i]
            if d not in regime:                  # 多頭市況濾網 (與 A/C/D 一致)
                continue
            try:
                rk = rs.at[d, code]
            except KeyError:
                rk = None
            if rk is not None and np.isnan(rk):
                rk = None
            # Stage-2 趨勢模板 + RS 門檻 (品質疊加條件)
            tmpl = (rk is not None and rk >= RS_OVERLAY
                    and not np.isnan(ma200[i]) and not np.isnan(ma150[i])
                    and close[i] > ma150[i] > ma200[i] and ma200[i] > ma200[i - 21])
            for name, fn, _ in ex.METHODS:
                s = fn(df, i, rk)
                if not s:
                    continue
                ed_i = i + 1
                if ed_i >= n:
                    continue
                ed = df["date"].iloc[ed_i]
                raw[name].setdefault(ed, []).append((s["score"], code, ed_i, s))
                if tmpl:
                    ovl[name].setdefault(ed, []).append((rk, code, ed_i, s))
    return raw, ovl


def run_method(data, entry_map, all_dates, date_idx, regime_tiers, vol_scalars):
    return ex.run_generic(data, entry_map, all_dates, date_idx,
                          regime_tiers, vol_scalars)


def curve_stats(curve, years, init=2_000_000):
    eq = curve.set_index("date")["equity"].sort_index()
    if len(eq) == 0:
        return None
    cagr = (eq.iloc[-1] / init) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return {"cagr": cagr, "mdd": dd, "mar": cagr / dd if dd > 0 else np.nan,
            "ret": eq.iloc[-1] / init - 1}


def combine(curves, years):
    """等權合併多 sleeve: 組合權益 = 對齊日期後各曲線平均."""
    panel = None
    for i, c in enumerate(curves):
        s = c.set_index("date")["equity"].rename(f"s{i}").sort_index()
        panel = s.to_frame() if panel is None else panel.join(s, how="outer")
    panel = panel.sort_index().ffill().dropna()
    eq = panel.mean(axis=1)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return {"cagr": cagr, "mdd": dd, "mar": cagr / dd if dd > 0 else np.nan,
            "ret": eq.iloc[-1] / eq.iloc[0] - 1}


def daily_returns(curve):
    eq = curve.set_index("date")["equity"].sort_index()
    return eq.pct_change().fillna(0.0)


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    data = {c: ex.extend(df) for c, df in data.items()}
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    print(f"{len(data)} 檔, {len(all_dates)} 交易日, 跨 {years:.1f} 年\n")

    # ── 防守核心: 策略 D (canonical 路徑) ──
    print("回測防守核心 D ...")
    risk_on = bt.load_regime()
    d_sigs = bt.collect_signals(data, "D")
    d_sigs = {dt: lst for dt, lst in d_sigs.items() if dt in risk_on}
    d_entry = bt.build_entry_map(d_sigs, data)
    d_trades, d_curve = bt.run_sub(data, d_entry, "D", 1.0, 2_000_000,
                                   all_dates=all_dates, date_idx=date_idx,
                                   regime_tiers=regime_tiers, vol_scalars=vol_scalars)

    # ── 5 個動能法: raw + overlay ──
    print("掃描 5 個動能法訊號 (raw + 品質疊加) ...")
    raw, ovl = collect_winners(data, rs, regime)

    raw_curves, ovl_curves = {}, {}
    print("\n=== 個別 sleeve 表現 (2M 標準帳, 1x, 同引擎/成本/風控) ===")
    print(f"  {'方法':<18}{'版本':<8}{'交易':>6}{'CAGR':>8}{'MaxDD':>8}{'MAR':>7}")
    dstat = curve_stats(d_curve, years)
    print(f"  {'[核心] D':<18}{'--':<8}{len(d_trades):>6}"
          f"{dstat['cagr']*100:>7.1f}%{dstat['mdd']*100:>7.1f}%{dstat['mar']:>7.2f}")
    for name, _, _ in ex.METHODS:
        tr, cv = run_method(data, raw[name], all_dates, date_idx, regime_tiers, vol_scalars)
        raw_curves[name] = cv
        st = curve_stats(cv, years)
        print(f"  {name:<18}{'raw':<8}{len(tr):>6}"
              f"{st['cagr']*100:>7.1f}%{st['mdd']*100:>7.1f}%{st['mar']:>7.2f}")
        tro, cvo = run_method(data, ovl[name], all_dates, date_idx, regime_tiers, vol_scalars)
        ovl_curves[name] = cvo
        sto = curve_stats(cvo, years)
        print(f"  {name:<18}{'+品質':<8}{len(tro):>6}"
              f"{sto['cagr']*100:>7.1f}%{sto['mdd']*100:>7.1f}%{sto['mar']:>7.2f}")

    # ── 相關性矩陣 (raw 動能法 + D) ──
    print("\n=== 日報酬相關性矩陣 (raw 動能法 + 防守核心 D) ===")
    series = {name: daily_returns(raw_curves[name]) for name, _, _ in ex.METHODS}
    series["D核心"] = daily_returns(d_curve)
    corr = pd.DataFrame(series).corr()
    short = {n: n[:6] for n in corr.columns}
    corr2 = corr.rename(index=short, columns=short)
    print(corr2.round(2).to_string())

    # ── 組合候選 (效率前緣) ──
    print("\n=== 組合候選比較 (等權合併, MAR = CAGR / MaxDD) ===")
    print(f"  {'組合':<34}{'CAGR':>8}{'MaxDD':>8}{'MAR':>7}{'總報酬':>9}")

    def show(label, curves):
        st = combine(curves, years)
        print(f"  {label:<34}{st['cagr']*100:>7.1f}%{st['mdd']*100:>7.1f}%"
              f"{st['mar']:>7.2f}{st['ret']*100:>8.0f}%")
        return st

    best_raw = max(ex.METHODS, key=lambda m: curve_stats(raw_curves[m[0]], years)["mar"])[0]
    best_ovl = max(ex.METHODS, key=lambda m: curve_stats(ovl_curves[m[0]], years)["mar"])[0]

    show("[基準] 防守核心 D 單獨", [d_curve])
    show("5法 raw 等權", [raw_curves[n] for n, _, _ in ex.METHODS])
    show("5法 +品質 等權", [ovl_curves[n] for n, _, _ in ex.METHODS])
    show(f"D + 最佳raw動能({best_raw[:6]})", [d_curve, raw_curves[best_raw]])
    show(f"D + 最佳品質動能({best_ovl[:6]})", [d_curve, ovl_curves[best_ovl]])
    show("D + 5法raw 等權", [d_curve] + [raw_curves[n] for n, _, _ in ex.METHODS])
    show("D + 5法+品質 等權", [d_curve] + [ovl_curves[n] for n, _, _ in ex.METHODS])

    print(f"\n最佳 raw 動能法 (MAR): {best_raw}")
    print(f"最佳 品質疊加動能法 (MAR): {best_ovl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
