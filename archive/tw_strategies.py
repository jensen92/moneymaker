"""根據 tw_habits.py 實證的台股習性, 反推 3 個選股策略並真實回測。

實測習性 -> 策略對應:
  T1 隔夜溢酬因子   <- 習性A: 報酬幾乎全在隔夜。選「累積隔夜報酬」最強股 (需求/籌碼
                       推升, 非盤中當沖追殺) — 月再平衡。另附隔夜vs全日資本曲線示範。
  T2 漲停續攻(短週期) <- 習性F: 漲停後5日顯著續強, 但效應是「天」。週再平衡, 選近5日
                       有漲停且仍強勢者 (對的持有週期)。
  T3 動能×季節×波動  <- 習性C+E+G: 動能單調有效, 高波動在多頭更強, 9-10月轉弱。
                       複合分數=動能+波動, 並在9-10月降風險 (季節擇時疊加大盤濾網)。
門檻: 對標 D MAR 0.99, 看是否超越前面30因子天花板(~0.5), 甚至逼近D。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402

THRESH = 0.99 * 1.3
DATA = xb.DATA


def load_oc():
    op, cl, vol = {}, {}, {}
    for fn in os.listdir(DATA):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(DATA, fn), parse_dates=["date"]).set_index("date")
        op[code] = df["open"]; cl[code] = df["close"]; vol[code] = df["volume"]
    C = pd.DataFrame(cl).sort_index()
    O = pd.DataFrame(op).reindex(C.index)
    V = pd.DataFrame(vol).reindex(C.index)
    return O, C, V


def metrics_pp(eq, rets, ppy):
    eq = eq.dropna(); n = len(rets)
    if n < 12 or eq.iloc[0] <= 0:
        return None
    years = n / ppy
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    sharpe = rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else 0
    return dict(cagr=cagr, dd=dd, mar=cagr / dd if dd > 0 else np.nan,
                sharpe=sharpe, wr=(rets > 0).mean())


def main():
    print("載入 OHLCV ...")
    O, C, V = load_oc()
    P = C
    TO = P * V
    R = P.pct_change()
    prevC = C.shift(1)
    overnight = (O / prevC - 1.0)
    ma200 = P.rolling(200).mean()
    turnover20 = TO.rolling(20).mean()
    tw = pd.read_csv(os.path.join(DATA, "_TWII.csv"), parse_dates=["date"]).set_index("date")["close"]
    tw = tw.reindex(P.index).ffill()
    regime_on = (tw > tw.rolling(60).mean())
    # 季節擇時: 9/10月關閉 (習性E實測弱)
    regime_seasonal = regime_on & (~P.index.month.isin([9, 10]))

    uptrend = P > ma200
    mom6 = P / P.shift(126) - 1.0
    vol20 = R.rolling(20).std()

    panels = {
        "T1 隔夜溢酬因子": {"on": overnight.rolling(60).sum().where(uptrend)},
        "T3 動能×波動(複合)": {"mom": mom6.where(uptrend), "vol": vol20.where(uptrend)},
    }
    # T2 漲停續攻 (週頻): 近5日漲停次數 + 仍在上升
    limitup5 = (R >= 0.09).rolling(5).sum()
    panels_w = {"T2 漲停續攻(週)": {"lu": limitup5.where(uptrend), "mom": (P / P.shift(20) - 1.0).where(uptrend)}}

    per = P.index.to_period("M")
    rebal_m = [d for d in pd.Series(P.index, index=per).groupby(level=0).max().tolist()
               if P.index.get_loc(d) >= xb.WARMUP]
    perw = P.index.to_period("W")
    rebal_w = [d for d in pd.Series(P.index, index=perw).groupby(level=0).max().tolist()
               if P.index.get_loc(d) >= xb.WARMUP]

    print(f"目標: 對標 D MAR 0.99 (1.3倍={THRESH:.2f}), 前30因子天花板~0.5\n")

    # ── 習性A 示範: 隔夜 vs 全日 資本曲線 (買強勢動能股池, 只算毛報酬示範效應大小) ──
    print("=== 習性A 示範: 同一動能股池, 隔夜持有 vs 全日持有 (毛報酬, 不計成本) ===")
    sel = xb.make_select_fn({"mom": mom6.where(uptrend)}, P, turnover20, 30)
    on_eq, full_eq = [1.0], [1.0]
    days = [d for d in P.index if P.index.get_loc(d) >= xb.WARMUP]
    # 月選股, 日持有
    holding = []
    rebal_set = set(rebal_m)
    for d in days[:-1]:
        if d in rebal_set or not holding:
            holding = sel(d)
        nxt = days[days.index(d) + 1] if days.index(d) + 1 < len(days) else None
        if nxt is None or not holding:
            continue
        on_r = overnight.loc[nxt, holding].mean()
        full_r = R.loc[nxt, holding].mean()
        on_eq.append(on_eq[-1] * (1 + (on_r if pd.notna(on_r) else 0)))
        full_eq.append(full_eq[-1] * (1 + (full_r if pd.notna(full_r) else 0)))
    print(f"  隔夜持有累積: {on_eq[-1]:.1f}x   全日持有累積: {full_eq[-1]:.1f}x")
    print("  (示範隔夜溢酬規模; 但日頻換手成本0.785%/次會吃掉, 故下方做月/週可行版)\n")

    best = (None, -999)
    print(f"  {'策略':<22}{'頻率':>4}{'N':>4}{'CAGR':>7}{'MaxDD':>7}{'MAR':>6}{'Sharpe':>7}{'勝率':>6}")
    bar = "  " + "-" * 62
    print(bar)
    for label, comp in panels.items():
        for N in [20, 50]:
            sel = xb.make_select_fn(comp, P, turnover20, N)
            reg = regime_seasonal if label.startswith("T3") else regime_on
            eq, rr = xb.run_backtest(sel, P, rebal_m, reg, use_regime=True)
            m = metrics_pp(eq, rr, 12)
            tag = label if N == 20 else ""
            flag = " <==達標!" if m["mar"] >= THRESH else (" <超前30" if m["mar"] > 0.52 else "")
            print(f"  {tag:<22}{'月':>4}{N:>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
                  f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['wr']*100:>5.0f}%{flag}")
            if m["mar"] > best[1]:
                best = (f"{label} 月N={N}", m["mar"])
        print(bar)
    for label, comp in panels_w.items():
        for N in [20, 50]:
            sel = xb.make_select_fn(comp, P, turnover20, N)
            eq, rr = xb.run_backtest(sel, P, rebal_w, regime_on, use_regime=True)
            m = metrics_pp(eq, rr, 52)
            tag = label if N == 20 else ""
            flag = " <==達標!" if m["mar"] >= THRESH else (" <超前30" if m["mar"] > 0.52 else "")
            print(f"  {tag:<22}{'週':>4}{N:>4}{m['cagr']*100:>6.1f}%{m['dd']*100:>6.1f}%"
                  f"{m['mar']:>6.2f}{m['sharpe']:>7.2f}{m['wr']*100:>5.0f}%{flag}")
            if m["mar"] > best[1]:
                best = (f"{label} 週N={N}", m["mar"])
        print(bar)
    print(f"\n本批最佳: {best[0]}  MAR={best[1]:.2f}  (前30因子天花板~0.5, D=0.99, 門檻1.29)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
