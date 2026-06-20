"""台股習性實證分析 (stylized facts): 用全市場日線量化幾個常見台股說法是否成立,
作為設計選股策略的依據 (不靠網友傳說, 靠數據)。

測項:
  A 隔夜 vs 盤中報酬分解  — 台股報酬究竟發生在隔夜跳空還是盤中? (隔夜溢酬假說)
  B 短期反轉 (1週)        — 上週漲多的下週是否反轉? (散戶過度反應)
  C 中期動能 (6月)        — 6月漲多的下月是否續強?
  D 週轉率/規模效應       — 高週轉/小型股下月報酬是否不同?
  E 月份季節性            — 哪幾個月台股個股平均報酬最好/最差? (除權息、作帳)
  F 漲停續攻              — 接近漲停(>9%)隔日與後5日報酬 vs 全市場基準
  G 波動狀態              — 高波動後 vs 低波動後, 下月報酬 (波動率擇時)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_backtest as xb  # noqa: E402

DATA = xb.DATA


def load_oc_panels():
    op, cl, hi, lo, vol = {}, {}, {}, {}, {}
    for fn in os.listdir(DATA):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(DATA, fn), parse_dates=["date"]).set_index("date")
        op[code] = df["open"]; cl[code] = df["close"]
        hi[code] = df["high"]; lo[code] = df["low"]; vol[code] = df["volume"]
    O = pd.DataFrame(op).sort_index(); C = pd.DataFrame(cl).reindex(O.index)
    H = pd.DataFrame(hi).reindex(O.index); L = pd.DataFrame(lo).reindex(O.index)
    V = pd.DataFrame(vol).reindex(O.index)
    return O, C, H, L, V


def main():
    print("載入 OHLCV 面板 ...")
    O, C, H, L, V = load_oc_panels()
    prevC = C.shift(1)
    overnight = (O / prevC - 1.0)             # 隔夜跳空報酬
    intraday = (C / O - 1.0)                   # 盤中報酬
    total = (C / prevC - 1.0)
    # 只取可交易 (價>=10, 有量)
    tradable = (C >= 10) & (V > 0) & (prevC > 0)
    on = overnight.where(tradable); idy = intraday.where(tradable); tot = total.where(tradable)

    print("\n=== A 隔夜 vs 盤中報酬分解 (全市場逐股逐日平均) ===")
    print(f"  隔夜(open/prevClose-1) 日均: {on.stack().mean()*100:+.4f}%  年化≈{on.stack().mean()*252*100:+.1f}%")
    print(f"  盤中(close/open-1)     日均: {idy.stack().mean()*100:+.4f}%  年化≈{idy.stack().mean()*252*100:+.1f}%")
    print(f"  全日(close/prevClose-1)日均: {tot.stack().mean()*100:+.4f}%")
    print("  -> 台股報酬主要來自隔夜還是盤中?")

    R = C.pct_change().where(tradable)
    print("\n=== B 短期反轉 (上週報酬 vs 下週報酬) ===")
    r5 = C / C.shift(5) - 1.0
    fwd5 = C.shift(-5) / C - 1.0
    rk = r5.rank(axis=1, pct=True)
    for q, lab in [(0.9, "上週最強10%"), (0.5, "中位"), (0.1, "上週最弱10%")]:
        mask = (rk >= q - 0.05) & (rk < q + 0.05)
        print(f"  {lab:<12} 下週均報酬: {fwd5.where(mask).stack().mean()*100:+.3f}%")

    print("\n=== C 中期動能 (6月報酬 vs 下月報酬) ===")
    r126 = C / C.shift(126) - 1.0
    fwd21 = C.shift(-21) / C - 1.0
    rk = r126.rank(axis=1, pct=True)
    for q, lab in [(0.9, "6月最強10%"), (0.5, "中位"), (0.1, "6月最弱10%")]:
        mask = (rk >= q - 0.05) & (rk < q + 0.05)
        print(f"  {lab:<12} 下月均報酬: {fwd21.where(mask).stack().mean()*100:+.3f}%")

    print("\n=== D 週轉率/規模效應 (20日成交值分位 vs 下月報酬) ===")
    TO20 = (C * V).rolling(20).mean()
    rk = TO20.rank(axis=1, pct=True)
    for q, lab in [(0.95, "最大型5%"), (0.75, "大型"), (0.5, "中型"), (0.25, "小型"), (0.1, "最小10%")]:
        mask = (rk >= q - 0.05) & (rk < q + 0.05)
        print(f"  {lab:<10} 下月均報酬: {fwd21.where(mask).stack().mean()*100:+.3f}%")

    print("\n=== E 月份季節性 (全市場個股各日曆月平均日報酬) ===")
    bym = R.stack().groupby(R.stack().index.get_level_values(0).month).mean() * 100
    for m in range(1, 13):
        bar = "█" * max(0, int(bym.get(m, 0) * 50))
        print(f"  {m:>2}月: {bym.get(m, float('nan')):+.4f}%  {bar}")

    print("\n=== F 漲停續攻 (接近漲停>9% 隔日與後5日 vs 基準) ===")
    limitup = (R >= 0.09)
    fwd1 = C.shift(-1) / C - 1.0
    print(f"  漲停隔日均報酬: {fwd1.where(limitup).stack().mean()*100:+.3f}%  (基準全市場: {fwd1.where(tradable).stack().mean()*100:+.3f}%)")
    print(f"  漲停後5日均報酬: {fwd5.where(limitup).stack().mean()*100:+.3f}%  (基準: {fwd5.where(tradable).stack().mean()*100:+.3f}%)")

    print("\n=== G 波動狀態 (20日波動分位 vs 下月報酬) ===")
    vol20 = R.rolling(20).std()
    rk = vol20.rank(axis=1, pct=True)
    for q, lab in [(0.9, "最高波動10%"), (0.5, "中位"), (0.1, "最低波動10%")]:
        mask = (rk >= q - 0.05) & (rk < q + 0.05)
        print(f"  {lab:<12} 下月均報酬: {fwd21.where(mask).stack().mean()*100:+.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
