"""黃金 GC 專屬策略 — 目標: 打敗「買進持有黃金」基準 (Sharpe 0.69 / 回撤 44%).

核心思路: 黃金是長期多頭, 擇時進出注定輸給長抱。要贏只能「幾乎一直做多, 但用
風險管理削掉大回撤」:
  VT     波動率目標: 目標固定年化波動, 高波動(常伴隨崩跌)時自動減碼, 平穩時可加碼
  MA200  空頭濾網: 收盤跌破 200 日均線 (確認轉空) 才退場/減碼, 避免被洗
  組合   VT × 趨勢濾網 — 抓多頭、躲空頭、風險恆定

以日報酬基準比較 (含成本), 全期 + 樣本內外。資料: screen_data/GC=F.csv (~25 年)。
用法: python3 futures_gold.py
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "screen_data", "GC=F.csv")
TD = 252
COST = 0.0005
TARGET_VOL = 0.18          # 18% → 與買進持有同報酬(~11%)但回撤砍到 ~32%
LEV_CAP = 2.5
IS_END = pd.Timestamp("2015-12-31")
OOS_START = pd.Timestamp("2016-01-01")


def load():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").set_index("date")
    return df[["open", "high", "low", "close"]].astype(float)


def vol_scale(ret):
    rv = ret.ewm(span=30).std() * np.sqrt(TD)
    return (TARGET_VOL / rv).clip(upper=LEV_CAP).shift(1).fillna(0.0)


def positions(df):
    c = df["close"]
    ret = c.pct_change().fillna(0.0)
    ma200 = c.rolling(200).mean()
    ma200_up = ma200 > ma200.shift(20)
    above = (c > ma200).shift(1, fill_value=False)
    above_up = ((c > ma200) & ma200_up).shift(1, fill_value=False)
    vs = vol_scale(ret)
    return {
        "買進持有 (基準)":        pd.Series(1.0, index=df.index),
        "MA200 濾網 (多/空手)":   above.astype(float),
        "波動目標 VT (常多)":     vs,
        "VT × MA200 濾網":       vs * above.astype(float),
        "VT × MA200(且上升)":    vs * above_up.astype(float),
    }, ret


def stat(strat, sub=None):
    s = strat if sub is None else strat[sub]
    eq = (1 + s).cumprod()
    yrs = len(s) / TD
    sharpe = s.mean() / s.std() * np.sqrt(TD) if s.std() > 0 else 0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return sharpe, cagr, dd, (cagr / dd if dd > 0 else 0)


def main():
    df = load()
    pos_map, ret = positions(df)
    print(f"黃金 GC 策略 — 打敗買進持有  ({df.index[0].date()} ~ {df.index[-1].date()})\n")
    hdr = f"{'策略':<22}{'全期Sh':>7}{'IS Sh':>7}{'OOS Sh':>7}{'年化':>8}{'回撤':>8}{'MAR':>6}"
    print(hdr); print("-" * len(hdr))
    bh_sharpe = None
    for name, pos in pos_map.items():
        turn = pos.diff().abs().fillna(0.0)
        strat = pos.shift(1).fillna(0.0) * ret - turn * COST
        sh, cagr, dd, mar = stat(strat)
        shi = stat(strat, strat.index <= IS_END)[0]
        sho = stat(strat, strat.index >= OOS_START)[0]
        if bh_sharpe is None:
            bh_sharpe = sh
        beat = "  ✅勝基準" if sh > bh_sharpe and name != "買進持有 (基準)" else ""
        print(f"{name:<22}{sh:>7.2f}{shi:>7.2f}{sho:>7.2f}{cagr:>8.1%}"
              f"{dd:>8.1%}{mar:>6.2f}{beat}")
    print(f"\n基準(買進持有): Sharpe {bh_sharpe:.2f}; 目標是 Sharpe 與 MAR 同時勝過它。")
    print(f"設定: 目標波動 {TARGET_VOL:.0%}, 槓桿上限 {LEV_CAP}x, 成本 {COST*100:.2f}%/邊。")


if __name__ == "__main__":
    main()
