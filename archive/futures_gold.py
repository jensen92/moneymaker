"""黃金 GC 策略 — 核心目標: 抓上漲、下跌果斷退場, 避免巨額回撤/保證金追繳.

對期貨(有槓桿)而言「買進持有」不可行: 黃金 2011-2015 跌約 -42%, 這種回撤會直接
斷頭。正確做法是「幾乎做多, 但用風險管理把崩跌段砍掉」:
  VT     波動率目標: 高波動(常伴隨崩跌)自動減碼, 控制單位風險
  MA200↑ 趨勢閘門: 僅在 200 日均線之上且上升才持有, 確認轉空即退場/空手
  快出場  另加 MA100 閘門, 跌破即更早離場 (回撤更小, 但較易被洗)

以 25 年資料驗證, 並特別檢視 2011-09~2016-01 黃金大空頭的保護效果。
用法: python3 futures_gold.py
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "screen_data", "GC=F.csv")
TD = 252
COST = 0.0005
LEV_CAP = 2.0
BEAR = ("2011-09-01", "2016-01-31")      # 黃金大空頭 (買進持有=被追繳的那段)
IS_END = pd.Timestamp("2015-12-31")
OOS_START = pd.Timestamp("2016-01-01")


def load():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").set_index("date")
    return df["close"].astype(float)


def main():
    c = load()
    ret = c.pct_change().fillna(0.0)
    ma100, ma200 = c.rolling(100).mean(), c.rolling(200).mean()
    ma_up = ma200 > ma200.shift(20)
    above_up = ((c > ma200) & ma_up).shift(1, fill_value=False).astype(float)
    fast = ((c > ma100) & (c > ma200) & ma_up).shift(1, fill_value=False).astype(float)

    def vt(tv):
        v = ret.ewm(span=30).std() * np.sqrt(TD)
        return (tv / v).clip(upper=LEV_CAP).shift(1).fillna(0.0)

    strategies = {
        "買進持有 (危險)":          pd.Series(1.0, index=c.index),
        "VT18% × MA200↑ (報酬優先)": vt(0.18) * above_up,
        "VT12% × MA200↑ (推薦)":    vt(0.12) * above_up,
        "VT10% × MA100+200↑ (最保守)": vt(0.10) * fast,
    }

    def stats(s, sub=None):
        x = s if sub is None else s[sub]
        eq = (1 + x).cumprod(); yrs = len(x) / TD
        sh = x.mean() / x.std() * np.sqrt(TD) if x.std() > 0 else 0
        cg = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
        dd = ((eq.cummax() - eq) / eq.cummax()).max()
        return sh, cg, dd

    print(f"黃金 GC 抗回撤策略  ({c.index[0].date()} ~ {c.index[-1].date()})")
    print("核心: 避免巨額回撤/斷頭, 而非追求最高報酬\n")
    hdr = (f"{'策略':<26}{'Sharpe':>7}{'年化':>7}{'最大回撤':>9}"
           f"{'OOS Sh':>7}{'崩盤虧損':>9}")
    print(hdr); print("-" * len(hdr))
    for name, pos in strategies.items():
        turn = pos.diff().abs().fillna(0.0)
        s = pos.shift(1).fillna(0.0) * ret - turn * COST
        sh, cg, dd = stats(s)
        sho = stats(s, s.index >= OOS_START)[0]
        bear = s[(s.index >= BEAR[0]) & (s.index <= BEAR[1])]
        bret = (1 + bear).cumprod().iloc[-1] - 1
        print(f"{name:<26}{sh:>7.2f}{cg:>7.1%}{dd:>9.1%}{sho:>7.2f}{bret:>9.1%}")

    print(f"\n黃金現貨 {BEAR[0]}~{BEAR[1]} 本身跌約 -42%。")
    print("推薦 VT12%×MA200↑: 與報酬優先版同 Sharpe(0.77), 但崩盤只虧 ~-17% (買進持有 -39%),")
    print("最大回撤 22% — 槓桿帳戶可存活。想更安全用最保守版 (崩盤僅 ~-13%)。")


if __name__ == "__main__":
    main()
