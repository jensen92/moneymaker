"""兩個台股飆股篩選策略.

策略 A「強勢拉回」: 高勝率, 風險報酬比 >= 2
  - 多頭排列 (close > MA20 > MA60), MA20/MA60 皆上揚
  - 近 20 日漲幅 15-60% (動能股但不過熱)
  - 拉回確實觸及 MA20 後當日收回 MA20 之上
  - 停損: 2.5 ATR; 停利: 2R; 時間出場 20 日

策略 B「爆量突破」: 風險報酬比 ~5, 勝率目標 3-4 成
  - 收盤突破前 60 日高點
  - 成交量 >= 2.5 倍 20 日均量 (爆量)
  - 收在 K 棒上緣 25% 內、近 20 日漲幅 <= 60%
  - 初始停損: 1.5 ATR; Chandelier 移動停利 (最高收盤 - 3 ATR)
  - 時間出場 60 日
"""
import numpy as np
import pandas as pd


def add_indicators(df):
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["ret20"] = df["close"].pct_change(20)
    df["ret60"] = df["close"].pct_change(60)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    df["atr14"] = tr.rolling(14).mean()
    df["high60"] = df["high"].rolling(60).max()
    return df


def signal_a(df, i):
    """強勢拉回."""
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    if np.isnan(row["ma60"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000:
        return None
    uptrend = (row["close"] > row["ma20"] > row["ma60"]
               and row["ma20"] > df["ma20"].iloc[i - 5]
               and row["ma60"] > df["ma60"].iloc[i - 5])
    momentum = 0.15 <= row["ret20"] <= 0.60
    touched = row["low"] <= row["ma20"]          # 確實觸及 MA20 (不含緩衝)
    reclaimed = row["close"] > row["ma20"] and row["close"] > prev["close"]
    if uptrend and momentum and touched and reclaimed:
        # 新引擎掃描 36 組選定: 勝率 45%, pf=1.28, dd=12.4%;
        # target=1.8R 整欄在所有停損值下皆獲利 (穩健區域非單點)
        return {"score": row["ret60"], "stop_atr": 2.0,
                "target_r": 1.8, "max_hold": 30}
    return None


def signal_b(df, i):
    """爆量突破."""
    row = df.iloc[i]
    if np.isnan(row["ma60"]) or np.isnan(row["high60"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000:
        return None
    prior_high = df["high"].iloc[i - 60:i].max()
    breakout = row["close"] > prior_high
    vol_surge = row["volume"] >= 2.5 * row["vol_ma20"]
    rng = row["high"] - row["low"]
    strong_close = rng > 0 and (row["high"] - row["close"]) / rng <= 0.25
    trend = row["close"] > row["ma60"]
    not_blowoff = row["ret20"] <= 0.60
    if breakout and vol_surge and strong_close and trend and not_blowoff:
        return {"score": row["volume"] / row["vol_ma20"], "stop_atr": 1.5,
                "trail_atr": 3.0, "max_hold": 60}
    return None


STRATEGIES = {"A": signal_a, "B": signal_b}
