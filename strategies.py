"""兩個台股飆股篩選策略.

策略 A「強勢拉回」: 高勝率, 風險報酬比 >= 2
  - 多頭排列 (close > MA20 > MA60), MA20 上揚
  - 近 20 日漲幅 >= 10% (動能股)
  - 拉回觸及 MA20 附近後當日收回 MA20 之上 (買回確認)
  - 停損: 進場價 - 1.5 ATR; 停利: 2.2R; 時間出場 15 日

策略 B「爆量突破」: 風險報酬比 ~5, 勝率目標 3-4 成
  - 收盤創 60 日新高
  - 成交量 >= 2 倍 20 日均量 (爆量)
  - 收盤接近當日最高 (收在 K 棒上緣 25% 內)
  - 股價 > MA60
  - 停損: 進場價 - 1.2 ATR; 停利: 5R; 時間出場 40 日
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
    """強勢拉回. 回傳 (score, stop_mult, target_r, max_hold) 或 None."""
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    if np.isnan(row["ma60"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000:  # 日成交值 5 千萬以上
        return None
    uptrend = (row["close"] > row["ma20"] > row["ma60"]
               and row["ma20"] > df["ma20"].iloc[i - 5]
               and row["ma60"] > df["ma60"].iloc[i - 5])
    momentum = 0.15 <= row["ret20"] <= 0.60
    touched = row["low"] <= row["ma20"]  # 確實觸及 MA20
    reclaimed = row["close"] > row["ma20"] and row["close"] > prev["close"]
    if uptrend and momentum and touched and reclaimed:
        # 參數掃描 (sweep_a.py) 選定: 寬停損 2.5 ATR 降低被洗掉機率,
        # 也減少成本對 R 的侵蝕; 動能下限 15% 過濾弱勢股
        return {"score": row["ret60"], "stop_atr": 2.5,
                "target_r": 2.0, "max_hold": 20}
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
        score = row["volume"] / row["vol_ma20"]
        # 寬鬆初始停損 + Chandelier 移動停利, 讓贏家跑出 5R 以上
        return {"score": score, "stop_atr": 1.5,
                "trail_atr": 3.0, "max_hold": 60}
    return None


STRATEGIES = {"A": signal_a, "B": signal_b}
