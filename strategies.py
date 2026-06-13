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
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma150"] = df["close"].rolling(150).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ma50"] = df["volume"].rolling(50).mean()
    df["ret20"] = df["close"].pct_change(20)
    df["ret60"] = df["close"].pct_change(60)
    df["ret126"] = df["close"].pct_change(126)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    df["atr14"] = tr.rolling(14).mean()
    df["atr5"] = tr.rolling(5).mean()
    df["high60"] = df["high"].rolling(60).max()
    df["high252"] = df["high"].rolling(252).max()
    df["low252"] = df["low"].rolling(252).min()
    df["high20"] = df["high"].rolling(20).max()
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


def signal_c(df, i, rs_rank=None):
    """Minervini 第二階段趨勢模板 + 波動收縮突破 (策略 C).

    模板 (8 條件, P/E 與 RS 線以可得資料近似):
      1/8. close > MA150, MA200, MA50
      2.   MA150 > MA200
      3.   MA200 上揚 >= 1 個月 (21 交易日)
      4.   MA50 > MA150 > MA200
      5.   close >= 52 週低點 * 1.25
      6.   close >= 52 週高點 * 0.75
      7.   RS rank >= 70 (126 日報酬全市場百分位, 由回測引擎預先計算)
    觸發: 波動收縮 (ATR5 < 0.75*ATR14) + 樞軸日量縮 (< MA50量)
          次日突破 20 日高點 + 放量 (>1.5x MA50量) -- 以收盤突破近似
    出場 (引擎特殊處理):
      - 初始停損 8% (上限 10%)
      - 獲利達 3R 停損上移至成本價
      - MA50 上穿成本價後改用 MA50 移動停損
      - +20% 賣出一半
      - 收盤跌破 MA50 全部出場; 最長持有 90 日
    """
    row = df.iloc[i]
    if np.isnan(row["ma200"]) or np.isnan(row["low252"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000:
        return None
    template = (
        row["close"] > row["ma150"] and row["close"] > row["ma200"]
        and row["ma150"] > row["ma200"]
        and row["ma200"] > df["ma200"].iloc[i - 21]          # 條件 3
        and row["ma50"] > row["ma150"]                        # 條件 4
        and row["close"] > row["ma50"]                        # 條件 8
        and row["close"] >= row["low252"] * 1.25              # 條件 5
        and row["close"] >= row["high252"] * 0.75             # 條件 6
    )
    if not template:
        return None
    if rs_rank is not None and rs_rank < 0.70:                # 條件 7
        return None
    # 波動收縮與量縮看突破前一日 (突破日本身必然放量)
    prev = df.iloc[i - 1]
    contraction = prev["atr5"] < 0.80 * prev["atr14"]
    vol_dryup = prev["volume"] < prev["vol_ma50"]
    prior_high20 = df["high"].iloc[i - 20:i].max()
    breakout = (row["close"] > prior_high20
                and row["volume"] > 1.5 * row["vol_ma50"])
    if contraction and vol_dryup and breakout:
        stop_pct = 0.08
        return {"score": rs_rank if rs_rank is not None else row["ret126"],
                "minervini": True, "stop_pct": stop_pct, "max_hold": 90}
    return None


def _vcp_swings(df, i, lookback=60, req=2, min_spacing=8):
    """以真實局部高低點偵測 VCP 收縮擺動.

    在突破日前 lookback 根 K 線內找局部高點 (5-bar local max),
    測量每次高點到後續低點的回檔深度. 要求:
      - 至少 req 次擺動且深度逐次遞減
      - 首次深度 >= 5% (擺動要有意義), 末次深度 <= 12%
      - 首次深度 <= 40%, 末次均量 < 首次均量 (供給枯竭)
    回傳 (pivot_high, pivot_low, last_depth) 或 None.
    """
    start = i - lookback
    if start < 8:
        return None
    swing_highs = []
    for j in range(start + 5, i - 4):
        h = df["high"].iloc[j]
        if h == df["high"].iloc[j - 5: j + 5].max():
            if not swing_highs or j - swing_highs[-1] >= min_spacing:
                swing_highs.append(j)

    if len(swing_highs) < req + 1:
        return None

    # 只取最後 req+1 個擺動高點
    swing_highs = swing_highs[-(req + 1):]
    swings = []
    for k in range(len(swing_highs) - 1):
        j0, j1 = swing_highs[k], swing_highs[k + 1]
        hi = df["high"].iloc[j0]
        lo = df["low"].iloc[j0: j1 + 1].min()
        if hi <= 0:
            continue
        vol = df["volume"].iloc[max(j0 - 2, 0): j0 + 3].mean()
        swings.append({"hi": hi, "lo": lo, "depth": (hi - lo) / hi, "vol": vol})

    if len(swings) < req:
        return None

    last = swings[-req:]
    if not all(last[k]["depth"] > last[k + 1]["depth"] for k in range(req - 1)):
        return None
    if last[0]["depth"] < 0.05 or last[0]["depth"] > 0.40:  # 首段需有意義
        return None
    if last[-1]["depth"] > 0.12:
        return None
    if last[-1]["vol"] >= last[0]["vol"]:
        return None

    return last[-1]["hi"], last[-1]["lo"], last[-1]["depth"]


def signal_d(df, i, rs_rank=None):
    """Mark Minervini SEPA 精髓版 (策略 D v3).

    在策略 C 已驗證的進場邏輯基礎上, 加入 Minervini 原版獨有的兩個機制:
      1. Pyramid 分批加碼: 突破建 25% 倉, 漲 2% 加 25%, 漲 4% 補滿 50%
      2. 三週法則: 突破後 15 日內漲 >= 20% → 延長持有 8 週

    進場與 C 相同但門檻更嚴:
      - RS rank >= 80 (C 是 70)
      - 突破日漲幅 <= 10% (不追高)
      - 停損 7% (C 是 8%)
    """
    row = df.iloc[i]
    if np.isnan(row["ma200"]) or np.isnan(row["low252"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000:
        return None
    template = (
        row["close"] > row["ma150"] and row["close"] > row["ma200"]
        and row["ma150"] > row["ma200"]
        and row["ma200"] > df["ma200"].iloc[i - 21]
        and row["ma50"] > row["ma150"]
        and row["close"] > row["ma50"]
        and row["close"] >= row["low252"] * 1.25
        and row["close"] >= row["high252"] * 0.75
    )
    if not template:
        return None
    if rs_rank is None or rs_rank < 0.80:
        return None
    prev = df.iloc[i - 1]
    contraction = prev["atr5"] < 0.80 * prev["atr14"]
    vol_dryup = prev["volume"] < prev["vol_ma50"]
    prior_high20 = df["high"].iloc[i - 20: i].max()
    breakout = (row["close"] > prior_high20
                and row["volume"] > 1.5 * row["vol_ma50"]
                and row["close"] / prev["close"] - 1 <= 0.10)
    if not (contraction and vol_dryup and breakout):
        return None
    return {
        "score":     rs_rank,
        "minervini": True,
        "stop_pct":  0.07,
        "max_hold":  90,
    }


STRATEGIES = {"A": signal_a, "B": signal_b, "C": signal_c, "D": signal_d}
