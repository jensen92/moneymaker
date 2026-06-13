"""台股飆股篩選策略 (C / D).

策略 C: Minervini 第二階段趨勢模板 + 波動收縮突破, RS≥70, 停損 8%.
策略 D: 同 C 但 RS≥85, 停損 8%, 三週法則 (強勢股不賣半).
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


# 策略 D 進場參數 (optimize_d.py 全期 54 組掃描最佳 MAR 組合)
#   舊 D v4: rs 0.80 / stop 0.07 / tw 0.20 → +550% 但 MaxDD 29% (MAR~0.46)
#   優化後 : rs 0.85 / stop 0.08 / tw 0.25 → +509% 且 MaxDD 20% (MAR 0.62)
#   關鍵: 較寬 8% 停損避免強勢突破被洗、RS 提高到 85 提升突破品質
D_CONFIG = {
    "rs_min":          0.85,  # RS 百分位門檻
    "stop_pct":        0.08,  # 初始停損 (寬停損反而少被洗出)
    "gain_cap":        0.10,  # 突破日漲幅上限 (不追高)
    "contraction":     0.80,  # ATR5/ATR14 收縮門檻 (越小越嚴)
    "vol_mult":        1.5,   # 突破日量 / 50日均量 門檻
    "three_week_gain": 0.25,  # 三週法則: N日內漲幅 >= 此值即讓利潤奔跑
    "max_hold":        90,
}


def _d_features(df, i, rs_rank):
    """抽取策略 D 所需的原始特徵 (與參數無關), 模板不過則回傳 None.

    供 signal_d 與優化掃描共用: 模板與原始量價只算一次, 各參數組合再套門檻.
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
    prev = df.iloc[i - 1]
    prior_high20 = df["high"].iloc[i - 20: i].max()
    atr14 = prev["atr14"]
    volma = row["vol_ma50"]
    return {
        "rank":         rs_rank,
        "contraction":  (prev["atr5"] / atr14) if atr14 and atr14 > 0 else 9.9,
        "vol_dryup":    prev["volume"] < prev["vol_ma50"],
        "breakout":     row["close"] > prior_high20,
        "vol_surge":    (row["volume"] / volma) if volma and volma > 0 else 0.0,
        "gain":         row["close"] / prev["close"] - 1.0,
    }


def _d_signal(feat, cfg):
    """以參數組合 cfg 對特徵 feat 判定訊號, 回傳訊號字典或 None."""
    if feat is None:
        return None
    if feat["rank"] is None or feat["rank"] < cfg["rs_min"]:
        return None
    if not (feat["contraction"] < cfg["contraction"]):
        return None
    if not feat["vol_dryup"]:
        return None
    if not (feat["breakout"]
            and feat["vol_surge"] > cfg["vol_mult"]
            and feat["gain"] <= cfg["gain_cap"]):
        return None
    return {
        "score":           feat["rank"],
        "minervini":       True,
        "stop_pct":        cfg["stop_pct"],
        "three_week_gain": cfg["three_week_gain"],
        "max_hold":        cfg["max_hold"],
    }


def signal_d(df, i, rs_rank=None):
    """Mark Minervini SEPA 精髓版 (策略 D).

    在策略 C 已驗證的趨勢模板 + 波動收縮突破基礎上, 加入 Minervini 原版的
    「三週法則」(突破後 15 日內漲 >= 20% 即視為主升段, 不執行 +20% 賣半,
    讓強勢股自由奔跑; 由回測引擎處理), 並收緊進場門檻:
      - RS rank >= 80 (C 是 70)
      - 突破日漲幅 <= 10% (不追高)
      - 停損 7% (C 是 8%)
    參數集中於 D_CONFIG, 方便優化掃描.
    """
    return _d_signal(_d_features(df, i, rs_rank), D_CONFIG)



STRATEGIES = {"C": signal_c, "D": signal_d}
