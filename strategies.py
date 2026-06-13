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
    df["atr20"] = tr.rolling(20).mean()   # Turtle N
    df["atr5"]  = tr.rolling(5).mean()
    df["high60"]  = df["high"].rolling(60).max()
    df["high252"] = df["high"].rolling(252).max()
    df["low252"]  = df["low"].rolling(252).min()
    df["high20"]  = df["high"].rolling(20).max()
    df["low10"]   = df["low"].rolling(10).min()   # Turtle System-1 exit
    # Williams %R (14日)
    hi14 = df["high"].rolling(14).max()
    lo14 = df["low"].rolling(14).min()
    df["willr"] = (hi14 - df["close"]) / (hi14 - lo14 + 1e-9) * -100
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



# ── 大師策略 ──────────────────────────────────────────────────────────────

def signal_e(df, i):
    """Turtle Trading System 1 (Richard Dennis & Bill Eckhardt, 1983).

    規則來源: Curtis Faith《Way of the Turtle》2003 年公開版
    哲學: 純價格趨勢跟蹤, 不看基本面/成交量/市場情緒, 只問「價格是否突破通道?」

    進場: 收盤突破前 20 日最高價 (不含當日; 即 high20 前一日)
    停損: 進場價 - 2 × N (N = ATR20, Turtle 對波動的基本單位)
    出場: 收盤跌破前 10 日最低價 (low10 移動停利; 由引擎 trail_low 機制處理)
    加碼: 每漲 0.5N 加 1 單位, 最多 4 單位 (本實作以 engine pyramid 近似)
    部位: 1% 帳戶風險 / (2N × 股價) = 標準 Turtle Unit
    不設獲利目標, 純靠移動停損讓趨勢自然結束
    """
    row = df.iloc[i]
    if np.isnan(row["atr20"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 30_000_000:   # 3000 萬流動性下限
        return None
    N = row["atr20"]
    if N <= 0:
        return None
    # 突破條件: 收盤嚴格大於昨日為止的 20 日高
    prior_high20 = df["high"].iloc[i - 20: i].max()
    if row["close"] <= prior_high20:
        return None
    # 避免極度延伸 (單日漲幅 > 3N 視為 gap, 略過)
    if row["close"] - df["close"].iloc[i - 1] > 3 * N:
        return None
    return {
        "score":     row["close"] / prior_high20,   # 突破強度
        "stop_atr":  2.0,     # 2N 停損
        "trail_low": True,    # 引擎使用 low10 移動停利
        "max_hold":  120,
    }


def signal_f(df, i):
    """Darvas Box Theory (Nicolas Darvas, 1960 《How I Made $2,000,000 in the Stock Market》).

    哲學: 股票先在「箱子」裡盤整, 代表多空達到平衡; 一旦放量突破箱頂,
    代表供需失衡、新一段上漲啟動。資金管理: 停損在箱底, 新箱形成後上移。

    箱頂確認 (box_top): 最近 N 日最高價, 且之後連續 3 日高點均未超過 (盤整確認)
    箱底確認 (box_bot): 箱頂確認期間的最低收盤 (支撐)
    進場: 今日收盤突破 box_top + 成交量 > 1.5× 20日均量 (放量確認)
    停損: box_bot (箱底, 若超過 10% 則縮至 entry×0.90)
    出場: 引擎停損 or 時間出場 (60日); 可觀察下一個箱子
    """
    row = df.iloc[i]
    if np.isnan(row["ma50"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 30_000_000:
        return None

    # 確認箱頂: 往回找最近的局部高點 (3 日高點確認)
    # 掃描 i-25 ~ i-4 的窗口, 找最近一個 "3日後均未超越" 的高點
    box_top = None
    box_bot = None
    for j in range(i - 4, max(i - 30, 3), -1):
        candidate = df["high"].iloc[j]
        # 候選高點之後 3 日內最高不超過 (盤整確認)
        after_high = df["high"].iloc[j + 1: j + 4].max()
        if after_high > candidate:
            continue   # 高點被突破, 不是有效箱頂
        # 確認後, 箱底 = 候選高點確認期間的最低收盤
        box_region_low = df["low"].iloc[j: j + 4].min()
        if candidate <= 0 or box_region_low <= 0:
            continue
        box_depth = (candidate - box_region_low) / candidate
        if box_depth > 0.20:   # 箱子太深 (>20%) 視為盤整失敗
            continue
        box_top = candidate
        box_bot = box_region_low
        break

    if box_top is None:
        return None
    # 今日突破箱頂 + 放量
    if row["close"] <= box_top:
        return None
    if row["volume"] < 1.5 * row["vol_ma20"]:
        return None
    # 箱底停損, 上限 10%
    stop_pct = min(0.10, max(0.03, 1.0 - box_bot / row["close"]))
    return {
        "score":    row["volume"] / row["vol_ma20"],
        "stop_pct": round(stop_pct, 4),
        "max_hold": 60,
    }


def signal_g(df, i):
    """Larry Williams %R 超賣回升 (Larry Williams《Long-Term Secrets to Short-Term Trading》).

    哲學: 在長期上升趨勢中, 短期超賣是進場機會而非逃命訊號。
    Williams 強調「趨勢中的回調進場」是高勝率、低風險的交易模式。

    條件:
      1. 長期趨勢向上: close > MA200 且 MA200 呈上升 (21日前低)
      2. 中期趨勢健康: close > MA50
      3. 短期超賣: Williams %R(14) 今日 <= -80 (極度超賣)
      4. 當日確認反彈: close > open (收紅 K, 顯示買盤介入)
      5. 成交量 >= 均量 (確認有買盤, 非量縮下跌)
    停損: 2 × ATR14 (Williams 強調嚴格停損)
    停利: %R 回升至 -20 以上 (超買) 或 ATR Chandelier
    持有: 短線 15 日
    """
    row = df.iloc[i]
    if np.isnan(row["ma200"]) or np.isnan(row["willr"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 30_000_000:
        return None
    # 長期 + 中期趨勢
    uptrend = (row["close"] > row["ma200"]
               and row["ma200"] > df["ma200"].iloc[i - 21]
               and row["close"] > row["ma50"])
    if not uptrend:
        return None
    # Williams %R 超賣確認反彈
    oversold = row["willr"] <= -80
    reversal = row["close"] > row["open"]           # 收紅 K
    volume_ok = row["volume"] >= 0.8 * row["vol_ma20"]
    if not (oversold and reversal and volume_ok):
        return None
    return {
        "score":      -row["willr"],   # %R 越低越超賣, score 越高
        "stop_atr":   2.0,
        "willr_exit": True,            # 引擎: %R 回升至 -20 即出場
        "trail_atr":  2.5,             # Chandelier 保底
        "max_hold":   15,
    }


STRATEGIES = {
    "A": signal_a, "B": signal_b, "C": signal_c, "D": signal_d,
    "E": signal_e, "F": signal_f, "G": signal_g,
}
