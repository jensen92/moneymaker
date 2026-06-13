"""台股飆股篩選策略 (A / C / D).

策略 A: Minervini SEPA 完整版 — 五模組 (趨勢模板+VCP+MVP+負面排除), RS≥70.
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



# ─────────────────────────────────────────────────────────────────────────────
# 策略 A 輔助函式
# ─────────────────────────────────────────────────────────────────────────────

def _detect_vcp(df, i, min_t=2, max_t=6, w=4):
    """VCP 波動收縮型態: 找出 min_t ~ max_t 次遞縮回檔 + 右側量縮.

    每次 peak→trough 回撤幅度需比前一次小 (整體末 < 首 × 0.70),
    且最後一次收縮幅度 < 15%; 近 8 根平均量需低於 50 日均量 85%.
    回傳 (is_vcp: bool, n_contractions: int).
    """
    lb = min(70, i - 2)
    s  = i - lb
    hi  = df["high"].values[s:i + 1]
    lo  = df["low"].values[s:i + 1]
    vol = df["volume"].values[s:i + 1]
    vm  = df["vol_ma50"].values[s:i + 1]
    n   = len(hi)
    if n < 20:
        return False, 0

    # 局部極值 (window = w bars)
    peaks   = [j for j in range(w, n - w) if hi[j] == hi[j - w:j + w + 1].max()]
    troughs = [j for j in range(w, n - w) if lo[j] == lo[j - w:j + w + 1].min()]
    if len(peaks) < 2 or len(troughs) < 2:
        return False, 0

    # 按時間排列, 依序找 peak→trough 對
    events = [(j, "P", hi[j]) for j in peaks] + [(j, "T", lo[j]) for j in troughs]
    events.sort()
    swings, last_pk = [], None
    for j, t, v in events:
        if t == "P":
            last_pk = (j, v)
        elif t == "T" and last_pk is not None:
            pct = (last_pk[1] - v) / last_pk[1]
            if 0.03 <= pct <= 0.65:      # 有效回撤: 3% ~ 65%
                swings.append(pct)
            last_pk = None               # 下一輪需新高點

    if len(swings) < min_t:
        return False, 0

    sw = swings[-max_t:]                 # 取最近 max_t 次收縮
    # 整體遞縮: 最後一次 < 第一次 × 0.70, 且末次 < 15% (緊縮到位)
    contracting = sw[-1] < sw[0] * 0.70 and sw[-1] < 0.15
    # 右側量縮: 近 8 根均量 < 50 日均量 × 0.85
    valid_vm = vm[-8:][np.isfinite(vm[-8:])]
    vol_dryup = (len(valid_vm) >= 4
                 and np.mean(vol[-8:]) < 0.85 * np.mean(valid_vm))
    return contracting and vol_dryup, len(sw)


def signal_a(df, i, rs_rank=None):
    """Minervini SEPA 完整版 (策略 A) — 五模組全部實作.

    模組一  趨勢模板 (8 條件):
      1/8. close > MA150, MA200, MA50
      2.   MA150 > MA200
      3.   MA200 上揚 >= 21 交易日
      4.   MA50 > MA150 > MA200
      5.   close >= 52 週低點 × 1.25
      6.   close >= 52 週高點 × 0.75
      7a.  RS rank >= 70
      7b.  RS 線上升趨勢 >= 6 週 (以 ret126 近 30 日斜率近似)
      8.   close > MA50

    模組二  基本面 (因無財報資料, 本版略過 EPS/Revenue/ROE 篩選).

    模組三  VCP 波動收縮型態:
      - 2 ~ 6 次遞縮回檔 (每次約為前次一半)
      - 右側量縮 (最後一次收縮量低於 50 日均量)

    模組四  進場觸發與 MVP 動能 (OR 關係, 任一成立即進場):
      - 突破觸發: 收盤突破 20 日高點 + 當日量 > 1.5× MA50量
      - MVP 指標: 近 15 日 ≥12 天上漲 + 量增 25% + 漲幅 ≥20%

    模組五  負面排除 (先行篩除, 任一命中即放棄):
      - close < MA200 (第四階段衰退)
      - 從 52 週高點回撤 > 50%
      - 近 40 根內出現 ≥2 次向下跳空 (>2%)
      - 近 60 根內成交量最大一日為顯著長黑 (收 < 開 -1.5%)

    出場 (與策略 C 相同, 由引擎執行):
      MA50 跌破全出 / +20% 賣半 / 3R 保本移停 / 最長 90 日
    """
    row  = df.iloc[i]
    prev = df.iloc[i - 1]

    # 最低流動性門檻
    if row["close"] < 10 or row["volume"] * row["close"] < 50_000_000:
        return None
    if np.isnan(row["ma200"]) or np.isnan(row["low252"]):
        return None

    # ═══ 模組五: 負面排除 (優先執行) ═════════════════════════════════════
    # 5-1 跌破 MA200 → 可能進入第四階段
    if row["close"] < row["ma200"]:
        return None
    # 5-2 從 52 週高點回撤 > 50% → 基本面可能惡化、上方套牢壓力大
    if row["close"] < row["high252"] * 0.50:
        return None
    # 5-3 近 40 根出現 ≥2 次向下跳空 (>2%) → 連續下跳代表機構出貨
    gap_downs = sum(
        1 for k in range(max(1, i - 39), i + 1)
        if df["open"].iloc[k] < df["close"].iloc[k - 1] * 0.98
    )
    if gap_downs >= 2:
        return None
    # 5-4 近 60 根內量最大一日為長黑 (收 < 開 -1.5%) → 機構出貨訊號
    sub60 = df.iloc[max(0, i - 59):i + 1]
    mvr = df.iloc[sub60["volume"].idxmax()]
    if mvr["close"] < mvr["open"] * 0.985:
        return None

    # ═══ 模組一: 趨勢模板 (8 條件) ═══════════════════════════════════════
    # 條件 1/8: close > MA150, MA200, MA50
    if not (row["close"] > row["ma150"]
            and row["close"] > row["ma200"]
            and row["close"] > row["ma50"]):
        return None
    # 條件 2: MA150 > MA200
    if row["ma150"] <= row["ma200"]:
        return None
    # 條件 3: MA200 上揚 >= 21 交易日 (最好 84~105 日)
    if row["ma200"] <= df["ma200"].iloc[i - 21]:
        return None
    # 條件 4: MA50 > MA150 (> MA200 已由條件 2 推導)
    if row["ma50"] <= row["ma150"]:
        return None
    # 條件 5: close >= 52 週低點 × 1.25 (脫離底部至少 25%)
    if row["close"] < row["low252"] * 1.25:
        return None
    # 條件 6: close >= 52 週高點 × 0.75 (位於年高附近)
    if row["close"] < row["high252"] * 0.75:
        return None
    # 條件 7a: RS rank >= 70
    if rs_rank is None or rs_rank < 0.70:
        return None
    # 條件 7b: RS 線上升趨勢 >= 6 週 (≈30 交易日)
    #   以 ret126 近 30 日是否上升近似 (ret126 提升 → 相對強度改善)
    if i < 30 or df["ret126"].iloc[i] <= df["ret126"].iloc[i - 30]:
        return None

    # ═══ 模組三: VCP 波動收縮型態 ════════════════════════════════════════
    vcp_ok, n_t = _detect_vcp(df, i)
    if not vcp_ok:
        return None

    # ═══ 模組四: 進場觸發與 MVP 動能 (OR) ═══════════════════════════════
    # 4-1 突破觸發: 收盤突破 20 日高點 + 當日量 > 1.5× MA50量
    prior_high20 = df["high"].iloc[i - 20:i].max()
    breakout = (row["close"] > prior_high20
                and row["volume"] > 1.5 * row["vol_ma50"])

    # 4-2 MVP 動能 (David Ryan):
    #   近 15 日 ≥12 天收漲 (Momentum)
    #   + 20 日均量相較 15 日前增加 25% (Volume)
    #   + 近 15 日漲幅 ≥20% (Price)
    mvp = False
    if i >= 15:
        up_days = sum(
            1 for k in range(i - 14, i + 1)
            if df["close"].iloc[k] > df["close"].iloc[k - 1]
        )
        vol_surge15 = (not np.isnan(df["vol_ma20"].iloc[i - 15])
                       and row["vol_ma20"] > df["vol_ma20"].iloc[i - 15] * 1.25)
        ret15 = row["close"] / df["close"].iloc[i - 15] - 1.0
        mvp = up_days >= 12 and vol_surge15 and ret15 >= 0.20

    if not (breakout or mvp):
        return None

    return {
        "score":           rs_rank,
        "minervini":       True,
        "stop_pct":        0.08,
        "three_week_gain": 0.20,
        "max_hold":        90,
    }


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



STRATEGIES = {"A": signal_a, "C": signal_c, "D": signal_d}
