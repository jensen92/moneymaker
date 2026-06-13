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

def _vcp_raw(df, i, window=4, lookback=120):
    """抽取 VCP 原始特徵 (與門檻無關), 供掃描重用. 結構無效則回傳 None.

    回傳 (swings, vol_ratio, pivot):
      swings    — 由左到右的回撤幅度 list (peak→trough, 有效範圍 3%~65%)
      vol_ratio — 近 10 根均量 / 整段均量 (量縮程度, 越小越乾)
      pivot     — 近 20 根 (不含當根) 最高價, 作為突破樞紐
    收縮次數/末段緊縮/深度/量縮等門檻一律由 _a_signal 套用, 此處只算原始量.
    """
    lb = min(lookback, i - 2)
    s = i - lb
    hi  = df["high"].values[s:i + 1]
    lo  = df["low"].values[s:i + 1]
    vol = df["volume"].values[s:i + 1]
    n   = len(hi)
    if n < 20:
        return None

    w = window
    peaks   = [j for j in range(w, n - w) if hi[j] == hi[j - w:j + w + 1].max()]
    troughs = [j for j in range(w, n - w) if lo[j] == lo[j - w:j + w + 1].min()]
    if len(peaks) < 2 or len(troughs) < 2:
        return None

    events = [(j, "P", hi[j]) for j in peaks] + [(j, "T", lo[j]) for j in troughs]
    events.sort()
    swings, last_pk = [], None
    for j, t, v in events:
        if t == "P":
            last_pk = (j, v)
        elif t == "T" and last_pk is not None:
            pct = (last_pk[1] - v) / last_pk[1]
            if 0.03 <= pct <= 0.65:
                swings.append(pct)
            last_pk = None

    if not swings:
        return None
    mean_all = np.nanmean(vol)
    mean_last10 = np.nanmean(vol[-10:])
    vol_ratio = (mean_last10 / mean_all) if mean_all and mean_all > 0 else 9.9
    pivot = float(np.nanmax(hi[-21:-1]) if len(hi) >= 21 else np.nanmax(hi[:-1]))
    return swings, vol_ratio, pivot


# 策略 A 進場參數 (optimize_a.py 消融 + 144 組掃描最佳 MAR 組合)
#   baseline (原版): CAGR 0.8% / DD 22.6% / MAR 0.04 / PF 1.05  ← 規則互相打架
#   優化後         : CAGR 6.2% / DD 19.1% / MAR 0.32 / PF 1.48  (近三年 CAGR 15.3%)
#   關鍵發現: 條件 7b (RS 線上升) 反而是最大拖累 — 關閉後 PF 1.05→1.23;
#            再配合 RS 0.85 + 8% 停損 + 三週法則 0.30 把 MAR 推到 0.32.
#            衝頂出場 (use_climax) 經消融證實「保留較好」(關閉反而變差).
A_CONFIG = {
    # L2 趨勢樣板
    "rs_min":              0.85,   # RS 百分位門檻 (掃描最佳, 提升突破品質)
    "use_rs_uptrend":      False,  # 條件 7b (RS 線上升): 消融證實為負貢獻, 關閉
    # L1 負面排除
    "use_l1":              True,   # 套用 L1 負面排除 (下跳空/高量長黑)
    # L5 VCP
    "vcp_T_min":           2,      # 最少收縮次數 T
    "vcp_T_max":           6,      # 最多收縮次數 T
    "vcp_last_max":        0.10,   # 末段收縮上限 10%（理想 < 5%）
    "vcp_depth_max":       0.35,   # 整體最深回撤上限 35%
    "vcp_contraction_ratio": 0.80, # 整體遞縮: 末次 < 首次 × 此值
    "vcp_dryup":           0.85,   # 末段均量 / 整段均量 上限 (台股流動性較低)
    # L6 進場
    "breakout_vol_x":      1.5,    # 突破放量倍數
    "use_mvp":             True,   # 啟用 MVP 追勢進場 (突破 OR MVP)
    # 停損 / 持有
    "stop_pct":            0.08,   # 初始停損 8% (寬停損反而少被洗出)
    "three_week_gain":     0.30,   # 三週法則: N 日內漲幅 >= 此值即讓利潤奔跑
    "max_hold":            90,
    # 出場：衝頂 (強勢賣出)
    "use_climax":          True,   # 是否啟用衝頂出場 (強勢賣在上漲中)
    "climax_sprint_pct":   0.25,   # 近 15 日漲幅 >= 25% (衝頂觸發)
    "climax_min_gain":     0.30,   # 持倉總獲利 >= 30% 才啟動衝頂賣出
    "extended_updays":     7,      # 近 10 日上漲天數 >= 7 (延長訊號)
    "extended_min_gain":   0.20,   # 持倉總獲利 >= 20% 才啟動延長賣出
}


def _a_features(df, i, rs_rank):
    """抽取策略 A 所需的原始特徵 (與可調門檻無關), 模板不過回傳 None.

    供 signal_a 與 optimize_a 共用: L0 + L2 模板 (條件 1-6,8 固定標準值) 與
    VCP swing 掃描只算一次; 7a/7b/L1/VCP 門檻/進場/出場 留給 _a_signal 套參數.
    """
    row = df.iloc[i]
    # ═══ L0: 可交易性 ════════════════════════════════════════════════════
    if row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000:
        return None
    if np.isnan(row["ma200"]) or np.isnan(row["low252"]):
        return None
    if i < 200:
        return None

    # ═══ L2: 趨勢樣板 (條件 1-6, 8 固定標準值; 7a/7b 留給 _a_signal) ══════
    if not (row["close"] > row["ma150"] and row["close"] > row["ma200"]):  # 1
        return None
    if row["ma150"] <= row["ma200"]:                                       # 2
        return None
    if row["ma200"] <= df["ma200"].iloc[i - 21]:                           # 3
        return None
    if row["ma50"] <= row["ma150"]:                                        # 4
        return None
    if row["close"] < row["low252"] * 1.25:                                # 5
        return None
    if row["close"] < row["high252"] * 0.75:                               # 6
        return None
    if row["close"] <= row["ma50"]:                                        # 8
        return None

    # ═══ L5: VCP 原始特徵 (swing 掃描, 重運算只做一次) ═══════════════════
    vcp = _vcp_raw(df, i)
    if vcp is None:
        return None
    swings, vol_ratio, pivot = vcp

    # ═══ L1: 負面排除原始量 (存 bool, 由 _a_signal 決定是否套用) ═════════
    gap_downs = sum(
        1 for k in range(max(1, i - 39), i + 1)
        if df["open"].iloc[k] < df["close"].iloc[k - 1] * 0.97
    )
    sub60 = df.iloc[max(0, i - 59):i + 1]
    mvr = df.iloc[sub60["volume"].idxmax()]
    big_red = mvr["close"] < mvr["open"] * 0.985
    l1_pass = gap_downs < 3 and not big_red

    # ═══ L2 條件 7b: RS 線上升趨勢 (ret126 近 30 日上升) ════════════════
    rs_uptrend = i >= 30 and df["ret126"].iloc[i] > df["ret126"].iloc[i - 30]

    # ═══ L6: MVP 原始量 (David Ryan 動能法) ════════════════════════════
    up_days, ret15, vol_ma20_ratio = 0, 0.0, 0.0
    if i >= 15:
        up_days = sum(
            1 for k in range(i - 14, i + 1)
            if df["close"].iloc[k] > df["close"].iloc[k - 1]
        )
        ret15 = row["close"] / df["close"].iloc[i - 15] - 1.0
        v0 = df["vol_ma20"].iloc[i - 15]
        vol_ma20_ratio = (row["vol_ma20"] / v0) if v0 and v0 > 0 else 0.0

    return {
        "rank":           rs_rank,
        "rs_uptrend":     rs_uptrend,
        "l1_pass":        l1_pass,
        "swings":         swings,
        "vol_ratio":      vol_ratio,
        "pivot":          pivot,
        "close":          row["close"],
        "volume":         row["volume"],
        "vol_ma50":       row["vol_ma50"],
        "up_days15":      up_days,
        "ret15":          ret15,
        "vol_ma20_ratio": vol_ma20_ratio,
    }


def _a_signal(feat, cfg):
    """以參數組合 cfg 對特徵 feat 判定策略 A 訊號, 回傳訊號字典或 None."""
    if feat is None:
        return None
    # 條件 7a: RS 門檻
    if feat["rank"] is None or feat["rank"] < cfg["rs_min"]:
        return None
    # 條件 7b: RS 線上升趨勢 (可選)
    if cfg.get("use_rs_uptrend", True) and not feat["rs_uptrend"]:
        return None
    # L1 負面排除 (可選)
    if cfg.get("use_l1", True) and not feat["l1_pass"]:
        return None
    # L5 VCP 門檻
    sw = feat["swings"][-cfg["vcp_T_max"]:]
    if len(sw) < cfg["vcp_T_min"]:
        return None
    if sw[-1] > cfg["vcp_last_max"]:                       # 末段夠緊
        return None
    if max(sw) > cfg["vcp_depth_max"]:                     # 整體不過深
        return None
    if sw[-1] >= sw[0] * cfg["vcp_contraction_ratio"]:     # 整體遞縮
        return None
    if feat["vol_ratio"] > cfg["vcp_dryup"]:               # 末段量縮
        return None
    # L6 進場: 突破 OR (MVP 啟用且成立)
    vm50 = feat["vol_ma50"]
    breakout = (feat["close"] > feat["pivot"]
                and vm50 and feat["volume"] > cfg["breakout_vol_x"] * vm50)
    mvp = (feat["up_days15"] >= 12 and feat["vol_ma20_ratio"] > 1.25
           and feat["ret15"] >= 0.20)
    if not (breakout or (cfg.get("use_mvp", True) and mvp)):
        return None

    sig = {
        "score":           feat["rank"],
        "minervini":       True,
        "stop_pct":        cfg["stop_pct"],
        "three_week_gain": cfg["three_week_gain"],
        "max_hold":        cfg["max_hold"],
    }
    # 衝頂出場 (可選): climax_exit:False 時引擎完全停用此分支
    if cfg.get("use_climax", True):
        sig.update({
            "climax_exit":       True,
            "climax_sprint_pct": cfg["climax_sprint_pct"],
            "climax_min_gain":   cfg["climax_min_gain"],
            "extended_updays":   cfg["extended_updays"],
            "extended_min_gain": cfg["extended_min_gain"],
        })
    return sig


def signal_a(df, i, rs_rank=None):
    """Minervini SEPA 完整版 (策略 A) — 六層漏斗.

    L0 可交易性 → L1 負面排除 → L2 趨勢樣板 8 條 → L3 相對強度(併入 7a/7b)
    → L4 基本面(無財報略) → L5 VCP 型態 → L6 進場(突破 OR MVP).
    參數集中於 A_CONFIG, 方便 optimize_a 掃描.
    """
    return _a_signal(_a_features(df, i, rs_rank), A_CONFIG)


# 策略 C 進場參數 (與 D 共用 _d_features/_d_signal, 僅 config 不同)
#   C = 較寬鬆版 D: 不限突破日漲幅 (gain_cap 放大)、三週法則用引擎預設 0.20.
#   optimize_cd.py 成長掃描 (首次正式優化):
#     舊 C (rs0.70/stop0.08): CAGR 10.2% / DD 24.5% / MAR 0.41 / PF 1.51
#     優化後 (rs0.80/stop0.10): CAGR 10.8% / DD 14.5% / MAR 0.75 / PF 1.78
#     關鍵: RS 提到 0.80 提升品質、停損放寬到 10% 少被洗 → 報酬略增而回撤近乎砍半.
C_CONFIG = {
    "rs_min":          0.80,  # RS 百分位門檻 (掃描最佳, 較舊版 0.70 嚴)
    "stop_pct":        0.10,  # 初始停損 (放寬到 10%, 少被洗出)
    "gain_cap":        9.99,  # 不限突破日漲幅 (C 的特色, 與 D 區隔)
    "contraction":     0.80,  # ATR5/ATR14 收縮門檻
    "vol_mult":        1.5,   # 突破日量 / 50日均量 門檻
    "three_week_gain": 0.20,  # 三週法則 (沿用引擎預設值)
    "max_hold":        90,
}


def signal_c(df, i, rs_rank=None):
    """Minervini 第二階段趨勢模板 + 波動收縮突破 (策略 C).

    與策略 D 共用 _d_features (模板+量價特徵) 與 _d_signal (門檻判定),
    差別僅在 C_CONFIG (RS 較寬、不限突破漲幅). 出場由引擎處理:
    初始停損 8% / 3R 保本 / MA50 移動停損 / +20% 賣半 / 跌破 MA50 全出 / 最長 90 日.
    """
    return _d_signal(_d_features(df, i, rs_rank), C_CONFIG)


# 策略 D 進場參數 (optimize_d.py 核心 + optimize_d2.py 深度掃描進場品質)
#   v5 (rs0.85/stop0.08/tw0.25, gain_cap0.10, max_hold90): +509% DD20% MAR0.62
#   v6 優化後 (gain_cap0.08, max_hold120)               : +418% DD12% MAR0.99 PF2.15
#   關鍵: 突破日漲幅上限收緊到 8% (不追過熱突破) 砍掉最深回撤; 最長持有拉到
#        120 日讓主升段奔跑. 報酬略降但 MaxDD 由 20% 砍半至 12%, 近三年 MAR 3.3.
D_CONFIG = {
    "rs_min":          0.85,  # RS 百分位門檻
    "stop_pct":        0.08,  # 初始停損 (寬停損反而少被洗出)
    "gain_cap":        0.08,  # 突破日漲幅上限 (收緊到 8%, 不追過熱突破)
    "contraction":     0.80,  # ATR5/ATR14 收縮門檻 (越小越嚴)
    "vol_mult":        1.5,   # 突破日量 / 50日均量 門檻
    "three_week_gain": 0.25,  # 三週法則: N日內漲幅 >= 此值即讓利潤奔跑
    "max_hold":        120,   # 最長持有 (拉長讓主升段奔跑)
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
    sig = {
        "score":           feat["rank"],
        "minervini":       True,
        "stop_pct":        cfg["stop_pct"],
        "three_week_gain": cfg["three_week_gain"],
        "max_hold":        cfg["max_hold"],
    }
    if cfg.get("pyramid"):                 # 漸進式曝險: 起始 1/4 倉, 順勢加碼
        sig["pyramid"] = True
    return sig


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
