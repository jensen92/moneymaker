"""台股飆股篩選策略 (A / C / D).

策略 A: Minervini SEPA 完整版 — 五模組 (趨勢模板+VCP+MVP+負面排除), RS≥70.
策略 C: Minervini 第二階段趨勢模板 + 波動收縮突破, RS≥70, 停損 8%.
策略 D: 同 C 但 RS≥85, 停損 8%, 三週法則 (強勢股不賣半).
"""
import numpy as np
import pandas as pd


def add_indicators(df):
    df = df.copy()
    df["ma5"] = df["close"].rolling(5).mean()
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
    # RSI(14) — Wilder 平滑, 供策略 E (回檔買進) 判斷超賣反彈
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
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
    # ── 書中(Minervini)出場心法: 總是 +20% 賣半鎖利, 不設時間出場 ──
    "use_three_week":  False,  # 關閉三週法則 → 達 +20% 一律賣半 (書中「賣半雙贏」)
    "three_week_gain": 0.20,  # (僅 use_three_week=True 時生效)
    "max_hold":        9999,  # 取消硬性持有上限, 改由 MA50 移動停損/衝頂/停損決定出場
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
#   v7 漏斗診斷 (解決長期現金滿手無部位問題): 全市場逐關卡漏斗分析發現,
#        在已通過趨勢模板+RS 0.85 的候選中, contraction<0.80 (ATR5/ATR14
#        收縮門檻) 砍掉 86.6%, 是除「突破」本身外最嚴苛的關卡——比 vol_dryup
#        (砍18.7%)、vol_surge (砍35.2%) 都嚴格得多, 是訊號太少、資金常閒置的
#        主因。逐檔測試 0.80→0.90→1.00→1.20→關閉, 全部走真實引擎(非事件研究):
#        0.80: 511筆 MAR0.67 / 0.90: 738筆 MAR0.79 / 1.00: 806筆 MAR0.99(最佳,
#        MaxDD反降至8.4%) / 1.20: 907筆 MAR0.38(品質開始崩潰) / 關閉: 905筆
#        MAR0.54。取 1.00 為甜蜜點: 交易數+58%、CAGR 7.6%→8.3%、MaxDD
#        11.2%→8.4%、MAR 0.67→0.99, 同時解決閒置資金與報酬問題。
D_CONFIG = {
    "rs_min":          0.85,  # RS 百分位門檻
    "stop_pct":        0.08,  # 初始停損 (寬停損反而少被洗出)
    "gain_cap":        0.08,  # 突破日漲幅上限 (收緊到 8%, 不追過熱突破)
    "contraction":     1.00,  # ATR5/ATR14 收縮門檻 (v7: 0.80→1.00, 解決訊號稀少)
    "vol_mult":        1.5,   # 突破日量 / 50日均量 門檻
    # ── 書中(Minervini)出場心法: 總是 +20% 賣半鎖利, 不設時間出場 ──
    "use_three_week":  False,  # 關閉三週法則 → 達 +20% 一律賣半 (書中「賣半雙贏」)
    "three_week_gain": 0.25,  # (僅 use_three_week=True 時生效)
    "max_hold":        9999,  # 取消硬性持有上限, 改由 MA50 移動停損/衝頂/停損決定出場
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
        # ── 策略 B 增強用特徵 (實證: 52週高點接近度最佳因子 / 量增價穩 / 季節性) ──
        "prox_52wh":    row["close"] / row["high252"] if row["high252"] > 0 else 0.0,
        "day_range":    (row["high"] - row["low"]) / row["close"] if row["close"] > 0 else 9.9,
        "month":        df["date"].iloc[i].month,
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
    # ── 策略 B 選用增強門檻 (D 不設這些 key, 完全向後相容) ──
    if cfg.get("prox_min") is not None and feat.get("prox_52wh", 0.0) < cfg["prox_min"]:
        return None
    if cfg.get("skip_months") and feat.get("month") in cfg["skip_months"]:
        return None
    if cfg.get("max_day_range") is not None and feat.get("day_range", 9.9) > cfg["max_day_range"]:
        return None
    sig = {
        "score":           feat["rank"],
        "minervini":       True,
        "stop_pct":        cfg["stop_pct"],
        "use_three_week":  cfg.get("use_three_week", False),
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


# ─────────────────────────────────────────────────────────────
# 策略 PA / PB / PC — 高 PF 選股 (D 引擎 + 進場品質加嚴, 為 2x 槓桿設計)
# ─────────────────────────────────────────────────────────────
# optimize_pf.py 108 組掃描的產物. 核心發現 (兩個旋鈕讓 PF 1.81→2.5~3.5):
#   1. gain_cap=0.06: 突破日漲幅 <= 6% 才買 (緊貼樞軸點, 絕不追高) —
#      Minervini「儘可能接近中樞點買入」的量化.
#   2. use_three_week=True: 讓贏家奔跑 (不太早賣半), 放大右尾.
# 三組差別在「進場鬆緊 ↔ 交易數/報酬」取捨; 皆為 1x 基準, 上 2x 槓桿前先確認回撤.
#   PA 最穩: 132筆 勝率57% PF3.53 年化4.2% 回撤5.5% MAR0.77 (槓桿最安全, 但資金常閒置)
#   PB 平衡: 208筆 勝率51% PF2.46 年化5.0% 回撤6.5% MAR0.76 (報酬與安全兼顧)
#   PC 積極: 328筆 勝率48% PF2.00 年化5.7% 回撤12.9% MAR0.44 (交易最多, 回撤較深)
_PF_BASE = {"vol_mult": 1.5, "three_week_gain": 0.20,
            "use_three_week": True, "max_hold": 9999}
PA_CONFIG = {**_PF_BASE, "rs_min": 0.85, "contraction": 0.70,
             "gain_cap": 0.06, "stop_pct": 0.10}
PB_CONFIG = {**_PF_BASE, "rs_min": 0.85, "contraction": 0.75,
             "gain_cap": 0.06, "stop_pct": 0.10}
PC_CONFIG = {**_PF_BASE, "rs_min": 0.85, "contraction": 0.75,
             "gain_cap": 0.08, "stop_pct": 0.10}


def signal_pa(df, i, rs_rank=None):
    """高 PF 選股 A — 最嚴進場 (PF 3.53 / 勝率 57%), 2x 槓桿最安全."""
    return _d_signal(_d_features(df, i, rs_rank), PA_CONFIG)


def signal_pb(df, i, rs_rank=None):
    """高 PF 選股 B — 平衡 (PF 2.46 / 勝率 51% / MAR 0.76), 報酬與安全兼顧."""
    return _d_signal(_d_features(df, i, rs_rank), PB_CONFIG)


def signal_pc(df, i, rs_rank=None):
    """高 PF 選股 C — 積極 (PF 2.00 / 交易 328 筆), 交易最多但回撤較深."""
    return _d_signal(_d_features(df, i, rs_rank), PC_CONFIG)



# ─────────────────────────────────────────────────────────────
# 策略 E — 強勢股回檔買進 (low-correlation 互補, 與 C/D 報酬來源不同)
# ─────────────────────────────────────────────────────────────
# 設計理念: C/D 在「突破創新高」進場 (買高); E 在「多頭股回檔到支撐反彈」進場
# (買低). 觸發日與 C/D 幾乎不重疊 → 與 C/D 報酬流低相關, 提供分散效益.
#   進場: 長多排列 + 高 RS + 近期自高點回檔 8~25% + 回測 MA20/MA50 支撐
#         + RSI(14) 曾超賣 (<40) 後出現反轉 K (收紅且收復前日高).
#   出場: 較短線 (mean-reversion 性質) — 停損 / 2.5R 停利 / ATR 移動停損 / 最長持有.
#         非 minervini 模式 (不套 MA50 全出與 +20% 賣半).
# optimize_e.py 掃描最佳 (正期望值 + 與 C/D 低相關). 進場品質是關鍵:
#   買「淺回檔到上升 MA20 的領導股」(健康回檔), 而非「深跌 RSI 崩到 MA50」(接刀);
#   並要求回檔時量縮 (賣壓乾涸) + 反轉日確認.
# optimize_e.py 128 組掃描最佳 (正期望值 + 與 C/D 低相關):
#   勝率 41% / PF 1.14 / CAGR +1.5% / DD 16.8% / MAR 0.09 / ρC 0.03 / ρD 0.14.
#   誠實註記: 回檔買進在台股動能宇宙僅有微弱正優勢 (遠輸 C/D 的突破動能);
#   價值在「報酬流與 C/D 近乎零相關」, 適合作小額衛星部位分散尾端風險,
#   不宜與 C/D 等權 (會稀釋報酬). 詳見 corr_analysis.py 的組合對比.
E_CONFIG = {
    "rs_min":         0.88,   # RS 門檻 (僅買最強領導股回檔)
    "pullback_min":   0.05,   # 自近 40 日高至少回檔 5%
    "pullback_max":   0.15,   # 回檔不超過 15% (淺回檔才健康, 深跌不接)
    "near_ma_pct":    0.03,   # 近 5 日最低觸及 MA20 ±3% (緊貼上升均線)
    "rsi_os":         50.0,   # RSI(14) 曾 < 50 (短線降溫即可, 不必崩跌)
    "require_dryup":  True,    # 回檔需量縮 (近 3 日均量 < 50 日均量)
    "require_ma20_up": True,   # MA20 須上升 (確認回檔發生在上升結構)
    "stop_pct":       0.07,   # 停損 7%
    "target_r":       2.0,     # 停利 2.0R (mean-reversion 較快了結)
    "trail_atr":      0.0,     # 0 = 不用 ATR 移動停損 (純停利/停損/時間)
    "max_hold":       25,      # 最長持有 25 日
}


def _e_features(df, i, rs_rank):
    """抽取策略 E 原始特徵 (長多排列 + 回檔/支撐/超賣/量縮), 不符回傳 None."""
    row = df.iloc[i]
    if np.isnan(row["ma200"]) or np.isnan(row["rsi14"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000:
        return None
    if i < 200:
        return None
    # 長多排列 (與 C/D 同樣只玩多頭股, 確保品質)
    uptrend = (
        row["ma50"] > row["ma150"] > row["ma200"]
        and row["ma200"] > df["ma200"].iloc[i - 21]
        and row["close"] > row["ma200"]
    )
    if not uptrend:
        return None
    # 自近 40 日高點的回檔幅度
    hi40 = df["high"].iloc[max(0, i - 39):i + 1].max()
    pullback = (hi40 - row["close"]) / hi40 if hi40 > 0 else 0.0
    # 近 5 日最低與 MA20 / MA50 的距離 (回測支撐程度)
    lo5 = df["low"].iloc[max(0, i - 4):i + 1].min()
    near_ma20 = abs(lo5 - row["ma20"]) / row["ma20"] if row["ma20"] > 0 else 9.9
    near_ma50 = abs(lo5 - row["ma50"]) / row["ma50"] if row["ma50"] > 0 else 9.9
    # RSI 近 5 日最低 (短線是否降溫)
    rsi_min5 = df["rsi14"].iloc[max(0, i - 4):i + 1].min()
    # 反轉 K: 今日收紅且收盤收復前一日高點 (買在反彈確認)
    prev = df.iloc[i - 1]
    reversal = row["close"] > row["open"] and row["close"] > prev["high"]
    # 回檔量縮: 近 3 日均量 < 50 日均量 (賣壓乾涸)
    vol3 = df["volume"].iloc[max(0, i - 2):i + 1].mean()
    dryup = vol3 < row["vol_ma50"] if row["vol_ma50"] > 0 else False
    # MA20 上升 (回檔發生在上升結構中)
    ma20_up = i >= 5 and row["ma20"] > df["ma20"].iloc[i - 5]
    return {
        "rank":      rs_rank,
        "pullback":  pullback,
        "near_ma20": near_ma20,
        "near_ma50": near_ma50,
        "rsi_min5":  rsi_min5,
        "reversal":  reversal,
        "dryup":     dryup,
        "ma20_up":   ma20_up,
    }


def _e_signal(feat, cfg):
    """套用 cfg 門檻判定策略 E 訊號, 回傳訊號字典或 None."""
    if feat is None:
        return None
    if feat["rank"] is None or feat["rank"] < cfg["rs_min"]:
        return None
    if not (cfg["pullback_min"] <= feat["pullback"] <= cfg["pullback_max"]):
        return None
    # 回測支撐: 取 MA20 距離 (預設) 判斷
    near = feat["near_ma20"] if cfg.get("support_ma", "ma20") == "ma20" else \
        min(feat["near_ma20"], feat["near_ma50"])
    if near > cfg["near_ma_pct"]:
        return None
    if feat["rsi_min5"] >= cfg["rsi_os"]:
        return None
    if not feat["reversal"]:
        return None
    if cfg.get("require_dryup", True) and not feat["dryup"]:
        return None
    if cfg.get("require_ma20_up", True) and not feat["ma20_up"]:
        return None
    sig = {
        "score":    feat["rank"],
        "stop_pct": cfg["stop_pct"],
        "target_r": cfg["target_r"],
        "max_hold": cfg["max_hold"],
    }
    if cfg.get("trail_atr", 0):
        sig["trail_atr"] = cfg["trail_atr"]
    return sig


def signal_e(df, i, rs_rank=None):
    """強勢股回檔買進 (策略 E) — 與 C/D 低相關的互補策略.

    在多頭領導股 (高 RS + 長多排列) 淺回檔到上升 MA20、量縮後出現反轉 K
    時進場 (買低), 與 C/D 的突破進場 (買高) 報酬來源互補.
    出場較短線: 7% 停損 / 2.0R 停利 / 最長 25 日.
    """
    return _e_signal(_e_features(df, i, rs_rank), E_CONFIG)



# ─────────────────────────────────────────────────────────────
# 策略 F — 海龜交易法則 (Turtle Trading System 2)
# ─────────────────────────────────────────────────────────────
# 設計理念: Richard Dennis / William Eckhardt 海龜系統 2.
#   進場: 收盤突破 55 日通道高點 (前 55 日最高價); 無 VCP / RS 篩選 (流動性即可).
#   停損: 入場價 × (1 - 2×ATR14/close), 上限 20%.
#   出場: 讓利潤奔跑 (gain_cap=9.99); 最長持有 60 日 (引擎保底).
#   特色: 與 C/D (VCP 突破 + RS 篩選) 進場邏輯截然不同 → 低相關性.
#   簡化: 不做加碼 (引擎為單倉模式); 不用 10/20 日通道低出場 (改用 ATR 停損).
F_CONFIG = {
    "channel_days": 55,    # 55 日通道突破 (System 2)
    "atr_stop_mult": 2.0,  # 停損 = 2×ATR14
    "stop_max":      0.20, # 停損上限 20%
    "gain_cap":      9.99, # 讓利潤奔跑 (無上限)
    "max_hold":      60,   # 最長持有 60 日 (保底出場)
    "min_turnover":  50_000_000,  # 最低成交額門檻
}


def signal_f(df, i, rs_rank=None):
    """海龜交易系統 2 (策略 F) — 55 日通道突破.

    不需要 RS 排名; 任何有流動性的股票均可. 進場條件:
      1. 最低成交額 > 5000 萬
      2. 今日收盤 > 前 55 日最高價 (不含今日)
      3. ATR14 有效 (至少 14 根 K 線)
    停損 = 2×ATR14 / close (上限 20%).
    """
    cfg = F_CONFIG
    if i < cfg["channel_days"] + 1:
        return None
    row = df.iloc[i]
    if row["close"] < 10:
        return None
    if row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    atr14 = row.get("atr14", float("nan"))
    if np.isnan(atr14) or atr14 <= 0:
        return None
    # 55 日通道高點 (不含今日)
    channel_high = df["high"].iloc[i - cfg["channel_days"]: i].max()
    if row["close"] <= channel_high:
        return None
    stop_pct = min(cfg["atr_stop_mult"] * atr14 / row["close"], cfg["stop_max"])
    return {
        "score":    0.5,          # 無 RS 排名, 給固定中間分
        "stop_pct": stop_pct,
        "gain_cap": cfg["gain_cap"],
        "max_hold": cfg["max_hold"],
    }



# ─────────────────────────────────────────────────────────────
# 策略 G — Darvas Box (Nicolas Darvas《我如何在股市賺了200萬》)
# ─────────────────────────────────────────────────────────────
# 設計理念: Darvas 在 1950s 靠此方法把 25,000 美元變 200 萬.
#   核心: 找「盒型整理」→ 盒頂突破 + 爆量 → 進場.
#   盒型定義:
#     1. 先找近期高點 (box_top): 前 N 日中, 有連續 box_confirm 根 K 未超過此高點
#     2. 盒底 (box_bottom): 整理期間最低點
#     3. 今日收盤突破 box_top → 進場
#     4. 量能: 今日成交量 > vol_ma50 × vol_mult 倍 (爆量確認)
#     5. RS 篩選: 只做高 RS 股 (rs_min), 避免弱股假突破
#   停損: 跌回 box_bottom 以下即出場 (盒底停損)
#   出場: 讓利潤奔跑; 最長 90 日
#   特色: 進場邏輯是「盒型整理後突破」而非 VCP (C/D 是波動收縮型態);
#         量能爆量 + 盒底停損 與 C/D 的 ATR 停損截然不同 → 低相關.
G_CONFIG = {
    "box_window":   20,    # 找盒型的回顧窗口
    "box_confirm":   3,    # 盒頂右側需連續幾根未超過
    "vol_mult":     1.5,   # 突破量 > vol_ma50 × 1.5
    "rs_min":       0.70,  # RS 排名下限 (70 百分位)
    "stop_pct":     0.08,  # 盒底停損 (若盒底太遠則用 8%)
    "gain_cap":     9.99,  # 讓利潤奔跑
    "max_hold":     90,    # 最長持有 90 日
    "min_price":    10.0,  # 最低股價門檻
}


def signal_g(df, i, rs_rank=None):
    """Darvas Box 突破 (策略 G).

    進場條件:
      1. RS 排名 >= rs_min
      2. 識別盒型整理 (box_window 天內找盒頂, 右側 box_confirm 根未突破)
      3. 今日收盤突破盒頂
      4. 今日成交量 > vol_ma50 × vol_mult
    停損: max(盒底以下 1%, 8% 固定停損)
    """
    cfg = G_CONFIG
    if i < cfg["box_window"] + cfg["box_confirm"] + 10:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    row = df.iloc[i]
    if row["close"] < cfg["min_price"]:
        return None
    vol_ma50 = row.get("vol_ma50", float("nan"))
    if np.isnan(vol_ma50) or vol_ma50 <= 0:
        return None
    if row["volume"] < vol_ma50 * cfg["vol_mult"]:
        return None

    # 識別盒型: 在 [i-box_window-box_confirm, i-box_confirm) 找最高點作為 box_top
    # 然後確認 [i-box_confirm, i) 這幾根都未超過 box_top
    search_start = i - cfg["box_window"] - cfg["box_confirm"]
    search_end   = i - cfg["box_confirm"]
    if search_start < 0:
        return None
    box_top = df["high"].iloc[search_start:search_end].max()
    # 右側確認: box_confirm 根都在 box_top 以下
    confirm_highs = df["high"].iloc[search_end:i]
    if (confirm_highs > box_top).any():
        return None
    # 今日突破盒頂
    if row["close"] <= box_top:
        return None
    # 盒底: 整理區間最低點
    box_bottom = df["low"].iloc[search_start:i].min()
    box_depth = (box_top - box_bottom) / box_top
    # 盒型太淺或太深視為無效
    if box_depth < 0.05 or box_depth > 0.35:
        return None
    # 停損取「盒底下 1%」或「固定 stop_pct」之較小者 (保護更緊)
    stop_from_box = (row["close"] - box_bottom * 0.99) / row["close"]
    stop_pct = min(stop_from_box, cfg["stop_pct"])
    stop_pct = max(stop_pct, 0.03)  # 最小 3%

    return {
        "score":    rs_rank,
        "stop_pct": stop_pct,
        "gain_cap": cfg["gain_cap"],
        "max_hold": cfg["max_hold"],
    }


# ─────────────────────────────────────────────────────────────
# 策略 H — 上升三角形突破 (Edwards & Magee《股價趨勢技術分析》)
# ─────────────────────────────────────────────────────────────
# 設計理念: R.D. Edwards & John Magee, "Technical Analysis of Stock Trends".
#   上升三角形 (Ascending Triangle) 是該書最可靠的多頭續勢/累積型態:
#     - 上方「水平壓力線」: 多次觸及同一高點區 (供給帶), 反覆被壓回
#     - 下方「上升支撐線」: 一波比一波高的低點 (需求逐步增強)
#     - 兩線收斂, 最終價格「向上突破水平壓力 + 放量」→ 進場
#   與 C/D (VCP 對稱波動收縮) 的差異:
#     * VCP 兩側都收斂; 上升三角是「上平下升」單向收斂 (頂部水平)
#     * 偵測用「壓力線觸及次數 + 低點線性回歸正斜率」, 而非波段收縮比
#   停損: 跌破上升支撐線 (回歸線在當日之值) 下方, 或固定 stop_pct 取較緊者.
#   出場: 量度目標常為三角高度, 但本引擎讓利潤奔跑 (gain_cap 大), 最長 90 日.
H_CONFIG = {
    "window":        40,    # 三角型態回顧窗口
    "min_touches":    3,    # 水平壓力線最少觸及次數
    "touch_tol":     0.03,  # 觸及壓力線的容差 (3% 內視為同一帶)
    "min_slope":     0.0,   # 低點回歸線最小斜率 (>0 = 上升)
    "max_height":    0.30,  # 三角高度上限 (壓力-起始低點)/壓力
    "min_height":    0.05,  # 三角高度下限
    "vol_mult":      1.3,   # 突破量 > vol_ma50 × 1.3
    "rs_min":        0.70,  # RS 排名下限
    "stop_pct":      0.08,  # 固定停損上限
    "gain_cap":      9.99,  # 讓利潤奔跑
    "max_hold":      90,    # 最長持有 90 日
    "min_price":     10.0,
}


def _ascending_triangle(df, i, cfg):
    """偵測 [i-window, i) 是否形成上升三角, 並於今日 i 向上突破水平壓力.

    回傳 (resistance, support_today) 或 None.
    resistance = 水平壓力位; support_today = 上升支撐線在今日的值 (停損參考).
    """
    w = cfg["window"]
    if i < w + 5:
        return None
    seg = df.iloc[i - w:i]
    highs = seg["high"].values
    lows = seg["low"].values
    n = len(highs)

    # 1. 水平壓力線: 取區間最高價為壓力帶上緣
    resistance = highs.max()
    # 觸及次數: 高點落在壓力帶 (resistance × (1-tol)) 以上
    touch_band = resistance * (1 - cfg["touch_tol"])
    touches = int((highs >= touch_band).sum())
    if touches < cfg["min_touches"]:
        return None

    # 2. 上升支撐線: 對「低點序列」做線性回歸, 斜率須 > min_slope
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, lows, 1)
    if slope <= cfg["min_slope"]:
        return None
    # 低點需大致貼合回歸線 (殘差小), 確認確為趨勢線而非雜訊
    fitted = slope * x + intercept
    rmse = float(np.sqrt(np.mean((lows - fitted) ** 2)))
    if rmse / resistance > 0.05:   # 殘差 > 5% 視為不成形
        return None

    # 3. 三角高度檢查 (壓力 - 起始支撐) / 壓力
    support_start = slope * 0 + intercept
    height = (resistance - support_start) / resistance
    if height < cfg["min_height"] or height > cfg["max_height"]:
        return None

    # 支撐線在「今日 (x=n)」的延伸值, 作為停損參考
    support_today = slope * n + intercept
    return resistance, support_today


def signal_h(df, i, rs_rank=None):
    """上升三角形突破 (策略 H) — Edwards & Magee.

    進場條件:
      1. RS 排名 >= rs_min
      2. [i-window, i) 形成上升三角 (水平壓力 + 上升支撐, 收斂)
      3. 今日收盤向上突破水平壓力線
      4. 突破量 > vol_ma50 × vol_mult
    停損: 跌破上升支撐線延伸值, 或固定 stop_pct, 取較緊者 (下限 3%).
    """
    cfg = H_CONFIG
    if i < cfg["window"] + 5:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    row = df.iloc[i]
    if row["close"] < cfg["min_price"]:
        return None
    vol_ma50 = row.get("vol_ma50", float("nan"))
    if np.isnan(vol_ma50) or vol_ma50 <= 0:
        return None
    if row["volume"] < vol_ma50 * cfg["vol_mult"]:
        return None

    tri = _ascending_triangle(df, i, cfg)
    if tri is None:
        return None
    resistance, support_today = tri

    # 今日收盤突破水平壓力
    if row["close"] <= resistance:
        return None
    # 突破不可過度延伸 (距壓力 > 8% 視為追高)
    if (row["close"] - resistance) / resistance > 0.08:
        return None

    # 停損: 支撐線延伸值 vs 固定停損, 取較緊
    stop_from_support = (row["close"] - support_today) / row["close"]
    stop_pct = min(stop_from_support, cfg["stop_pct"])
    stop_pct = max(stop_pct, 0.03)

    return {
        "score":    rs_rank,
        "stop_pct": stop_pct,
        "gain_cap": cfg["gain_cap"],
        "max_hold": cfg["max_hold"],
    }


# ─────────────────────────────────────────────────────────────
# 策略 I — Cup with Handle (William O'Neil《笑傲股市 CANSLIM》)
# ─────────────────────────────────────────────────────────────
# O'Neil 研究 1950-2008 年百大飆股後發現杯柄型態是最常見的成功突破前型態.
#   杯型 (Cup): U 形整理, 深度 12~33%, 底部圓滑 (非 V 型), 7~20 週
#   柄部 (Handle): 杯口右側小幅回檔 5~15%, 量縮 (賣壓乾涸), 至少 5 根 K
#   突破 (Breakout): 收盤突破柄部高點 + 量 ≥ 1.5 倍 50 日均量
#   停損: 柄部最低點以下 7~8% (O'Neil 書中明確規定)
#   RS 門檻: O'Neil 建議 RS 排名 ≥ 80 (前 20% 強勢股)
#   高勝率設計: 杯底圓弧 + 柄部縮量嚴格過濾 → 歷史勝率約 60~70%
#   與 C/D 差異: C/D 是 VCP 波動收縮型態 (ATR 收縮比); CwH 是完整杯柄幾何結構
I_CONFIG = {
    "cup_min_weeks":  7,     # 杯型最短 7 週 (35 個交易日)
    "cup_max_weeks": 20,     # 杯型最長 20 週 (100 個交易日)
    "cup_min_depth": 0.12,   # 杯型最淺 12%
    "cup_max_depth": 0.35,   # 杯型最深 35%
    "handle_min":    5,      # 柄部最短 5 根 K
    "handle_max":   15,      # 柄部最長 15 根 K (太長則型態失效)
    "handle_depth":  0.15,   # 柄部回檔上限 15%
    "handle_pos":    0.50,   # 柄部必須在杯高 50% 以上 (O'Neil: 上半部)
    "vol_mult":      1.5,    # 突破量 ≥ vol_ma50 × 1.5
    "rs_min":        0.80,   # RS 排名下限 (O'Neil 建議 80+)
    "stop_pct":      0.08,   # 初始停損 8% (柄部最低點以下)
    "gain_cap":      9.99,   # 讓利潤奔跑
    "max_hold":      9999,   # MA50 移動停損決定出場 (minervini 模式)
    "min_price":     10.0,
}


def _cup_with_handle(df, i, cfg):
    """偵測 [i-handle_len, i) 為柄部、柄部之前為杯型.

    回傳 (cup_left_high, handle_low) 或 None.
    cup_left_high: 杯型左緣高點 (= 突破點)
    handle_low: 柄部最低點 (停損參考)
    """
    # 嘗試所有合法柄部長度
    for h_len in range(cfg["handle_min"], cfg["handle_max"] + 1):
        cup_end = i - h_len      # 杯型右緣 (= 柄部開始前一根)
        # 嘗試所有合法杯型長度
        cup_min_bars = cfg["cup_min_weeks"] * 5
        cup_max_bars = cfg["cup_max_weeks"] * 5
        for c_len in range(cup_min_bars, cup_max_bars + 1, 5):
            cup_start = cup_end - c_len
            if cup_start < 10:
                continue
            cup_seg = df.iloc[cup_start:cup_end + 1]
            # 杯型左緣高點 (第一根附近)
            left_high = cup_seg["high"].iloc[:5].max()
            # 杯型右緣高點 (最後 5 根)
            right_high = cup_seg["high"].iloc[-5:].max()
            # 右緣需接近左緣 (不能差距 > 5%, 代表回到原高點附近)
            if right_high < left_high * 0.95:
                continue
            # 杯底
            cup_low = cup_seg["low"].min()
            # 杯型深度: 相對左緣高點
            depth = (left_high - cup_low) / left_high
            if depth < cfg["cup_min_depth"] or depth > cfg["cup_max_depth"]:
                continue
            # 底部圓滑性: 最低點不在前 20% 或後 20% (避免 V 型)
            low_idx = cup_seg["low"].values.argmin()
            rel_pos = low_idx / len(cup_seg)
            if rel_pos < 0.20 or rel_pos > 0.80:
                continue

            # 柄部驗證
            handle_seg = df.iloc[cup_end:i]
            handle_high = handle_seg["high"].max()
            handle_low  = handle_seg["low"].min()
            # 柄部高點需在杯高 handle_pos 以上
            cup_range = left_high - cup_low
            if handle_low < cup_low + cup_range * cfg["handle_pos"]:
                continue
            # 柄部回檔深度
            handle_depth = (handle_high - handle_low) / handle_high
            if handle_depth > cfg["handle_depth"]:
                continue
            # 柄部量縮: 平均量 < 杯型均量
            cup_vol_avg = cup_seg["volume"].mean()
            handle_vol_avg = handle_seg["volume"].mean()
            if handle_vol_avg >= cup_vol_avg:
                continue
            # 突破點 = 柄部高點
            return handle_high, handle_low

    return None


def signal_i(df, i, rs_rank=None):
    """Cup with Handle 突破 (策略 I) — O'Neil CANSLIM.

    進場條件:
      1. RS 排名 >= 0.80
      2. 識別完整杯柄型態
      3. 今日收盤突破柄部高點 (= 突破點)
      4. 突破量 ≥ vol_ma50 × 1.5
    停損: 柄部最低點以下 3%, 或固定 8%, 取較緊者.
    出場: minervini 模式 (+20% 賣半 / 3R 保本 / MA50 移動停損).
    """
    cfg = I_CONFIG
    min_i = cfg["cup_max_weeks"] * 5 + cfg["handle_max"] + 20
    if i < min_i:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    row = df.iloc[i]
    if row["close"] < cfg["min_price"]:
        return None
    vol_ma50 = row.get("vol_ma50", float("nan"))
    if np.isnan(vol_ma50) or vol_ma50 <= 0:
        return None
    if row["volume"] < vol_ma50 * cfg["vol_mult"]:
        return None

    result = _cup_with_handle(df, i, cfg)
    if result is None:
        return None
    handle_high, handle_low = result

    # 今日收盤突破柄部高點
    if row["close"] <= handle_high:
        return None
    # 不追高: 突破幅度 ≤ 8%
    if (row["close"] - handle_high) / handle_high > 0.08:
        return None

    # 停損: 柄部最低點以下 3%
    stop_from_handle = (row["close"] - handle_low * 0.97) / row["close"]
    stop_pct = min(stop_from_handle, cfg["stop_pct"])
    stop_pct = max(stop_pct, 0.04)

    return {
        "score":    rs_rank,
        "minervini": True,
        "stop_pct": stop_pct,
        "gain_cap": cfg["gain_cap"],
        "max_hold": cfg["max_hold"],
    }


# ─────────────────────────────────────────────────────────────
# 策略 J — 動能點火 (Volume-Thrust Ignition) · 專抓飆股, 與 C/D 互補
# ─────────────────────────────────────────────────────────────
# 設計動機 (diag_rocket_block.py 診斷結論): C/D 的 VCP 哲學「先量縮、再突破」
# 本質上排斥真飆股 — 飆股起漲當天就是「爆量 + 大漲 + 創新高」(量能/波動「擴張」,
# 正是 contraction<1.0 與 vol_dryup 兩個濾網會擋掉的特徵)。近三年 C/D 對 1766
# 次飆股事件 (60 日內漲 >=80%) 只抓到 2.4%。J 不動 C/D, 改用相反的訊號原型:
#
#   核心 = 量能擴張 (vol_surge) + 動能點火 (大漲) + 脫離整理創新高,
#          只在最強勢股 (RS>=0.90) 且多頭結構中, 且「起漲初期」(未過度延伸) 才進。
#
# 「精準、不抓一堆非飆股」靠四道同時成立的硬條件 (任一不過即略過):
#   1. RS >= 0.90      — 只在全市場最強的 10% (飆股必為強勢股)
#   2. 爆量 >= 3×均量  — 真正的資金點火, 過濾無量假突破 (這是 VCP 反向特徵)
#   3. 動能點火        — 單日漲 >= min_gain 或 thrust_window 日內急漲 >= thrust_gain
#   4. 創 base 日新高 + 未過度延伸 (close <= ma50×max_ext) — 抓「起漲」非「追高」
# 出場: 寬停損 (飆股波動大, 太緊會被洗) + ATR 移動停損讓主升段奔跑 (不賣半,
#       不套 MA50 全出, 不衝頂出場 — 目標就是吃下整段噴出)。
#
# 驗證 (eval_j.py / tune_j.py, 真實引擎, 近三年 1766 次飆股事件):
#   飆股捕捉率 J 8.0% vs C/D 2.4% (3.3 倍); 進場精準度 J 44% > C 38% > D 29%
#   (J 的進場「更」集中在飆股波段, 不是亂槍打鳥)。
#   獨立績效 (全期): PF 1.21 / 勝率 35% / CAGR 5.5% / MaxDD 19.5% / MAR 0.28 —
#   遠低於 C/D (MAR 0.59/0.99)。這是飆股交易的「本質取捨」: 爆發型突破命中率
#   天生低 (多數點火失敗), 報酬完全靠少數大贏家的右尾。
#   tune_j.py 掃 7 組 (rs 0.90→0.92 / vol 3→4 / stop / trail / ext / 強點火) 全
#   不如 baseline — 收緊進場反而同時砍掉捕捉率與精準度。故採此 baseline。
#   定位: 不宜當大額主力 (會稀釋 C/D 的 MAR), 適合小額衛星部位專吃 C/D 結構上
#   抓不到的飆股, 與 C/D 報酬來源互補。
J_CONFIG = {
    "rs_min":         0.90,        # 只做最強勢股
    "min_turnover":   30_000_000,  # 流動性門檻 (略低於 C/D 的 5000 萬, 以稍早卡位)
    "min_price":      10.0,
    "vol_surge":      3.0,         # 當日量 >= 3× 50 日均量 (量能擴張 = VCP 反向)
    "min_gain":       0.06,        # 單日點火漲幅 >= 6%
    "thrust_window":  3,           # 或 N 日內
    "thrust_gain":    0.15,        # 急漲 >= 15% (多日點火)
    "base_lookback":  50,          # 突破前 50 日整理區高點 (脫離整理創新高)
    "max_ext_ma50":   1.35,        # close <= ma50×1.35 (起漲初期, 不追過度延伸)
    "stop_pct":       0.12,        # 寬停損 (飆股波動大)
    "trail_atr":      3.0,         # 3×ATR 移動停損 (讓主升段奔跑)
    "gain_cap":       9.99,        # 不限漲幅 (讓利潤奔跑)
    "max_hold":       120,         # 最長持有 120 日
}


def signal_j(df, i, rs_rank=None):
    """動能點火 (策略 J) — 爆量大漲突破, 專抓飆股起漲, 與 C/D 互補.

    與 C/D 的 VCP「量縮收縮後突破」相反: J 要的是「量能擴張 + 動能點火」,
    捕捉 C/D 結構上抓不到的爆發型飆股。靠 RS>=0.90 + 3× 爆量 + 創新高 + 未延伸
    四道硬條件維持精準 (訊號稀少, 不濫抓非飆股)。出場用寬停損 + ATR 移動停損,
    目標吃下整段噴出 (不賣半/不 MA50 全出)。
    """
    cfg = J_CONFIG
    if i < 200:
        return None
    row = df.iloc[i]
    if row["close"] < cfg["min_price"]:
        return None
    if row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if np.isnan(row["ma200"]) or np.isnan(row["vol_ma50"]) or row["vol_ma50"] <= 0:
        return None
    if np.isnan(row["atr14"]) or row["atr14"] <= 0:
        return None
    # 多頭結構: 過濾下降趨勢中的假爆量 (但不套 C/D 的嚴格 VCP 模板)
    if not (row["ma50"] > row["ma150"] and row["close"] > row["ma150"]
            and row["ma200"] > df["ma200"].iloc[i - 21]):
        return None
    # 爆量: 量能擴張 (與 VCP vol_dryup 相反) — 真正的資金點火
    vol_surge = row["volume"] / row["vol_ma50"]
    if vol_surge < cfg["vol_surge"]:
        return None
    # 動能點火: 單日大漲 或 多日急漲
    day_gain = row["close"] / df["close"].iloc[i - 1] - 1.0
    thrust = row["close"] / df["close"].iloc[i - cfg["thrust_window"]] - 1.0
    if day_gain < cfg["min_gain"] and thrust < cfg["thrust_gain"]:
        return None
    # 突破: 創 base_lookback 日新高 (脫離整理區, 主升段起點)
    prior_high = df["high"].iloc[i - cfg["base_lookback"]:i].max()
    if row["close"] <= prior_high:
        return None
    # 不追過度延伸: 已大漲一段就不追, 確保抓的是「起漲」
    if row["close"] > row["ma50"] * cfg["max_ext_ma50"]:
        return None
    return {
        "score":     rs_rank + min(vol_surge, 20) / 100.0,  # 量越大排序越前
        "stop_pct":  cfg["stop_pct"],
        "trail_atr": cfg["trail_atr"],
        "gain_cap":  cfg["gain_cap"],
        "max_hold":  cfg["max_hold"],
        "vol_surge": vol_surge,                         # 供 tune_j_volrank.py 重排序實驗用
        "turnover":  row["volume"] * row["close"],
    }


# ─────────────────────────────────────────────────────────────
# 策略 B — D 引擎 + 全市場實證增強 (52週高點接近度 / 量增價穩 / 季節避險)
# ─────────────────────────────────────────────────────────────
# 來自 35 因子 + 台股習性大規模驗證 (見 FIB_RETRACE_TEST.md / STRATEGY_AUDIT.md):
#   - 純橫斷面選股因子天花板 ~MAR 0.5, 全部輸 D 的事件風控引擎 (MAR 0.99)。
#   - 其中最佳「選股訊號」是 52週高點接近度 (M/X2, MAR 0.50~0.52)。
#   - 量增價穩 (Z1) 是全場第二低回撤特徵; 9-10月個股報酬實測轉弱 (季節習性E)。
# 故 B = 不另起爐灶, 直接站在 D 的進場+風控引擎上, 疊加上述三個「能提升 MAR 才留」
# 的增強門檻。實際保留哪些, 由 optimize_b.py 消融實測決定 (見該檔結論)。
# optimize_b.py 消融實測結論 (真實引擎, 全期 2011-2026):
#   D 基準                 : 806筆 PF1.46 CAGR8.3% MaxDD8.4% MAR0.99
#   +prox_min 0.85 (採用)  : 788筆 PF1.54 CAGR8.9% MaxDD8.1% MAR1.10  ← 全維度勝 D
#   +prox_min 0.95         : 734筆 PF1.64 CAGR10.2% MaxDD9.3% MAR1.10 (CAGR高但回撤反升)
#   +max_day_range/skip月  : 全部使 MAR 下降 (振幅/季節屬防禦或時序效應, 不宜當進場硬門檻)
# 取 prox_min=0.85: 唯一在 CAGR/MaxDD/PF/勝率「同時」優於 D 的設定 (嚴格佔優),
# 交易數仍 788 (最不易過擬合)。增強來源: 52週高點接近度為 35因子驗證中最佳選股訊號。
B_CONFIG = {**D_CONFIG, "prox_min": 0.85}


def signal_b(df, i, rs_rank=None):
    """策略 B = D 事件引擎 + 52週高點接近度增強 (收盤 >= 52週高 * 0.85).

    在 D 已驗證的趨勢模板+波動收縮突破+全套風控 (停損/集中/市況/波動) 上, 疊加
    全市場實證最佳選股因子 (52週高點接近度) 作進場品質門檻。消融顯示此單一增強即
    使 MAR 0.99→1.10 且回撤不升反降 (8.4%→8.1%), 為嚴格優於 D 的版本。詳見 optimize_b.py."""
    return _d_signal(_d_features(df, i, rs_rank), B_CONFIG)


# ─────────────────────────────────────────────────────────────
# 策略 K / L — B 引擎再優化 (爆量確認 + 緊貼樞軸 + 寬停損, 大幅壓低回撤)
# ─────────────────────────────────────────────────────────────
# 來源: optimize_b2.py / b3_grid 在 B (D引擎+prox0.85) 之上做的二輪消融與網格。
# 關鍵發現 (真實引擎, 全期 2011-2026, 基準 B = 788筆 PF1.54 CAGR8.9% MaxDD8.1% MAR1.10):
#   三個旋鈕「同向」收緊即把 MAR 從 1.10 推到 1.4~1.5, 且 MaxDD 砍半:
#     vol_mult 1.5→2.0 : 突破日量能要更爆 (1.5→2倍均量), 確認真有資金點火, 濾掉假突破。
#     gain_cap 0.08→0.06: 突破日漲幅 <=6% 才買 (緊貼樞軸點, 絕不追高), 提高進場品質。
#     stop_pct 0.08→0.10/0.12: 配合上面更乾淨的進場, 放寬停損反而少被洗、留住主升段。
#   為何 CAGR 略降卻更優: 交易數從 788 降到 ~560-580 (只取最高品質訊號), 年化由 8.9%
#   降到 6.6~7.6%, 但 MaxDD 由 8.1% 砍到 4.6~5.1% → 風險調整後 (MAR) 與 PF 全面超越 B。
#
# K (核心高品質版)  : vol_mult2.0 + gain0.06 + stop0.10 + prox0.80
#                     581筆 PF1.85 CAGR7.6% MaxDD5.1% 勝率54% MAR1.49  ← 全網格最高 MAR
#   (進場已被爆量+緊樞軸+寬停損篩到極乾淨, prox 門檻可放鬆到 0.80 仍維持品質, 換取樣本數)
# L (最低回撤版)    : vol_mult2.0 + gain0.06 + stop0.12 (prox 維持 0.85)
#                     561筆 PF1.86 CAGR6.6% MaxDD4.6% 勝率55% MAR1.44  ← 全網格最低回撤
#   (更寬的 12% 停損 + 維持 B 的 prox0.85, 換取最小回撤與最高勝率, 最穩健的曲線)
# 兩者皆在 PF / MaxDD / 勝率 / MAR 四項同時優於 B, 僅 CAGR 較低 (回撤砍半的必然取捨)。
K_CONFIG = {**B_CONFIG, "vol_mult": 2.0, "gain_cap": 0.06, "stop_pct": 0.10, "prox_min": 0.80}
L_CONFIG = {**B_CONFIG, "vol_mult": 2.0, "gain_cap": 0.06, "stop_pct": 0.12}


def signal_k(df, i, rs_rank=None):
    """策略 K = B 引擎再優化 (爆量2.0x + 緊貼樞軸6% + 寬停損10%, prox0.80).

    在 B 的 D事件引擎+52週高點接近度上, 把三個進場品質旋鈕同向收緊: 突破日量能門檻
    1.5→2.0倍、漲幅上限 8%→6% (緊貼樞軸不追高)、停損放寬到 10%。進場被篩到極乾淨後
    prox 可鬆到 0.80 仍維持品質。真實引擎全期: PF1.85 MaxDD5.1% 勝率54% MAR1.49,
    在 PF/回撤/勝率/MAR 四項全面優於 B (回撤砍近半)。詳見 optimize_b2.py."""
    return _d_signal(_d_features(df, i, rs_rank), K_CONFIG)


def signal_l(df, i, rs_rank=None):
    """策略 L = B 引擎再優化 (爆量2.0x + 緊貼樞軸6% + 最寬停損12%, prox0.85).

    與 K 同源但維持 B 的 prox0.85, 並把停損再放寬到 12% 換取最小回撤。真實引擎全期:
    PF1.86 MaxDD4.6% (全網格最低) 勝率55% MAR1.44, 為最穩健曲線。詳見 optimize_b2.py."""
    return _d_signal(_d_features(df, i, rs_rank), L_CONFIG)


# ─────────────────────────────────────────────────────────────
# 策略 M — 道氏理論主趨勢延續 (Dow Theory primary-trend continuation)
# ─────────────────────────────────────────────────────────────
# Charles Dow 六大原則中可量化者:
#   * 主要趨勢 = 一系列「更高的高點 + 更高的低點」(HH/HL 結構)。
#   * 量能須確認趨勢 (上升段量增)。
#   * 兩種平均須互相印證 — 個股版以「大盤 (TWII) 同處多頭」代理, 由引擎的市況
#     濾網 (risk_on) 統一套用 (與所有策略一致), 故 signal 端不另接指數。
#   * 趨勢延續至明確反轉 (較低高點後跌破前低) → 以「跌破最近的更高低點」為結構性
#     停損, 並用 ATR 移動停損讓主趨勢奔跑 (近似道氏「持有至反轉訊號」)。
# 與 C/D (VCP 波動收縮突破) 不同: M 不要求量縮/收縮, 而是辨識擺動點 HH/HL 結構,
# 於「突破最近擺動高點 (延續主趨勢) 且量能確認」時進場。前視防護: 擺動點 j 須為
# [j-w, j+w] 區間極值, 故只認 j<=i-w 的已確認擺動點 (右側確認窗已閉合)。
M_CONFIG = {
    "lookback":     120,   # 擺動點掃描窗
    "swing_w":      5,     # 擺動點左右確認根數 (j 須為 [j-w,j+w] 極值)
    "vol_confirm":  1.0,   # 突破日量 > vol_ma50 × 此值 (量能確認趨勢)
    "max_ext":      0.06,  # 突破不超過擺動高點 6% (抓延續起點, 不追高)
    "stop_min":     0.04,
    "stop_max":     0.12,
    "trail_atr":    3.5,   # ATR 移動停損 (趨勢延續至反轉)
    "max_hold":     150,
    "min_price":    10.0,
    "min_turnover": 50_000_000,
}


def _dow_swings(df, i, lookback, w):
    """近期「已確認」擺動高/低點 (由舊到新)。只認 j<=i-w 的點 → 不看未來。"""
    s = max(0, i - lookback)
    hi = df["high"].values
    lo = df["low"].values
    highs, lows = [], []
    for j in range(s + w, i - w + 1):
        if hi[j] == hi[j - w:j + w + 1].max():
            highs.append((j, float(hi[j])))
        if lo[j] == lo[j - w:j + w + 1].min():
            lows.append((j, float(lo[j])))
    return highs, lows


def signal_m(df, i, rs_rank=None):
    """道氏理論主趨勢延續 (策略 M) — HH/HL 結構 + 擺動高點突破 + 量能確認.

    進場: 多頭背景 (close>ma200, ma50>ma200, ma200 上升) + 最近兩個擺動高點遞升
    (HH) + 最近兩個擺動低點遞升 (HL) + 今日收盤『當日新破』最近擺動高點 + 突破量
    > 50 日均量。停損: 跌破最近更高低點 (結構受損), 4~12% 內。出場: ATR 移動停損
    讓主趨勢奔跑, 最長 150 日 (非 minervini, 不賣半 — 道氏騎主趨勢)。
    """
    cfg = M_CONFIG
    if i < 200:
        return None
    row = df.iloc[i]
    if row["close"] < cfg["min_price"]:
        return None
    if row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if np.isnan(row["ma200"]) or np.isnan(row["vol_ma50"]) or row["vol_ma50"] <= 0:
        return None
    if np.isnan(row["atr14"]) or row["atr14"] <= 0:
        return None
    if not (row["close"] > row["ma200"] and row["ma50"] > row["ma200"]
            and row["ma200"] > df["ma200"].iloc[i - 21]):
        return None
    highs, lows = _dow_swings(df, i, cfg["lookback"], cfg["swing_w"])
    if len(highs) < 2 or len(lows) < 2:
        return None
    if not (highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]):
        return None
    last_high, last_low = highs[-1][1], lows[-1][1]
    prev_close = df["close"].iloc[i - 1]
    if not (row["close"] > last_high and prev_close <= last_high):
        return None
    if (row["close"] - last_high) / last_high > cfg["max_ext"]:
        return None
    if row["volume"] <= cfg["vol_confirm"] * row["vol_ma50"]:
        return None
    stop_pct = (row["close"] - last_low) / row["close"]
    stop_pct = min(max(stop_pct, cfg["stop_min"]), cfg["stop_max"])
    return {
        "score":     rs_rank if rs_rank is not None else 0.5,
        "stop_pct":  stop_pct,
        "trail_atr": cfg["trail_atr"],
        "gain_cap":  9.99,
        "max_hold":  cfg["max_hold"],
    }


# ─────────────────────────────────────────────────────────────
# 策略 N — 傑西·李佛摩「最小阻力線」(Livermore line of least resistance)
# ─────────────────────────────────────────────────────────────
# 《股票作手回憶錄》核心:
#   * 關鍵樞紐點 (pivotal point): 反覆測試的前高/供給帶。被『決定性』突破 = 阻力
#     被清除, 股價沿「最小阻力線」前進。
#   * 不預測、等確認: 樞紐被突破才動手 (不買勉強掠過的假突破)。
#   * 順勢加碼 (pyramiding): 沿最小阻力線前進時分批加碼, 同步上移停損 (引擎:
#     初始 1/4 倉, +2%/+4% 補上, 滿倉後最大虧損 < 初始單批風險)。
#   * 迅速停損 (~8%) 但對贏家「坐穩不動」(sit tight) — 靠少數大波段獲利。
# 與 G (Darvas 盒) 區別: N 的樞紐須多次觸及同一前高 (供給帶, base 較長 60 日),
#   採加碼 + 寬 ATR 移動停損坐穩主升段; Darvas 是 20 日短盒 + 盒底停損 + 無加碼。
N_CONFIG = {
    "base":         60,    # 整理區 (樞紐點) 回顧窗
    "tol":          0.03,  # 觸及樞紐帶容差
    "min_touches":  3,     # 樞紐點最少觸及次數 (真供給帶)
    "buffer":       0.005, # 決定性突破: close > pivot×(1+buffer)
    "vol_mult":     1.3,   # 突破量 > vol_ma50 × 此值
    "max_ext":      0.08,  # 不追高: 距樞紐 <= 8%
    "rs_min":       0.80,  # 只做領導股
    "stop_pct":     0.08,  # 迅速停損
    "trail_atr":    4.0,   # 寬 ATR 移動停損 (sit tight)
    "max_hold":     150,
    "min_price":    10.0,
    "min_turnover": 50_000_000,
}


def _liv_pivotal(df, i, cfg):
    """關鍵樞紐點 = [i-base, i) 整理區反覆觸及的水平壓力高點。回傳 pivot 或 None.

    只用今日之前的歷史 (不含今日突破), 杜絕前視。
    """
    s = max(0, i - cfg["base"])
    hi = df["high"].values[s:i]
    lo = df["low"].values[s:i]
    if len(hi) < 20:
        return None
    pivot = float(hi.max())
    if pivot <= 0:
        return None
    if int((hi >= pivot * (1 - cfg["tol"])).sum()) < cfg["min_touches"]:
        return None
    base_low = float(lo.min())
    depth = (pivot - base_low) / pivot
    if depth < 0.05 or depth > 0.30:   # 須為真整理 (非崩跌, 非全無波動)
        return None
    return pivot


def signal_n(df, i, rs_rank=None):
    """李佛摩最小阻力線突破 (策略 N) — 樞紐突破 + 順勢加碼 + 坐穩.

    進場: 領導股 (RS>=0.80) + 明確多頭排列 + 今日收盤『決定性、當日新破』關鍵樞紐
    (近 60 日反覆觸及 >=3 次的前高) + 突破量 > 1.3× 均量 + 不追高 (距樞紐<=8%)。
    部位: pyramid 順勢加碼。停損 8% 迅速止損, 寬 ATR (4×) 移動停損坐穩主升段。
    """
    cfg = N_CONFIG
    if i < 200:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    row = df.iloc[i]
    if row["close"] < cfg["min_price"]:
        return None
    if row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if np.isnan(row["ma200"]) or np.isnan(row["vol_ma50"]) or row["vol_ma50"] <= 0:
        return None
    if not (row["close"] > row["ma50"] > row["ma150"] > row["ma200"]
            and row["ma200"] > df["ma200"].iloc[i - 21]):
        return None
    piv = _liv_pivotal(df, i, cfg)
    if piv is None:
        return None
    prev_close = df["close"].iloc[i - 1]
    if not (row["close"] > piv * (1 + cfg["buffer"]) and prev_close <= piv):
        return None
    if (row["close"] - piv) / piv > cfg["max_ext"]:
        return None
    if row["volume"] <= cfg["vol_mult"] * row["vol_ma50"]:
        return None
    return {
        "score":     rs_rank,
        "stop_pct":  cfg["stop_pct"],
        "pyramid":   True,
        "trail_atr": cfg["trail_atr"],
        "gain_cap":  9.99,
        "max_hold":  cfg["max_hold"],
    }


# ─────────────────────────────────────────────────────────────
# 策略 O — 歐尼爾 CAN SLIM (William O'Neil《笑傲股市》, 完整版含基本面)
# ─────────────────────────────────────────────────────────────
# CAN SLIM 七要素, 本策略可量化者:
#   C 當季 EPS 年增 >= 25%  ─┐ 基本面 (由 fundamentals.py 以 point-in-time 提供;
#   A 年度 ROE     >= 17%  ─┘ 於回測/掃描端用 FundamentalDB 疊加, 杜絕前視)
#   N 創新高 / 突破整理 base: 收盤接近 52 週高 + 突破近 25 日 base 高 (買點 pivot)
#   S 量能 (供需): 突破量 >= 1.5× 50 日均量
#   L 領導股: RS 排名 >= 0.80
#   I 法人認養: 價量/財報資料不足, 略過 (誠實註記)
#   M 大盤方向: 由引擎市況濾網統一處理 (與其他策略一致)
# signal_o 只負責技術面 (N/S/L + Stage2); 基本面 C/A 因 signal 函式無 DB/code 存取,
# 改由 philo_backtest 以 FundamentalDB 疊加成「完整 CAN SLIM」, 並另出純技術版對照,
# 量化基本面 C/A 的實際貢獻。與既有策略 I (杯柄幾何) 互補: I 偵測 CwH 型態, O 是
# base 突破 + 領導 + 基本面篩 (CAN SLIM 的選股精神)。
O_CONFIG = {
    "base_high":    25,    # base 突破回顧 (O'Neil pivot buy point)
    "prox_52wh":    0.90,  # 收盤 >= 52 週高 × 此值 (創新高區)
    "vol_mult":     1.5,   # S: 突破量 >= 1.5× 均量
    "rs_min":       0.80,  # L: 領導股 (RS rating 80+)
    "max_ext":      0.05,  # 不追高: 距 pivot <= 5% (O'Neil 5% 買區)
    "stop_pct":     0.08,  # 7-8% 停損 (O'Neil 鐵律)
    "max_hold":     9999,  # minervini 模式: MA50 移動停損 / +20% 賣半
    "min_price":    10.0,
    "min_turnover": 50_000_000,
    # 基本面門檻 (供 philo_backtest 疊加; 不在 signal 內套, 因 signal 無 DB/code)
    "eps_yoy_min":  0.25,
    "roe_min":      0.17,
}


def signal_o(df, i, rs_rank=None):
    """歐尼爾 CAN SLIM 技術面 base (策略 O) — N/S/L + Stage2; 基本面 C/A 另疊.

    進場 (技術): Stage 2 趨勢模板 (領導股在多頭) + 收盤接近 52 週高 (>=90%) +
    今日收盤『當日新破』近 25 日 base 高 (買點 pivot) + 不追高 (距 pivot<=5%) +
    突破量 >= 1.5× 均量。出場: minervini 模式 (MA50 移動停損 / 3R 保本 / +20% 賣半)。
    完整 CAN SLIM 另需 EPS 年增>=25% 且 ROE>=17% (philo_backtest 以 PIT 疊加)。
    """
    cfg = O_CONFIG
    if i < 210:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    row = df.iloc[i]
    if row["close"] < cfg["min_price"]:
        return None
    if row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if np.isnan(row["ma200"]) or np.isnan(row["high252"]) or np.isnan(row["vol_ma50"]):
        return None
    if row["vol_ma50"] <= 0 or row["high252"] <= 0:
        return None
    if not (row["close"] > row["ma50"] > row["ma150"] > row["ma200"]
            and row["ma200"] > df["ma200"].iloc[i - 21]):
        return None
    if row["close"] < row["high252"] * cfg["prox_52wh"]:
        return None
    pivot = df["high"].iloc[i - cfg["base_high"]:i].max()
    prev_close = df["close"].iloc[i - 1]
    if not (row["close"] > pivot and prev_close <= pivot):
        return None
    if (row["close"] - pivot) / pivot > cfg["max_ext"]:
        return None
    if row["volume"] < cfg["vol_mult"] * row["vol_ma50"]:
        return None
    return {
        "score":           rs_rank,
        "minervini":       True,
        "stop_pct":        cfg["stop_pct"],
        "use_three_week":  False,
        "three_week_gain": 0.20,
        "gain_cap":        9.99,
        "max_hold":        cfg["max_hold"],
    }


STRATEGIES = {"A": signal_a, "B": signal_b, "C": signal_c, "D": signal_d, "E": signal_e,
              "F": signal_f, "G": signal_g, "H": signal_h, "I": signal_i,
              "J": signal_j, "K": signal_k, "L": signal_l,
              "M": signal_m, "N": signal_n, "O": signal_o,
              "PA": signal_pa, "PB": signal_pb, "PC": signal_pc}
