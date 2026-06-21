"""實驗室策略 (lab) — 使用者 120 策略清單中「資料可行 (純日線 OHLCV) 且現有
研究 (round1-5 / xs / tw / chip_cost / research_expert10) 尚未涵蓋」的真空地帶。

每個 signal_* 回傳與 backtest.run_sub 相容的訊號 dict 或 None, 由 validate_lab.py
餵進與正式策略完全相同的事件風控引擎 (停損/MA50移動停損/市況分級/波動標量/集中度)
做真實回測, 與現行冠軍 K/L/PA/PB 同台比較。只留勝過現行者, 其餘封存測試數據。

對應使用者清單:
  PP  策略84  Pocket Pivot (Morales & Kacher) — 隱性買點: 上漲日量 > 前10日最大跌量
  HF  策略86  High Tight Flag (O'Neil/Zanger) — 8週飆 90%+ 後緊旗 (<25%回檔) 突破
  TS  策略91  Turtle Soup (Raschke)           — 假跌破反轉 (逆勢短打做多)
  OO  策略93  OOPS (Larry Williams)           — 開盤跳空跌破昨低後拉回反轉
  VB  策略92  Volatility Breakout (LW)        — 開盤 + k×前一日區間 當日突破
  KA  策略24  KAMA / 效率比率 (Kaufman)        — ER>0.6 高效率趨勢中突破
  AX  策略59  ATR 擴張突破                     — ATR 由低檔轉升 + 突破 (波動 regime)
  QM  策略82  Momentum 整理突破 (Qullamaggie)  — 動能腿 + 高ADR% + 沿EMA緊縮後突破

註: data_adj 僅日線 OHLCV。盤中觸發型 (VB/OO) 以日線近似 (訊號日次日開盤進場),
會低估其盤中即時進場的邊際, 但足以判斷「是否值得進一步用分鐘資料驗證」。
"""
import numpy as np


def _bull(df, i, row):
    """共用多頭結構濾網 (與 D 家族同精神, 但不套嚴格 VCP 模板)."""
    return (not np.isnan(row["ma200"])
            and row["ma50"] > row["ma150"]
            and row["close"] > row["ma150"]
            and row["ma200"] > df["ma200"].iloc[i - 21])


# ── PP 策略84: Pocket Pivot ──────────────────────────────────────────────
PP_CONFIG = {"rs_min": 0.80, "near_ma_pct": 0.07, "lookback_dn": 10,
             "min_turnover": 50_000_000, "stop_pct": 0.08, "max_hold": 9999}


def signal_pp(df, i, rs_rank=None):
    cfg = PP_CONFIG
    if i < 200:
        return None
    row = df.iloc[i]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if np.isnan(row["vol_ma50"]) or row["vol_ma50"] <= 0:
        return None
    if not _bull(df, i, row):
        return None
    # 上漲日
    if row["close"] <= df["close"].iloc[i - 1]:
        return None
    # 量 > 前 10 日「下跌日」最大量 (口袋支點核心訊號)
    dn_vols = [df["volume"].iloc[k] for k in range(i - cfg["lookback_dn"], i)
               if df["close"].iloc[k] < df["close"].iloc[k - 1]]
    if not dn_vols or row["volume"] <= max(dn_vols):
        return None
    # 貼近 10/50 日均線 (建設性整理區內, 非延伸追高)
    near20 = abs(row["close"] - row["ma20"]) / row["ma20"] if row["ma20"] > 0 else 9
    near50 = abs(row["close"] - row["ma50"]) / row["ma50"] if row["ma50"] > 0 else 9
    if min(near20, near50) > cfg["near_ma_pct"]:
        return None
    return {"score": rs_rank, "minervini": True,
            "stop_pct": cfg["stop_pct"], "max_hold": cfg["max_hold"]}


# ── HF 策略86: High Tight Flag ───────────────────────────────────────────
HF_CONFIG = {"rs_min": 0.85, "pole_days": 40, "pole_gain": 0.90,
             "flag_min": 10, "flag_max": 25, "flag_pullback_max": 0.25,
             "vol_mult": 1.5, "stop_pct": 0.12, "max_hold": 120,
             "min_turnover": 50_000_000}


def signal_hf(df, i, rs_rank=None):
    cfg = HF_CONFIG
    if i < cfg["pole_days"] + cfg["flag_max"] + 5:
        return None
    row = df.iloc[i]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if np.isnan(row["vol_ma50"]) or row["vol_ma50"] <= 0:
        return None
    if row["volume"] < row["vol_ma50"] * cfg["vol_mult"]:
        return None
    # 嘗試各種旗形長度
    for flag_len in range(cfg["flag_min"], cfg["flag_max"] + 1):
        flag_start = i - flag_len
        pole_start = flag_start - cfg["pole_days"]
        if pole_start < 0:
            continue
        pole_low = df["low"].iloc[pole_start:flag_start].min()
        pole_high = df["high"].iloc[pole_start:flag_start + 1].max()
        if pole_low <= 0:
            continue
        # 旗桿: 8 週飆 >= 90%
        if pole_high / pole_low - 1.0 < cfg["pole_gain"]:
            continue
        flag_seg = df.iloc[flag_start:i]      # 不含今日
        flag_high = flag_seg["high"].max()
        flag_low = flag_seg["low"].min()
        # 緊旗: 回檔 < 25%
        if (flag_high - flag_low) / flag_high > cfg["flag_pullback_max"]:
            continue
        # 旗形需位於旗桿高檔 (不是已跌深)
        if flag_low < pole_high * (1 - cfg["flag_pullback_max"]):
            continue
        # 今日突破旗頂
        if row["close"] <= flag_high:
            continue
        if (row["close"] - flag_high) / flag_high > 0.10:   # 不追高
            continue
        return {"score": rs_rank, "stop_pct": cfg["stop_pct"],
                "trail_atr": 3.0, "gain_cap": 9.99, "max_hold": cfg["max_hold"]}
    return None


# ── TS 策略91: Turtle Soup (假跌破反轉, 逆勢短打) ─────────────────────────
TS_CONFIG = {"lookback": 20, "min_age": 4, "rs_min": 0.50,
             "stop_pct": 0.06, "target_r": 2.0, "max_hold": 8,
             "min_turnover": 50_000_000}


def signal_ts(df, i, rs_rank=None):
    cfg = TS_CONFIG
    if i < 210:
        return None
    row = df.iloc[i]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    # 多頭回檔中 (非空頭接刀)
    if np.isnan(row["ma200"]) or row["close"] < row["ma200"]:
        return None
    lb = cfg["lookback"]
    prior_low = df["low"].iloc[i - lb:i].min()
    # 該 20 日低點需於 min_age 日前形成 (Raschke 原則)
    recent_low = df["low"].iloc[i - cfg["min_age"]:i].min()
    if recent_low <= prior_low:
        return None
    # 今日假跌破: 盤中破前低, 收盤拉回前低之上
    if not (row["low"] < prior_low and row["close"] > prior_low):
        return None
    return {"score": rs_rank if (rs_rank and not np.isnan(rs_rank)) else 0.5,
            "stop_pct": cfg["stop_pct"], "target_r": cfg["target_r"],
            "max_hold": cfg["max_hold"]}


# ── OO 策略93: OOPS (跳空跌破昨低後反轉) ─────────────────────────────────
OO_CONFIG = {"rs_min": 0.50, "stop_pct": 0.06, "target_r": 2.0,
             "max_hold": 6, "min_turnover": 50_000_000}


def signal_oo(df, i, rs_rank=None):
    cfg = OO_CONFIG
    if i < 210:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    # 強勢股 (站上 MA50), 避免接刀
    if np.isnan(row["ma50"]) or row["close"] < row["ma50"]:
        return None
    # 開盤跳空跌破昨低, 收盤拉回昨低之上 (恐慌反轉)
    if not (row["open"] < prev["low"] and row["close"] > prev["low"]):
        return None
    return {"score": rs_rank if (rs_rank and not np.isnan(rs_rank)) else 0.5,
            "stop_pct": cfg["stop_pct"], "target_r": cfg["target_r"],
            "max_hold": cfg["max_hold"]}


# ── VB 策略92: Larry Williams 波動率突破 ─────────────────────────────────
VB_CONFIG = {"k": 0.5, "rs_min": 0.70, "stop_pct": 0.08, "trail_atr": 2.5,
             "gain_cap": 9.99, "max_hold": 20, "min_turnover": 50_000_000}


def signal_vb(df, i, rs_rank=None):
    cfg = VB_CONFIG
    if i < 210:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if np.isnan(row["ma20"]) or row["close"] < row["ma20"]:
        return None
    prev_range = prev["high"] - prev["low"]
    if prev_range <= 0:
        return None
    # 前一日非極端長黑 (過濾崩跌續弱)
    if prev["close"] < prev["open"] * 0.95:
        return None
    trigger = row["open"] + cfg["k"] * prev_range
    # 當日盤中觸發突破 (日線近似: high 觸及 trigger)
    if row["high"] < trigger:
        return None
    return {"score": rs_rank, "stop_pct": cfg["stop_pct"],
            "trail_atr": cfg["trail_atr"], "gain_cap": cfg["gain_cap"],
            "max_hold": cfg["max_hold"]}


# ── KA 策略24: KAMA / 效率比率自適應趨勢 ─────────────────────────────────
KA_CONFIG = {"er_window": 10, "er_min": 0.60, "rs_min": 0.75, "breakout": 20,
             "vol_mult": 1.3, "stop_pct": 0.08, "trail_atr": 3.0,
             "gain_cap": 9.99, "max_hold": 120, "min_turnover": 50_000_000}


def signal_ka(df, i, rs_rank=None):
    cfg = KA_CONFIG
    if i < 210:
        return None
    row = df.iloc[i]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if np.isnan(row["ma60"]) or row["close"] < row["ma60"]:
        return None
    if np.isnan(row["vol_ma50"]) or row["vol_ma50"] <= 0:
        return None
    n = cfg["er_window"]
    closes = df["close"].iloc[i - n:i + 1].values
    direction = abs(closes[-1] - closes[0])
    volatility = np.sum(np.abs(np.diff(closes)))
    if volatility <= 0:
        return None
    er = direction / volatility
    if er < cfg["er_min"]:
        return None
    # 高效率趨勢中突破近 20 日高 + 量增
    prior_high = df["high"].iloc[i - cfg["breakout"]:i].max()
    if row["close"] <= prior_high:
        return None
    if row["volume"] < row["vol_ma50"] * cfg["vol_mult"]:
        return None
    return {"score": rs_rank, "stop_pct": cfg["stop_pct"],
            "trail_atr": cfg["trail_atr"], "gain_cap": cfg["gain_cap"],
            "max_hold": cfg["max_hold"]}


# ── AX 策略59: ATR 擴張突破 (波動 regime 由低轉升) ───────────────────────
AX_CONFIG = {"atr_low_window": 60, "rs_min": 0.75, "breakout": 50,
             "vol_mult": 1.5, "stop_pct": 0.10, "trail_atr": 3.0,
             "gain_cap": 9.99, "max_hold": 120, "min_turnover": 50_000_000}


def signal_ax(df, i, rs_rank=None):
    cfg = AX_CONFIG
    if i < 210:
        return None
    row = df.iloc[i]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if np.isnan(row["ma60"]) or row["close"] < row["ma60"]:
        return None
    if np.isnan(row["atr14"]) or row["atr14"] <= 0 or np.isnan(row["vol_ma50"]):
        return None
    # ATR 由近 60 日低檔轉升: 今日 ATR > 5 日前, 且 5 日前曾接近 60 日 ATR 低點
    atr_now = row["atr14"]
    atr_prev = df["atr14"].iloc[i - 5]
    atr_low = df["atr14"].iloc[i - cfg["atr_low_window"]:i - 4].min()
    if not (atr_now > atr_prev and atr_prev <= atr_low * 1.15):
        return None
    # 突破近 50 日高 + 量增
    prior_high = df["high"].iloc[i - cfg["breakout"]:i].max()
    if row["close"] <= prior_high:
        return None
    if row["volume"] < row["vol_ma50"] * cfg["vol_mult"]:
        return None
    return {"score": rs_rank, "stop_pct": cfg["stop_pct"],
            "trail_atr": cfg["trail_atr"], "gain_cap": cfg["gain_cap"],
            "max_hold": cfg["max_hold"]}


# ── QM 策略82: Qullamaggie 動能整理突破 (動能腿 + 高ADR% + 沿EMA緊縮) ────
QM_CONFIG = {"rs_min": 0.85, "leg_days": 40, "leg_gain": 0.30,
             "tight_days": 15, "tight_pullback": 0.15, "adr_min": 0.04,
             "vol_mult": 1.3, "stop_pct": 0.10, "trail_atr": 3.0,
             "gain_cap": 9.99, "max_hold": 120, "min_turnover": 50_000_000}


def signal_qm(df, i, rs_rank=None):
    cfg = QM_CONFIG
    if i < 210:
        return None
    row = df.iloc[i]
    if row["close"] < 10 or row["volume"] * row["close"] < cfg["min_turnover"]:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if np.isnan(row["vol_ma50"]) or row["vol_ma50"] <= 0:
        return None
    # 動能腿: 前期 (整理之前) 40 日漲 >= 30%
    tstart = i - cfg["tight_days"]
    leg_start = tstart - cfg["leg_days"]
    if leg_start < 0:
        return None
    leg_gain = df["close"].iloc[tstart] / df["close"].iloc[leg_start] - 1.0
    if leg_gain < cfg["leg_gain"]:
        return None
    # 整理: 近 tight_days 沿均線緊縮, 回檔淺
    tight_seg = df.iloc[tstart:i]
    t_high, t_low = tight_seg["high"].max(), tight_seg["low"].min()
    if (t_high - t_low) / t_high > cfg["tight_pullback"]:
        return None
    # 高 ADR% (近 20 日平均日振幅)
    seg20 = df.iloc[i - 20:i]
    adr = ((seg20["high"] - seg20["low"]) / seg20["close"]).mean()
    if adr < cfg["adr_min"]:
        return None
    # 突破整理高 + 量增
    if row["close"] <= t_high:
        return None
    if row["volume"] < row["vol_ma50"] * cfg["vol_mult"]:
        return None
    return {"score": rs_rank, "stop_pct": cfg["stop_pct"],
            "trail_atr": cfg["trail_atr"], "gain_cap": cfg["gain_cap"],
            "max_hold": cfg["max_hold"]}


LAB_STRATEGIES = {
    "PP": signal_pp, "HF": signal_hf, "TS": signal_ts, "OO": signal_oo,
    "VB": signal_vb, "KA": signal_ka, "AX": signal_ax, "QM": signal_qm,
}
