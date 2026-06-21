"""個股期貨 (STF) 飆股策略 — 規格書技術面 5 策略 (一/五/六/七/八).

依《臺灣飆股篩選策略規格書》逐條量化, 進場條件忠實對應規格, 出場由 stf_engine 統一
套用規格 §0.3 (三段式 R 移動停利 → Chandelier + 時間停損 + 大盤濾網)。各策略只負責:
  1. 進場判定 (回傳訊號 dict 或 None)
  2. 提供「初始停損」(以 signal 當根收盤的百分比表示, 引擎於次日開盤套用)
  3. 提供結構性出場均線 ma_exit (如「跌破 MA50/MA60 收盤出場」)

signal dict 欄位:
  score      排序分數 (通常用 RS rank, 多訊號同日時優先取高分)
  stop_pct   初始停損 (相對 signal 收盤, 引擎以 entry×(1-stop_pct) 設停損)
  ma_exit    結構性出場均線欄位名 ("ma20"/"ma50"/"ma60"/"ma150") 或 None
  max_hold   最長持有日
  name       策略代號 (S1/S5/S6/S7/S8)

RS 排名 (rs_rank) 由引擎用 126 日報酬橫斷面百分位傳入 (規格的「近6月相對強度」)。
S7 需「近60日報酬 > 大盤近60日報酬 × 1.5」, 大盤報酬經 set_market() 注入。
"""
import numpy as np

from strategies import _vcp_raw

# 大盤 (TWII) 報酬序列, 由 stf_backtest.set_market() 注入: {date: ret60}
_MARKET_RET60 = {}


def set_market(ret60_map):
    """注入大盤近60日報酬 {date: float}, 供 S7 相對強度比較."""
    global _MARKET_RET60
    _MARKET_RET60 = ret60_map


def _liquid(row, min_turnover=50_000_000, min_price=10.0):
    """流動性 / 可交易性濾網 (規格 §0.1: 對應日均量>3000張 之名目額代理)."""
    if row["close"] < min_price:
        return False
    if row["volume"] * row["close"] < min_turnover:
        return False
    return True


def _stage2_template(df, i, row):
    """規格 Stage 2 趨勢樣板: Close>MA50>MA150>MA200 且 MA200 上彎 + 位階."""
    if (np.isnan(row["ma200"]) or np.isnan(row["ma150"])
            or np.isnan(row["low252"]) or i < 200):
        return False
    return (row["close"] > row["ma50"] > row["ma150"] > row["ma200"]
            and row["ma200"] > df["ma200"].iloc[i - 21]
            and row["close"] >= row["high252"] * 0.75
            and row["close"] >= row["low252"] * 1.30)


# ── 策略一: VCP 波動收縮突破 ────────────────────────────────────────────────
S1_CFG = dict(rs_min=0.70, vcp_T_min=2, vcp_last_max=0.10, vcp_depth_max=0.35,
              contraction_ratio=0.80, dryup=0.70, breakout_vol_x=1.5,
              atr_stop=1.5, max_hold=90)


def signal_s1(df, i, rs_rank=None):
    """策略一 VCP: Stage2 + 波動逐次收縮至樞紐 + 帶量突破 (規格忠實版)."""
    cfg = S1_CFG
    row = df.iloc[i]
    if not _liquid(row) or not _stage2_template(df, i, row):
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:  # RS 前30%
        return None
    vcp = _vcp_raw(df, i)
    if vcp is None:
        return None
    swings, vol_ratio, pivot = vcp
    sw = swings[-6:]
    if len(sw) < cfg["vcp_T_min"]:
        return None
    if sw[-1] > cfg["vcp_last_max"]:                       # 末段收縮 < 10%
        return None
    if max(sw) > cfg["vcp_depth_max"]:                     # 整體不過深
        return None
    if sw[-1] >= sw[0] * cfg["contraction_ratio"]:         # 逐次遞縮 (末<首×0.8)
        return None
    if vol_ratio > cfg["dryup"]:                           # 樞紐前量縮 < 0.7×
        return None
    vm50 = row["vol_ma50"]
    breakout = (row["close"] > pivot * 1.001
                and vm50 and row["volume"] > cfg["breakout_vol_x"] * vm50)
    if not breakout:
        return None
    atr = df["atr14"].iloc[i]
    # 初始停損: max(pivot×0.92, 進場−1.5×ATR) → 取相對收盤的較緊停損%
    stop_price = max(pivot * 0.92, row["close"] - cfg["atr_stop"] * atr)
    stop_pct = _stop_pct(row["close"], stop_price)
    return _sig(rs_rank, stop_pct, "ma50", cfg["max_hold"], "S1")


# ── 策略五: 創 52 週新高箱型突破 (Darvas Box) ──────────────────────────────
S5_CFG = dict(rs_min=0.0, near_52wh=0.95, box_min=10, box_max=20, box_width=0.15,
              breakout_vol_x=1.5, atr_stop=2.0, max_hold=90)


def signal_s5(df, i, rs_rank=None):
    """策略五 Darvas: 近52週高 + 盒型收斂 + 帶量突破箱頂 (規格忠實版)."""
    cfg = S5_CFG
    row = df.iloc[i]
    if not _liquid(row) or i < cfg["box_max"] + 5:
        return None
    if np.isnan(row["high252"]) or row["close"] < row["high252"] * cfg["near_52wh"]:
        return None  # 創52週新高或接近 (≥95%)
    # 盒型: 近 box_max 日 (不含今日) 高低區間收斂 < box_width
    seg = df.iloc[i - cfg["box_max"]:i]
    box_top = seg["high"].max()
    box_bottom = seg["low"].min()
    if box_top <= 0 or (box_top - box_bottom) / box_top > cfg["box_width"]:
        return None
    vm50 = row["vol_ma50"]
    if not (row["close"] > box_top * 1.002
            and vm50 and row["volume"] > cfg["breakout_vol_x"] * vm50):
        return None
    atr = df["atr14"].iloc[i]
    # 初始停損: max(箱底, 進場−2×ATR)
    stop_price = max(box_bottom, row["close"] - cfg["atr_stop"] * atr)
    stop_pct = _stop_pct(row["close"], stop_price)
    return _sig(rs_rank if rs_rank else 0.5, stop_pct, "ma50", cfg["max_hold"], "S5")


# ── 策略六: 強勢均線回測進場 (Pullback) ─────────────────────────────────────
S6_CFG = dict(rs_min=0.50, ma_tol=0.02, atr_stop=1.5, max_hold=60)


def signal_s6(df, i, rs_rank=None):
    """策略六 Pullback: 多頭排列回檔至 MA20 不破 + 止跌站回 (買回檔)."""
    cfg = S6_CFG
    row = df.iloc[i]
    if not _liquid(row) or i < 70:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    if any(np.isnan(row[c]) for c in ("ma5", "ma20", "ma60")):
        return None
    # 多頭排列 MA5>MA20>MA60 且三者上彎
    rising = (row["ma5"] > df["ma5"].iloc[i - 5]
              and row["ma20"] > df["ma20"].iloc[i - 5]
              and row["ma60"] > df["ma60"].iloc[i - 10])
    if not (row["ma5"] > row["ma20"] > row["ma60"] and rising):
        return None
    # 前波創波段新高 (近60日內曾接近高252)
    if row["close"] < df["high60"].iloc[i] * 0.85:
        return None
    # 回檔至 MA20 附近: 近5日最低觸及 MA20 ±2%
    lo5 = df["low"].iloc[i - 4:i + 1].min()
    if abs(lo5 - row["ma20"]) / row["ma20"] > cfg["ma_tol"]:
        return None
    # 回檔量縮: 近3日均量 < 50日均量
    if not (df["volume"].iloc[i - 2:i + 1].mean() < row["vol_ma50"]):
        return None
    # 止跌訊號: 今日收紅且收盤站回 MA20
    if not (row["close"] > row["open"] and row["close"] > row["ma20"]):
        return None
    atr = df["atr14"].iloc[i]
    # 初始停損: max(回檔低點, MA60, 進場−1.5×ATR)
    stop_price = max(lo5, row["ma60"], row["close"] - cfg["atr_stop"] * atr)
    stop_pct = _stop_pct(row["close"], stop_price)
    return _sig(rs_rank, stop_pct, "ma60", cfg["max_hold"], "S6")


# ── 策略七: 量能突破 + 相對強度領先 (RS + Volume) ──────────────────────────
S7_CFG = dict(rs_min=0.80, mkt_mult=1.5, breakout_days=50, breakout_vol_x=2.0,
              atr_stop=1.8, max_hold=90)


def signal_s7(df, i, rs_rank=None):
    """策略七 RS+Vol: RS前20% + 60日報酬>大盤×1.5 + 帶量突破50日高."""
    cfg = S7_CFG
    row = df.iloc[i]
    if not _liquid(row) or i < cfg["breakout_days"] + 5:
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:  # 前20%
        return None
    # 近60日報酬 > 大盤近60日報酬 × 1.5
    ret60 = row.get("ret60", np.nan)
    if np.isnan(ret60):
        return None
    mkt = _MARKET_RET60.get(df["date"].iloc[i])
    if mkt is None:
        return None
    thr = mkt * cfg["mkt_mult"] if mkt > 0 else 0.0
    if ret60 <= thr:
        return None
    # 突破近50日高 (不含今日) + 顯著放量
    prior_high = df["high"].iloc[i - cfg["breakout_days"]:i].max()
    vm50 = row["vol_ma50"]
    if not (row["close"] > prior_high
            and vm50 and row["volume"] > cfg["breakout_vol_x"] * vm50):
        return None
    atr = df["atr14"].iloc[i]
    stop_price = row["close"] - cfg["atr_stop"] * atr   # 進場−1.8×ATR
    stop_pct = _stop_pct(row["close"], stop_price)
    return _sig(rs_rank, stop_pct, "ma20", cfg["max_hold"], "S7")


# ── 策略八: Stage 2 階段轉換 (Weinstein 30 週均線突破) ──────────────────────
S8_CFG = dict(rs_min=0.60, slope_win=20, base_min=120, base_band=0.30,
              breakout_vol_x=1.2, max_hold=120)


def signal_s8(df, i, rs_rank=None):
    """策略八 Weinstein: 30週(150日)均線由下彎轉平/上彎 + 站上 + 帶量 + 長底."""
    cfg = S8_CFG
    row = df.iloc[i]
    if not _liquid(row) or i < cfg["base_min"] + cfg["slope_win"] + 5:
        return None
    if np.isnan(row["ma150"]):
        return None
    if rs_rank is None or np.isnan(rs_rank) or rs_rank < cfg["rs_min"]:
        return None
    ma150_now = row["ma150"]
    ma150_prev = df["ma150"].iloc[i - cfg["slope_win"]]
    ma150_prev2 = df["ma150"].iloc[i - 2 * cfg["slope_win"]]
    if np.isnan(ma150_prev) or np.isnan(ma150_prev2):
        return None
    # 斜率由負(前段下彎)轉非負(近段走平/上彎)
    prev_slope = ma150_prev - ma150_prev2
    now_slope = ma150_now - ma150_prev
    if not (prev_slope < 0 <= now_slope):
        return None
    # 站上 30 週均線 (突破階段底部)
    if row["close"] <= ma150_now:
        return None
    # 突破帶量
    vm50 = row["vol_ma50"]
    if not (vm50 and row["volume"] > cfg["breakout_vol_x"] * vm50):
        return None
    # 前期長時間 Stage 1 底部: base_min 日內價格在 ±base_band 窄幅橫盤
    base = df["close"].iloc[i - cfg["base_min"]:i]
    if (base.max() - base.min()) / base.mean() > cfg["base_band"]:
        return None
    # 初始停損: 跌回 30 週均線下方 → 以 (close-ma150)/close 為停損距離 (下限保護)
    stop_price = min(ma150_now, row["close"] * 0.90)
    stop_pct = _stop_pct(row["close"], stop_price)
    return _sig(rs_rank, stop_pct, "ma150", cfg["max_hold"], "S8")


# ── 共用工具 ────────────────────────────────────────────────────────────────
def _stop_pct(close, stop_price, lo=0.03, hi=0.20):
    """把絕對停損價換成相對收盤的停損% (夾在 3%~20%, 對應規格停損結構)."""
    if close <= 0 or stop_price <= 0 or stop_price >= close:
        return hi
    return float(min(max((close - stop_price) / close, lo), hi))


def _sig(score, stop_pct, ma_exit, max_hold, name):
    return {"score": float(score), "stop_pct": stop_pct, "ma_exit": ma_exit,
            "max_hold": max_hold, "name": name}


STF_STRATEGIES = {
    "S1": signal_s1,   # VCP 波動收縮突破
    "S5": signal_s5,   # Darvas Box 52週新高箱型突破
    "S6": signal_s6,   # 強勢均線回測 (Pullback)
    "S7": signal_s7,   # 量能突破 + RS 領先
    "S8": signal_s8,   # Weinstein Stage2 階段轉換
}

STF_NAMES = {
    "S1": "VCP波動收縮突破", "S5": "Darvas箱型突破", "S6": "強勢回檔Pullback",
    "S7": "量能突破RS領先", "S8": "Weinstein階段轉換",
}
