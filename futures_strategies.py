"""穀物期貨交易策略 (黃豆 / 小麥 / 玉米) — 可做多亦可做空.

商品期貨與股票最大不同: 沒有「只能做多」的限制, 且趨勢跟蹤在農產品上長期有效
(Turtle 海龜系統當年就是在商品上發跡). 本檔提供五個風格互補的策略:

  T  唐奇安 20/10 突破 (Turtle System-1)   — 中期趨勢跟蹤, 多空雙向
  D  唐奇安 55/20 突破 (Turtle System-2)   — 長期趨勢跟蹤, 多空雙向
  M  雙均線趨勢 (50/100, always-in 反手)   — 平滑趨勢跟蹤, 多空雙向
  B  布林通道均值回歸 (逆勢)               — 區間行情, 多空雙向
  S  穀物季節性做多 (收割低點進場)         — 利用農產品播種/天候溢價

訊號函式回傳 dict (或 None):
  dir          +1 做多 / -1 做空
  stop_atr     初始停損 = entry - dir * stop_atr * ATR20  (災難停損)
  trail_atr    Chandelier 移動停損係數 (可選)
  exit_channel n: 反向 n 日通道出場 (多單跌破 n 日低 / 空單突破 n 日高)
  exit_mid     True: 回到中軌 (MA20) 出場 (均值回歸用)
  exit_ma      True: MA50/MA100 反向交叉時出場 (雙均線策略用)
  reverse      True: 出現反向訊號時直接反手 (開啟做空後 always-in 用)
  max_hold     最長持有交易日
"""
import numpy as np
import pandas as pd

# 25 年回測證據: 穀物做空在 T/D/M/B 上一致虧損 (做空 PF 0.54~0.70, 做多 PF 1.07~1.87).
# 原因 — 農產品長期有上行漂移 (通膨/需求成長), 天候衝擊使價格「向上」跳空,
# 且趨勢濾網 (MA100/MA200) 讓多單自動避開空頭段。故預設僅做多, 此旗標可開啟做空。
ALLOW_SHORT = False


def add_indicators(df):
    df = df.copy()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma100"] = df["close"].rolling(100).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    # ATR (Wilder 以簡單均值近似, 與股票引擎一致)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    df["atr20"] = tr.rolling(20).mean()

    # 唐奇安通道 (以「前一日為止」的 N 日高低, 不含當日)
    df["dc_hi20"] = df["high"].rolling(20).max().shift(1)
    df["dc_lo20"] = df["low"].rolling(20).min().shift(1)
    df["dc_hi10"] = df["high"].rolling(10).max().shift(1)
    df["dc_lo10"] = df["low"].rolling(10).min().shift(1)
    df["dc_hi55"] = df["high"].rolling(55).max().shift(1)
    df["dc_lo55"] = df["low"].rolling(55).min().shift(1)
    df["dc_hi20cur"] = df["high"].rolling(20).max()   # 出場用 (含當日前的當期)
    df["dc_lo20cur"] = df["low"].rolling(20).min()

    # 布林通道
    std20 = df["close"].rolling(20).std()
    df["bb_up"] = df["ma20"] + 2 * std20
    df["bb_dn"] = df["ma20"] - 2 * std20

    return df


def signal_t(df, i):
    """唐奇安 20/10 突破 — Turtle System-1 + MA100 趨勢濾網, 多空雙向.

    原版海龜在 ~20 個市場分散下有效; 單在 3 個穀物上 20 日突破假訊號過多,
    故加 MA100 順勢濾網: 只順著中期趨勢方向接突破, 大幅減少逆勢被洗。
    """
    row = df.iloc[i]
    if np.isnan(row["ma100"]) or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    if row["close"] > row["dc_hi20"] and row["close"] > row["ma100"]:
        return {"dir": +1, "stop_atr": 2.0, "exit_channel": 10, "max_hold": 200}
    if ALLOW_SHORT and row["close"] < row["dc_lo20"] and row["close"] < row["ma100"]:
        return {"dir": -1, "stop_atr": 2.0, "exit_channel": 10, "max_hold": 200}
    return None


def signal_d(df, i):
    """唐奇安 55/20 突破 — Turtle System-2 (長線) + MA200 趨勢濾網, 多空雙向."""
    row = df.iloc[i]
    if np.isnan(row["ma200"]) or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    if row["close"] > row["dc_hi55"] and row["close"] > row["ma200"]:
        return {"dir": +1, "stop_atr": 2.5, "exit_channel": 20, "max_hold": 300}
    if ALLOW_SHORT and row["close"] < row["dc_lo55"] and row["close"] < row["ma200"]:
        return {"dir": -1, "stop_atr": 2.5, "exit_channel": 20, "max_hold": 300}
    return None


def signal_m(df, i):
    """雙均線趨勢 (50/100) — always-in 反手, 多空雙向; ATR 災難停損保底."""
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    if np.isnan(row["ma100"]) or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    cross_up = prev["ma50"] <= prev["ma100"] and row["ma50"] > row["ma100"]
    cross_dn = prev["ma50"] >= prev["ma100"] and row["ma50"] < row["ma100"]
    # exit_ma: 多單於 ma50 跌破 ma100 時出場 (空單相反), 取代 always-in 反手
    if cross_up:
        return {"dir": +1, "stop_atr": 3.0, "exit_ma": True, "max_hold": 400}
    if ALLOW_SHORT and cross_dn:
        return {"dir": -1, "stop_atr": 3.0, "exit_ma": True, "max_hold": 400}
    return None


def signal_b(df, i):
    """布林通道均值回歸 (逆勢) — 收盤穿出 ±2σ 後反向進場, 回到中軌出場.

    過濾: 只在「非強趨勢」時做逆勢 (close 與 MA100 乖離 < 15%), 避免在大趨勢中
    硬接刀; 趨勢過強時交給 T/D/M 處理.
    """
    row = df.iloc[i]
    if np.isnan(row["bb_dn"]) or np.isnan(row["ma100"]) or row["atr20"] <= 0:
        return None
    # 區間濾網: 只在「中期趨勢平緩」時做逆勢, 趨勢過陡時讓路給 T/D/M
    slope = abs(row["ma100"] / df["ma100"].iloc[i - 20] - 1)
    if slope > 0.04:
        return None
    devi = abs(row["close"] / row["ma100"] - 1)
    if devi > 0.15:
        return None
    if row["close"] < row["bb_dn"]:
        return {"dir": +1, "stop_atr": 2.0, "exit_mid": True, "max_hold": 20}
    if ALLOW_SHORT and row["close"] > row["bb_up"]:
        return {"dir": -1, "stop_atr": 2.0, "exit_mid": True, "max_hold": 20}
    return None


def signal_s(df, i):
    """穀物季節性做多 — 每年 11 月初 (收割賣壓尾聲) 進場, 持有至隔年初夏.

    農產品常於北美收割期 (9-10 月) 形成季節低點, 隨後因庫存消化、隔年播種與
    天候溢價而走強。此策略每年 11 月第一個交易日做多, 持有約 150 個交易日
    (≈ 隔年 6 月), 以寬 ATR 停損防黑天鵝。
    """
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    if np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    d, pd_ = row["date"], prev["date"]
    if d.month == 11 and pd_.month != 11:          # 進入 11 月的第一個交易日
        return {"dir": +1, "stop_atr": 4.0, "max_hold": 150}
    return None


STRATEGIES = {
    "T": signal_t, "D": signal_d, "M": signal_m, "B": signal_b, "S": signal_s,
}

STRATEGY_NAMES = {
    "T": "唐奇安20/10突破",
    "D": "唐奇安55/20突破",
    "M": "雙均線趨勢",
    "B": "布林均值回歸",
    "S": "穀物季節做多",
}
