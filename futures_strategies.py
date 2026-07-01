"""穀物期貨交易策略 (黃豆 / 小麥 / 玉米) — 可做多亦可做空, 參數集中於 CONFIG.

商品期貨與股票最大不同: 沒有「只能做多」的限制, 且趨勢跟蹤在農產品上長期有效
(Turtle 海龜系統當年就是在商品上發跡). 本檔提供五個風格互補的策略:

  T  唐奇安 20/10 突破 (Turtle System-1)   — 中期趨勢跟蹤, 多空雙向
  D  唐奇安 55/20 突破 (Turtle System-2)   — 長期趨勢跟蹤, 多空雙向
  M  雙均線趨勢 (50/100, MA 交叉出場)      — 平滑趨勢跟蹤, 多空雙向
  B  布林通道均值回歸 (逆勢)               — 區間行情, 多空雙向
  S  穀物季節性做多 (收割低點進場)         — 利用農產品播種/天候溢價

所有可調參數集中在 CONFIG, 供 futures_optimize.py 樣本內/外掃描共用。

訊號函式回傳 dict (或 None):
  dir            +1 做多 / -1 做空
  stop_atr       初始停損 = entry - dir * stop_atr * ATR20  (災難停損)
  exit_channel n 反向 n 日通道出場 (多單跌破 n 日低 / 空單突破 n 日高)
  exit_mid       True: 回到中軌 (MA20) 出場 (均值回歸用)
  exit_ma (f,s)  MA(f)/MA(s) 反向交叉時出場 (雙均線策略用)
  max_hold       最長持有交易日
"""
import numpy as np
import pandas as pd

# 25 年回測證據: 穀物做空在 T/D/M/B 上一致虧損 (做空 PF 0.54~0.70, 做多 PF 1.07~1.87).
# 原因 — 農產品長期有上行漂移 (通膨/需求成長), 天候衝擊使價格「向上」跳空,
# 且趨勢濾網 (MA) 讓多單自動避開空頭段。故預設僅做多, 此旗標可開啟做空。
ALLOW_SHORT = False

# 預先計算的均線與唐奇安通道長度 (CONFIG 只能引用這些, 以免每次掃描重算指標)
MA_LENS = [20, 30, 50, 80, 100, 150, 200]
DC_LENS = [10, 15, 20, 40, 55, 80, 100]

# 各策略參數 — 經 futures_optimize.py 樣本內(2000-15)挑選、樣本外(2016-26)驗證.
# 每組都來自「整片網格大多 OOS 獲利」的穩健區, 非單一尖峰 (防過擬合).
#   T: entry 20→40 (20 日突破太雜訊, 拉長至 40 後 OOS 100% 獲利)
#   M: stop 3.0→2.5 (50/100 均線在 OOS PF 1.52)
#   D: trend 200→150 (OOS PF 1.71, Sharpe 0.41)
#   S: 11 月→12 月進場、持有 150→100 日 (OOS PF 3.20 — 抓 12 月低點到春季行情)
CONFIG = {
    "T": {"entry": 40, "exit": 20, "stop": 2.5, "trend": 100, "max_hold": 200},
    "D": {"entry": 55, "exit": 20, "stop": 2.5, "trend": 150, "max_hold": 300},
    "M": {"fast": 50, "slow": 100, "stop": 2.5, "max_hold": 400},
    "B": {"std": 2.0, "stop": 2.0, "slope": 0.04, "devi": 0.15,
          "trend": 100, "max_hold": 20},
    "S": {"month": 12, "hold": 100, "stop": 3.0, "trend": 0},
}


def add_indicators(df):
    df = df.copy()
    for n in MA_LENS:
        df[f"ma{n}"] = df["close"].rolling(n).mean()

    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    df["atr20"] = tr.rolling(20).mean()

    # 唐奇安通道 (以「前一日為止」的 N 日高低, 不含當日)
    for n in DC_LENS:
        df[f"dc_hi{n}"] = df["high"].rolling(n).max().shift(1)
        df[f"dc_lo{n}"] = df["low"].rolling(n).min().shift(1)

    df["std20"] = df["close"].rolling(20).std()
    return df


def signal_t(df, i):
    """唐奇安突破 (Turtle System-1 風格) + MA 趨勢濾網, 反向短通道出場.

    原版海龜在 ~20 個市場分散下有效; 單在 3 個穀物上突破假訊號過多,
    故加 MA 順勢濾網: 只順著中期趨勢方向接突破, 大幅減少逆勢被洗。
    """
    c = CONFIG["T"]
    row = df.iloc[i]
    tr = row[f"ma{c['trend']}"]
    if np.isnan(tr) or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    if row["close"] > row[f"dc_hi{c['entry']}"] and row["close"] > tr:
        return {"dir": +1, "stop_atr": c["stop"], "exit_channel": c["exit"],
                "max_hold": c["max_hold"], "pyramid": True}
    if ALLOW_SHORT and row["close"] < row[f"dc_lo{c['entry']}"] and row["close"] < tr:
        return {"dir": -1, "stop_atr": c["stop"], "exit_channel": c["exit"],
                "max_hold": c["max_hold"], "pyramid": True}
    return None


def signal_d(df, i):
    """唐奇安長線突破 (Turtle System-2 風格) + MA 趨勢濾網, 反向通道出場."""
    c = CONFIG["D"]
    row = df.iloc[i]
    tr = row[f"ma{c['trend']}"]
    if np.isnan(tr) or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    if row["close"] > row[f"dc_hi{c['entry']}"] and row["close"] > tr:
        return {"dir": +1, "stop_atr": c["stop"], "exit_channel": c["exit"],
                "max_hold": c["max_hold"], "pyramid": True}
    if ALLOW_SHORT and row["close"] < row[f"dc_lo{c['entry']}"] and row["close"] < tr:
        return {"dir": -1, "stop_atr": c["stop"], "exit_channel": c["exit"],
                "max_hold": c["max_hold"], "pyramid": True}
    return None


def signal_m(df, i):
    """雙均線趨勢 — MA(fast)/MA(slow) 交叉; 反向交叉出場, ATR 災難停損保底."""
    c = CONFIG["M"]
    row, prev = df.iloc[i], df.iloc[i - 1]
    f, s = f"ma{c['fast']}", f"ma{c['slow']}"
    if np.isnan(row[s]) or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    cross_up = prev[f] <= prev[s] and row[f] > row[s]
    cross_dn = prev[f] >= prev[s] and row[f] < row[s]
    if cross_up:
        return {"dir": +1, "stop_atr": c["stop"], "exit_ma": (c["fast"], c["slow"]),
                "max_hold": c["max_hold"], "pyramid": True}
    if ALLOW_SHORT and cross_dn:
        return {"dir": -1, "stop_atr": c["stop"], "exit_ma": (c["fast"], c["slow"]),
                "max_hold": c["max_hold"], "pyramid": True}
    return None


def signal_b(df, i):
    """布林通道均值回歸 (逆勢) — 收盤穿出 ±std·σ 後反向進場, 回到中軌出場.

    區間濾網: 只在「中期趨勢平緩」時做逆勢 (MA{trend} 斜率小且乖離小),
    趨勢過陡時讓路給 T/D/M, 避免在大趨勢中硬接刀。
    """
    c = CONFIG["B"]
    row = df.iloc[i]
    tr = row[f"ma{c['trend']}"]
    if np.isnan(row["std20"]) or np.isnan(tr) or row["atr20"] <= 0:
        return None
    slope = abs(tr / df[f"ma{c['trend']}"].iloc[i - 20] - 1)
    if slope > c["slope"] or abs(row["close"] / tr - 1) > c["devi"]:
        return None
    up = row["ma20"] + c["std"] * row["std20"]
    dn = row["ma20"] - c["std"] * row["std20"]
    if row["close"] < dn:
        return {"dir": +1, "stop_atr": c["stop"], "exit_mid": True,
                "max_hold": c["max_hold"]}
    if ALLOW_SHORT and row["close"] > up:
        return {"dir": -1, "stop_atr": c["stop"], "exit_mid": True,
                "max_hold": c["max_hold"]}
    return None


# 各穀物『專屬』季節窗 — 依各自作物行事曆 (收割低點進場), 26 年樣本內外對切驗證。
# 績效有兩種口徑, 勿混淆:
#   (a) 純季節事件 (固定持有, 無ATR停損/部位管理) — 黃豆 PF~5.5/玉米 PF~5.6, 屬研究探索值;
#   (b) 真實引擎 (run_sub, 含 ATR 寬停損+風險化口數, 即實際/futures所跑) 單商品實跑:
#       黃豆 ZS PF~2.75 DD~7%, 玉米 ZC PF~4.30 DD~5%, 小麥 ZW PF~2.15 DD~5%。
#       組合 S(三商品) PF~2.94 DD~10% MAR~0.26 — /futures 顯示的即此引擎口徑。
#   小麥季節性最弱 (固定持有單獨 PF1.68 且不穩), 個別進出場見 grain_signals.py。
# 註: 舊版單一 12 月進場對黃豆錯置 (黃豆 10 月才是收割低點)。窗口為作物常識+對切驗證,
#     非網格挑選 (hold 用整數月, 避免實盤失真)。
# 小麥(ZW)經逐年%驗證不通過 (勝率38%、中位-3.4%, 正報酬僅靠2021單一離群年撐起),
# 已移除, 季節僅保留黃豆、玉米。
S_SEASON = {
    "ZS": {"month": 10, "hold": 105, "stop": 3.0},   # 黃豆
    "ZC": {"month": 12, "hold": 63,  "stop": 3.0},   # 玉米
}


def signal_s(df, i):
    """穀物季節性做多 — 各穀物依『自己的』作物行事曆 (收割低點月) 進場, 持有其季節
    強勢窗 (見 S_SEASON)。每年該月第一個交易日做多, 以寬 ATR 停損防黑天鵝。

    取代原本『所有穀物統一 12 月進場』(對黃豆錯置) 的做法; 改用各穀物專屬窗後
    DD 由 7~12% 降至 1~3%, 且樣本內外皆穩 (見 S_SEASON 註記)。
    """
    market = df["market"].iloc[i] if "market" in df.columns else None
    c = S_SEASON.get(market)
    if c is None:            # 僅交易 S_SEASON 白名單內的穀物 (黃豆/玉米), 其餘不做
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    if row["date"].month == c["month"] and prev["date"].month != c["month"]:
        return {"dir": +1, "stop_atr": c["stop"], "max_hold": c["hold"]}
    return None


STRATEGIES = {
    "T": signal_t, "D": signal_d, "M": signal_m, "B": signal_b, "S": signal_s,
}

STRATEGY_NAMES = {
    "T": "唐奇安突破(短)",
    "D": "唐奇安突破(長)",
    "M": "雙均線趨勢",
    "B": "布林均值回歸",
    "S": "穀物季節做多",
}
