"""台指期 (TXF) 多時間框架順勢回檔波段 — 核心邏輯 (單一程式碼真相)。

與 main 既有策略的區別 (刻意不同):
  既有 `txf_strategy` = 前日高低『突破』的『當沖』(追價買強、13:30 平倉), 訊號/出場都在小時 K。
  本策略 `txf_mtf`   = 多時間框架『順勢回檔』的『波段』(在既定趨勢中買弱、可留倉):
    ‧ 日 K   → 定趨勢 (日線收盤 vs 日線 EMA20): 只做順大勢的方向 (多頭日只做多)。
    ‧ 小時線 → 定波段結構 (EMA20/50): 微趨勢方向 + 回檔基準 + 趨勢破壞出場 (回測主錨, 3 年歷史)。
    ‧ 15 分 → 進出場時機微調 (回檔到快線再轉強才進場; 實單用, 見 txf_mtf_strategy)。

經濟直覺 (誰把錢輸給你): 在已確立的日線多頭裡, 盤中/短線回檔多來自當沖獲利了結與掃短
停損; 有紀律的買方在回檔吸收後行情續行。本策略買『回檔後轉強』而非追突破 —— 進場價較好、
停損較近, edge 與突破策略互補 (突破買延續動能, 回檔買折價)。因此兩者報酬流不同向, 可分散。

刻意的資料誠實聲明 (strategy-pipeline G1/G4):
  Yahoo ^TWII 的 15 分 K 只有 ~60 天歷史, 不足以做多年統計檢定。故『相同 run() 邏輯』:
    ‧ 統計主錨 = 小時 K (3 年) → IS/OOS + 參數平台 + 逐年 + bootstrap;
    ‧ 15 分 K (60 天) → 只做『執行頻率確認』, 小樣本結果誠實標註, 判定上限『觀察』。
  兩者共用本檔 run(), 不另寫一份邏輯 (避免邏輯分叉 bug)。實單於 TXF/MXF 執行; ^TWII 為
  開發代理 (日內相關 > 0.99), 需另計基差與夜盤留倉跳空風險 (本策略可留倉, 尤須留意)。

參數 (回測 G3.3 已驗證為『平台』而非單點尖峰, 取平台中心而非最佳格):
  EMA_FAST=20 / EMA_SLOW=50 (連續小時線) / EMA_DAILY=20 (日線濾網) / STOP_PT=100。
"""
from collections import defaultdict

EMA_FAST = 20       # 連續快 EMA (回檔基準 + 微趨勢方向)
EMA_SLOW = 50       # 連續慢 EMA (微趨勢方向 + 收盤破線 → 趨勢破壞出場)
EMA_DAILY = 20      # 日線 EMA (大趨勢方向濾網)
STOP_PT = 100.0     # 固定停損點數
POINT_VALUE = 200.0 # 大台 NT$/點


def ema(values, n):
    """標準指數移動平均。回傳與輸入等長的 list。"""
    if not values:
        return []
    a = 2.0 / (n + 1)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(a * float(v) + (1 - a) * out[-1])
    return out


def daily_bias_map(daily_dates, daily_closes, ema_n=EMA_DAILY):
    """日線趨勢濾網。回傳 {date: +1 多 / -1 空 / 0 無明確}。

    當日方向由『前一日收盤 vs 前一日 日線EMA』決定 (只用已收盤資訊, 避免未來函數)。
    """
    de = ema(daily_closes, ema_n)
    out = {daily_dates[0]: 0}
    for i in range(1, len(daily_dates)):
        out[daily_dates[i]] = 1 if daily_closes[i - 1] > de[i - 1] else -1
    return out


def prepare(rows, fast=EMA_FAST, slow=EMA_SLOW):
    """rows: list of (datetime_str 'YYYY-MM-DD HH:MM', o,h,l,c) 已按時間排序。

    在『整段連續序列』上算快/慢 EMA (不逐日重置), 回傳:
      bars: list of {dt,date,hm,o,h,l,c,ef,es}
      daily_dates, daily_closes: 每日 (該日最後一根收盤) 供日線濾網。
    """
    closes = [r[4] for r in rows]
    ef, es = ema(closes, fast), ema(closes, slow)
    bars = []
    by_day = defaultdict(list)
    for k, (dt, o, h, l, c) in enumerate(rows):
        d, hm = dt.split(" ")
        bar = {"dt": dt, "date": d, "hm": hm, "o": float(o), "h": float(h),
               "l": float(l), "c": float(c), "ef": ef[k], "es": es[k]}
        bars.append(bar)
        by_day[d].append(bar)
    daily_dates = sorted(by_day)
    daily_closes = [by_day[d][-1]["c"] for d in daily_dates]
    return bars, daily_dates, daily_closes


def run(bars, bias_map, stop_pt=STOP_PT):
    """順勢回檔波段狀態機。逐根走過連續 bars, 回傳已平倉交易 list + 目前未平倉部位。

    進場 (long, 當根 date 的 bias=+1):
      需 ef>es (微多頭) 且『前一根收盤 < 前一根快線、當根收盤 ≥ 當根快線』(回檔後轉強),
      於當根收盤價進場。short 對稱 (bias=-1, ef<es, 前根收在快線上、當根收回快線下)。
    出場 (先到者為準, 可跨日留倉):
      1. 觸固定停損 (盤中觸價, entry ∓ stop_pt);
      2. 收盤跌破/升破慢線 es (趨勢破壞)。
    回傳:
      trades: list of dict(side,dir,entry,exit,entry_dt,exit_dt,reason,gross_pt)
      open_pos: dict 或 None (目前仍持有的部位, 供實單/即時報告)。
    """
    trades = []
    pos = None  # {side, entry, stop, entry_dt}
    for i in range(1, len(bars)):
        b, prev = bars[i], bars[i - 1]
        if pos is not None:
            side, entry, stop = pos["side"], pos["entry"], pos["stop"]
            # 1) 固定停損 (盤中觸價)
            if side > 0 and b["l"] <= stop:
                trades.append(_close(pos, stop, b["dt"], "stopped")); pos = None; continue
            if side < 0 and b["h"] >= stop:
                trades.append(_close(pos, stop, b["dt"], "stopped")); pos = None; continue
            # 2) 收盤破慢線 → 趨勢破壞出場
            if side > 0 and b["c"] < b["es"]:
                trades.append(_close(pos, b["c"], b["dt"], "trend_exit")); pos = None
            elif side < 0 and b["c"] > b["es"]:
                trades.append(_close(pos, b["c"], b["dt"], "trend_exit")); pos = None
            if pos is not None:
                continue
        bias = bias_map.get(b["date"], 0)
        if bias > 0 and b["ef"] > b["es"] and prev["c"] < prev["ef"] and b["c"] >= b["ef"]:
            pos = {"side": 1, "entry": b["c"], "stop": b["c"] - stop_pt, "entry_dt": b["dt"]}
        elif bias < 0 and b["ef"] < b["es"] and prev["c"] > prev["ef"] and b["c"] <= b["ef"]:
            pos = {"side": -1, "entry": b["c"], "stop": b["c"] + stop_pt, "entry_dt": b["dt"]}
    return trades, pos


def _close(pos, exit_px, exit_dt, reason):
    side = pos["side"]
    return {"side": side, "dir": "多" if side > 0 else "空",
            "entry": pos["entry"], "exit": exit_px, "entry_dt": pos["entry_dt"],
            "exit_dt": exit_dt, "reason": reason,
            "gross_pt": side * (exit_px - pos["entry"])}
