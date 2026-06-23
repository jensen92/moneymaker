"""研究: 區間突破後 3 個交易日內上漲 >= 5% 的勝率有多高?

使用者需求: 「突破區間後 3 天內能上漲 5% 以上, 勝率達 90%」的篩選策略。

本腳本以實證方式回答「90% 勝率是否可達」, 並找出能把勝率推到最高的濾網組合。

事件定義:
  進場日 i = 收盤突破前 N 日區間高 (close > 前 N 日最高, 不含當日)。
  進場價 = 次日開盤 (T+1 open, 貼近實際可成交價)。
  勝 (win) = 進場後 3 個交易日 (T+1..T+3) 內, 收盤相對進場價曾 >= +TARGET。
            另記錄以「最高價觸及」為準的寬鬆勝率 (win_touch)。

掃描多組濾網, 印出 [事件數 / 收盤勝率 / 觸及勝率 / 平均3日報酬]。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402
from strategies import add_indicators  # noqa: E402

bt.DATA_DIR = os.environ.get("MM_DATA_DIR",
                             os.path.join(os.path.dirname(__file__), "data_adj"))

TARGET = 0.05          # 目標漲幅
HORIZON = 3            # 交易日數
RANGE_N = 20           # 突破區間長度
MIN_TURNOVER = 50_000_000   # 最低成交額 (流動性)


def load():
    print(f"載入資料 ({bt.DATA_DIR}) ...")
    data, names = bt.load_all()
    print(f"  {len(data)} 檔。計算 RS rank ...")
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    return data, names, rs, regime


def collect_events(data, rs, regime):
    """掃描所有突破事件, 回傳結構化陣列供濾網篩選與勝率統計。"""
    events = []
    for code, df in data.items():
        n = len(df)
        if n < 260:
            continue
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        op = df["open"].values
        vol = df["volume"].values
        vol_ma50 = df["vol_ma50"].values
        ma50 = df["ma50"].values
        ma150 = df["ma150"].values
        ma200 = df["ma200"].values
        atr14 = df["atr14"].values
        atr5 = df["atr5"].values
        high252 = df["high252"].values
        low252 = df["low252"].values
        rsi14 = df["rsi14"].values
        dates = df["date"].values
        rmap = rs.get(code, {})

        # 前 N 日區間高 (不含當日)
        prior_high = np.full(n, np.nan)
        for i in range(RANGE_N, n):
            prior_high[i] = high[i - RANGE_N:i].max()

        # 進場需有 T+1..T+3, 故 i 最多到 n-4
        for i in range(250, n - HORIZON):
            if np.isnan(prior_high[i]) or np.isnan(ma200[i]):
                continue
            if close[i] <= prior_high[i]:        # 必須突破
                continue
            if vol[i] * close[i] < MIN_TURNOVER:
                continue
            if close[i] < 10:
                continue
            entry = op[i + 1]                    # 次日開盤進場
            if entry <= 0:
                continue
            fwd_close = close[i + 1:i + 1 + HORIZON]
            fwd_high = high[i + 1:i + 1 + HORIZON]
            max_close_ret = fwd_close.max() / entry - 1.0
            max_high_ret = fwd_high.max() / entry - 1.0
            ret3_close = fwd_close[-1] / entry - 1.0   # 第3日收盤報酬

            a14 = atr14[i - 1] if atr14[i - 1] and atr14[i - 1] > 0 else np.nan
            events.append({
                "code": code,
                "date": dates[i],
                "rs": rmap.get(dates[i], np.nan),
                "regime": dates[i] in regime,
                "gain_day": close[i] / close[i - 1] - 1.0,    # 突破日漲幅
                "vol_surge": (vol[i] / vol_ma50[i]) if vol_ma50[i] > 0 else 0.0,
                "above_ma50": close[i] > ma50[i],
                "above_ma150": close[i] > ma150[i] if not np.isnan(ma150[i]) else False,
                "trend": (not np.isnan(ma200[i]) and close[i] > ma200[i]
                          and ma50[i] > ma150[i] > ma200[i]),
                "contraction": (atr5[i - 1] / a14) if a14 and a14 > 0 else 9.9,
                "near_high": (close[i] >= high252[i] * 0.90) if not np.isnan(high252[i]) else False,
                "off_low": (close[i] >= low252[i] * 1.25) if not np.isnan(low252[i]) else False,
                "rsi": rsi14[i],
                "gap_up": op[i + 1] / close[i] - 1.0,          # 次日跳空
                "win": max_close_ret >= TARGET,
                "win_touch": max_high_ret >= TARGET,
                "ret3": ret3_close,
            })
    return events


def report(events, label, mask_fn):
    sub = [e for e in events if mask_fn(e)]
    if not sub:
        print(f"  {label:<46} 事件數 0")
        return
    n = len(sub)
    wr = sum(e["win"] for e in sub) / n
    wrt = sum(e["win_touch"] for e in sub) / n
    avg = np.mean([e["ret3"] for e in sub])
    print(f"  {label:<46} 事件 {n:>6}　收盤勝率 {wr:5.1%}　觸及勝率 {wrt:5.1%}　"
          f"均3日 {avg:+5.2%}")


def main():
    data, names, rs, regime = load()
    print("掃描突破事件 ...")
    events = collect_events(data, rs, regime)
    print(f"共 {len(events)} 筆突破事件 (區間={RANGE_N}日, 目標+{TARGET:.0%}/{HORIZON}日)\n")

    print("=== 基準與單一濾網 (收盤勝率 = 嚴格; 觸及勝率 = 盤中曾達標) ===")
    report(events, "基準 (所有突破)", lambda e: True)
    report(events, "多頭市況 (TAIEX>MA60)", lambda e: e["regime"])
    report(events, "趨勢模板 (站上MA50>150>200)", lambda e: e["trend"])
    report(events, "RS>=0.80", lambda e: e["rs"] >= 0.80)
    report(events, "RS>=0.90", lambda e: e["rs"] >= 0.90)
    report(events, "突破日量 >= 2x均量", lambda e: e["vol_surge"] >= 2.0)
    report(events, "突破日量 >= 3x均量", lambda e: e["vol_surge"] >= 3.0)
    report(events, "突破日漲幅 <= 4% (貼樞軸)", lambda e: e["gain_day"] <= 0.04)
    report(events, "波動收縮 ATR5/ATR14<0.8", lambda e: e["contraction"] < 0.80)
    report(events, "近52週高 (>=90%)", lambda e: e["near_high"])
    report(events, "RSI 50~70 (不過熱)", lambda e: 50 <= e["rsi"] <= 70)
    report(events, "次日不跳空高開 (<2%)", lambda e: e["gap_up"] < 0.02)

    print("\n=== 多濾網疊加 (追求高勝率) ===")
    report(events, "多頭+趨勢+RS80", lambda e: e["regime"] and e["trend"] and e["rs"] >= 0.80)
    report(events, "多頭+趨勢+RS80+量2x",
           lambda e: e["regime"] and e["trend"] and e["rs"] >= 0.80 and e["vol_surge"] >= 2.0)
    report(events, "多頭+趨勢+RS90+量2x+收縮",
           lambda e: e["regime"] and e["trend"] and e["rs"] >= 0.90
           and e["vol_surge"] >= 2.0 and e["contraction"] < 0.80)
    report(events, "全部疊加+漲幅<4%+RSI<70",
           lambda e: e["regime"] and e["trend"] and e["rs"] >= 0.90
           and e["vol_surge"] >= 2.0 and e["contraction"] < 0.80
           and e["gain_day"] <= 0.04 and e["rsi"] <= 70)
    report(events, "極嚴: +量3x+近高+不跳空",
           lambda e: e["regime"] and e["trend"] and e["rs"] >= 0.90
           and e["vol_surge"] >= 3.0 and e["contraction"] < 0.80
           and e["near_high"] and e["gap_up"] < 0.02 and e["gain_day"] <= 0.04)

    print(f"\n說明: 收盤勝率 = T+1~T+{HORIZON} 內任一日「收盤」相對次日開盤進場價 >= +{TARGET:.0%}")
    print(f"      觸及勝率 = 期間「最高價」曾觸及 +{TARGET:.0%} (盤中達標即算, 較寬鬆)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
