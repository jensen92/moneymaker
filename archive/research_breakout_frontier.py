"""勝率前緣: 在最佳濾網下, 不同 (目標漲幅 × 持有天數) 的觸及勝率。

回答「要達到 ~90% 勝率, 目標與天數該怎麼設」。
最佳濾網沿用 research_breakout3d.py 結論: 多頭市況 + 趨勢模板 + RS>=0.90。
進場 = 突破次日開盤。win_touch = 期間最高價觸及目標。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

bt.DATA_DIR = os.environ.get("MM_DATA_DIR",
                             os.path.join(os.path.dirname(__file__), "data_adj"))

RANGE_N = 20
MIN_TURNOVER = 50_000_000
MAXH = 20
TARGETS = [0.02, 0.03, 0.05, 0.08]
HORIZONS = [3, 5, 10, 20]


def main():
    data, _ = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()

    # 每事件記錄 T+1..T+MAXH 的最高價相對進場價報酬, 以及達標停損情況
    recs = []   # (target_touch_dict) 先存進場價與未來最高序列
    for code, df in data.items():
        n = len(df)
        if n < 260:
            continue
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        op = df["open"].values
        vol = df["volume"].values
        ma50 = df["ma50"].values
        ma150 = df["ma150"].values
        ma200 = df["ma200"].values
        dates = df["date"].values
        rmap = rs.get(code, {})
        for i in range(250, n - MAXH):
            if np.isnan(ma200[i]) or close[i] < 10:
                continue
            ph = high[i - RANGE_N:i].max()
            if close[i] <= ph or vol[i] * close[i] < MIN_TURNOVER:
                continue
            if not (dates[i] in regime and close[i] > ma200[i]
                    and ma50[i] > ma150[i] > ma200[i]):
                continue
            if rmap.get(dates[i], np.nan) < 0.90:
                continue
            entry = op[i + 1]
            if entry <= 0:
                continue
            fh = high[i + 1:i + 1 + MAXH]
            fl = low[i + 1:i + 1 + MAXH]
            # 累積最高 / 最低 (用於觸及與停損)
            cummax = np.maximum.accumulate(fh) / entry - 1.0
            cummin = np.minimum.accumulate(fl) / entry - 1.0
            recs.append((cummax, cummin))

    print(f"最佳濾網事件數: {len(recs)} (多頭+趨勢+RS>=0.90)\n")
    print("觸及勝率 (期間最高價曾達目標) — 列=目標漲幅, 欄=持有天數")
    hdr = "  目標\\天數 " + "".join(f"{h:>8d}d" for h in HORIZONS)
    print(hdr)
    for t in TARGETS:
        row = f"  +{t:>4.0%}   "
        for h in HORIZONS:
            wr = np.mean([(r[0][:h].max() >= t) for r in recs])
            row += f"{wr:>8.1%} "
        print(row)

    print("\n參考: 期間最低價跌破 -5% (停損觸發率) — 欄=持有天數")
    row = "  -5%停損 "
    for h in HORIZONS:
        sr = np.mean([(r[1][:h].min() <= -0.05) for r in recs])
        row += f"{sr:>8.1%} "
    print(row)
    row = "  -8%停損 "
    for h in HORIZONS:
        sr = np.mean([(r[1][:h].min() <= -0.08) for r in recs])
        row += f"{sr:>8.1%} "
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
