"""系統性測試: 高點拉回若守在 0.618~0.667 黃金切割帶, 之後突破的表現是否顯著優於
其他深度的拉回 (對照組: 守在更淺 <0.5 或更深 >0.786 的拉回帶)。

事件定義 (純收盤價, 因果性 — 不窺視未來):
  1. swing_low: 過去 LOOKBACK 日內最低點 (上漲段起點)
  2. swing_high: swing_low 之後、HIGH_WIN 日內滾動高點 (確認後即固定, 用 shift 避免未來函數)
  3. retrace = (swing_high - close) / (swing_high - swing_low)   (拉回幅度比例)
  4. 整理確認: retrace 連續 HOLD_DAYS 天都落在某帶內 (golden=[0.618,0.667], 對照帶另設)
  5. 突破: 整理結束後, close 突破整理期高點 → 進場, 量 N 日後報酬
比較 golden 帶 vs shallow(<0.50) / deep(>0.786) 帶的勝率與平均報酬, 用同一套真實
(簡化) 成本與口徑, 並用 t 檢定看顯著性。
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "data_adj")
LOOKBACK = 120      # 找 swing_low 的回溯窗
HIGH_WIN = 60        # swing_low 之後找 swing_high 的窗
HOLD_DAYS = 5         # 守在帶內至少 N 天才算「平台整理」
MAX_HOLD_WINDOW = 15  # 整理期最長容許天數 (找 hold 是否發生於此窗內)
FWD = [10, 20, 40]
BANDS = {
    "golden(0.618-0.667)": (0.618, 0.667),
    "shallow(<0.50)": (0.0, 0.50),
    "mid(0.50-0.618)": (0.50, 0.618),
    "deep(0.667-0.786)": (0.667, 0.786),
    "very_deep(>0.786)": (0.786, 1.0),
}


def find_events(df):
    close = df["close"].values
    high = df["high"].values
    n = len(close)
    events = []
    i = LOOKBACK
    while i < n - MAX_HOLD_WINDOW - max(FWD):
        lo_win = close[i - LOOKBACK:i]
        swing_low = lo_win.min()
        lo_idx = i - LOOKBACK + lo_win.argmin()
        if lo_idx >= i:
            i += 1; continue
        hi_win = high[lo_idx:i]
        if len(hi_win) == 0:
            i += 1; continue
        swing_high = hi_win.max()
        rng = swing_high - swing_low
        if rng <= 0 or swing_low <= 0:
            i += 1; continue
        # 從目前點起算往後 MAX_HOLD_WINDOW 天, 看 retrace 帶, 找連續 HOLD_DAYS 天落在某帶
        for band_name, (lo_r, hi_r) in BANDS.items():
            window = close[i:i + MAX_HOLD_WINDOW]
            if len(window) < HOLD_DAYS:
                continue
            retrace = (swing_high - window) / rng
            in_band = (retrace >= lo_r) & (retrace < hi_r)
            # 找第一個連續 HOLD_DAYS True 的位置
            run = 0
            hold_end = None
            for j, b in enumerate(in_band):
                run = run + 1 if b else 0
                if run >= HOLD_DAYS:
                    hold_end = j
                    break
            if hold_end is None:
                continue
            entry_i = i + hold_end + 1   # 整理確認後一天進場 (簡化: 不強制等突破, 看是否就此守住反彈)
            if entry_i >= n - max(FWD):
                continue
            entry_px = close[entry_i]
            rec = dict(band=band_name, entry_idx=entry_i, entry_px=entry_px,
                       swing_high=swing_high, swing_low=swing_low)
            for f in FWD:
                rec[f"ret{f}"] = close[entry_i + f] / entry_px - 1.0
            events.append(rec)
        i += 1
    return events


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".csv") and not f.startswith("_"))
    print(f"掃描 {len(files)} 檔股票 ...")
    all_events = []
    for k, fn in enumerate(files):
        df = pd.read_csv(os.path.join(DATA, fn))
        if len(df) < LOOKBACK + MAX_HOLD_WINDOW + max(FWD) + 10:
            continue
        evs = find_events(df)
        for e in evs:
            e["code"] = fn[:-4]
        all_events.extend(evs)
        if (k + 1) % 500 == 0:
            print(f"  ... {k+1}/{len(files)} 檔, 累積事件 {len(all_events)}")

    edf = pd.DataFrame(all_events)
    print(f"\n總事件數: {len(edf)}\n")
    print(f"{'帶':<24}{'n':>7}" + "".join(f"{f'ret{f}均':>9}{f'ret{f}勝率':>9}" for f in FWD))
    base = edf[edf["band"] == "golden(0.618-0.667)"]
    for band in BANDS:
        g = edf[edf["band"] == band]
        row = f"  {band:<22}{len(g):>7}"
        for f in FWD:
            row += f"{g[f'ret{f}'].mean()*100:>8.2f}%{(g[f'ret{f}']>0).mean()*100:>8.1f}%"
        print(row)

    print("\n=== golden 帶 vs 其他帶: t 檢定 (ret20) ===")
    for band in BANDS:
        if band == "golden(0.618-0.667)":
            continue
        g = edf[edf["band"] == band]
        if len(g) < 10 or len(base) < 10:
            continue
        t, p = stats.ttest_ind(base["ret20"], g["ret20"], equal_var=False)
        print(f"  golden vs {band:<20} t={t:>6.2f}  p={p:.4f}  "
              f"(golden均{base['ret20'].mean()*100:.2f}% vs {band}均{g['ret20'].mean()*100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
