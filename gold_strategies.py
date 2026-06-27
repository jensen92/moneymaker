"""黃金期貨 (GC=F COMEX) 小時線 — 順勢突破策略 (生產版).

沿革: 原始需求為「兩高兩低定區間, 上緣空/下緣多, 停損反手」(區間逆勢),
經 gold_range_backtest.py / gold_optimize.py 實測證實**結構性負期望**
(毛利近 0, 反手機制空轉吃光成本, 對抗黃金多年上行趨勢):

  V1 區間逆勢+停損反手          總損益 -$787,020   PF 0.83   MAR -0.98
  V3 順勢突破(只做多) 24/3.0ATR  總損益 +$190,020   PF 1.90   MAR  6.40  (前版)
  V3 順勢突破(只做多) 24/2.5ATR  總損益 +$209,590   PF 2.00   MAR  9.40  ← 採用

故改採方向相反的「順勢突破」: 收盤突破最近 N 小時高點進場做多, ATR 倍數
移動停損出場 (對稱跌破做空在黃金上經驗證為負期望, 預設關閉)。
參數 (breakout=24, atr_stop=2.5) 經乾淨資料 (只含已完成棒) 細網格優化:
  - atr_stop: 2.25~2.75 為平滑平台 (MAR 8.6~9.4), 收緊至 2.5 較原 3.0
    每年皆改善、回撤由 29.7k 降至 22.3k (見下方 atr_stop 細掃)。
  - breakout: 18~30 為平台 (MAR 6.9~10.5), 24 居中。
  - atr_n=14: 8~14 為平台 (MAR 9~11); 改 10 可再增 +24k/MAR→10.96, 但 ATR
    較快對雜訊敏感, 保守維持 14。
逐年 (2024:+40,200 / 2025:+104,600 / 2026:+64,790, 含黃金回落的 2026)
全正, 非單一過擬合尖峰, 見 gold_optimize.py。

抗過擬合驗證 (已確認此參數為穩健最優, 請勿再調 breakout/atr_stop/atr_n):
經多面向研究 + 對抗式檢驗, 一致裁定維持 24/2.5/14, 證據四點 —
  (1) Walk-forward: 滾動每折用 train-MAR 重挑參數, 累積樣本外損益 164,120
      反而『輸給』始終用基準的 169,390 → 再優化這 3 個參數弊大於利。
  (2) 樣本外不變號: 純用樣本內(2024+2025)在 160 組參數網格(n≥80)選優,
      160/160 組的樣本外(2026)全部為正、零轉負 → 基準落在穩健區中心非僥倖尖峰。
  (3) 增強假設全否決: MA 趨勢過濾(突破本含動能, 近乎 no-op, 日線空頭年未翻正)、
      保本/分離停損(日線受害或單調惡化)、突破邊際 margin(單調砍利潤)皆無
      『樣本內+樣本外+平台+日線26年』四關全過者。
  (4) 唯一帳面贏基準的候選(縮短 atr_n 14→10)優勢全來自 atr_n, 但在
      日線 26 年(含 2013-2015 空頭)完全不複現 → 屬 2024-26 近期 regime 產物。
注意: 樣本僅 2.4 年、樣本外桶僅 ~28 筆, 本就不該據以微調; 跨 regime 長期
基準請參考 gold_daily_backtest.py (日線 26 年, PF 2.26/2024前)。

資料: futures_data/GC_60m.csv (Yahoo 連續近月合約 1 小時線, 台北時間,
只含已完成整點棒 — load_bars 已濾掉即時快照半成品棒與非整點棒)。
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "futures_data", "GC_60m.csv")

POINT_VALUE = 100.0   # US$ / 每 1.0 報價變動 / 口 (COMEX GC, 100 oz)
COST_PER_SIDE = 0.30  # 滑價+手續費估計 (報價單位), 進出各一次

# 經 gold_optimize.py 參數網格 + 逐年驗證選定 (見上方說明)
CONFIG = {
    "breakout":    24,    # 突破窗: 收盤 > 過去 N 小時高點 (不含當小時)
    "atr_n":       14,
    "atr_stop":    2.5,   # 移動停損 = 最高收盤後高水位 - atr_stop * ATR (細網格優化, 平台中心)
    "allow_short": False, # 對稱做空在黃金上驗證為負期望 (拖累 PF 1.83→1.11), 預設關閉
}


def load_bars(path=CSV, completed_only=True):
    """讀取小時 K 線。

    completed_only=True (預設): 只保留「已完成」的整點 (:00) K 棒, 濾掉
      (a) Yahoo 每次抓取附在尾端的『即時快照』半成品棒 (時間戳非整點如 09:56,
          OHLC 退化為單一即時價), 與 (b) 假日/夏令時造成的 :30 等非整點棒。
    這是回測與訊號判斷的正確母體 — 半成品棒會污染突破高點視窗、ATR 與
    最近一根的進出場判定 (詳見 gold_signals/gold_monitor 的修正)。
    """
    o, h, l, c, dt = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = row["date"]
            if completed_only and not ts.endswith(":00"):
                continue
            dt.append(ts)
            o.append(float(row["open"])); h.append(float(row["high"]))
            l.append(float(row["low"]));  c.append(float(row["close"]))
    return dt, np.array(o), np.array(h), np.array(l), np.array(c)


def breakout_level(h, i, bo):
    """『正在形成中的下一根』要突破的多單參考價 = 最近 bo 根『已完成』棒的最高價
    (i 為最後一根已完成棒的索引, 含 i)。供即時監控比對即時價之用。

    注意: 與 signal_breakout 的歷史判定不同 — 後者判斷『已完成棒 i 本身』是否
    突破其『之前』bo 根的高點, 故視窗為 h[i-bo:i] (不含 i)。即時監控判斷的是
    『下一根 (形成中)』, 故參考價需含最近這根已完成棒, 視窗為 h[i-bo+1:i+1]。
    """
    return float(h[i - bo + 1:i + 1].max())


def prev_day_high_low(dt, h, l, i=None):
    """今日固定參考錨點 = 前一交易日(台北日)的最高/最低價, 全天不變。

    供盯盤用的『今天要留意的固定價』(現行滾動突破門檻每根棒會變, 無固定值);
    僅作顯示參考, 不影響滾動突破訊號邏輯。回測比較顯示『前1日高突破』
    風險報酬略遜滾動版 (MAR 6.81 vs 9.40), 故訊號仍用滾動, 此處只給錨點。
    回傳 (前一日日期, 前一日高, 前一日低) 或 (None, None, None)。
    """
    if i is None:
        i = len(h) - 1
    cur_day = dt[i][:10]
    prev_day = None
    for j in range(i, -1, -1):
        if dt[j][:10] != cur_day:
            prev_day = dt[j][:10]
            break
    if prev_day is None:
        return None, None, None
    hi = lo = None
    for k in range(len(dt)):
        if dt[k][:10] == prev_day:
            hi = h[k] if hi is None else max(hi, h[k])
            lo = l[k] if lo is None else min(lo, l[k])
    return prev_day, float(hi), float(lo)


def atr(h, l, c, n=14):
    """Wilder ATR (簡化遞迴版, 與 gold_optimize.py 一致)。"""
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    a = np.full(len(c), np.nan)
    if len(c) >= n:
        a[n - 1] = tr[:n].mean()
        for i in range(n, len(c)):
            a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def signal_breakout(h, l, c, i, cfg=CONFIG, a=None):
    """順勢突破訊號 — 收盤突破過去 breakout 小時高點 (不含當小時) 即做多.

    對稱跌破做空 (allow_short=True 時) 同邏輯反向。回傳進場方向 (+1/-1/None);
    出場由呼叫端以 ATR 移動停損逐根追蹤 (見 backtest())。
    """
    bo = cfg["breakout"]
    if i < bo + cfg["atr_n"] + 1:
        return None
    if a is None:
        a = atr(h, l, c, cfg["atr_n"])
    if np.isnan(a[i]):
        return None
    hh = h[i - bo:i].max()
    ll = l[i - bo:i].min()
    if c[i] > hh:
        return 1
    if cfg["allow_short"] and c[i] < ll:
        return -1
    return None


def backtest(cfg=CONFIG, csv_path=CSV):
    """單口、無加碼的順勢突破回測 (ATR 移動停損), 回傳逐筆交易清單。"""
    dt, o, h, l, c = load_bars(csv_path)
    n = len(c)
    a = atr(h, l, c, cfg["atr_n"])
    pos = 0
    entry = 0.0
    trail = 0.0
    trades = []

    def close(exit_px, j):
        nonlocal pos
        pnl = (exit_px - entry) * pos * POINT_VALUE - COST_PER_SIDE * POINT_VALUE * 2
        trades.append({"dir": pos, "entry": entry, "exit": exit_px,
                       "pnl": pnl, "entry_date": dt[entry_i], "exit_date": dt[j]})
        pos = 0

    entry_i = 0
    for i in range(cfg["breakout"] + cfg["atr_n"] + 1, n):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            sig = signal_breakout(h, l, c, i, cfg, a)
            if sig == 1:
                pos, entry, entry_i = 1, px, i
                trail = px - cfg["atr_stop"] * a[i]
            elif sig == -1:
                pos, entry, entry_i = -1, px, i
                trail = px + cfg["atr_stop"] * a[i]
        elif pos == 1:
            trail = max(trail, px - cfg["atr_stop"] * a[i])
            if px <= trail:
                close(px, i)
        else:
            trail = min(trail, px + cfg["atr_stop"] * a[i])
            if px >= trail:
                close(px, i)
    if pos:
        close(c[-1], n - 1)
    return trades


def metrics(trades):
    if not trades:
        return None
    p = np.array([t["pnl"] for t in trades])
    w = p[p > 0]; ls = p[p < 0]
    gp = w.sum(); gl = -ls.sum()
    eq = np.cumsum(p)
    dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) else 0.0
    return {"n": len(trades), "win": len(w) / len(trades),
            "pf": gp / gl if gl > 0 else float("inf"),
            "total": p.sum(), "dd": dd,
            "mar": (eq[-1] / dd) if dd > 0 else float("inf")}


def yearly(trades):
    """逐年績效拆解 (以出場年份歸戶) — 看策略在不同行情/regime的表現分布。
    回傳 [(年, 筆數, 總損益, 勝率), ...] 依年份排序。"""
    by = {}
    for t in trades:
        y = t["exit_date"][:4]
        by.setdefault(y, []).append(t["pnl"])
    out = []
    for y in sorted(by):
        p = np.array(by[y])
        out.append((y, len(p), float(p.sum()), float((p > 0).mean())))
    return out


def yearly_line(trades):
    """精簡單行逐年損益 (供 /gold 等通知用), 例: '2024:+39k 2025:+105k 2026:+65k'。"""
    return "  ".join(f"{y}:{tot/1000:+.0f}k" for y, _, tot, _ in yearly(trades))


def main():
    trades = backtest()
    m = metrics(trades)
    print(f"黃金順勢突破 (breakout={CONFIG['breakout']}, atr_stop={CONFIG['atr_stop']}, "
          f"allow_short={CONFIG['allow_short']})")
    print(f"交易 {m['n']}  勝率 {m['win']:.1%}  PF {m['pf']:.2f}  "
          f"總損益 ${m['total']:+,.0f}  最大DD ${m['dd']:,.0f}  MAR {m['mar']:.2f}")
    print("\n逐年績效拆解 (不同行情下的表現分布):")
    for y, n, tot, win in yearly(trades):
        bar = "█" * int(abs(tot) / 3000)
        print(f"  {y}  {tot:>+10,.0f}  ({n:>3}筆, 勝率{win:>5.1%})  "
              f"{'+' if tot >= 0 else '-'}{bar}")


if __name__ == "__main__":
    main()
