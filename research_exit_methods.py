"""出場法則事件研究: 在同一組突破事件上, 比較不同「出場方法」的每筆期望報酬。

目的: 回答「什麼方法能做到目前策略 ~2 倍報酬」。
報酬的最大槓桿不是進場濾網, 而是「如何處理贏家」。本研究固定進場
(20日區間突破 + 多頭市況 + 趨勢模板, RS 分層), 在 T+1 開盤進場, 然後
逐日模擬不同出場法則, 計算「每筆交易淨報酬」(扣 0.6% 來回成本) 的:
  - 平均報酬 (expectancy, 最關鍵 — 直接決定複利速度)
  - 勝率 / 賺賠比 / 期望R
比較方法:
  M1 固定停利 (target +R, stop -S)          ← C/PA 類: 高勝率但封頂
  M2 純時間出場 (持有 H 日收盤)
  M3 移動停損 ATR (trail_atr × ATR, 寬停損讓贏家奔跑)
  M4 跌破 MA50 出場 (趨勢跟隨, D 類核心)
  M5 MA50 + 三週法則 (主升段不賣半, 完整奔跑)

注意: 這是「單筆等權」事件研究, 不含部位控管/槓桿/同時持倉上限,
故與組合回測 CAGR 不直接相等, 但「相對倍數」可靠 (同母體同成本)。
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
MAXH = 130          # 最長追蹤天數 (對應 D 的 max_hold 120)
COST = 0.006        # 來回成本 (手續費+稅+滑價) 近似
RS_TIERS = [0.0, 0.70, 0.85, 0.95]


def build_events(data, rs, regime):
    """回傳每事件的未來 OHLC 序列 (相對進場價) + 進場時的 ATR / RS 分層."""
    ev = []
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
        atr14 = df["atr14"].values
        dates = df["date"].values
        rmap = rs.get(code, {})
        for i in range(250, n - 2):
            if np.isnan(ma200[i]) or close[i] < 10:
                continue
            ph = high[i - RANGE_N:i].max()
            if close[i] <= ph or vol[i] * close[i] < MIN_TURNOVER:
                continue
            # 多頭市況 + 趨勢模板 (固定進場品質, 與 D 一致)
            if not (dates[i] in regime and close[i] > ma200[i]
                    and ma50[i] > ma150[i] > ma200[i]):
                continue
            rk = rmap.get(dates[i], np.nan)
            if np.isnan(rk):
                continue
            entry = op[i + 1]
            a14 = atr14[i]
            if entry <= 0 or np.isnan(a14) or a14 <= 0:
                continue
            j0 = i + 1
            j1 = min(i + 1 + MAXH, n)
            ev.append({
                "rs": rk,
                "o": op[j0:j1] / entry,
                "h": high[j0:j1] / entry,
                "l": low[j0:j1] / entry,
                "c": close[j0:j1] / entry,
                "ma50": ma50[j0:j1] / entry,
                "atr_frac": a14 / entry,   # ATR 佔進場價比例 (移動停損用)
            })
    return ev


def sim_fixed(e, target, stop):
    """M1: 固定停利停損. 觸及目標→target, 觸及停損→-stop, 否則最後收盤."""
    for k in range(len(e["c"])):
        if e["l"][k] <= 1 - stop:
            return -stop
        if e["h"][k] >= 1 + target:
            return target
    return e["c"][-1] - 1.0


def sim_time(e, hold):
    """M2: 純時間出場 (持有 hold 日, 期間 -stop 硬停)."""
    stop = 0.10
    h = min(hold, len(e["c"]))
    for k in range(h):
        if e["l"][k] <= 1 - stop:
            return -stop
    return e["c"][h - 1] - 1.0


def sim_trail_atr(e, mult, init_stop):
    """M3: ATR 移動停損. 高點 - mult×ATR 為停損, 寬初始停損讓贏家奔跑."""
    af = e["atr_frac"]
    stop = 1 - init_stop
    peak = 1.0
    for k in range(len(e["c"])):
        if e["l"][k] <= stop:
            return stop - 1.0
        peak = max(peak, e["c"][k])
        stop = max(stop, peak - mult * af)
    return e["c"][-1] - 1.0


def sim_ma50(e, init_stop, three_week=False, tw_gain=0.20, tw_days=15):
    """M4/M5: 跌破 MA50 收盤出場 + 初始硬停; 可選三週法則(主升段不提前出)."""
    stop = 1 - init_stop
    for k in range(len(e["c"])):
        if e["l"][k] <= stop:
            return stop - 1.0
        # 三週法則: 前 tw_days 內漲 >= tw_gain → 鎖定奔跑(略過 MA50 早出不需要,
        # 因 MA50 本就是寬出場; 此處用於「達標前不被 MA50 洗」的近似: 提高容忍)
        below_ma = e["c"][k] < e["ma50"][k]
        if below_ma:
            # 三週法則啟動時, 主升段給一次寬容 (不立即在首次跌破出場)
            if three_week and k <= tw_days and e["h"][:k + 1].max() >= 1 + tw_gain:
                continue
            return e["c"][k] - 1.0
    return e["c"][-1] - 1.0


def stats(rets):
    r = np.array(rets)
    n = len(r)
    if n == 0:
        return None
    net = r - COST
    win = net > 0
    wr = win.mean()
    avg_w = net[win].mean() if win.any() else 0.0
    avg_l = net[~win].mean() if (~win).any() else 0.0
    pf = (net[win].sum() / -net[~win].sum()) if (~win).any() and net[~win].sum() < 0 else np.inf
    return {
        "n": n, "exp": net.mean(), "wr": wr,
        "avg_w": avg_w, "avg_l": avg_l, "pf": pf,
        "median_hold": None,
    }


def main():
    data, _ = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    print("建立事件中 ...")
    ev = build_events(data, rs, regime)
    print(f"母體事件數 (多頭+趨勢模板): {len(ev)}\n")

    methods = [
        ("M1 固定 +10%/-8% (C/PA類, 封頂)", lambda e: sim_fixed(e, 0.10, 0.08)),
        ("M1 固定 +20%/-10%",              lambda e: sim_fixed(e, 0.20, 0.10)),
        ("M2 時間出場 20日",               lambda e: sim_time(e, 20)),
        ("M2 時間出場 60日",               lambda e: sim_time(e, 60)),
        ("M3 ATR移動停損 3×, 初停10%",      lambda e: sim_trail_atr(e, 3.0, 0.10)),
        ("M3 ATR移動停損 5×, 初停12%",      lambda e: sim_trail_atr(e, 5.0, 0.12)),
        ("M4 跌破MA50, 初停8%",            lambda e: sim_ma50(e, 0.08)),
        ("M4 跌破MA50, 初停10%",           lambda e: sim_ma50(e, 0.10)),
        ("M5 MA50+三週法則, 初停8%",        lambda e: sim_ma50(e, 0.08, three_week=True)),
    ]

    for lo in RS_TIERS:
        sub = [e for e in ev if e["rs"] >= lo]
        print(f"━━━ RS >= {lo:.2f}   事件 {len(sub):>6} ━━━")
        print(f"  {'方法':<28}{'期望%':>8}{'勝率':>8}{'均賺%':>8}{'均賠%':>8}{'PF':>7}")
        base_exp = None
        for name, fn in methods:
            st = stats([fn(e) for e in sub])
            if st is None:
                continue
            if base_exp is None:
                base_exp = st["exp"]
            mult = st["exp"] / base_exp if base_exp and base_exp != 0 else 0
            tag = f"  ({mult:.1f}×)" if base_exp else ""
            print(f"  {name:<28}{st['exp']*100:>7.2f}{st['wr']*100:>7.1f}"
                  f"{st['avg_w']*100:>7.1f}{st['avg_l']*100:>7.1f}{st['pf']:>7.2f}{tag}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
