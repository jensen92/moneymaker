"""TWII 日線規律探勘與策略回測 (1997-2026, 29年)。

流程與誠實揭露:
- Part A 規律統計: 規律在全樣本探勘, 天生有 in-sample 偏誤。緩解: 每條規律
  同時列出「前半 (1997-2011) / 後半 (2012-2026)」的次日均報酬, 只採信
  前後半同號且量級相近的規律; 單邊漂亮另一邊翻臉的視為雜訊。
- Part B 策略回測: 由穩定規律構成的候選策略, 訊號收盤決定、次日開盤執行,
  成本來回 2 點 (TXF 手續費+稅+滑價概算), 點數計量 + mark-to-market 回撤。
  附樣本內外 70/30、逐年損益、參數敏感度。
- 標的以 ^TWII 現貨代理 TXF: 無夜盤/基差/結算, 隔夜跳空以次日開盤實價成交
  (代理已含跳空), 但實單期貨跳空可能更劇烈, 數字當上限看。

用法: python3 daily_lab.py            # Part A + Part B
      python3 daily_lab.py --patterns # 只跑規律統計
"""
import csv
import datetime as dt
import math
import os
import sys

from smc_lab import fmt, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "futures_data", "TWII_1d_max.csv")


def fetch():
    """Yahoo range=max 會被降成月線, 故用 period1/period2 每5年分段抓日線。"""
    import time

    import requests
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII"
    all_rows = {}
    seg = dt.datetime(1997, 1, 1)
    end = dt.datetime.now() + dt.timedelta(days=1)
    while seg < end:
        seg_end = min(seg + dt.timedelta(days=1825), end)
        r = requests.get(url, params={"period1": int(seg.timestamp()),
                                      "period2": int(seg_end.timestamp()),
                                      "interval": "1d"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        ts, q = res.get("timestamp") or [], res["indicators"]["quote"][0]
        for i, t in enumerate(ts):
            o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            if None in (o, h, l, c) or not c or c <= 0 or h < l:
                continue
            d = time.strftime("%Y-%m-%d", time.gmtime(t + 8 * 3600))
            all_rows[d] = (d, round(o, 2), round(h, 2), round(l, 2), round(c, 2))
        seg = seg_end
        time.sleep(0.5)
    rows = [all_rows[k] for k in sorted(all_rows)]
    if not rows:
        raise RuntimeError("Yahoo 回空資料, 不覆寫既有檔")
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close"])
        w.writerows(rows)
    return len(rows)


def load():
    if not os.path.exists(CSV):
        print(f"抓取 ^TWII 日線 1997~ ... ({fetch()} 筆)")
    bars = []
    with open(CSV) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                bars.append((row[0], float(row[1]), float(row[2]),
                             float(row[3]), float(row[4])))
    return bars


# ================================================================ Part A
def sma(vals, n):
    out = [None] * len(vals)
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= n:
            s -= vals[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def pattern_stats(bars):
    n = len(bars)
    dates = [b[0] for b in bars]
    o = [b[1] for b in bars]
    h = [b[2] for b in bars]
    l = [b[3] for b in bars]
    c = [b[4] for b in bars]
    ret = [None] + [c[i] / c[i - 1] - 1 for i in range(1, n)]
    ma200 = sma(c, 200)
    ibs = [(c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.5 for i in range(n)]
    wd = [dt.date(*map(int, d.split("-"))).weekday() for d in dates]
    month = [int(d[5:7]) for d in dates]

    half = n // 2

    def stat(idx, label):
        """idx: 訊號日 i 的集合, 統計次日報酬 (i+1) 與後5日。"""
        idx = [i for i in idx if i + 1 < n]
        if len(idx) < 30:
            return
        r1 = [ret[i + 1] for i in idx]
        r5 = [c[min(i + 5, n - 1)] / c[i] - 1 for i in idx]
        m1 = sum(r1) / len(r1)
        win = sum(1 for x in r1 if x > 0) / len(r1) * 100
        sd = (sum((x - m1) ** 2 for x in r1) / len(r1)) ** 0.5
        t = m1 / (sd / len(r1) ** 0.5) if sd > 0 else 0
        a = [ret[i + 1] for i in idx if i < half]
        b = [ret[i + 1] for i in idx if i >= half]
        ma = sum(a) / len(a) * 100 if a else float("nan")
        mb = sum(b) / len(b) * 100 if b else float("nan")
        stable = "✓" if a and b and (sum(a) > 0) == (sum(b) > 0) else "✗"
        print(f"  {label:<34} n={len(idx):>5}  次日{m1*100:+.3f}% (t={t:+.1f}) "
              f"勝{win:4.1f}%  5日{sum(r5)/len(r5)*100:+.2f}%  "
              f"前半{ma:+.3f}/後半{mb:+.3f} {stable}")

    base = [i for i in range(200, n - 1)]
    print(f"全樣本基準: 次日均報酬 {sum(ret[i+1] for i in base)/len(base)*100:+.3f}%"
          f"  (n={len(base)}, {dates[0]}~{dates[-1]})")

    print("\n[1] 週間效應 (訊號日=某星期, 統計次日)")
    for d, name in enumerate(["一", "二", "三", "四", "五"]):
        stat([i for i in base if wd[i + 1] == d], f"星期{name} 當日")

    print("\n[2] 月份季節性 (該月所有日的次日報酬)")
    for m in range(1, 13):
        stat([i for i in base if month[i + 1] == m], f"{m}月")

    print("\n[3] 月轉換效應 (月底2日+月初3日 vs 其他)")
    tom = set()
    for i in range(1, n):
        if month[i] != month[i - 1]:          # 月初第1日
            for k in range(0, 3):
                if i + k < n:
                    tom.add(i + k)
            for k in range(1, 3):             # 前月最後2日
                if i - k >= 0:
                    tom.add(i - k)
    stat([i for i in base if i + 1 in tom], "月轉換窗 (末2+初3)")
    stat([i for i in base if i + 1 not in tom], "非月轉換窗")

    print("\n[4] 連續漲跌後 (訊號日=連N跌/漲收盤日)")
    def streak(i, sign):
        k = 0
        j = i
        while j >= 1 and ret[j] is not None and (ret[j] > 0) == sign and ret[j] != 0:
            k += 1
            j -= 1
        return k
    for k in (2, 3, 4):
        stat([i for i in base if streak(i, False) >= k], f"連跌≥{k}日後")
        stat([i for i in base if streak(i, True) >= k], f"連漲≥{k}日後")

    print("\n[5] IBS (收盤在當日區間位置)")
    stat([i for i in base if ibs[i] < 0.2], "IBS<0.2 (收在低檔)")
    stat([i for i in base if ibs[i] > 0.8], "IBS>0.8 (收在高檔)")

    print("\n[6] MA200 趨勢 regime")
    stat([i for i in base if c[i] > ma200[i]], "收盤 > MA200")
    stat([i for i in base if c[i] <= ma200[i]], "收盤 <= MA200")

    print("\n[7] 動能視窗 (過去N日漲跌 → 次日)")
    for w in (5, 20, 60, 120):
        stat([i for i in base if i >= w and c[i] > c[i - w]], f"過去{w}日上漲")
        stat([i for i in base if i >= w and c[i] < c[i - w]], f"過去{w}日下跌")

    print("\n[8] 突破 (收盤創20日新高/新低)")
    stat([i for i in base if i >= 20 and c[i] >= max(c[i - 20:i + 1])], "創20日新高")
    stat([i for i in base if i >= 20 and c[i] <= min(c[i - 20:i + 1])], "創20日新低")

    print("\n[9] 大跌日後 (單日 < -2%)")
    stat([i for i in base if ret[i] is not None and ret[i] < -0.02], "單日跌>2%後")
    stat([i for i in base if ret[i] is not None and ret[i] < -0.02
          and c[i] > ma200[i]], "跌>2%後 且在MA200上")

    print("\n[10] 組合: 多頭 regime 內的短線超賣")
    stat([i for i in base if c[i] > ma200[i] and ibs[i] < 0.2], "MA200上 + IBS<0.2")
    stat([i for i in base if c[i] > ma200[i] and streak(i, False) >= 3], "MA200上 + 連跌3日")
    stat([i for i in base if c[i] <= ma200[i] and ibs[i] < 0.2], "MA200下 + IBS<0.2")


# ================================================================ Part B
COST = 2.0  # 來回點數


def _finish(bars, trades, curve, pos, eq):
    if pos is not None:
        pts = bars[-1][4] - pos["entry"] - COST
        eq += pts
        trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                           exit_dt=bars[-1][0], exit=bars[-1][4],
                           reason="eod", pts=pts))
        curve[-1] = eq
    return trades, curve


def sim_regime(bars, ma_n=200):
    """S1 趨勢 regime: 收盤>MA200 持有, 跌破出場。"""
    c = [b[4] for b in bars]
    ma = sma(c, ma_n)
    trades, curve = [], []
    pos, eq, sig = None, 0.0, None
    for i, b in enumerate(bars):
        if sig == "in" and pos is None:
            pos = dict(entry=b[1], dt=b[0])
        elif sig == "out" and pos is not None:
            pts = b[1] - pos["entry"] - COST
            eq += pts
            trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                               exit_dt=b[0], exit=b[1], reason="regime", pts=pts))
            pos = None
        sig = None
        if ma[i] is not None:
            if pos is None and c[i] > ma[i]:
                sig = "in"
            elif pos is not None and c[i] <= ma[i]:
                sig = "out"
        curve.append(eq + (c[i] - pos["entry"] if pos else 0.0))
    return _finish(bars, trades, curve, pos, eq)


def sim_ibs(bars, ibs_in=0.2, ibs_out=0.8, max_hold=5, ma_filter=True, ma_n=200):
    """S2 多頭內短線超賣: MA200上 + IBS<門檻 買進;
    IBS>出場門檻 或 持有滿 max_hold 日 出場。"""
    c = [b[4] for b in bars]
    ma = sma(c, ma_n)
    trades, curve = [], []
    pos, eq, sig = None, 0.0, None
    for i, b in enumerate(bars):
        if sig == "in" and pos is None:
            pos = dict(entry=b[1], dt=b[0], held=0)
        elif sig == "out" and pos is not None:
            pts = b[1] - pos["entry"] - COST
            eq += pts
            trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                               exit_dt=b[0], exit=b[1], reason="ibs", pts=pts))
            pos = None
        sig = None
        _, o_, h_, l_, c_ = b
        cur_ibs = (c_ - l_) / (h_ - l_) if h_ > l_ else 0.5
        trend_ok = (not ma_filter) or (ma[i] is not None and c_ > ma[i])
        if pos is None:
            if trend_ok and cur_ibs < ibs_in:
                sig = "in"
        else:
            pos["held"] += 1
            if cur_ibs > ibs_out or pos["held"] >= max_hold:
                sig = "out"
        curve.append(eq + (c_ - pos["entry"] if pos else 0.0))
    return _finish(bars, trades, curve, pos, eq)


def sim_donchian(bars, n_in=20, n_out=10, ma_filter=True, ma_n=200):
    """S3 突破動能: 收盤創 n_in 日新高買進, 跌破 n_out 日新低出場。"""
    c = [b[4] for b in bars]
    ma = sma(c, ma_n)
    trades, curve = [], []
    pos, eq, sig = None, 0.0, None
    for i, b in enumerate(bars):
        if sig == "in" and pos is None:
            pos = dict(entry=b[1], dt=b[0])
        elif sig == "out" and pos is not None:
            pts = b[1] - pos["entry"] - COST
            eq += pts
            trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                               exit_dt=b[0], exit=b[1], reason="donch", pts=pts))
            pos = None
        sig = None
        if i >= max(n_in, ma_n if ma_filter else 0):
            trend_ok = (not ma_filter) or (ma[i] is not None and c[i] > ma[i])
            if pos is None and trend_ok and c[i] >= max(c[i - n_in:i + 1]):
                sig = "in"
            elif pos is not None and c[i] <= min(c[i - n_out:i]):
                sig = "out"
        curve.append(eq + (c[i] - pos["entry"] if pos else 0.0))
    return _finish(bars, trades, curve, pos, eq)


def buyhold_stats(bars):
    c = [b[4] for b in bars]
    pts = c[-1] - c[0]
    peak, dd = -1e18, 0.0
    for x in c:
        peak = max(peak, x)
        dd = max(dd, peak - x)
    return pts, dd


def yearly(trades):
    out = {}
    for t in trades:
        y = t["exit_dt"][:4]
        out[y] = out.get(y, 0.0) + t["pts"]
    return out


def main():
    bars = load()
    print("=" * 100)
    print(f"TWII 日線規律探勘  {bars[0][0]} ~ {bars[-1][0]}  ({len(bars)} 交易日)")
    print("=" * 100)
    pattern_stats(bars)
    if "--patterns" in sys.argv:
        return

    print("\n" + "=" * 100)
    print("Part B 策略回測 (訊號收盤決定、次日開盤執行, 成本來回2點, 只做多)")
    print("=" * 100)
    bh_pts, bh_dd = buyhold_stats(bars)
    print(f"buy&hold 對照: {bh_pts:+,.0f}點  MaxDD {bh_dd:,.0f}點  "
          f"總/DD {bh_pts/bh_dd:.2f}")

    sims = [
        ("S1 MA200 regime", lambda b: sim_regime(b)),
        ("S2 IBS超賣 (MA200上)", lambda b: sim_ibs(b)),
        ("S2b IBS超賣 (無濾網)", lambda b: sim_ibs(b, ma_filter=False)),
        ("S3 Donchian20/10 (MA200上)", lambda b: sim_donchian(b)),
        ("S3b Donchian20/10 (無濾網)", lambda b: sim_donchian(b, ma_filter=False)),
    ]
    results = {}
    for name, fn in sims:
        t, cv = fn(bars)
        results[name] = (t, cv)
        print(fmt(name, metrics(t, cv)))

    # 樣本內外 70/30
    print("\n### 樣本內外 70/30")
    cut = int(len(bars) * 0.7)
    for name, fn in sims:
        for tag, seg in (("內70%", bars[:cut]), ("外30%", bars[cut:])):
            print(fmt(f"  {name} {tag}", metrics(*fn(seg))))

    # S2 參數敏感度
    print("\n### S2 參數敏感度 (ibs_in × max_hold, MA200濾網)")
    for ii in (0.15, 0.2, 0.25):
        for mh in (3, 5, 8):
            t, cv = sim_ibs(bars, ibs_in=ii, max_hold=mh)
            print(fmt(f"  in<{ii} hold≤{mh}", metrics(t, cv)))

    # 百分比複利統計 (去除指數規模偏誤: 早年指數8000/近年46000,
    # 點數計量會機械性放大近年權重; %複利才能跨時代公平比較)
    print("\n### 百分比複利統計 (每筆 ret% 複利, 成本單邊0.01%)")
    yrs = len(bars) / 245
    for name, (t, cv) in results.items():
        eq = 1.0
        expo_days = 0
        for x in t:
            eq *= 1 + (x["exit"] / x["entry"] - 1) - 0.0002
        for x in t:
            d0 = dt.date(*map(int, x["entry_dt"].split("-")))
            d1 = dt.date(*map(int, x["exit_dt"].split("-")))
            expo_days += max((d1 - d0).days * 5 / 7, 1)
        cagr = eq ** (1 / yrs) - 1
        print(f"  {name:<26} 總報酬 {(eq-1)*100:+9.0f}%  CAGR {cagr*100:+5.1f}%  "
              f"持倉時間占比 {expo_days/(yrs*245)*100:4.1f}%")
    c0, c1 = bars[0][4], bars[-1][4]
    bh_cagr = (c1 / c0) ** (1 / yrs) - 1
    print(f"  {'buy&hold':<26} 總報酬 {(c1/c0-1)*100:+9.0f}%  CAGR {bh_cagr*100:+5.1f}%  "
          f"持倉時間占比 100.0%")

    # 逐年 (最佳者)
    print("\n### 逐年損益 (S2 IBS超賣 MA200上)")
    yr = yearly(results["S2 IBS超賣 (MA200上)"][0])
    neg = sum(1 for v in yr.values() if v < 0)
    for y in sorted(yr):
        print(f"  {y}: {yr[y]:+8.0f}點")
    print(f"  虧損年: {neg}/{len(yr)}")

    print("\n### 逐年損益 (S1 MA200 regime)")
    yr = yearly(results["S1 MA200 regime"][0])
    neg = sum(1 for v in yr.values() if v < 0)
    for y in sorted(yr):
        print(f"  {y}: {yr[y]:+8.0f}點")
    print(f"  虧損年: {neg}/{len(yr)}")


if __name__ == "__main__":
    main()
