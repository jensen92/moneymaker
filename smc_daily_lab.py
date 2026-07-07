"""SMC 方法論的日線規律探勘與策略回測 (TWII 1997-2026, 29年)。

與 daily_lab.py (K線型態/日曆效應) 不同, 本檔用 SMC 的語言找規律:
結構 (CHoCH/BOS)、流動性 (掃蕩前低/前高後收回)、失衡 (FVG)、訂單區塊回踩。

Part A SMC 事件研究: 每種結構事件發生後的 1/5/10/20 日前瞻報酬,
  附 t 值與「前半 1997-2011 / 後半 2012-2026」穩定性 (同號才採信)。
Part B 策略回測: 由穩定事件構成的日線 SMC 策略。訊號收盤決定、次日開盤
  執行, 成本來回 2 點, 點數 + %複利雙計量, 樣本內外 70/30、逐年、敏感度。

事件定義 (pivot length L=5, 轉折需 L 日後確認, 無未來函數):
  bull/bear CHoCH : 收盤突破前轉折高/低 → 結構翻轉
  bull/bear BOS   : 趨勢方向上收盤突破「新確認的」轉折高/低 (續勢)
  sweep low/high  : 盤中刺破最近已確認轉折低/高, 收盤收回其上/下
                    (SMC 流動性掃蕩 / 假突破獵殺停損)
  FVG bull/bear   : low[i] > high[i-2] / high[i] < low[i-2] (三根K失衡)
  OB retest       : 多頭 CHoCH 後 30 日內首次回踩訂單區塊上緣
                    (前轉折低 + 0.5*ATR)

用法: python3 smc_daily_lab.py            # Part A + Part B
      python3 smc_daily_lab.py --events   # 只跑事件研究
"""
import datetime as dt
import sys

from daily_lab import load
from smc_backtest import atr_series, pivot_flags
from smc_lab import fmt, metrics, simulate

COST = 2.0
L = 5  # pivot length


# ================================================================ 事件掃描
def scan_events(bars, length=L):
    """回傳 dict: 事件名 -> [bar index]; 以及每日結構趨勢狀態 trend[i]。"""
    n = len(bars)
    phs, pls = pivot_flags(bars, length)
    atr = atr_series(bars)
    c = [b[4] for b in bars]
    h = [b[2] for b in bars]
    l = [b[3] for b in bars]

    ev = {k: [] for k in ("bull_choch", "bear_choch", "bull_bos", "bear_bos",
                          "sweep_low", "sweep_high", "fvg_bull", "fvg_bear",
                          "ob_retest")}
    trend_state = [0] * n

    last_ph = last_pl = None
    conf_ph = conf_pl = None
    trend = 0
    bos_lvl = None
    ob_top = None       # 多頭 CHoCH 後的訂單區塊上緣
    ob_deadline = -1
    ob_armed = False

    for i in range(n):
        if phs[i] is not None:
            last_ph = phs[i]
            conf_ph = phs[i]
            if trend == 1:
                bos_lvl = phs[i]
        if pls[i] is not None:
            last_pl = pls[i]
            conf_pl = pls[i]
            if trend == -1:
                bos_lvl = pls[i]

        a = atr[i]

        # 流動性掃蕩: 刺破已確認轉折但收回 (排除當日確認的轉折避免自我參照)
        if conf_pl is not None and l[i] < conf_pl and c[i] > conf_pl and pls[i] is None:
            ev["sweep_low"].append(i)
        if conf_ph is not None and h[i] > conf_ph and c[i] < conf_ph and phs[i] is None:
            ev["sweep_high"].append(i)

        # FVG 三根失衡
        if i >= 2 and l[i] > h[i - 2]:
            ev["fvg_bull"].append(i)
        if i >= 2 and h[i] < l[i - 2]:
            ev["fvg_bear"].append(i)

        # CHoCH
        bull = trend <= 0 and last_ph is not None and last_pl is not None and c[i] > last_ph
        bear = trend >= 0 and last_pl is not None and last_ph is not None and c[i] < last_pl
        if bull:
            ev["bull_choch"].append(i)
            trend = 1
            bos_lvl = None
            if a is not None and last_pl is not None:
                ob_top = last_pl + 0.5 * a
                ob_deadline = i + 30
                ob_armed = True
            last_ph = None
        elif bear:
            ev["bear_choch"].append(i)
            trend = -1
            bos_lvl = None
            ob_armed = False
            last_pl = None
        else:
            # BOS 續勢
            if bos_lvl is not None:
                if trend == 1 and c[i] > bos_lvl:
                    ev["bull_bos"].append(i)
                    bos_lvl = None
                elif trend == -1 and c[i] < bos_lvl:
                    ev["bear_bos"].append(i)
                    bos_lvl = None

        # OB 回踩 (多頭結構內首次觸及區塊上緣)
        if ob_armed and trend == 1 and i <= ob_deadline and l[i] <= ob_top:
            ev["ob_retest"].append(i)
            ob_armed = False
        if i > ob_deadline:
            ob_armed = False

        trend_state[i] = trend
    return ev, trend_state


def event_study(bars, ev, trend_state):
    n = len(bars)
    c = [b[4] for b in bars]
    half = n // 2

    def fwd(idx, k):
        r = [c[min(i + k, n - 1)] / c[i] - 1 for i in idx if i + 1 < n]
        return r

    def line(label, idx):
        idx = [i for i in idx if i + 1 < n and i >= 200]
        if len(idx) < 25:
            print(f"  {label:<26} n={len(idx):>5}  (樣本不足)")
            return
        r1 = fwd(idx, 1)
        m1 = sum(r1) / len(r1)
        sd = (sum((x - m1) ** 2 for x in r1) / len(r1)) ** 0.5
        t = m1 / (sd / len(r1) ** 0.5) if sd > 0 else 0
        win = sum(1 for x in r1 if x > 0) / len(r1) * 100
        parts = []
        for k in (5, 10, 20):
            rk = fwd(idx, k)
            parts.append(f"{k}日{sum(rk)/len(rk)*100:+.2f}%")
        a = [x for i, x in zip(idx, r1) if i < half]
        b = [x for i, x in zip(idx, r1) if i >= half]
        st = "✓" if a and b and (sum(a) > 0) == (sum(b) > 0) else "✗"
        ma = sum(a) / len(a) * 100 if a else float("nan")
        mb = sum(b) / len(b) * 100 if b else float("nan")
        print(f"  {label:<26} n={len(idx):>5}  次日{m1*100:+.3f}% (t={t:+.1f}) "
              f"勝{win:4.1f}%  {'  '.join(parts)}  前{ma:+.3f}/後{mb:+.3f} {st}")

    base = [i for i in range(200, n - 1)]
    print(f"全樣本基準: 次日 {sum(c[i+1]/c[i]-1 for i in base)/len(base)*100:+.3f}%"
          f"  5日 {sum(c[min(i+5,n-1)]/c[i]-1 for i in base)/len(base)*100:+.2f}%")

    print("\n[結構趨勢 regime] (SMC 純結構定義, 無均線)")
    line("結構多頭 (trend=+1) 日", [i for i in base if trend_state[i] == 1])
    line("結構空頭 (trend=-1) 日", [i for i in base if trend_state[i] == -1])

    print("\n[結構事件]")
    line("多頭 CHoCH (翻多日)", ev["bull_choch"])
    line("空頭 CHoCH (翻空日)", ev["bear_choch"])
    line("多頭 BOS (續勢突破)", ev["bull_bos"])
    line("空頭 BOS (續勢跌破)", ev["bear_bos"])

    print("\n[流動性掃蕩] (刺破已確認轉折後收回)")
    line("掃蕩前低收回 (假破底)", ev["sweep_low"])
    line("掃蕩前高收回 (假破頂)", ev["sweep_high"])
    ts = trend_state
    line("假破底 且結構多頭", [i for i in ev["sweep_low"] if ts[i] == 1])
    line("假破底 且結構空頭", [i for i in ev["sweep_low"] if ts[i] == -1])

    print("\n[FVG 三根失衡]")
    line("看漲 FVG", ev["fvg_bull"])
    line("看跌 FVG", ev["fvg_bear"])
    line("看漲 FVG 且結構多頭", [i for i in ev["fvg_bull"] if ts[i] == 1])

    print("\n[訂單區塊回踩] (多頭 CHoCH 後 30 日內首觸)")
    line("OB 回踩日", ev["ob_retest"])


# ================================================================ 策略
def sim_structure_regime(bars, length=L):
    """D1 純結構 regime: 多頭 CHoCH 進場持有, 空頭 CHoCH 出場 (無均線/無停損)。"""
    ev, ts = scan_events(bars, length)
    flips = sorted([(i, 1) for i in ev["bull_choch"]] +
                   [(i, -1) for i in ev["bear_choch"]])
    trades, curve = [], []
    pos, eq = None, 0.0
    fi = 0
    pend = None
    for i, b in enumerate(bars):
        if pend == "in" and pos is None:
            pos = dict(entry=b[1], dt=b[0])
        elif pend == "out" and pos is not None:
            pts = b[1] - pos["entry"] - COST
            eq += pts
            trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                               exit_dt=b[0], exit=b[1], reason="choch", pts=pts))
            pos = None
        pend = None
        while fi < len(flips) and flips[fi][0] == i:
            pend = "in" if flips[fi][1] == 1 else "out"
            fi += 1
        curve.append(eq + (b[4] - pos["entry"] if pos else 0.0))
    if pos is not None:
        pts = bars[-1][4] - pos["entry"] - COST
        eq += pts
        trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                           exit_dt=bars[-1][0], exit=bars[-1][4],
                           reason="eod", pts=pts))
        curve[-1] = eq
    return trades, curve


def sim_sweep(bars, length=L, max_hold=10, bull_only=True):
    """D3 流動性掃蕩反轉: 假破底收回 → 次日開盤買進;
    停損 = 掃蕩日低點; 出場 = 持有 max_hold 日 或 停損。"""
    ev, ts = scan_events(bars, length)
    sweeps = set(ev["sweep_low"])
    trades, curve = [], []
    pos, eq, pend = None, 0.0, None
    for i, b in enumerate(bars):
        d_, o, h, l, c = b
        # 停損/時間出場 (以前一日收盤掛好的狀態)
        if pos is not None and pos["bar"] < i:
            if o <= pos["stop"]:
                pts = o - pos["entry"] - COST
                eq += pts
                trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                                   exit_dt=d_, exit=o, reason="stop", pts=pts))
                pos = None
            elif l <= pos["stop"]:
                pts = pos["stop"] - pos["entry"] - COST
                eq += pts
                trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                                   exit_dt=d_, exit=pos["stop"], reason="stop", pts=pts))
                pos = None
        if pos is not None and pend == "time_out":
            pts = o - pos["entry"] - COST
            eq += pts
            trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                               exit_dt=d_, exit=o, reason="time", pts=pts))
            pos = None
        if pos is None and pend and pend != "time_out":
            pos = dict(entry=o, dt=d_, stop=pend, bar=i, held=0)
        pend = None
        # 收盤訊號
        if pos is None:
            if i in sweeps and (not bull_only or ts[i] == 1):
                pend = l  # 停損掛掃蕩日低點
        else:
            pos["held"] += 1
            if pos["held"] >= max_hold:
                pend = "time_out"
        curve.append(eq + (c - pos["entry"] if pos else 0.0))
    if pos is not None:
        pts = bars[-1][4] - pos["entry"] - COST
        eq += pts
        trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                           exit_dt=bars[-1][0], exit=bars[-1][4],
                           reason="eod", pts=pts))
        curve[-1] = eq
    return trades, curve


def sim_fvg(bars, length=L, max_hold=20, bull_only=True):
    """D4 FVG 動能: 收盤形成看漲 FVG (low[i] > high[i-2]) 且結構多頭
    → 次日開盤買進; 出場 = 收盤跌破 FVG 下緣 (失衡被回補) 或持有 max_hold 日。"""
    ev, ts = scan_events(bars, length)
    fvg_days = set(ev["fvg_bull"])
    h = [b[2] for b in bars]
    trades, curve = [], []
    pos, eq, pend = None, 0.0, None
    for i, b in enumerate(bars):
        d_, o, hh_, ll_, c = b
        if pend == "out" and pos is not None:
            pts = o - pos["entry"] - COST
            eq += pts
            trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                               exit_dt=d_, exit=o, reason="fvg", pts=pts))
            pos = None
        elif pend == "in" and pos is None:
            pos = dict(entry=o, dt=d_, floor=pend_floor[0], held=0)
        pend = None
        if pos is None:
            if i in fvg_days and (not bull_only or ts[i] == 1):
                pend = "in"
                pend_floor = (h[i - 2],)   # FVG 下緣
        else:
            pos["held"] += 1
            if c < pos["floor"] or pos["held"] >= max_hold:
                pend = "out"
        curve.append(eq + (c - pos["entry"] if pos else 0.0))
    if pos is not None:
        pts = bars[-1][4] - pos["entry"] - COST
        eq += pts
        trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                           exit_dt=bars[-1][0], exit=bars[-1][4],
                           reason="eod", pts=pts))
        curve[-1] = eq
    return trades, curve


def pct_stats(trades, bars):
    """%複利 + 持倉占比 + %MaxDD (mark-to-market)。"""
    yrs = len(bars) / 245
    eq = 1.0
    expo = 0.0
    for x in trades:
        eq *= 1 + (x["exit"] / x["entry"] - 1) - 0.0002
        d0 = dt.date(*map(int, x["entry_dt"].split("-")))
        d1 = dt.date(*map(int, x["exit_dt"].split("-")))
        expo += max((d1 - d0).days * 5 / 7, 1)
    cagr = eq ** (1 / yrs) - 1
    # %MaxDD
    tr = sorted(trades, key=lambda x: x["entry_dt"])
    ti, active, e = 0, None, 1.0
    peak, dd = 0.0, 0.0
    for b in bars:
        while ti < len(tr) and tr[ti]["entry_dt"] == b[0]:
            active = tr[ti]
            ti += 1
        if active and b[0] == active["exit_dt"]:
            e *= 1 + (active["exit"] / active["entry"] - 1) - 0.0002
            active = None
        cur = e * (b[4] / active["entry"] if active else 1.0)
        peak = max(peak, cur)
        dd = max(dd, 1 - cur / peak if peak > 0 else 0)
    return cagr, dd, expo / (yrs * 245)


def yearly(trades):
    out = {}
    for t in trades:
        y = t["exit_dt"][:4]
        out[y] = out.get(y, 0.0) + t["pts"]
    return out


# ================================================================ 主流程
def main():
    bars = load()
    print("=" * 104)
    print(f"SMC 日線規律探勘 (結構/流動性/失衡/訂單區塊)  "
          f"{bars[0][0]} ~ {bars[-1][0]}  ({len(bars)} 交易日, L={L})")
    print("=" * 104)
    ev, ts = scan_events(bars)
    event_study(bars, ev, ts)
    if "--events" in sys.argv:
        return

    print("\n" + "=" * 104)
    print("Part B 日線 SMC 策略回測 (訊號收盤決定、次日開盤執行, 成本來回2點, 只做多)")
    print("=" * 104)
    c = [b[4] for b in bars]
    peak, bdd = 0.0, 0.0
    for x in c:
        peak = max(peak, x)
        bdd = max(bdd, 1 - x / peak)
    yrs = len(bars) / 245
    print(f"buy&hold 對照: {c[-1]-c[0]:+,.0f}點  CAGR {((c[-1]/c[0])**(1/yrs)-1)*100:+.1f}%  "
          f"%MaxDD {bdd*100:.1f}%")

    sims = [
        ("D1 純結構regime (CHoCH開關)", lambda b: sim_structure_regime(b)),
        ("D2 BOS+吊燈 (smc_lab勝出組)", lambda b: simulate(b, "bos", "atrail",
                                                      allow_short=False)),
        ("D3 假破底反轉 (結構多頭內)", lambda b: sim_sweep(b)),
        ("D3b 假破底反轉 (不限結構)", lambda b: sim_sweep(b, bull_only=False)),
        ("D4 FVG動能 (結構多頭內)", lambda b: sim_fvg(b)),
        ("D4b FVG動能 (不限結構)", lambda b: sim_fvg(b, bull_only=False)),
    ]
    results = {}
    for name, fn in sims:
        t, cv = fn(bars)
        results[name] = t
        cagr, dd, expo = pct_stats(t, bars)
        print(fmt(name, metrics(t, cv)))
        print(f"    └ %複利: CAGR {cagr*100:+.1f}%  %MaxDD {dd*100:.1f}%  "
              f"持倉占比 {expo*100:.1f}%")

    print("\n### 樣本內外 70/30")
    cut = int(len(bars) * 0.7)
    for name, fn in sims:
        for tag, seg in (("內70%", bars[:cut]), ("外30%", bars[cut:])):
            print(fmt(f"  {name} {tag}", metrics(*fn(seg))))

    print("\n### 參數敏感度 (pivot length)")
    for name, mk in (("D1", lambda b, LL: sim_structure_regime(b, length=LL)),
                     ("D2", lambda b, LL: simulate(b, "bos", "atrail",
                                                   length=LL, allow_short=False)),
                     ("D3", lambda b, LL: sim_sweep(b, length=LL))):
        for LL in (3, 5, 8):
            print(fmt(f"  {name} L={LL}", metrics(*mk(bars, LL))))

    print("\n### D3 持有天數敏感度")
    for mh in (5, 10, 15, 20):
        print(fmt(f"  D3 hold={mh}", metrics(*sim_sweep(bars, max_hold=mh))))

    print("\n### D4 敏感度 (max_hold × pivot L)")
    for mh in (10, 20, 40):
        for LL in (3, 5, 8):
            print(fmt(f"  D4 hold={mh} L={LL}",
                      metrics(*sim_fvg(bars, length=LL, max_hold=mh))))

    for name in ("D1 純結構regime (CHoCH開關)", "D2 BOS+吊燈 (smc_lab勝出組)",
                 "D3 假破底反轉 (結構多頭內)", "D4 FVG動能 (結構多頭內)"):
        yr = yearly(results[name])
        neg = sum(1 for v in yr.values() if v < 0)
        row = "  ".join(f"{y[2:]}:{v:+,.0f}" for y, v in sorted(yr.items()))
        print(f"\n### 逐年 ({name})  虧損年 {neg}/{len(yr)}")
        print("  " + row)


if __name__ == "__main__":
    main()
