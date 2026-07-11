"""SMC 正統開發方式 — 由上而下多時間框架 (Top-Down MTF) 策略回測。

SMC 交易員的實際工作流程 (本檔的機械化版本):
  1. HTF 日線: 判定結構偏向 (CHoCH/BOS → trend), 標記需求區 POI:
     - 日線看漲 FVG 區 [high[d-2], low[d]] (事件研究已驗證最強, 見 SMC_DAILY.md)
     - 正統訂單區塊 OB: 多頭突破日 (CHoCH/BOS) 前最後一根陰線的 [low, high]
     POI 失效: 日收盤跌破區塊下緣, 或超過 max_age 日未被觸及。
  2. LTF 60m: 日線偏多 + 價格回落觸及日線 POI → 武裝 (arm);
     武裝期間 60m 收盤向上突破最近已確認轉折高 (LTF 結構翻多確認) → 次根開盤進場。
  3. 停損 = POI 下緣 − 0.5×ATR60; 出場 = 固定 RR 或 ATR 吊燈 (兩種都測)。

只做多 (先前多空拆解空單全負)。日線由 60m 資料就地重採樣 (保證對齊、
只用「已完成的日線」— 無未來函數)。成本來回 2 點。

資料限制 (誠實揭露): TWII 60m 僅 3 年 → 本檔為原型驗證, 樣本不大;
以 GC 60m (2.4年) 交叉檢查 + 參數敏感度緩解。

用法: python3 smc_mtf.py
"""
import math
import os

from smc_backtest import DATA, atr_series, load_csv, pivot_flags
from smc_lab import fmt, metrics, simulate

COST = 2.0


# ---------------------------------------------------------------- 日線層
def resample_daily(bars60):
    """60m → 日線 (只含完整交易日), 回傳 daily bars + 每根60m對應的日索引。"""
    days = []
    day_of_bar = []
    cur = None
    for dt_, o, h, l, c in bars60:
        d = dt_[:10]
        if cur is None or cur[0] != d:
            if cur is not None:
                days.append(tuple(cur))
            cur = [d, o, h, l, c]
        else:
            cur[2] = max(cur[2], h)
            cur[3] = min(cur[3], l)
            cur[4] = c
        day_of_bar.append(len(days))  # 當日 index (尚未完成)
    if cur is not None:
        days.append(tuple(cur))
    return days, day_of_bar


def daily_analysis(days, length=5):
    """回傳 trend[d] (該日收盤後的結構方向) 與 zones_created[d] (該日收盤後
    新增的需求區 POI 清單: dict(bottom, top, kind, day))。"""
    n = len(days)
    phs, pls = pivot_flags(days, length)
    o = [x[1] for x in days]
    h = [x[2] for x in days]
    l = [x[3] for x in days]
    c = [x[4] for x in days]

    trend_after = [0] * n
    zones_created = [[] for _ in range(n)]

    last_ph = last_pl = None
    trend = 0
    bos_lvl = None

    def last_down_candle(i):
        for j in range(i - 1, max(i - 10, -1), -1):
            if c[j] < o[j]:
                return dict(bottom=l[j], top=h[j], kind="ob", day=i)
        return None

    for i in range(n):
        if phs[i] is not None:
            last_ph = phs[i]
            if trend == 1:
                bos_lvl = phs[i]
        if pls[i] is not None:
            last_pl = pls[i]
            if trend == -1:
                bos_lvl = pls[i]

        # 看漲 FVG 需求區
        if i >= 2 and l[i] > h[i - 2]:
            zones_created[i].append(dict(bottom=h[i - 2], top=l[i],
                                         kind="fvg", day=i))

        bull = trend <= 0 and last_ph is not None and last_pl is not None and c[i] > last_ph
        bear = trend >= 0 and last_pl is not None and last_ph is not None and c[i] < last_pl
        if bull:
            trend = 1
            bos_lvl = None
            ob = last_down_candle(i)
            if ob:
                zones_created[i].append(ob)
            last_ph = None
        elif bear:
            trend = -1
            bos_lvl = None
            last_pl = None
        elif bos_lvl is not None and trend == 1 and c[i] > bos_lvl:
            ob = last_down_candle(i)
            if ob:
                zones_created[i].append(ob)
            bos_lvl = None
        trend_after[i] = trend
    return trend_after, zones_created


# ---------------------------------------------------------------- MTF 引擎
def sim_mtf(bars60, poi="both", exit_="rr", rr=2.0, arm_window=30,
            l_daily=5, l60=5, trail_k=3.0, max_age=60, atr_buf=0.5):
    days, day_of = resample_daily(bars60)
    d_trend, d_zones = daily_analysis(days, l_daily)
    phs60, pls60 = pivot_flags(bars60, l60)
    atr60 = atr_series(bars60)
    dclose = [x[4] for x in days]

    active = []            # 生效中的 POI
    conf_ph = None         # 最近已確認 60m 轉折高
    armed = None           # dict(zone, until)
    pend = None            # dict(stop0) → 次根開盤市價
    pos = None
    trades, curve = [], []
    eq = 0.0
    last_done_day = -1     # 已完成的最後一個日索引

    for i, (dt_, o, h, l, c) in enumerate(bars60):
        # ---- 換日: 前一日完成 → 更新日線層 ----
        cur_day = day_of[i]
        if cur_day - 1 > last_done_day:
            for d in range(last_done_day + 1, cur_day):
                for z in d_zones[d]:
                    if poi in ("both", z["kind"]):
                        active.append(z)
                # 失效: 日收盤跌破下緣 / 過齡
                active = [z for z in active
                          if dclose[d] >= z["bottom"] and d - z["day"] <= max_age]
            last_done_day = cur_day - 1

        # ---- 盤中撮合 ----
        if pos is not None and pos["bar"] < i:
            d_, S, T = 1, pos["stop"], pos["target"]
            done = None
            if S is not None and o <= S:
                done = (o, "stop")
            elif S is not None and l <= S:
                done = (S, "stop")
            elif T is not None and o >= T:
                done = (o, "target")
            elif T is not None and h >= T:
                done = (T, "target")
            if done:
                pts = done[0] - pos["entry"] - COST
                eq += pts
                trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                                   exit_dt=dt_, exit=done[0], reason=done[1],
                                   pts=pts))
                pos = None
        if pos is None and pend is not None and pend["bar"] < i:
            pos = dict(entry=o, dt=dt_, stop=pend["stop0"], target=None,
                       hi=o, bar=i)
            if exit_ == "rr":
                risk = o - pend["stop0"]
                pos["target"] = o + risk * rr if risk > 0 else None
            pend = None

        # ---- 收盤邏輯 ----
        prev_conf_ph = conf_ph
        if phs60[i] is not None:
            conf_ph = phs60[i]

        a = atr60[i]
        bias = d_trend[last_done_day] if last_done_day >= 0 else 0

        # 觸及 POI → 武裝 (日線偏多才做)
        if bias == 1 and pos is None and pend is None and active:
            touched = [z for z in active if l <= z["top"]]
            if touched:
                z = max(touched, key=lambda z: z["top"])  # 最近的區
                if c >= z["bottom"]:                       # 收盤守住區才武裝
                    armed = dict(zone=z, until=i + arm_window)
        if armed is not None:
            if i > armed["until"] or c < armed["zone"]["bottom"] or bias != 1:
                armed = None

        # LTF 確認: 武裝中收盤向上突破最近已確認轉折高 (需為當根新突破)
        if (armed is not None and pos is None and pend is None
                and a is not None and prev_conf_ph is not None
                and c > prev_conf_ph and bars60[i - 1][4] <= prev_conf_ph):
            stop0 = armed["zone"]["bottom"] - a * atr_buf
            if stop0 < c:
                pend = dict(stop0=stop0, bar=i)
                active = [z for z in active if z is not armed["zone"]]
                armed = None

        # 吊燈更新
        if pos is not None and exit_ == "atrail" and a is not None:
            pos["hi"] = max(pos["hi"], c)
            cand = pos["hi"] - trail_k * a
            if cand > pos["stop"]:
                pos["stop"] = cand

        curve.append(eq + (c - pos["entry"] if pos else 0.0))

    if pos is not None:
        pts = bars60[-1][4] - pos["entry"] - COST
        eq += pts
        trades.append(dict(dir=1, entry_dt=pos["dt"], entry=pos["entry"],
                           exit_dt=bars60[-1][0], exit=bars60[-1][4],
                           reason="eod", pts=pts))
        curve[-1] = eq
    return trades, curve


# ---------------------------------------------------------------- 主流程
def run_market(name, bars, cost_note=""):
    print(f"\n### {name}  ({bars[0][0]} .. {bars[-1][0]}, {len(bars)} bars"
          f"{cost_note})")
    c0, c1 = bars[0][4], bars[-1][4]
    print(f"  buy&hold {c1-c0:+,.0f}點")
    for poi in ("fvg", "ob", "both"):
        for ex, tag in (("rr", "RR2.0"), ("atrail", "吊燈k3")):
            t, cv = sim_mtf(bars, poi=poi, exit_=ex)
            print(fmt(f"  MTF {poi:<4} + {tag}", metrics(t, cv)))
    # 對照: 60m 單框架勝出組 (bos+atrail 只多)
    t, cv = simulate(bars, "bos", "atrail", allow_short=False)
    print(fmt("  (對照) 60m bos+吊燈", metrics(t, cv)))


def main():
    p60 = load_csv(os.path.join(DATA, "TWII_60m.csv"))
    print("=" * 104)
    print("SMC Top-Down 多時間框架回測 — 日線定偏向+POI (FVG/OB), 60m 回落確認進場")
    print("=" * 104)
    run_market("TWII 60m", p60)

    pgc = os.path.join(DATA, "GC_60m.csv")
    if os.path.exists(pgc):
        run_market("GC 60m (交叉檢查)", load_csv(pgc))

    # 敏感度 (主市場, both+rr 與 both+atrail)
    print("\n### 參數敏感度 (TWII, poi=both)")
    for ex in ("rr", "atrail"):
        for aw in (15, 30, 60):
            for ld in (3, 5):
                t, cv = sim_mtf(p60, poi="both", exit_=ex, arm_window=aw,
                                l_daily=ld)
                print(fmt(f"  {ex} arm={aw} Ld={ld}", metrics(t, cv)))
    print("\n### RR 敏感度 (TWII, poi=both, exit=rr)")
    for r in (1.5, 2.0, 3.0):
        t, cv = sim_mtf(p60, poi="both", exit_="rr", rr=r)
        print(fmt(f"  RR={r}", metrics(t, cv)))

    # 樣本內外
    print("\n### 樣本內外 70/30 (TWII, poi=both)")
    cut = int(len(p60) * 0.7)
    for ex in ("rr", "atrail"):
        for tag, seg in (("內70%", p60[:cut]), ("外30%", p60[cut:])):
            t, cv = sim_mtf(seg, poi="both", exit_=ex)
            print(fmt(f"  {ex} {tag}", metrics(t, cv)))


if __name__ == "__main__":
    main()
