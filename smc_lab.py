"""SMC 指標進出場實驗室 — 保留原策略指標 (pivot結構/CHoCH/訂單區塊/ATR),
系統化測試不同進出場組合, 尋找適合此指標的用法。

指標核心 (與 smc_strategy.pine 相同, 無未來函數):
- ta.pivothigh/pivotlow(L,L) 轉折點 (確認延遲 L 根)
- CHoCH: 收盤突破前轉折高/低 → 結構翻轉 (trend ±1)
- 訂單區塊: 前轉折點 ± ATR14 × mult
- BOS: 趨勢方向上突破「新確認的」轉折高/低 (續勢)

進場 (Entry):
  E1 breakout : CHoCH 確認 → 次根開盤市價進場 (順新結構方向)
  E3 retest   : CHoCH 後掛限價於「被突破的前高/前低」等回測 (非深處OB)
  E4 ob       : 原版 — 限價回踩訂單區塊 (基準)
  E5 bos      : E1 + 趨勢持續中每次 BOS 亦進場 (僅空手時)

出場 (Exit):
  X1 rr     : 固定停損 (結構點±ATR緩衝) + RR 停利 (基準)
  X2 choch  : 固定停損 + 反向 CHoCH 次根開盤出場 (純結構)
  X3 strail : 結構跟蹤 — 停損跟隨新確認轉折低(高)點 ± ATR 緩衝, 只升不降
  X4 atrail : ATR 吊燈 — 停損 = 進場後最高(低)收盤 ∓ k×ATR

橫盤濾網 (range_mult, 預設0=關): 過濾短區間洗盤。CHoCH/BOS 觸發時,
量測「本次結構擺幅」= |突破的轉折點 − 對側轉折點 (或BOS基準點)|;
若擺幅 < range_mult × ATR14, 視為盤整區間內的雜訊擺動, 結構狀態照樣翻轉
(定義不變, 不影響指標本身), 但**跳過這次進場**。橫盤期間相鄰轉折點彼此
靠得很近 (擺幅小), 是典型的假突破/洗盤特徵；趨勢期間擺幅自然放大,
不受影響。

計量: 以「指數點」為單位 (mark-to-market 逐根權益), 成本以點數計
(TWII/TXF 預設來回 2 點 ≈ 手續費+期交稅+1點滑價)。
換算: 大台 1 點 = NT$200。

用法:
  python3 smc_lab.py            # 完整: 網格 + 跨市場/樣本外驗證 + 金額換算
  python3 smc_lab.py --grid     # 只跑 TWII 60m 網格
"""
import math
import os
import sys

from smc_backtest import DATA, adx_series, atr_series, load_csv, pivot_flags

ENTRIES = ("breakout", "retest", "ob", "bos")
EXITS = ("rr", "choch", "strail", "atrail")

P = dict(length=5, atr_mult=0.5, rr=2.0, trail_k=3.0, expiry=30)


# ---------------------------------------------------------------- 引擎
def simulate(bars, entry, exit_, length=5, atr_mult=0.5, rr=2.0,
             trail_k=3.0, expiry=30, cost_pts=2.0, allow_long=True,
             allow_short=True, range_mult=0.0, adx_min=0.0):
    """回傳 (trades, equity_pts) — equity 為逐根 mark-to-market 累計點數。"""
    phs, pls = pivot_flags(bars, length)
    atr = atr_series(bars)
    adx = adx_series(bars) if adx_min > 0 else None

    last_ph = last_pl = None          # CHoCH 用 (突破後重置)
    conf_ph = conf_pl = None          # 最新已確認轉折 (不重置; trail/BOS 用)
    trend = 0
    bos_lvl = None                    # 待突破的續勢轉折價
    pend = None                       # dict(dir, kind, limit, bar, stop0)
    pos = None                        # dict(dir, entry, entry_dt, stop, target, hi, lo, fill_bar, exit_mkt)
    trades = []
    eq = 0.0
    curve = []

    def close_pos(i, price, reason):
        nonlocal pos, eq
        pts = (price - pos["entry"]) * pos["dir"] - cost_pts
        eq += pts
        trades.append(dict(dir=pos["dir"], entry_dt=pos["entry_dt"],
                           entry=pos["entry"], exit_dt=bars[i][0],
                           exit=price, reason=reason, pts=pts))
        pos = None

    for i, (dt, o, h, l, c) in enumerate(bars):
        # ---- 1) 盤中撮合 (前一根收盤決定的委託) ----
        if pos is not None and pos["fill_bar"] < i:
            if pos.get("exit_mkt"):
                close_pos(i, o, "choch")
            else:
                d, S, T = pos["dir"], pos["stop"], pos["target"]
                if d == 1:
                    if S is not None and o <= S:
                        close_pos(i, o, "stop")
                    elif S is not None and l <= S:
                        close_pos(i, S, "stop")
                    elif T is not None and o >= T:
                        close_pos(i, o, "target")
                    elif T is not None and h >= T:
                        close_pos(i, T, "target")
                else:
                    if S is not None and o >= S:
                        close_pos(i, o, "stop")
                    elif S is not None and h >= S:
                        close_pos(i, S, "stop")
                    elif T is not None and o <= T:
                        close_pos(i, o, "target")
                    elif T is not None and l <= T:
                        close_pos(i, T, "target")

        if pos is None and pend is not None and pend["bar"] < i:
            d, L = pend["dir"], pend["limit"]
            fill = None
            if L is None:                       # 市價
                fill = o
            elif d == 1 and o <= L:
                fill = o
            elif d == 1 and l <= L:
                fill = L
            elif d == -1 and o >= L:
                fill = o
            elif d == -1 and h >= L:
                fill = L
            if fill is not None:
                pos = dict(dir=d, entry=fill, entry_dt=dt, stop=pend["stop0"],
                           target=None, hi=fill, lo=fill, fill_bar=i)
                if exit_ == "rr" and pend["stop0"] is not None:
                    risk = (fill - pend["stop0"]) * d
                    if risk > 0:
                        pos["target"] = fill + d * risk * rr
                pend = None

        # ---- 2) 收盤邏輯 ----
        if phs[i] is not None:
            last_ph = phs[i]
            conf_ph = phs[i]
        if pls[i] is not None:
            last_pl = pls[i]
            conf_pl = pls[i]
        # BOS 追蹤: 多頭中新確認的轉折高 = 續勢突破價; 空頭中新確認的轉折低
        if entry == "bos":
            if trend == 1 and phs[i] is not None:
                bos_lvl = phs[i]
            if trend == -1 and pls[i] is not None:
                bos_lvl = pls[i]

        a = atr[i]
        if a is None:
            curve.append(eq + ((c - pos["entry"]) * pos["dir"] if pos else 0.0))
            continue

        bull = trend <= 0 and last_ph is not None and last_pl is not None and c > last_ph
        bear = trend >= 0 and last_pl is not None and last_ph is not None and c < last_pl

        sig_dir = 0
        if bull:
            sig_dir = 1
            broken, base = last_ph, last_pl
            trend = 1
            bos_lvl = None
            last_ph = None
        elif bear:
            sig_dir = -1
            broken, base = last_pl, last_ph
            trend = -1
            bos_lvl = None
            last_pl = None

        # CHoCH → 依進場模式建立委託 / 出場 (橫盤濾網: 擺幅太小視為洗盤, 跳過進場)
        if sig_dir != 0:
            if pos is not None and exit_ in ("choch",) and pos["dir"] == -sig_dir:
                pos["exit_mkt"] = True
            choppy = range_mult > 0 and abs(broken - base) < range_mult * a
            weak_trend = adx_min > 0 and (adx[i] is None or adx[i] < adx_min)
            if pos is None and not choppy and not weak_trend and ((sig_dir == 1 and allow_long) or (sig_dir == -1 and allow_short)):
                stop0 = base - a * atr_mult if sig_dir == 1 else base + a * atr_mult
                if entry == "breakout" or entry == "bos":
                    pend = dict(dir=sig_dir, limit=None, bar=i, stop0=stop0)
                elif entry == "retest":
                    pend = dict(dir=sig_dir, limit=broken, bar=i, stop0=stop0)
                elif entry == "ob":
                    lvl = base + a * atr_mult if sig_dir == 1 else base - a * atr_mult
                    pend = dict(dir=sig_dir, limit=lvl, bar=i, stop0=stop0)
            elif pos is not None:
                pend = None  # 持倉中不留反向/同向舊掛單

        # BOS 續勢進場 (E5): 趨勢方向上收盤突破新確認轉折價 (同橫盤濾網)
        if entry == "bos" and pos is None and pend is None and bos_lvl is not None:
            weak_trend = adx_min > 0 and (adx[i] is None or adx[i] < adx_min)
            if trend == 1 and c > bos_lvl and allow_long:
                stop0 = (conf_pl - a * atr_mult) if conf_pl is not None else None
                choppy = (range_mult > 0 and conf_pl is not None
                         and abs(bos_lvl - conf_pl) < range_mult * a)
                if stop0 is not None and stop0 < c and not choppy and not weak_trend:
                    pend = dict(dir=1, limit=None, bar=i, stop0=stop0)
                bos_lvl = None
            elif trend == -1 and c < bos_lvl and allow_short:
                stop0 = (conf_ph + a * atr_mult) if conf_ph is not None else None
                choppy = (range_mult > 0 and conf_ph is not None
                         and abs(bos_lvl - conf_ph) < range_mult * a)
                if stop0 is not None and stop0 > c and not choppy and not weak_trend:
                    pend = dict(dir=-1, limit=None, bar=i, stop0=stop0)
                bos_lvl = None

        # 限價掛單過期 / 結構失效撤單
        if pend is not None and pend["limit"] is not None and pos is None:
            expired = expiry > 0 and i - pend["bar"] > expiry
            violated = pend["stop0"] is not None and (
                (pend["dir"] == 1 and c < pend["stop0"]) or
                (pend["dir"] == -1 and c > pend["stop0"]))
            if expired or violated:
                pend = None

        # 持倉: 更新跟蹤停損
        if pos is not None:
            pos["hi"] = max(pos["hi"], c)
            pos["lo"] = min(pos["lo"], c)
            if exit_ == "strail":
                if pos["dir"] == 1 and pls[i] is not None:
                    cand = pls[i] - a * atr_mult
                    if pos["stop"] is None or cand > pos["stop"]:
                        pos["stop"] = cand
                elif pos["dir"] == -1 and phs[i] is not None:
                    cand = phs[i] + a * atr_mult
                    if pos["stop"] is None or cand < pos["stop"]:
                        pos["stop"] = cand
            elif exit_ == "atrail":
                if pos["dir"] == 1:
                    cand = pos["hi"] - trail_k * a
                    if pos["stop"] is None or cand > pos["stop"]:
                        pos["stop"] = cand
                else:
                    cand = pos["lo"] + trail_k * a
                    if pos["stop"] is None or cand < pos["stop"]:
                        pos["stop"] = cand

        curve.append(eq + ((c - pos["entry"]) * pos["dir"] if pos else 0.0))

    if pos is not None:
        close_pos(len(bars) - 1, bars[-1][4], "eod")
        curve[-1] = eq
    return trades, curve


# ---------------------------------------------------------------- 指標
def metrics(trades, curve):
    if not trades:
        return dict(n=0, win=0, pf=0, pts=0, dd=0, rdd=0, avg=0)
    wins = [t["pts"] for t in trades if t["pts"] > 0]
    losses = [t["pts"] for t in trades if t["pts"] <= 0]
    gp, gl = sum(wins), -sum(losses)
    peak, dd = -1e18, 0.0
    for e in curve:
        peak = max(peak, e)
        dd = max(dd, peak - e)
    tot = curve[-1]
    return dict(n=len(trades), win=len(wins) / len(trades) * 100,
                pf=(gp / gl) if gl > 0 else math.inf,
                pts=tot, dd=dd, rdd=(tot / dd) if dd > 0 else math.inf,
                avg=tot / len(trades))


def fmt(name, m):
    pf = f"{m['pf']:.2f}" if m['pf'] != math.inf else "inf"
    rdd = f"{m['rdd']:.2f}" if m['rdd'] != math.inf else "inf"
    return (f"{name:<22} 筆數{m['n']:>4}  勝率{m['win']:5.1f}%  PF {pf:>5}  "
            f"均{m['avg']:+7.1f}點  總{m['pts']:+9.0f}點  "
            f"DD{m['dd']:8.0f}點  總/DD {rdd:>6}")


# ---------------------------------------------------------------- 主流程
def main():
    grid_only = "--grid" in sys.argv
    p60 = load_csv(os.path.join(DATA, "TWII_60m.csv"))
    p1d = os.path.join(DATA, "TWII_1d.csv")
    d1d = load_csv(p1d) if os.path.exists(p1d) else None
    pgc = os.path.join(DATA, "GC_60m.csv")
    dgc = load_csv(pgc) if os.path.exists(pgc) else None

    print("=" * 104)
    print("SMC 指標進出場實驗室 — TWII 60m (2023-07~2026-07) | 成本來回2點 | "
          f"L={P['length']} ATRx{P['atr_mult']} RR={P['rr']} 吊燈k={P['trail_k']}")
    print("=" * 104)

    results = {}
    for e in ENTRIES:
        for x in EXITS:
            t, cv = simulate(p60, e, x, **P)
            m = metrics(t, cv)
            results[(e, x)] = m
            print(fmt(f"{e:>8} + {x}", m))
        print("-" * 104)

    if grid_only:
        return

    # 依 總/DD 與 PF 綜合挑前 3 組做驗證 (需 n>=30 避免小樣本)
    ranked = sorted((k for k, m in results.items() if m["n"] >= 30),
                    key=lambda k: (results[k]["rdd"]), reverse=True)[:3]
    print(f"\n### 驗證前 3 組 (依 總/DD, n>=30): "
          f"{', '.join(f'{e}+{x}' for e, x in ranked)}")

    for e, x in ranked:
        print(f"\n--- {e} + {x} ---")
        cut = int(len(p60) * 0.7)
        for tag, seg in (("TWII60m 樣本內70%", p60[:cut]), ("TWII60m 樣本外30%", p60[cut:])):
            m = metrics(*simulate(seg, e, x, **P))
            print(fmt(f"  {tag}", m))
        if d1d:
            m = metrics(*simulate(d1d, e, x, **P))
            print(fmt("  TWII 日線15年", m))
        if dgc:
            m = metrics(*simulate(dgc, e, x, cost_pts=0.5,
                                  **{k: v for k, v in P.items()}))
            print(fmt("  GC 60m (成本0.5點)", m))
        # 多空拆解
        t, _ = simulate(p60, e, x, **P)
        for d, tag in ((1, "多"), (-1, "空")):
            sub = [y["pts"] for y in t if y["dir"] == d]
            if sub:
                w = sum(1 for s in sub if s > 0)
                print(f"    {tag}單 {len(sub):>3} 筆  勝率 {w/len(sub)*100:5.1f}%  "
                      f"累計 {sum(sub):+8.0f}點")
        # 參數敏感度 (只看鄰近格)
        for L in (3, 5, 8):
            for am in (0.5, 1.0):
                m = metrics(*simulate(p60, e, x, length=L, atr_mult=am,
                                      rr=P["rr"], trail_k=P["trail_k"],
                                      expiry=P["expiry"]))
                print(fmt(f"    L={L} ATRx{am}", m))

    # 橫盤濾網 (擺幅/ADX) 對最佳組的效果 — 過濾短區間洗盤是否有穩健幫助
    if ranked:
        e0, x0 = ranked[0]
        print(f"\n### 橫盤濾網效果 ({e0}+{x0} 只做多)")
        for tag, bars, cost in (("TWII60m", p60, 2.0),
                                ("TWII1d", d1d, 2.0) if d1d else (None, None, None),
                                ("GC60m", dgc, 0.5) if dgc else (None, None, None)):
            if bars is None:
                continue
            for rm, am in ((0.0, 0.0), (1.5, 0.0), (0.0, 20.0)):
                q = dict(P); q["range_mult"] = rm; q["adx_min"] = am
                m = metrics(*simulate(bars, e0, x0, allow_short=False,
                                      cost_pts=cost, **q))
                print(fmt(f"  {tag} range={rm} adx={am}", m))

    # 金額換算 (最佳組, 10口大台)
    if ranked:
        e, x = ranked[0]
        t, cv = simulate(p60, e, x, **P)
        m = metrics(t, cv)
        yrs = len(p60) / (5 * 245)  # 每日5根60m
        print(f"\n### 金額換算 — {e}+{x}, 大台 10 口 (1點=NT$200)")
        print(f"  3年總損益 {m['pts']*200*10/1e4:+,.0f} 萬 | 年均 "
              f"{m['pts']*200*10/1e4/yrs:+,.0f} 萬 | 最大回撤 "
              f"{m['dd']*200*10/1e4:,.0f} 萬 | 所需保證金約 10口×36萬 = 360萬")


if __name__ == "__main__":
    main()
