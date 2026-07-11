"""SMC Structure Strategy (CHoCH + 訂單區塊回踩) 回測 — 原版 vs 修正版對照。

策略邏輯 (對應 smc_strategy.pine):
- ta.pivothigh/pivotlow(length, length) 追蹤最近轉折高低點。
- 收盤突破前轉折高 → 多頭 CHoCH: 以前轉折低 ± ATR*mult 定義訂單區塊,
  掛限價單等回踩區塊上緣進多; 停損=區塊下緣, 停利=風險*RR。空頭對稱。

mode="original" 忠實模擬原始 Pine 腳本的行為 (含 bug):
  B1 CHoCH 時 OB 來源轉折點已被重置為 na → strategy.entry(limit=na) 退化為
     市價單, 且 strategy.exit(stop=na, limit=na) 不掛出場單 → 無停損裸倉。
  B2 趨勢翻轉不撤反向殘留限價單, 多空掛單並存, 舊單可能在反轉後成交。
  B3 持倉中停損/停利每根用「當前」ob_top/ob_bottom 重算, 新 CHoCH 使停損
     漂移到新區塊價位 (常在市價另一側 → 立即市價出場)。
  B4 掛單無期限。
mode="fixed" 為修正版: CHoCH 需兩轉折點存在、翻轉撤反向單、進場鎖定
  停損/停利、掛單過期(expiry_bars)與區塊收破失效、risk<=0 不掛停利。
mode="struct" 為修正版+結構出場: 進場衛生同 fixed, 但不掛固定 RR 停利,
  改為「反向 CHoCH 出現 → 次根開盤市價出場」+ 初始停損。此機制源自
  回測發現: 原版 bug B3 的停損漂移意外近似結構跟蹤出場, 讓獲利跑到
  結構反轉為止, 優於固定 2R 上蓋 — struct 模式把它做成正確的規則。
mode="struct_rr" 同 struct 但保留 RR 停利 (先到先出)。

回測假設 (保守):
- 訊號在收盤決定, 委託自次根生效; 限價/停損穿價以開盤價成交 (跳空劣化)。
- 同根同時觸及停損與停利 → 以停損計 (保守)。
- 全額進出 (100% equity 複利), 單邊成本 0.01% (期指手續費+稅+滑價概算)。

用法:
  python3 smc_backtest.py            # 三組資料 原版vs修正版 + 敏感度 + 樣本內外
  python3 smc_backtest.py --quick    # 只跑預設參數對照
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "futures_data")

COST_SIDE = 0.0001  # 單邊 0.01%

DEFAULTS = dict(length=5, rr=2.0, atr_mult=0.5, expiry=30)


# ---------------------------------------------------------------- 資料
def load_csv(path):
    bars = []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if not row:
                continue
            bars.append((row[0], float(row[1]), float(row[2]),
                         float(row[3]), float(row[4])))
    return bars


def fetch_twii_daily(path):
    """抓 ^TWII 日線 (長歷史) 供跨時間框架驗證。"""
    import time

    import requests
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII"
    r = requests.get(url, params={"range": "15y", "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts, q = res["timestamp"], res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c):
            continue
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t + 8 * 3600)),
                     round(o, 2), round(h, 2), round(l, 2), round(c, 2)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close"])
        w.writerows(rows)
    return rows


def datasets():
    out = []
    p60 = os.path.join(DATA, "TWII_60m.csv")
    if os.path.exists(p60):
        out.append(("TWII 60m", load_csv(p60)))
    p1d = os.path.join(DATA, "TWII_1d.csv")
    if not os.path.exists(p1d):
        try:
            fetch_twii_daily(p1d)
        except Exception as e:  # noqa: BLE001
            print(f"TWII 日線下載失敗: {e}")
    if os.path.exists(p1d):
        out.append(("TWII 1d", load_csv(p1d)))
    pgc = os.path.join(DATA, "GC_60m.csv")
    if os.path.exists(pgc):
        out.append(("GC 60m", load_csv(pgc)))
    return out


# ---------------------------------------------------------------- 指標
def atr_series(bars, n=14):
    """Wilder RMA ATR, 對齊 Pine ta.atr。"""
    out = [None] * len(bars)
    trs = []
    prev_c = None
    rma = None
    for i, (_, o, h, l, c) in enumerate(bars):
        tr = h - l if prev_c is None else max(h - l, abs(h - prev_c), abs(l - prev_c))
        prev_c = c
        if rma is None:
            trs.append(tr)
            if len(trs) == n:
                rma = sum(trs) / n
                out[i] = rma
        else:
            rma = (rma * (n - 1) + tr) / n
            out[i] = rma
    return out


def adx_series(bars, n=14):
    """Wilder ADX, 對齊 Pine ta.dmi 的 ADX 輸出。用於橫盤/趨勢強度濾網:
    ADX 低 = 橫盤雜訊, ADX 高 = 有效趨勢。"""
    ln = len(bars)
    out = [None] * ln
    tr_rma = plus_rma = minus_rma = dx_rma = None
    trs = plus_list = minus_list = None
    dxs = []
    for i in range(1, ln):
        _, o, h, l, c = bars[i]
        _, po, ph, pl_, pc = bars[i - 1]
        up = h - ph
        down = pl_ - l
        plus_dm = up if (up > down and up > 0) else 0.0
        minus_dm = down if (down > up and down > 0) else 0.0
        tr = max(h - l, abs(h - pc), abs(l - pc))
        if tr_rma is None:
            trs = [tr] if trs is None else trs + [tr]
            plus_list = [plus_dm] if plus_list is None else plus_list + [plus_dm]
            minus_list = [minus_dm] if minus_list is None else minus_list + [minus_dm]
            if len(trs) == n:
                tr_rma = sum(trs)
                plus_rma = sum(plus_list)
                minus_rma = sum(minus_list)
        else:
            tr_rma = tr_rma - tr_rma / n + tr
            plus_rma = plus_rma - plus_rma / n + plus_dm
            minus_rma = minus_rma - minus_rma / n + minus_dm
        if tr_rma:
            pdi = 100 * plus_rma / tr_rma
            mdi = 100 * minus_rma / tr_rma
            dx = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0.0
            dxs.append(dx)
            if dx_rma is None:
                if len(dxs) == n:
                    dx_rma = sum(dxs) / n
                    out[i] = dx_rma
            else:
                dx_rma = (dx_rma * (n - 1) + dx) / n
                out[i] = dx_rma
    return out


def pivot_flags(bars, length):
    """回傳兩個 list: 在 bar i 「確認」的 pivot high/low 價 (轉折點在 i-length)。"""
    n = len(bars)
    phs = [None] * n
    pls = [None] * n
    for i in range(2 * length, n):
        p = i - length
        hp = bars[p][2]
        lp = bars[p][3]
        window = range(p - length, p + length + 1)
        if all(bars[j][2] < hp for j in window if j != p):
            phs[i] = hp
        if all(bars[j][3] > lp for j in window if j != p):
            pls[i] = lp
    return phs, pls


# ---------------------------------------------------------------- 引擎
def simulate(bars, mode, length=5, rr=2.0, atr_mult=0.5, expiry=30):
    """逐根模擬。回傳 (trades, equity_curve)。

    trade: dict(dir, entry_dt, entry, exit_dt, exit, reason, ret)
    """
    struct_exit = mode in ("struct", "struct_rr")
    use_target = mode in ("original", "fixed", "struct_rr")
    hygiene = mode != "original"  # 進場衛生: na防護/撤反向單/過期失效

    phs, pls = pivot_flags(bars, length)
    atr = atr_series(bars)

    last_ph = last_pl = None
    trend = 0
    ob_top = ob_bottom = None   # 可能為 None (original 的 na 區塊)
    awaiting = False
    order_bar = None

    # 掛單: id -> dict(dir, limit[None=市價], bar)
    pending = {}
    pos = None  # dict(dir, entry, entry_dt, stop, target, fill_bar)
    equity = 1.0
    curve = []
    trades = []

    def close_pos(i, price, reason):
        nonlocal pos, equity
        d = pos["dir"]
        ret = (price / pos["entry"] - 1.0) * d - 2 * COST_SIDE
        equity *= 1.0 + ret
        trades.append(dict(dir=d, entry_dt=pos["entry_dt"], entry=pos["entry"],
                           exit_dt=bars[i][0], exit=price, reason=reason, ret=ret,
                           na_entry=pos.get("na_entry", False),
                           stale=pos.get("stale", 0),
                           nostop_bars=pos.get("nostop_bars", 0)))
        pos = None

    for i, (dt, o, h, l, c) in enumerate(bars):
        # ---- 1) 盤中撮合 (依前一根收盤掛好的委託) ----
        if pos is not None and pos["fill_bar"] < i and pos.get("exit_market"):
            close_pos(i, o, "struct")  # 反向 CHoCH → 次根開盤市價出場
        if pos is not None and pos["fill_bar"] < i:
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

        if pos is None and pending:
            for oid in list(pending):
                od = pending[oid]
                if od["bar"] >= i:
                    continue
                d, L = od["dir"], od["limit"]
                fill = None
                if L is None:               # original B1: na 限價 → 市價
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
                    pos = dict(dir=d, entry=fill, entry_dt=dt, stop=None,
                               target=None, fill_bar=i, na_entry=L is None,
                               stale=i - od["bar"], nostop_bars=0)
                    pending.clear()
                    break

        # ---- 2) 收盤邏輯 (訊號 / 掛單管理) ----
        if phs[i] is not None:
            last_ph = phs[i]
        if pls[i] is not None:
            last_pl = pls[i]
        a = atr[i]
        if a is None:
            curve.append(equity)
            continue

        if hygiene:
            bull = trend <= 0 and last_ph is not None and last_pl is not None and c > last_ph
            bear = trend >= 0 and last_pl is not None and last_ph is not None and c < last_pl
        else:
            bull = trend <= 0 and last_ph is not None and c > last_ph
            bear = trend >= 0 and last_pl is not None and c < last_pl

        if bull:
            trend = 1
            ob_top = last_pl + a * atr_mult if last_pl is not None else None
            ob_bottom = last_pl - a * atr_mult if last_pl is not None else None
            awaiting = True
            order_bar = i
            if hygiene:
                pending.pop("short", None)
            if struct_exit and pos is not None and pos["dir"] == -1:
                pos["exit_market"] = True
            last_ph = None
        if bear:
            trend = -1
            ob_top = last_ph + a * atr_mult if last_ph is not None else None
            ob_bottom = last_ph - a * atr_mult if last_ph is not None else None
            awaiting = True
            order_bar = i
            if hygiene:
                pending.pop("long", None)
            if struct_exit and pos is not None and pos["dir"] == 1:
                pos["exit_market"] = True
            last_pl = None

        # 修正版: 掛單過期 / 區塊被收破遠端 → 撤單
        if hygiene and awaiting and pos is None:
            expired = expiry > 0 and i - order_bar > expiry
            violated = (trend == 1 and ob_bottom is not None and c < ob_bottom) or \
                       (trend == -1 and ob_top is not None and c > ob_top)
            if expired or violated:
                awaiting = False
                pending.clear()

        # 下/更新進場掛單
        if awaiting and pos is None:
            if trend == 1:
                if hygiene and ob_top is None:
                    pass
                else:
                    pending["long"] = dict(dir=1, limit=ob_top,
                                           bar=pending.get("long", {}).get("bar", i))
            if trend == -1:
                if hygiene and ob_bottom is None:
                    pass
                else:
                    pending["short"] = dict(dir=-1, limit=ob_bottom,
                                            bar=pending.get("short", {}).get("bar", i))

        # 持倉: 設定出場價
        if pos is not None:
            if awaiting:
                awaiting = False
            if hygiene:
                if pos["stop"] is None and pos["target"] is None and not pos.get("locked"):
                    pos["locked"] = True
                    if pos["dir"] == 1:
                        pos["stop"] = ob_bottom
                        risk = pos["entry"] - ob_bottom
                        if use_target and risk > 0:
                            pos["target"] = pos["entry"] + risk * rr
                    else:
                        pos["stop"] = ob_top
                        risk = ob_top - pos["entry"]
                        if use_target and risk > 0:
                            pos["target"] = pos["entry"] - risk * rr
            else:
                # original B3: 每根收盤用「當前」區塊重算 (可能為 None → 不掛出場單)
                if pos["dir"] == 1:
                    pos["stop"] = ob_bottom
                    if ob_bottom is not None:
                        risk = pos["entry"] - ob_bottom
                        pos["target"] = pos["entry"] + risk * rr
                    else:
                        pos["target"] = None
                else:
                    pos["stop"] = ob_top
                    if ob_top is not None:
                        risk = ob_top - pos["entry"]
                        pos["target"] = pos["entry"] - risk * rr
                    else:
                        pos["target"] = None

        if pos is not None and pos["stop"] is None:
            pos["nostop_bars"] += 1  # 無停損裸倉 (original B1)

        curve.append(equity)

    # 期末強制平倉 (統計用)
    if pos is not None:
        close_pos(len(bars) - 1, bars[-1][4], "eod")
        curve[-1] = equity
    return trades, curve


# ---------------------------------------------------------------- 指標
def metrics(trades, curve):
    if not trades:
        return dict(n=0, win=0.0, pf=0.0, ret=0.0, dd=0.0, rdd=0.0, avg=0.0)
    wins = [t["ret"] for t in trades if t["ret"] > 0]
    losses = [t["ret"] for t in trades if t["ret"] <= 0]
    gp, gl = sum(wins), -sum(losses)
    peak, dd = 1.0, 0.0
    for e in curve:
        peak = max(peak, e)
        dd = max(dd, 1 - e / peak)
    total = curve[-1] - 1.0
    return dict(n=len(trades), win=len(wins) / len(trades) * 100,
                pf=(gp / gl) if gl > 0 else math.inf,
                ret=total * 100, dd=dd * 100,
                rdd=(total / dd) if dd > 0 else math.inf,
                avg=sum(t["ret"] for t in trades) / len(trades) * 100)


def fmt(name, m):
    pf = f"{m['pf']:.2f}" if m['pf'] != math.inf else "inf"
    rdd = f"{m['rdd']:.2f}" if m['rdd'] != math.inf else "inf"
    return (f"{name:<26} 筆數{m['n']:>4}  勝率{m['win']:5.1f}%  PF {pf:>5}  "
            f"單筆均{m['avg']:+6.2f}%  總報酬{m['ret']:+8.1f}%  "
            f"MaxDD{m['dd']:5.1f}%  報酬/回撤 {rdd:>5}")


# ---------------------------------------------------------------- 主流程
def main():
    quick = "--quick" in sys.argv
    ds = datasets()
    if not ds:
        print("找不到資料, 先執行 python3 txf_data.py")
        return

    print("=" * 100)
    print("SMC Structure Strategy 回測 — 原版(含bug) vs 修正版")
    print(f"預設參數: length={DEFAULTS['length']} RR={DEFAULTS['rr']} "
          f"atr_mult={DEFAULTS['atr_mult']} expiry={DEFAULTS['expiry']} | "
          f"成本 單邊{COST_SIDE*100:.2f}% | 同根雙觸以停損計(保守)")
    print("=" * 100)

    for name, bars in ds:
        print(f"\n### {name}  ({bars[0][0]} .. {bars[-1][0]}, {len(bars)} bars, "
              f"buy&hold {((bars[-1][4]/bars[0][4])-1)*100:+.1f}%)")
        for mode, tag in (("original", "原版(含bug)"), ("fixed", "修正版(固定RR)"),
                          ("struct_rr", "修正版(RR+結構出場)"), ("struct", "修正版(純結構出場)")):
            t, cv = simulate(bars, mode, **DEFAULTS)
            m = metrics(t, cv)
            naked = sum(1 for x in t if x["reason"] == "eod")
            extra = f"  [期末強平 {naked} 筆]" if naked else ""
            print(fmt(tag, m) + extra)

    if quick:
        return

    # ---- 敏感度 (結構出場版, 主要標的 TWII 60m; 非最佳化, 看參數穩定性) ----
    name, bars = ds[0]
    print(f"\n### 參數敏感度 (純結構出場版, {name}) — 檢視穩定性, 非挑參數")
    for length in (3, 5, 8):
        for rr in (2.0,):
            for am in (0.5, 1.0):
                t, cv = simulate(bars, "struct", length=length, rr=rr,
                                 atr_mult=am, expiry=30)
                m = metrics(t, cv)
                print(fmt(f"  L={length} ATRx{am}", m))

    # ---- 樣本內外 70/30 ----
    print("\n### 樣本內外 70/30 (純結構出場版, 預設參數)")
    for name, bars in ds:
        cut = int(len(bars) * 0.7)
        for tag, seg in (("樣本內", bars[:cut]), ("樣本外", bars[cut:])):
            t, cv = simulate(seg, "struct", **DEFAULTS)
            m = metrics(t, cv)
            print(fmt(f"  {name} {tag}", m))

    # ---- 只做多 vs 多空 (台股長期偏多, 檢查空單貢獻) ----
    print("\n### 多空拆解 (純結構出場版, 預設參數)")
    for name, bars in ds:
        t, cv = simulate(bars, "struct", **DEFAULTS)
        for d, tag in ((1, "多單"), (-1, "空單")):
            sub = [x for x in t if x["dir"] == d]
            if not sub:
                continue
            wins = [x["ret"] for x in sub if x["ret"] > 0]
            losses = [x["ret"] for x in sub if x["ret"] <= 0]
            gl = -sum(losses)
            pf = sum(wins) / gl if gl > 0 else math.inf
            print(f"  {name:<10} {tag}: {len(sub):>3} 筆  "
                  f"勝率 {len(wins)/len(sub)*100:5.1f}%  PF {pf:5.2f}  "
                  f"累計 {sum(x['ret'] for x in sub)*100:+7.1f}%")


if __name__ == "__main__":
    main()
