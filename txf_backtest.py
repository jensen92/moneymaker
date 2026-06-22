"""台指期 (TXF) 日內策略回測 — 以 ^TWII 小時 K 為開發標的。

時間框架: 小時 K (Yahoo ^TWII 60m, 唯一有 ~3 年歷史者; 5m/15m 僅 60 天不做回測)。
每個交易日 5 根小時 K: 09:00 / 10:00 / 11:00 / 12:00 / 13:00(收 13:30)。

成本模型 (大台 TXF, 每點 NT$200):
  每邊 = 滑價 1 點 + 手續費 NT$50(=0.25點) + 期交稅 0.00002×指數點
  來回約 3.3 點 (隨指數高低浮動)。回測一律以「淨點數」評估。

評估: 淨總點數 / 獲利因子(PF) / 勝率 / 每筆期望 / 最大回撤(點) / 年化 Sharpe,
並做 70/30 時間切分的「樣本內 (IS) / 樣本外 (OOS)」穩健度驗證 — 只有 IS 與
OOS 皆淨獲利且 PF>1 的策略才算可用。
"""
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "futures_data", "TWII_60m.csv")

POINT_VALUE = 200.0    # 大台 NT$/點
SLIP_PT = 1.0          # 每邊滑價 (點)
FEE_NT = 50.0          # 每邊手續費 (NT$/口)
TAX_RATE = 0.00002     # 期交稅率 (每邊, 對名目價值)


def cost_pts(price):
    """單邊成本 (點). 來回需 ×2。"""
    return SLIP_PT + FEE_NT / POINT_VALUE + TAX_RATE * price


# ── 載入 + 分日 ──────────────────────────────────────────────────────────────

def load_days():
    rows = []
    with open(DATA) as f:
        r = csv.reader(f)
        next(r)
        for dt, o, h, l, c, v in r:
            rows.append((dt, float(o), float(h), float(l), float(c)))
    days = defaultdict(list)
    for dt, o, h, l, c in rows:
        d, hm = dt.split(" ")
        days[d].append({"hm": hm, "o": o, "h": h, "l": l, "c": c})
    # 只保留正常 5 根 (剔除半日/異常)
    out = []
    for d in sorted(days):
        bars = days[d]
        # 合併同日的 13:00 與 13:30 (取 13:00 為主, 用 13:30 收盤更新 close/high/low)
        norm = [b for b in bars if b["hm"] in ("09:00", "10:00", "11:00", "12:00", "13:00")]
        extra = [b for b in bars if b["hm"] == "13:30"]
        if extra and norm and norm[-1]["hm"] == "13:00":
            e = extra[0]
            norm[-1]["c"] = e["c"]
            norm[-1]["h"] = max(norm[-1]["h"], e["h"])
            norm[-1]["l"] = min(norm[-1]["l"], e["l"])
        if len(norm) == 5:
            out.append((d, norm))
    return out


# ── 交易結果 → 指標 ──────────────────────────────────────────────────────────

def metrics(trades, daily_pnl):
    """trades: list of net points; daily_pnl: dict date->net points."""
    if not trades:
        return None
    arr = np.array(trades)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    # 權益曲線 (累積淨點) + 最大回撤
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    maxdd = (peak - eq).max() if len(eq) else 0.0
    # 日 Sharpe
    dp = np.array(list(daily_pnl.values()))
    sharpe = (dp.mean() / dp.std() * np.sqrt(252)) if dp.std() > 0 else 0.0
    return {
        "trades": len(arr), "win": (arr > 0).mean(), "total": arr.sum(),
        "avg": arr.mean(), "pf": pf, "maxdd": maxdd, "sharpe": sharpe,
        "nt": arr.sum() * POINT_VALUE,
    }


# ── 策略: 每日回傳 0 或 1 筆交易 (方向, 進場價, 出場價) ────────────────────────
# 以小時 K 近似日內: 進場價 = 觸發價; 之後逐根檢查停損, 否則收盤(13:30)平倉。

def _simulate_after(bars, start_i, direction, entry, stop):
    """從 start_i 起逐根模擬, 回傳出場價 (碰停損則停損價, 否則最後一根收盤)。"""
    for j in range(start_i, len(bars)):
        b = bars[j]
        if direction > 0 and b["l"] <= stop:
            return stop
        if direction < 0 and b["h"] >= stop:
            return stop
    return bars[-1]["c"]


def strat_orb(bars, prev, p):
    """開盤區間突破 (Opening Range Breakout)。
    第1根(09:00)為開盤區間 [H1,L1]; 之後任一根突破 H1+buf 做多 / 跌破 L1-buf 做空,
    停損設在區間另一端 (或固定點數 stop_pt 取較近者), 收盤平倉。p: buf_frac, stop_pt, side。"""
    H1, L1 = bars[0]["h"], bars[0]["l"]
    rng = H1 - L1
    if rng <= 0:
        return None
    buf = p["buf_frac"] * rng
    up, dn = H1 + buf, L1 - buf
    for i in range(1, len(bars)):
        b = bars[i]
        long_trig = b["h"] >= up
        short_trig = b["l"] <= dn
        if long_trig and (p["side"] in ("both", "long")):
            entry = max(b["o"], up)     # 跳空越過則以開盤成交 (不取得不可能的更佳價)
            stop = max(L1, entry - p["stop_pt"])
            ex = _simulate_after(bars, i, +1, entry, stop)
            return (+1, entry, ex)
        if short_trig and (p["side"] in ("both", "short")):
            entry = min(b["o"], dn)
            stop = min(H1, entry + p["stop_pt"])
            ex = _simulate_after(bars, i, -1, entry, stop)
            return (-1, entry, ex)
    return None


def strat_firsthour_mom(bars, prev, p):
    """首根動能延續: 09:00 根 (收-開) 方向, 10:00 開盤進場, 停損 stop_pt, 收盤平倉。
    p: stop_pt, min_move (首根最小幅度, 點)。"""
    first = bars[0]
    move = first["c"] - first["o"]
    if abs(move) < p["min_move"]:
        return None
    direction = 1 if move > 0 else -1
    if p["side"] == "long" and direction < 0:
        return None
    if p["side"] == "short" and direction > 0:
        return None
    entry = bars[1]["o"]
    stop = entry - direction * p["stop_pt"]
    ex = _simulate_after(bars, 1, direction, entry, stop)
    return (direction, entry, ex)


def strat_firsthour_fade(bars, prev, p):
    """首根反轉 (均值回歸): 09:00 根漲/跌過大則反向, 10:00 進場, 停損, 收盤平倉。"""
    first = bars[0]
    move = first["c"] - first["o"]
    if abs(move) < p["min_move"]:
        return None
    direction = -1 if move > 0 else 1   # 反向
    entry = bars[1]["o"]
    stop = entry - direction * p["stop_pt"]
    ex = _simulate_after(bars, 1, direction, entry, stop)
    return (direction, entry, ex)


def strat_prevday_breakout(bars, prev, p):
    """前日高低突破: 盤中突破前日高做多 / 跌破前日低做空, 停損 stop_pt, 收盤平倉。"""
    if prev is None:
        return None
    ph, pl = prev["h"], prev["l"]
    for i in range(0, len(bars)):
        b = bars[i]
        if b["h"] >= ph and p["side"] in ("both", "long"):
            entry = max(b["o"], ph)     # 跳空越過前日高 → 以開盤成交, 非前日高
            stop = entry - p["stop_pt"]
            ex = _simulate_after(bars, i, +1, entry, stop)
            return (+1, entry, ex)
        if b["l"] <= pl and p["side"] in ("both", "short"):
            entry = min(b["o"], pl)
            stop = entry + p["stop_pt"]
            ex = _simulate_after(bars, i, -1, entry, stop)
            return (-1, entry, ex)
    return None


# ── 回測 driver ──────────────────────────────────────────────────────────────

def run(days, strat, p, trend_filter=None):
    """trend_filter: None / 'ema' (只順大勢: 收盤>EMA20日做多側, <做空側)。"""
    closes = [d[1][-1]["c"] for d in days]
    ema = _ema(closes, 20)
    trades = []
    daily = {d[0]: 0.0 for d in days}   # 含無交易日 (=0), Sharpe 才反映實際曆日
    prev_day = None
    for k, (d, bars) in enumerate(days):
        prev = {"h": max(b["h"] for b in prev_day[1]),
                "l": min(b["l"] for b in prev_day[1]),
                "c": prev_day[1][-1]["c"]} if prev_day else None
        res = strat(bars, prev, p)
        if res:
            direction, entry, ex = res
            # 趨勢濾網: 只做與大勢同向
            allow = True
            if trend_filter == "ema" and k > 0:
                bias = 1 if closes[k - 1] > ema[k - 1] else -1
                allow = (direction == bias)
            if allow:
                gross = direction * (ex - entry)
                net = gross - (cost_pts(entry) + cost_pts(ex))
                trades.append(net)
                daily[d] = daily.get(d, 0.0) + net
        prev_day = (d, bars)
    return metrics(trades, daily), trades, daily


def _ema(x, n):
    x = np.array(x, float)
    a = 2 / (n + 1)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def split_eval(days, strat, p, trend_filter=None, frac=0.7):
    cut = int(len(days) * frac)
    m_all, *_ = run(days, strat, p, trend_filter)
    m_is, *_ = run(days[:cut], strat, p, trend_filter)
    m_oos, *_ = run(days[cut:], strat, p, trend_filter)
    return m_all, m_is, m_oos


# ── 參數網格 ─────────────────────────────────────────────────────────────────

GRIDS = {
    "ORB": (strat_orb, [
        {"buf_frac": bf, "stop_pt": sp, "side": sd}
        for bf in (0.0, 0.1, 0.25) for sp in (40, 60, 80, 120) for sd in ("both", "long")]),
    "FH-MOM": (strat_firsthour_mom, [
        {"stop_pt": sp, "min_move": mm, "side": sd}
        for sp in (40, 60, 80, 120) for mm in (0, 30, 50, 80) for sd in ("both", "long")]),
    "FH-FADE": (strat_firsthour_fade, [
        {"stop_pt": sp, "min_move": mm, "side": "both"}
        for sp in (40, 60, 80, 120) for mm in (50, 80, 120, 160)]),
    "PDAY-BO": (strat_prevday_breakout, [
        {"stop_pt": sp, "side": sd}
        for sp in (40, 60, 80, 120) for sd in ("both", "long")]),
}


def main():
    days = load_days()
    print(f"載入 {len(days)} 個交易日 ({days[0][0]} .. {days[-1][0]}), "
          f"小時K 5根/日\n")

    results = []
    for name, (strat, grid) in GRIDS.items():
        for tf in (None, "ema"):
            for p in grid:
                m_all, m_is, m_oos = split_eval(days, strat, p, tf)
                if not (m_all and m_is and m_oos):
                    continue
                robust = (m_is["total"] > 0 and m_oos["total"] > 0
                          and m_is["pf"] > 1.0 and m_oos["pf"] > 1.0)
                results.append({
                    "name": name, "tf": tf or "-", "p": p,
                    "trades": m_all["trades"], "win": m_all["win"],
                    "total": m_all["total"], "avg": m_all["avg"],
                    "pf": m_all["pf"], "maxdd": m_all["maxdd"],
                    "sharpe": m_all["sharpe"], "nt": m_all["nt"],
                    "oos_total": m_oos["total"], "oos_pf": m_oos["pf"],
                    "is_total": m_is["total"], "robust": robust,
                })

    # 先看穩健 (IS+OOS 皆獲利), 再依 Sharpe 排序
    results.sort(key=lambda r: (r["robust"], r["sharpe"]), reverse=True)
    print(f"{'策略':<9}{'濾網':<5}{'交易':>5}{'勝率':>6}{'淨點':>8}{'每筆':>6}"
          f"{'PF':>6}{'回撤':>7}{'Sharpe':>7}{'OOS淨':>8}  穩健")
    print("-" * 86)
    for r in results[:25]:
        print(f"{r['name']:<9}{r['tf']:<5}{r['trades']:>5}{r['win']*100:>5.0f}%"
              f"{r['total']:>8.0f}{r['avg']:>6.1f}{r['pf']:>6.2f}{r['maxdd']:>7.0f}"
              f"{r['sharpe']:>7.2f}{r['oos_total']:>8.0f}  {'✅' if r['robust'] else '✗'}")

    best = next((r for r in results if r["robust"]), None)
    print("\n" + "=" * 60)
    if best:
        print("最佳 (穩健且 Sharpe 最高):")
        print(f"  策略 {best['name']} 濾網={best['tf']} 參數={best['p']}")
        print(f"  全期: {best['trades']} 筆, 勝率 {best['win']:.0%}, 淨 {best['total']:.0f} 點 "
              f"(NT${best['nt']:,.0f}/大台), PF {best['pf']:.2f}, "
              f"最大回撤 {best['maxdd']:.0f} 點, Sharpe {best['sharpe']:.2f}")
        print(f"  IS 淨 {best['is_total']:.0f} 點 / OOS 淨 {best['oos_total']:.0f} 點 "
              f"(PF {best['oos_pf']:.2f}) — 樣本外仍獲利")
    else:
        print("沒有任何參數組在 IS+OOS 皆穩健獲利 (扣成本後)。")
    return results


if __name__ == "__main__":
    main()
