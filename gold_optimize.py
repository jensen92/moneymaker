"""黃金期貨小時線策略優化探索 — 檢視區間策略可否轉正, 並對照順勢突破。

依先前診斷 (毛利近 0、成本+空轉吃光、強趨勢逆勢不利) 測試四條結構性路線:
  V1 區間逆勢+停損反手      (baseline, 原需求)
  V2 區間逆勢+停損出場+冷卻 (砍掉反手空轉, 加 cooldown 降頻)
  V3 順勢突破 (上破做多/下破做空) + ATR 移動停損   ← 反向假設
  V4 區間/突破『擇時切換』: 趨勢盤(ADX高)只順勢突破, 盤整盤(ADX低)只區間逆勢
並對 V3 做停損/突破窗參數網格, 看是否有穩健正期望。
"""
import csv
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "futures_data", "GC_60m.csv")
PV = 100.0
COST = 0.30   # 每邊報價成本


def load():
    o, h, l, c, dt = [], [], [], [], []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            dt.append(r["date"])
            o.append(float(r["open"])); h.append(float(r["high"]))
            l.append(float(r["low"]));  c.append(float(r["close"]))
    return dt, np.array(o), np.array(h), np.array(l), np.array(c)


def atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                              np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    a = np.full(len(c), np.nan)
    if len(c) >= n:
        a[n - 1] = tr[:n].mean()
        for i in range(n, len(c)):
            a[i] = (a[i - 1] * (n - 1) + tr[i]) / n
    return a


def stats(trades, eq0=0.0):
    if not trades:
        return None
    p = np.array([t["pnl"] for t in trades])
    w = p[p > 0]; ls = p[p < 0]
    gp = w.sum(); gl = -ls.sum()
    eq = np.cumsum(p)
    dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) else 0.0
    return {"n": len(trades), "win": len(w) / len(trades),
            "pf": gp / gl if gl > 0 else float("inf"),
            "tot": p.sum(), "dd": dd,
            "mar": (eq[-1] / dd) if dd > 0 else float("inf")}


def find_range(h, l, i, pw=6, minw=0.006, maxw=0.05):
    lo_i = max(pw, i - 200)
    hp, lp = [], []
    for j in range(i - pw, lo_i - 1, -1):
        if h[j] == h[j - pw:j + pw + 1].max():
            hp.append(h[j])
        if l[j] == l[j - pw:j + pw + 1].min():
            lp.append(l[j])
        if len(hp) >= 2 and len(lp) >= 2:
            break
    if len(hp) < 2 or len(lp) < 2:
        return None
    res = max(hp[0], hp[1]); sup = min(lp[0], lp[1])
    if res <= sup:
        return None
    width = (res - sup) / sup
    if width < minw or width > maxw:
        return None
    return sup, res


# ── V1 區間逆勢+停損反手 ────────────────────────────────────────
def v1(dt, o, h, l, c, stop_buf=0.003, reverse=True, cooldown=0):
    n = len(c); pos = 0; entry = 0.0; sup = res = 0.0; cd = 0
    tr = []
    def close(px, i, why):
        nonlocal pos
        tr.append({"dir": pos, "pnl": (px - entry) * pos * PV - COST * PV * 2,
                   "i": i, "why": why}); pos = 0
    for i in range(30, n):
        px = c[i]
        if pos == 0:
            if cd > 0:
                cd -= 1; continue
            rng = find_range(h, l, i)
            if rng is None:
                continue
            sup, res = rng
            if px >= res:
                pos, entry = -1, px
            elif px <= sup:
                pos, entry = 1, px
            continue
        if pos == -1:
            if px >= res * (1 + stop_buf):
                close(px, i, "stop")
                if reverse:
                    pos, entry = 1, px
                else:
                    cd = cooldown
            elif px <= sup:
                close(px, i, "tgt")
                if reverse:
                    pos, entry = 1, px
                else:
                    cd = cooldown
        else:
            if px <= sup * (1 - stop_buf):
                close(px, i, "stop")
                if reverse:
                    pos, entry = -1, px
                else:
                    cd = cooldown
            elif px >= res:
                close(px, i, "tgt")
                if reverse:
                    pos, entry = -1, px
                else:
                    cd = cooldown
    if pos:
        close(c[-1], n - 1, "eod")
    return tr


# ── V3 順勢突破 + ATR 移動停損 ──────────────────────────────────
def v3(dt, o, h, l, c, breakout=24, atr_stop=3.0, allow_short=True):
    n = len(c); a = atr(h, l, c)
    pos = 0; entry = 0.0; trail = 0.0; tr = []
    def close(px, i):
        nonlocal pos
        tr.append({"dir": pos, "pnl": (px - entry) * pos * PV - COST * PV * 2, "i": i}); pos = 0
    for i in range(breakout + 20, n):
        px = c[i]
        if np.isnan(a[i]):
            continue
        hh = h[i - breakout:i].max(); ll = l[i - breakout:i].min()
        if pos == 0:
            if px > hh:
                pos, entry, trail = 1, px, px - atr_stop * a[i]
            elif allow_short and px < ll:
                pos, entry, trail = -1, px, px + atr_stop * a[i]
        elif pos == 1:
            trail = max(trail, px - atr_stop * a[i])
            if px <= trail:
                close(px, i)
        else:
            trail = min(trail, px + atr_stop * a[i])
            if px >= trail:
                close(px, i)
    if pos:
        close(c[-1], n - 1)
    return tr


# ── V4 ADX 擇時: 趨勢盤順勢突破, 盤整盤區間逆勢 ────────────────
def adx(h, l, c, n=14):
    up = h[1:] - h[:-1]; dn = l[:-1] - l[1:]
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    def smooth(x):
        s = np.full(len(x), np.nan); s[n - 1] = x[:n].sum()
        for i in range(n, len(x)):
            s[i] = s[i - 1] - s[i - 1] / n + x[i]
        return s
    atr_ = smooth(tr); pdi = 100 * smooth(plus) / atr_; mdi = 100 * smooth(minus) / atr_
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx_ = np.full(len(c), np.nan); valid = ~np.isnan(dx)
    # 簡化: dx 的 n 期 Wilder 平滑
    out = np.full(len(dx), np.nan)
    first = np.argmax(valid) + n
    if first < len(dx):
        out[first] = np.nanmean(dx[first - n + 1:first + 1])
        for i in range(first + 1, len(dx)):
            if not np.isnan(dx[i]):
                out[i] = (out[i - 1] * (n - 1) + dx[i]) / n
    adx_[1:] = out
    return adx_


def v4(dt, o, h, l, c, adx_thr=25, stop_buf=0.003, breakout=24, atr_stop=3.0):
    n = len(c); a = atr(h, l, c); ad = adx(h, l, c)
    pos = 0; entry = 0.0; mode = None; sup = res = trail = 0.0; tr = []
    def close(px, i):
        nonlocal pos
        tr.append({"dir": pos, "pnl": (px - entry) * pos * PV - COST * PV * 2, "i": i}); pos = 0
    for i in range(breakout + 30, n):
        px = c[i]
        if np.isnan(a[i]) or np.isnan(ad[i]):
            continue
        trending = ad[i] >= adx_thr
        if pos == 0:
            if trending:
                hh = h[i - breakout:i].max(); ll = l[i - breakout:i].min()
                if px > hh:
                    pos, entry, trail, mode = 1, px, px - atr_stop * a[i], "trend"
                elif px < ll:
                    pos, entry, trail, mode = -1, px, px + atr_stop * a[i], "trend"
            else:
                rng = find_range(h, l, i)
                if rng:
                    sup, res = rng
                    if px >= res:
                        pos, entry, mode = -1, px, "range"
                    elif px <= sup:
                        pos, entry, mode = 1, px, "range"
        elif mode == "trend":
            if pos == 1:
                trail = max(trail, px - atr_stop * a[i])
                if px <= trail:
                    close(px, i)
            else:
                trail = min(trail, px + atr_stop * a[i])
                if px >= trail:
                    close(px, i)
        else:  # range, 停損出場不反手
            if pos == -1 and (px >= res * (1 + stop_buf) or px <= sup):
                close(px, i)
            elif pos == 1 and (px <= sup * (1 - stop_buf) or px >= res):
                close(px, i)
    if pos:
        close(c[-1], n - 1)
    return tr


def row(label, tr):
    m = stats(tr)
    if not m:
        print(f"{label:<28} 無交易"); return
    print(f"{label:<28}{m['n']:>6}{m['win']:>8.1%}{m['pf']:>7.2f}"
          f"{m['tot']:>+12,.0f}{m['dd']:>11,.0f}{m['mar']:>7.2f}")


def main():
    dt, o, h, l, c = load()
    print(f"GC 小時線 {len(c)} 根  {dt[0]} ~ {dt[-1]}\n")
    print(f"{'策略':<28}{'交易':>6}{'勝率':>8}{'PF':>7}{'總損益$':>12}{'最大DD$':>11}{'MAR':>7}")
    row("V1 區間逆勢+停損反手", v1(dt, o, h, l, c, reverse=True))
    row("V2 區間+出場+冷卻24", v1(dt, o, h, l, c, reverse=False, cooldown=24))
    row("V3 順勢突破(多空) 24/3ATR", v3(dt, o, h, l, c, 24, 3.0, True))
    row("V3 順勢突破(只多) 24/3ATR", v3(dt, o, h, l, c, 24, 3.0, False))
    row("V4 ADX擇時(趨勢突破+盤整逆勢)", v4(dt, o, h, l, c))

    print("\nV3 順勢突破參數網格 (多空, 總損益$ / PF):")
    hdr = "breakout\\atr"
    print(f"{hdr:<14}" + "".join(f"{s:>14}" for s in (2.0, 3.0, 4.0, 5.0)))
    for bo in (12, 24, 48, 96):
        cells = []
        for st_ in (2.0, 3.0, 4.0, 5.0):
            m = stats(v3(dt, o, h, l, c, bo, st_, True))
            cells.append(f"{m['tot']:>+8,.0f}/{m['pf']:.2f}")
        print(f"{bo:<14}" + "".join(f"{x:>14}" for x in cells))


if __name__ == "__main__":
    main()
