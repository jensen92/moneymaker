"""資金流向研究 (2/3): 宏觀資金環境濾網 — 名目殖利率 / 美元指數 能否領先提升績效。

假設: 黃金與『實質殖利率、美元』反向。殖利率下行 / 美元轉弱 = 資金流向黃金的順風,
反之逆風。做法: 在既有突破進場外, 多要求宏觀處於『順風』(如 10y殖利率 < 其MA, 或
DXY < 其MA) 才允許做多。定位為『日線 regime 脈絡』, 非小時時機。

註: 最理想是『實質殖利率(TIPS/FRED DFII10)』, 但此環境 FRED 不可達; 改用 Yahoo 可抓的
名目 10y 殖利率 (^TNX) 與美元指數 (DX-Y.NYB) 作宏觀代理 (兩者皆黃金公認驅動)。

四關: 主測日線26年 (macro=日訊號), base敏感度(採用配置 vs 弱base), 並看子期一致性。
"""
import csv
import os
import time
import numpy as np

import gold_strategies as gs
import gold_daily_backtest as gd

H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
DIR = os.path.join(gs.HERE, "futures_data")


def fetch_yahoo_daily(symbol, fname):
    """抓 Yahoo 日線收盤 → {date: close}; 存快取 fname。"""
    path = os.path.join(DIR, fname)
    if os.path.exists(path):
        return {r["date"]: float(r["close"]) for r in csv.DictReader(open(path))}
    import requests
    p1 = 946684800  # 2000-01-01
    for attempt in range(3):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                             params={"period1": p1, "period2": int(time.time()), "interval": "1d"},
                             headers=H, timeout=40)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]; q = res["indicators"]["quote"][0]["close"]
            rows = [(time.strftime("%Y-%m-%d", time.gmtime(t)), cl)
                    for t, cl in zip(ts, q) if cl is not None]
            with open(path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["date", "close"]); w.writerows(rows)
            return {d: cl for d, cl in rows}
        except Exception as e:
            if attempt == 2:
                print(f"  {symbol} 抓取失敗: {type(e).__name__}"); return None
            time.sleep(2 ** attempt)


def align(gold_dates, macro):
    """把 macro {date:val} 以『as-of』對齊到 gold 交易日 (用 <= 當日的最近一筆, 無lookahead)。"""
    mkeys = sorted(macro)
    out = np.full(len(gold_dates), np.nan)
    j = 0
    for i, d in enumerate(gold_dates):
        while j + 1 < len(mkeys) and mkeys[j + 1] <= d:
            j += 1
        if mkeys and mkeys[j] <= d:
            out[i] = macro[mkeys[j]]
    return out


def sma_series(x, n):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - n + 1):i + 1]; seg = seg[~np.isnan(seg)]
        if len(seg) >= max(2, n // 2):
            out[i] = seg.mean()
    return out


def bt(dt, h, l, c, *, tailwind=None, bo=40, st=4.0, an=20):
    """日線突破+ATR停損; tailwind[i]=True 才允許進場 (None=無濾網)。"""
    a = gs.atr(h, l, c, an)
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; tr = []
    for i in range(bo + an + 1, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            if px > h[i - bo:i].max() and (tailwind is None or tailwind[i]):
                pos, entry, ei, trail = 1, px, i, px - st * a[i]
        else:
            trail = max(trail, px - st * a[i])
            if px <= trail:
                tr.append({"pnl": (px - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                           "exit_date": dt[i]}); pos = 0
    if pos:
        tr.append({"pnl": (c[-1] - entry) * gs.POINT_VALUE - gs.COST_PER_SIDE * gs.POINT_VALUE * 2,
                   "exit_date": dt[-1]})
    return tr


def M(tr, lo=None, hi=None):
    if lo or hi:
        tr = [t for t in tr if (not lo or t["exit_date"] >= lo) and (not hi or t["exit_date"] < hi)]
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr]); eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()); w = p[p > 0]; ls = p[p < 0]
    return dict(n=len(tr), win=float((p > 0).mean()),
                pf=float(w.sum() / -ls.sum()) if len(ls) else 9.99,
                tot=float(p.sum()), dd=dd, mar=float(eq[-1] / dd) if dd > 0 else 0,
                sharpe=float(p.mean() / p.std()) if p.std() > 0 else 0)


def row(lbl, m):
    if not m:
        print(f"{lbl:<22} 無交易"); return
    print(f"{lbl:<22}{m['n']:>5}{m['win']:>7.1%}{m['pf']:>6.2f}{m['tot']:>+11,.0f}"
          f"{m['dd']:>10,.0f}{m['mar']:>7.2f}{m['sharpe']:>7.3f}")


HDR = f"{'':<22}{'筆':>5}{'勝率':>7}{'PF':>6}{'總損益$':>11}{'最大DD$':>10}{'報酬/DD':>7}{'Sharpe':>7}"


def main():
    print("═══ 宏觀資金環境濾網 (名目10y殖利率 ^TNX / 美元 DXY) ═══\n")
    print("抓取 Yahoo 宏觀資料...")
    tnx = fetch_yahoo_daily("%5ETNX", "TNX_1d.csv")
    dxy = fetch_yahoo_daily("DX-Y.NYB", "DXY_1d.csv")
    ddt, do, dh, dl, dc = gd.load()
    print(f"黃金日線 {ddt[0]}~{ddt[-1]}; ^TNX {'OK' if tnx else 'X'}({len(tnx) if tnx else 0}); "
          f"DXY {'OK' if dxy else 'X'}({len(dxy) if dxy else 0})\n")

    configs = [("採用 bo40/atr4/ATR20", dict(bo=40, st=4.0, an=20)),
               ("弱base bo24/atr2.5/ATR14", dict(bo=24, st=2.5, an=14))]

    for macro, name in [(tnx, "^TNX 名目10y殖利率"), (dxy, "美元指數 DXY")]:
        if not macro:
            print(f"── {name}: 資料不可用, 跳過 ──\n"); continue
        val = align(ddt, macro)
        for win in (50, 100, 200):
            ma = sma_series(val, win)
            tail = (val < ma)   # 順風: 殖利率/美元 低於其 MA (下行/轉弱)
            cov = np.mean(tail[~np.isnan(ma)])
            print(f"── 濾網: {name} < MA{win}  (順風佔比 {cov:.0%}) ──")
            for cname, kw in configs:
                b = M(bt(ddt, dh, dl, dc, **kw))
                f = M(bt(ddt, dh, dl, dc, tailwind=tail, **kw))
                print(f"  【{cname}】"); print("  " + HDR)
                row("  無濾網", b); row(f"  +{name}<MA{win}", f)
            print()


if __name__ == "__main__":
    main()
