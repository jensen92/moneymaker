"""資金流向研究 (3/3): COT 持倉 — 大戶資金部位流向能否提前預警或跟隨提升績效。

COT = CFTC 每週持倉報告 (週二數據, 週五公布)。黃金 (COMEX, 代碼 088691) 的
Managed Money (基金/投機大戶) 淨部位是『資金部位流向』的正牌指標。測兩種用法:
  (A) 跟隨/確認: MM 淨部位站上其 MA (資金持續流入) 才做多。
  (B) 反向預警: MM 淨部位在極端高檔 (過度擁擠) 時不做多 — 『提前知道』反轉風險。

去lookahead: 每個黃金交易日對齊到『已公布』的最近一期 COT (報告日+3日 ≤ 當日)。
定位為日線 regime 脈絡 (COT 週級, 不可能是小時訊號)。四關: 日線26年 + base敏感度。
"""
import csv
import io
import os
import time
import zipfile
from datetime import datetime, timedelta

import numpy as np

import gold_strategies as gs
import gold_daily_backtest as gd

H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
DIR = os.path.join(gs.HERE, "futures_data")
COT_CACHE = os.path.join(DIR, "COT_gold.csv")
GOLD_CODE = "088691"   # GOLD - COMMODITY EXCHANGE INC.


def fetch_cot(years=range(2006, 2027)):
    """抓 CFTC disaggregated 期貨持倉年檔, 抽 COMEX 黃金 MM 多空與 OI → 快取。"""
    if os.path.exists(COT_CACHE):
        rows = list(csv.DictReader(open(COT_CACHE)))
        if rows:
            return rows
    import requests
    out = []
    for y in years:
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip"
        try:
            r = requests.get(url, headers=H, timeout=60)
            if r.status_code != 200:
                continue
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            name = zf.namelist()[0]
            txt = zf.read(name).decode("latin-1")
            for row in csv.DictReader(io.StringIO(txt)):
                code = (row.get("CFTC_Contract_Market_Code") or "").strip()
                if code != GOLD_CODE:
                    continue
                rd = (row.get("Report_Date_as_YYYY-MM-DD") or "").strip()[:10]
                try:
                    ml = float(row.get("M_Money_Positions_Long_All") or "nan")
                    ms = float(row.get("M_Money_Positions_Short_All") or "nan")
                    oi = float(row.get("Open_Interest_All") or "nan")
                except ValueError:
                    continue
                out.append({"date": rd, "mm_long": ml, "mm_short": ms, "oi": oi})
            print(f"  {y}: 累計 {len(out)} 期")
        except Exception as e:
            print(f"  {y}: 失敗 {type(e).__name__}")
    out.sort(key=lambda x: x["date"])
    if out:
        with open(COT_CACHE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "mm_long", "mm_short", "oi"])
            w.writeheader(); w.writerows(out)
    return out


def build_series(cot):
    """→ (release_dates, mm_net_pct_oi)。以報告日+3日近似公布日 (去lookahead)。"""
    rel, net = [], []
    for r in cot:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d")
        except ValueError:
            continue
        released = (d + timedelta(days=3)).strftime("%Y-%m-%d")
        oi = r["oi"] if r["oi"] > 0 else np.nan
        rel.append(released)
        net.append((r["mm_long"] - r["mm_short"]) / oi if oi == oi else np.nan)
    return rel, np.array(net)


def align_asof(gold_dates, rel_dates, series):
    out = np.full(len(gold_dates), np.nan)
    j = 0
    for i, d in enumerate(gold_dates):
        while j + 1 < len(rel_dates) and rel_dates[j + 1] <= d:
            j += 1
        if rel_dates and rel_dates[j] <= d:
            out[i] = series[j]
    return out


def rolling_ma(x, n):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - n + 1):i + 1]; seg = seg[~np.isnan(seg)]
        if len(seg) >= max(2, n // 2):
            out[i] = seg.mean()
    return out


def rolling_pctile(x, i, win=104):
    seg = x[max(0, i - win + 1):i + 1]; seg = seg[~np.isnan(seg)]
    if len(seg) < 20 or np.isnan(x[i]):
        return np.nan
    return (seg < x[i]).mean()


def bt(dt, h, l, c, *, allow=None, bo=40, st=4.0, an=20):
    a = gs.atr(h, l, c, an)
    pos = 0; entry = 0.0; trail = 0.0; ei = 0; tr = []
    for i in range(bo + an + 1, len(c)):
        px = c[i]
        if np.isnan(a[i]):
            continue
        if pos == 0:
            if px > h[i - bo:i].max() and (allow is None or allow[i]):
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


def M(tr):
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
        print(f"{lbl:<24} 無交易"); return
    print(f"{lbl:<24}{m['n']:>5}{m['win']:>7.1%}{m['pf']:>6.2f}{m['tot']:>+11,.0f}"
          f"{m['dd']:>10,.0f}{m['mar']:>7.2f}{m['sharpe']:>7.3f}")


HDR = f"{'':<24}{'筆':>5}{'勝率':>7}{'PF':>6}{'總損益$':>11}{'最大DD$':>10}{'報酬/DD':>7}{'Sharpe':>7}"


def main():
    print("═══ COT Managed Money 持倉研究 (COMEX 黃金 088691) ═══\n抓取 CFTC...")
    cot = fetch_cot()
    if not cot:
        print("COT 資料不可用, 結束。"); return
    rel, net = build_series(cot)
    print(f"COT 期數 {len(rel)}  {rel[0]}~{rel[-1]}  (MM淨部位/OI)\n")
    ddt, do, dh, dl, dc = gd.load()
    netd = align_asof(ddt, rel, net)
    # COT 從 2006 才有 → 回測起點對齊到 COT 涵蓋期
    have = ~np.isnan(netd)
    print(f"黃金日線對齊到 COT 的有效覆蓋: {have.mean():.0%} (2006 前無 COT)\n")

    ma26 = rolling_ma(netd, 26)   # ~半年 MA
    pct = np.array([rolling_pctile(netd, i, 104) for i in range(len(netd))])  # 2年百分位
    trend = (netd > ma26)                       # (A) 資金流入: MM淨 > 半年均
    not_crowded = ~(pct > 0.90)                 # (B) 非極端擁擠 (<90百分位) 才做多

    configs = [("採用 bo40/atr4/ATR20", dict(bo=40, st=4.0, an=20)),
               ("弱base bo24/atr2.5/ATR14", dict(bo=24, st=2.5, an=14))]
    # 只在 COT 覆蓋期比較 (2006+): 用 allow 過濾 + 基準也限縮同期才公平 → 以 netd 有值為門檻
    for lbl, allow in [("(A) MM淨 > 半年MA (資金流入才做多)", trend & have),
                       ("(B) MM淨 <90百分位 (非擁擠才做多)", not_crowded & have),
                       ("基準同期 (2006+, 無COT濾網)", have)]:
        print(f"── {lbl} ──")
        for cname, kw in configs:
            print(f"  【{cname}】"); print("  " + HDR)
            row("  結果", M(bt(ddt, dh, dl, dc, allow=allow, **kw)))
        print()


if __name__ == "__main__":
    main()
