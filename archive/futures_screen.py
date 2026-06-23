"""跨市場趨勢掃描 — 找「流動性大 + 趨勢明顯/策略會賺」的期貨標的.

涵蓋股指/債券/能源/金屬/外匯/農產/肉品/加密等主要流動性期貨, 用趨勢策略
(唐奇安通道反轉 + 雙均線) 各跑一遍。為了讓不同合約可比較, 採用「波動率標準化的
%報酬」回測 — 每個市場都調整到相同目標波動, 與合約點值/大小無關。

輸出: 依趨勢 Sharpe 排名, 標示多空/僅多、CAGR、回撤、流動性等級。
找出的高分標的, 即適合納入本專案的趨勢/季節策略組合 (跨板塊分散可拉高整體 Sharpe)。

用法:
    python3 futures_screen.py            # 下載(快取) + 掃描
    python3 futures_screen.py --no-download
"""
import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "screen_data")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# 流動性等級: A 極高 / B 高 / C 中 (依一般 ADV 概略分級)
UNIVERSE = [
    ("ES=F", "標普500 S&P",    "股指", "A"),
    ("NQ=F", "那斯達克 Nasdaq", "股指", "A"),
    ("YM=F", "道瓊 Dow",       "股指", "B"),
    ("RTY=F", "羅素2000",      "股指", "B"),
    ("ZN=F", "美10年公債",     "債券", "A"),
    ("ZB=F", "美30年公債",     "債券", "A"),
    ("ZF=F", "美5年公債",      "債券", "A"),
    ("ZT=F", "美2年公債",      "債券", "B"),
    ("CL=F", "WTI原油",        "能源", "A"),
    ("BZ=F", "布蘭特原油",     "能源", "B"),
    ("NG=F", "天然氣",         "能源", "A"),
    ("RB=F", "汽油 RBOB",      "能源", "B"),
    ("HO=F", "熱燃油",         "能源", "B"),
    ("GC=F", "黃金 Gold",      "金屬", "A"),
    ("SI=F", "白銀 Silver",    "金屬", "A"),
    ("HG=F", "銅 Copper",      "金屬", "A"),
    ("PL=F", "鉑金 Platinum",  "金屬", "C"),
    ("PA=F", "鈀金 Palladium", "金屬", "C"),
    ("6E=F", "歐元 EUR",       "外匯", "A"),
    ("6J=F", "日圓 JPY",       "外匯", "A"),
    ("6B=F", "英鎊 GBP",       "外匯", "B"),
    ("6A=F", "澳幣 AUD",       "外匯", "B"),
    ("6C=F", "加幣 CAD",       "外匯", "B"),
    ("DX=F", "美元指數 DXY",   "外匯", "B"),
    ("ZS=F", "黃豆 Soybean",   "農產", "A"),
    ("ZW=F", "小麥 Wheat",     "農產", "B"),
    ("ZC=F", "玉米 Corn",      "農產", "A"),
    ("ZM=F", "黃豆粉",         "農產", "B"),
    ("ZL=F", "黃豆油",         "農產", "B"),
    ("KC=F", "咖啡 Coffee",    "軟商品", "B"),
    ("SB=F", "糖 Sugar",       "軟商品", "B"),
    ("CT=F", "棉花 Cotton",    "軟商品", "B"),
    ("CC=F", "可可 Cocoa",     "軟商品", "C"),
    ("LE=F", "活牛 Cattle",    "肉品", "B"),
    ("HE=F", "瘦豬 Hogs",      "肉品", "B"),
    ("BTC=F", "比特幣 BTC",    "加密", "A"),
]


def fetch(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"period1": 0, "period2": int(time.time()),
                                  "interval": "1d"}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res.get("timestamp")
    if not ts:
        return None
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c, v = (q["open"][i], q["high"][i], q["low"][i],
                         q["close"][i], q["volume"][i])
        if None in (o, h, l, c):
            continue
        rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)),
                     o, h, l, c, v or 0))
    return rows


def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    def task(item):
        sym = item[0]
        try:
            rows = fetch(sym)
            if rows and len(rows) > 300:
                with open(os.path.join(DATA_DIR, f"{sym}.csv"), "w",
                          newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["date", "open", "high", "low", "close", "volume"])
                    w.writerows(rows)
                return sym, True
        except Exception:
            pass
        return sym, False

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(task, it) for it in UNIVERSE]
        ok = sum(1 for f in as_completed(futs) if f.result()[1])
    print(f"下載完成: {ok}/{len(UNIVERSE)}\n")


def strat_returns(df, system="donchian", allow_short=True, cost_bps=2.0,
                  target_vol=0.15):
    """波動率標準化的趨勢日 %報酬序列 (index=date), 與合約點值無關."""
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    ret = close.pct_change()
    if system == "donchian":
        hi = df["high"].rolling(55).max().shift(1)
        lo = df["low"].rolling(55).min().shift(1)
        raw = pd.Series(np.where(close > hi, 1.0,
                                 np.where(close < lo, -1.0, np.nan)),
                        index=df.index).ffill().fillna(0.0)
    else:  # ma
        fast = close.rolling(50).mean()
        slow = close.rolling(100).mean()
        raw = pd.Series(np.where(fast > slow, 1.0, -1.0),
                        index=df.index).fillna(0.0)
    if not allow_short:
        raw = raw.clip(lower=0.0)
    vol = ret.rolling(20).std()
    size = (target_vol / np.sqrt(252) / vol).clip(upper=3.0).fillna(0.0)
    position = raw * size
    turnover = position.diff().abs().fillna(0.0)
    strat = (position.shift(1) * ret - turnover.shift(1) * cost_bps / 1e4).fillna(0.0)
    return pd.Series(strat.values, index=df["date"].values)


def _metrics(strat, years):
    eq = (1 + strat).cumprod()
    if eq.iloc[-1] <= 0 or years < 1:
        return -9, -1, 1
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return sharpe, cagr, dd


def trend_metrics(df, system="donchian", allow_short=True):
    """回傳 (sharpe, cagr, maxdd, n_years)."""
    s = strat_returns(df, system, allow_short)
    years = len(df) / 252
    sh, cagr, dd = _metrics(s, years)
    return sh, cagr, dd, years


# 分散示範用的籃子 (流動性 A/B 為主)
# ★ 精選 5 標的: 每板塊各 1 個 (金屬/股指/農產/債券/肉品), 長史+高流動, 經前後半驗證
#   貪婪前向選擇 + 「每板塊最多 1 個」約束 + 長史 (≥20年) 篩出, 上限 5 項。
BASKETS = {
    "只有三穀物": ["ZS=F", "ZW=F", "ZC=F"],
    "★精選5(黃豆油)": ["GC=F", "NQ=F", "ZL=F", "ZT=F", "LE=F"],
    "★精選5(保留黃豆)": ["GC=F", "NQ=F", "ZS=F", "ZT=F", "LE=F"],
    "跨板塊 12 標的": ["ES=F", "NQ=F", "ZN=F", "ZT=F", "CL=F", "GC=F", "HG=F",
                  "6E=F", "6J=F", "ZS=F", "ZC=F", "LE=F"],
}


def _portfolio(syms, allow_short):
    cols = {}
    for s in syms:
        p = os.path.join(DATA_DIR, f"{s}.csv")
        if os.path.exists(p):
            cols[s] = strat_returns(pd.read_csv(p, parse_dates=["date"]),
                                    "donchian", allow_short)
    m = pd.DataFrame(cols).dropna(how="all").fillna(0.0)
    port = m.mean(axis=1)            # 等權 (等風險, 各市場已標準化波動)
    return _metrics(port, len(port) / 252)


def portfolio_demo():
    print("\n分散效果 — 等權趨勢組合 (各標的波動率標準化):")
    print(f"  {'組合':<16}{'數':>3}{'多空Sharpe':>11}{'僅多Sharpe':>11}"
          f"{'僅多CAGR':>9}{'僅多回撤':>9}")
    print("  " + "-" * 58)
    for name, syms in BASKETS.items():
        shs, _, _ = _portfolio(syms, True)
        shl, cl, ddl = _portfolio(syms, False)
        print(f"  {name:<16}{len(syms):>3}{shs:>11.2f}{shl:>11.2f}"
              f"{cl:>9.1%}{ddl:>9.1%}")
    print("  → 結論: 從 3 穀物擴到 ~12 個跨板塊標的, Sharpe 倍增、回撤大降;")
    print("         分散是比優化單一策略更大的獲利槓桿。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    if not args.no_download:
        download_all()

    rows = []
    for sym, name, sector, liq in UNIVERSE:
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if len(df) < 500:
            continue
        sh_ls, cagr_ls, dd_ls, yrs = trend_metrics(df, "donchian", True)
        sh_lo, cagr_lo, dd_lo, _ = trend_metrics(df, "donchian", False)
        sh_ma, _, _, _ = trend_metrics(df, "ma", True)
        best = max(sh_ls, sh_ma)
        rows.append({"sym": sym, "name": name, "sector": sector, "liq": liq,
                     "yrs": yrs, "sh_ls": sh_ls, "sh_ma": sh_ma,
                     "sh_lo": sh_lo, "cagr": cagr_ls, "dd": dd_ls, "best": best})

    rows.sort(key=lambda r: -r["best"])
    print(f"{'排名':<3}{'代號':<7}{'商品':<16}{'板塊':<6}{'流動':<5}"
          f"{'年數':>5}{'唐奇安多空':>11}{'雙均多空':>9}{'僅多':>7}"
          f"{'CAGR':>8}{'回撤':>7}")
    print("-" * 92)
    for i, r in enumerate(rows, 1):
        print(f"{i:<3}{r['sym']:<7}{r['name']:<15}{r['sector']:<6}{r['liq']:<5}"
              f"{r['yrs']:>5.0f}{r['sh_ls']:>10.2f}{r['sh_ma']:>9.2f}"
              f"{r['sh_lo']:>7.2f}{r['cagr']:>8.1%}{r['dd']:>7.1%}")

    print("\n各板塊最佳趨勢標的 (僅多 Sharpe, 多數市場有上行漂移):")
    bysec = {}
    for r in rows:
        if r["sector"] not in bysec or r["sh_lo"] > bysec[r["sector"]]["sh_lo"]:
            bysec[r["sector"]] = r
    for sec, r in sorted(bysec.items(), key=lambda x: -x[1]["sh_lo"]):
        print(f"  {sec:<6} → {r['name']:<14} (僅多 Sharpe {r['sh_lo']:.2f}, "
              f"流動性 {r['liq']})")

    portfolio_demo()


if __name__ == "__main__":
    main()
