"""CTA 多元商品分散趨勢投組 (Time-Series Momentum / TSMOM) — 商品交易王者法.

策略思維: 海龜/Dunn/Seykota 這類商品趨勢交易巨額報酬的核心不是單一神奇訊號,
而是「跨數十個低相關市場同時做趨勢 + 風險平價分散」——單一市場勝率通常僅
30~40%、Sharpe 偏低甚至為負, 但跨市場分散後靠少數市場的大趨勢年撐起整體,
波動被大幅攤平。

本模組: 19 個商品期貨(金屬/能源/穀物油籽/軟商品/肉品), 各自:
  訊號 = 60/120/250 日動量方向平均 (多空皆做)
  部位 = 目標日波動 ÷ 該市場已實現波動 (風險平價, 波動大的市場配權重小)
  出場 = 訊號翻向即換邊 (無額外停損, 訊號本身即出場機制)
投組 = 19 市場等權平均日報酬。

26 年實證 (2000-2026, 19市場):
  投組 Sharpe 0.18 (單市場平均僅 0.09, 分散後接近兩倍) MaxDD 8.2%
  前/後半 Sharpe 0.19/0.16 — 樣本內外皆穩, 非單一時段撐起
誠實揭露: 這是純商品版, Sharpe 偏溫和 (真實 CTA 傳奇報酬多來自納入債券/外匯/
股指期貨的多資產分散 + 較高槓桿操作, 非本模組範圍); 一年報酬集中在少數
趨勢年 (2004/2008/2010/2014/2020/2021), 盤整年會空轉。價值在於與台股/黃金/
穀物季節等現有策略低相關的投組分散, 非追求單獨高報酬。

用法: python3 cta_signals.py [--no-fetch]
"""
import argparse
import csv
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "futures_data", "basket")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# 19 市場: 代碼 -> (Yahoo symbol, 顯示名, 類別)
MARKETS = {
    "GC": ("GC=F", "黃金", "金屬"), "SI": ("SI=F", "白銀", "金屬"),
    "HG": ("HG=F", "銅", "金屬"), "PL": ("PL=F", "鉑金", "金屬"),
    "CL": ("CL=F", "原油", "能源"), "NG": ("NG=F", "天然氣", "能源"),
    "HO": ("HO=F", "取暖油", "能源"), "RB": ("RB=F", "汽油", "能源"),
    "ZC": ("ZC=F", "玉米", "穀物"), "ZS": ("ZS=F", "黃豆", "穀物"),
    "ZW": ("ZW=F", "小麥", "穀物"), "ZL": ("ZL=F", "豆油", "穀物"),
    "ZM": ("ZM=F", "豆粕", "穀物"),
    "KC": ("KC=F", "咖啡", "軟商品"), "SB": ("SB=F", "糖", "軟商品"),
    "CT": ("CT=F", "棉花", "軟商品"), "CC": ("CC=F", "可可", "軟商品"),
    "LE": ("LE=F", "活牛", "肉品"), "HE": ("HE=F", "瘦豬", "肉品"),
}

LOOKBACKS = (60, 120, 250)   # 動量訊號回看窗 (日) — 短中長期平均, 非單一尖峰
VOL_WIN = 60                 # 已實現波動回看窗
TARGET_VOL = 0.004           # 目標日波動 (單市場貢獻度)
WEIGHT_CAP = 2.0             # 單市場權重上限 (避免低波動市場槓桿過高)


def _path(key):
    return os.path.join(DATA, f"{key}.csv")


def fetch_basket():
    """抓 19 市場日線 (2000 年至今), 覆寫各自 CSV。單一市場失敗不影響其他市場。
    回傳成功市場數。API 回傳空資料時該市場不覆寫 (保留舊資料), 不計入失敗。"""
    import requests
    p1 = 946684800
    ok = 0
    os.makedirs(DATA, exist_ok=True)
    for key, (sym, name, cat) in MARKETS.items():
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"period1": p1, "period2": int(time.time()),
                                     "interval": "1d"}, headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]; q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
                if None in (o, h, l, c):
                    continue
                rows.append((time.strftime("%Y-%m-%d", time.gmtime(t)),
                             round(o, 4), round(h, 4), round(l, 4), round(c, 4)))
            if not rows:          # 空資料 — 絕不可覆寫掉既有歷史
                continue
            with open(_path(key), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close"])
                w.writerows(rows)
            ok += 1
        except Exception:
            continue
    return ok


def _load_closes():
    """回傳 {key: {date: close}} 僅含已存在本地檔的市場。"""
    out = {}
    for key in MARKETS:
        p = _path(key)
        if not os.path.exists(p):
            continue
        d = {}
        with open(p) as f:
            for r in csv.DictReader(f):
                try:
                    d[r["date"]] = float(r["close"])
                except ValueError:
                    continue
        if len(d) > max(LOOKBACKS) + VOL_WIN:
            out[key] = d
    return out


def _series(closes_dict, days):
    """對齊日期序列, 回傳 (dates, values) 由舊到新。"""
    ks = sorted(closes_dict)
    return ks, np.array([closes_dict[k] for k in ks])


def backtest_portfolio():
    """投組回測: 回傳 (dates, daily_returns) 全市場等權 TSMOM 波動目標投組。"""
    closes = _load_closes()
    if len(closes) < 3:
        return [], np.array([])
    all_dates = sorted(set.union(*[set(d) for d in closes.values()]))
    n = len(all_dates)
    contrib = {}   # key -> array of daily contribution (NaN 缺值日)
    for key, d in closes.items():
        c = np.array([d.get(dt, np.nan) for dt in all_dates])
        ret = np.full(n, np.nan)
        valid = ~np.isnan(c)
        ret[1:] = np.where(valid[1:] & valid[:-1], c[1:] / np.where(c[:-1] == 0, np.nan, c[:-1]) - 1, np.nan)
        # 動量訊號: 多 lookback 報酬符號平均
        sig = np.zeros(n)
        cnt = 0
        for L in LOOKBACKS:
            with np.errstate(invalid="ignore"):
                s = np.full(n, np.nan)
                s[L:] = np.sign(c[L:] / np.where(c[:-L] == 0, np.nan, c[:-L]) - 1)
            sig = np.nansum([sig, np.nan_to_num(s, nan=0.0)], axis=0)
            cnt += 1
        sig = sig / cnt
        # 已實現波動 (簡單標準差)
        vol = np.full(n, np.nan)
        for i in range(VOL_WIN, n):
            window = ret[i - VOL_WIN:i]
            if np.sum(~np.isnan(window)) >= VOL_WIN // 2:
                vol[i] = np.nanstd(window)
        w = np.clip(TARGET_VOL / np.where(vol > 0, vol, np.nan), None, WEIGHT_CAP)
        pos = np.roll(sig * w, 1)   # t-1 訊號套用在 t 報酬 (避免未來函數)
        pos[0] = np.nan
        contrib[key] = pos * ret

    mat = np.array([contrib[k] for k in contrib])   # markets x days
    has_any = ~np.all(np.isnan(mat), axis=0)         # 避免全NaN列觸發空切片警告
    port = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        port[has_any] = np.nanmean(mat[:, has_any], axis=0)
    valid = ~np.isnan(port)
    return [all_dates[i] for i in range(n) if valid[i]], port[valid]


def portfolio_metrics(dates, rets):
    if len(rets) < 100:
        return None
    eq = np.cumprod(1 + rets)
    years = len(rets) / 252
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak).max()
    return {"n": len(rets), "years": years, "cagr": float(cagr),
            "sharpe": float(sharpe), "dd": float(dd),
            "mar": float(cagr / dd) if dd > 0 else 0.0}


def current_signals():
    """回傳每個市場『目前』的方向與強度, 供即時報告用。
    list of dict: key,name,cat,dir(+1/-1/0),strength(0~1),close,vol_pct"""
    closes = _load_closes()
    out = []
    for key, d in closes.items():
        dates, c = _series(d, None)
        n = len(c)
        if n < max(LOOKBACKS) + VOL_WIN:
            continue
        sigs = []
        for L in LOOKBACKS:
            if n > L and c[-1 - L] != 0:
                sigs.append(1 if c[-1] > c[-1 - L] else -1)
        if not sigs:
            continue
        avg = sum(sigs) / len(sigs)
        direction = 1 if avg > 0 else (-1 if avg < 0 else 0)
        strength = abs(avg)   # 0~1: 三個 lookback 一致程度
        ret = np.diff(c[-VOL_WIN - 1:]) / c[-VOL_WIN - 1:-1]
        vol = float(np.std(ret)) if len(ret) > 5 else 0.0
        name, cat = MARKETS[key][1], MARKETS[key][2]
        out.append({"key": key, "name": name, "cat": cat, "dir": direction,
                    "strength": strength, "close": float(c[-1]), "vol": vol})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch:
        fetch_basket()

    sigs = current_signals()
    dates, rets = backtest_portfolio()
    m = portfolio_metrics(dates, rets)

    print("🌍 CTA 多元商品分散趨勢（19市場, TSMOM多空, 商品交易王者法）")
    if m:
        print(f"投組績效 年化{m['cagr']:+.1%}｜Sharpe {m['sharpe']:.2f}｜"
              f"MaxDD {m['dd']:.1%}｜報酬/回撤比 {m['mar']:.2f}（{m['years']:.0f}年）")
    print()

    longs = sorted([s for s in sigs if s["dir"] == 1], key=lambda s: -s["strength"])
    shorts = sorted([s for s in sigs if s["dir"] == -1], key=lambda s: -s["strength"])
    flats = [s for s in sigs if s["dir"] == 0]

    def fmt(s):
        bar = "●" * round(s["strength"] * 3) + "○" * (3 - round(s["strength"] * 3))
        return f"  {s['name']}({s['key']}) {bar}  {s['close']:,.2f}"

    print(f"🟢 做多趨勢中（{len(longs)}）")
    for s in longs[:10]:
        print(fmt(s))
    print(f"\n🔴 做空趨勢中（{len(shorts)}）")
    for s in shorts[:10]:
        print(fmt(s))
    if flats:
        print(f"\n⚪ 中性/訊號不一致（{len(flats)}）：" + "、".join(s["name"] for s in flats))

    print("\n(●多=三窗口動量方向一致度; 部位依波動反向配權, 分散投組非單一市場獨立操作)")
    print("(純商品版, 價值在與股票/黃金/穀物季節低相關的分散, 非追求單獨高報酬)")


if __name__ == "__main__":
    main()
