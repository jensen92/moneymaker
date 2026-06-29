"""台指期 (^TWII 代理) 波浪結構指標 — 時線為主, 月/週/日線定位.

誠實聲明: 艾略特波浪『自動辨識』本質主觀、無法 100% 客觀。本指標用『客觀可重現』
的組件給出輔助判讀, 不是定論:
  1. ZigZag 轉折點 (固定反轉門檻) — 客觀抓出各時間框架的顯著高低點。
  2. 趨勢結構 (高低點序列): 高點漸高+低點漸高=多頭(HH/HL); 反之空頭(LH/LL); 混合=盤整。
  3. 簡化波浪標記 (啟發式): 由轉折序列數『推進段』數估目前在第幾浪 (1-5 推動 / A-B-C 修正),
     並以費波那契回撤/延伸給可能目標。此為輔助, 浪數可能因新轉折而重算。
  4. 多時間框架: 月線定大方向、週線定中期、日線定波段、時線定當下進出。

用法: python3 txf_wave.py [--no-fetch]
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "futures_data", "twii_wave_cache.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# 各時間框架 ZigZag 反轉門檻 (分數) — 越大框架要越大波動才算一個轉折
THR = {"月": 0.10, "週": 0.06, "日": 0.035, "時": 0.018}


def _fetch(rng, iv):
    import requests
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII",
                     params={"range": rng, "interval": iv}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]; q = res["indicators"]["quote"][0]
    out = []
    for i, t in enumerate(ts):
        hh, ll, cc = q["high"][i], q["low"][i], q["close"][i]
        if None in (hh, ll, cc):
            continue
        out.append((int(t), float(hh), float(ll), float(cc)))
    return out


def fetch_all(path=CACHE):
    """抓時線(730d/1h)與日線(10y/1d), 重採樣出週/月線, 存快取。回傳 dict 或 None。"""
    try:
        hourly = _fetch("730d", "1h")
        daily = _fetch("10y", "1d")
    except Exception:
        return None
    data = {"時": hourly, "日": daily,
            "週": _resample(daily, 7), "月": _resample(daily, 30)}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(data, open(path, "w"))
    except Exception:
        pass
    return data


def _resample(daily, days):
    """把日線聚合成週(7日曆日)/月(自然月)K。回傳 (ts,h,l,c) list。"""
    if not daily:
        return []
    out = []
    bucket = None; key = None
    for t, h, l, c in daily:
        lt = time.gmtime(t)
        k = (lt.tm_year, lt.tm_mon) if days >= 30 else (lt.tm_year, lt.tm_yday // 7)
        if k != key:
            if bucket:
                out.append(bucket)
            bucket = [t, h, l, c]; key = k
        else:
            bucket[1] = max(bucket[1], h); bucket[2] = min(bucket[2], l); bucket[3] = c
    if bucket:
        out.append(bucket)
    return out


def zigzag(bars, thr):
    """回傳轉折點 [(index, price, type)], type=+1 高 / -1 低; 末端附當前運行極值。"""
    if len(bars) < 3:
        return []
    h = np.array([b[1] for b in bars]); l = np.array([b[2] for b in bars])
    piv = []; trend = 0
    hi = h[0]; hi_i = 0; lo = l[0]; lo_i = 0
    for i in range(1, len(bars)):
        if trend >= 0:
            if h[i] >= hi:
                hi, hi_i = h[i], i
            if l[i] <= hi * (1 - thr):
                piv.append((hi_i, float(hi), 1)); trend = -1; lo, lo_i = l[i], i
                continue
        if trend <= 0:
            if l[i] <= lo:
                lo, lo_i = l[i], i
            if h[i] >= lo * (1 + thr):
                piv.append((lo_i, float(lo), -1)); trend = 1; hi, hi_i = h[i], i
    # 末端當前運行極值 (暫定轉折; 與上一個不同型才加)
    tail = (hi_i, float(hi), 1) if trend >= 0 else (lo_i, float(lo), -1)
    if not piv or piv[-1][2] != tail[2]:
        piv.append(tail)
    return piv


def trend_of(piv):
    """由最後兩高兩低判趨勢: 多頭(HH/HL)/空頭(LH/LL)/盤整。"""
    highs = [p[1] for p in piv if p[2] == 1]
    lows = [p[1] for p in piv if p[2] == -1]
    if len(highs) < 2 or len(lows) < 2:
        return "資料不足"
    hh = highs[-1] > highs[-2]; hl = lows[-1] > lows[-2]
    if hh and hl:
        return "多頭"
    if (not hh) and (not hl):
        return "空頭"
    return "盤整"


def wave_label(piv, price):
    """簡化波浪啟發式: 從最近主轉折數推進段, 估目前浪數 + 費波那契目標。
    回傳 (浪標籤, 說明, 目標dict)。"""
    if len(piv) < 3:
        return "—", "轉折不足", {}
    trend = trend_of(piv)
    # 取最近一段顯著推進 (最後兩個轉折構成『上一完成段』) 算費波那契
    last = piv[-1]; prev = piv[-2]
    leg = abs(last[1] - prev[1])
    base = last[1]
    # 數最近同向推進段數 (粗估浪位)
    # 取最後 ~7 個轉折, 計算趨勢方向的推進段數
    recent = piv[-7:]
    up_legs = sum(1 for a, b in zip(recent, recent[1:]) if b[1] > a[1])
    dn_legs = sum(1 for a, b in zip(recent, recent[1:]) if b[1] < a[1])
    tgt = {}
    if trend == "多頭":
        # 目前若在上升推進段
        n = min(up_legs, 5)
        if last[2] == -1:  # 剛打出低點, 準備往上 → 修正末端/起漲
            label = f"上升修正末端(可能第{min(2*((up_legs))+0,4) or 2}浪低後轉強)"
            note = "回檔可能告一段落, 守此低點偏多"
            tgt = {"反彈目標(0.618回)": base + 0.618 * leg, "前高": prev[1]}
        else:  # 剛打出高點
            label = f"上升推進第{n}段(多頭, 啟發式)"
            note = "推進段; 留意是否進入修正"
            tgt = {"延伸1.272": prev[1] + 1.272 * leg if False else base,
                   "回檔0.382": base - 0.382 * leg, "回檔0.618": base - 0.618 * leg}
    elif trend == "空頭":
        n = min(dn_legs, 5)
        if last[2] == 1:
            label = "下跌反彈末端(可能反彈後再探低)"
            note = "反彈可能告一段落, 跌破前低偏空"
            tgt = {"回測0.618": base - 0.618 * leg, "前低": prev[1]}
        else:
            label = f"下跌推進第{n}段(空頭, 啟發式)"
            note = "下跌段; 留意是否進入反彈"
            tgt = {"反彈0.382": base + 0.382 * leg, "反彈0.618": base + 0.618 * leg}
    else:
        label = "盤整(波浪不明確)"
        note = "區間震盪, 等突破前高/前低再定方向"
        tgt = {"區間上緣": max(prev[1], last[1]), "區間下緣": min(prev[1], last[1])}
    return label, note, tgt


def _fmt_pivot(bars, p, is_hour):
    idx = p[0]
    off = 8 * 3600 if is_hour else 0
    d = time.strftime("%m-%d %H:%M" if is_hour else "%y-%m-%d",
                      time.gmtime(bars[idx][0] + off))
    return f"{'高' if p[2] == 1 else '低'}{p[1]:,.0f}@{d}"


def report(data=None):
    """回傳台指波浪結構文字 (供 /txf 內嵌)。失敗回 None。"""
    if data is None:
        data = fetch_all()
        if data is None and os.path.exists(CACHE):
            data = json.load(open(CACHE))
    if not data:
        return None
    price = data["時"][-1][3] if data["時"] else (data["日"][-1][3] if data["日"] else 0)
    lines = ["🌊 台指波浪結構（輔助判讀, 非定論）",
             f"現價 {price:,.0f}"]
    # 多時間框架趨勢
    tf_line = []
    for tf in ("月", "週", "日"):
        piv = zigzag(data[tf], THR[tf])
        tf_line.append(f"{tf}線 {trend_of(piv)}")
    lines.append("　".join(tf_line))
    # 時線波浪 (主角)
    hp = zigzag(data["時"], THR["時"])
    label, note, tgt = wave_label(hp, price)
    lines.append(f"時線 {trend_of(hp)}｜目前: {label}")
    lines.append(f"　{note}")
    if hp:
        lines.append("近期轉折 " + " ".join(_fmt_pivot(data["時"], p, True) for p in hp[-3:]))
    if tgt:
        lines.append("關鍵價位 " + "｜".join(f"{k} {v:,.0f}" for k, v in tgt.items()))
    lines.append("(波浪屬主觀判讀, 僅輔助; 以實際突破/跌破關鍵價位為準)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    data = None
    if args.no_fetch and os.path.exists(CACHE):
        data = json.load(open(CACHE))
    print(report(data) or "❌ 無法取得台指資料")


if __name__ == "__main__":
    main()
