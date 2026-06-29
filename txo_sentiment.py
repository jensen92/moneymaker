"""台指選擇權 (TXO) Put/Call 未平倉比率 — 反向情緒指標, 輔助台指期 (TXF) 進出.

資料: 台灣期交所 (TAIFEX) 每日 P/C ratio (pcRatioDown), 取『買賣權未平倉量比率%』
= 賣權OI / 買權OI ×100。

驗證 (2021-2026, 1225 日, 對台指隔日/後5日報酬):
  - P/C 水準有反向 edge: 賣權堆越多(恐慌)→ 之後越偏多。
      P/C 115-132 隔日 +0.19% 勝率61% (vs 低 P/C 55-90 隔日 +0.00% 勝率53%)
      極端恐慌 P/C≥138 (前15%): 後5日 +1.06% 勝率66% (基準 +0.51%)
      極端樂觀 P/C≤87 (前15%): 後5日 +0.23% (偏弱)
  - P/C『日變化』(分布移動) 相關 -0.009, 無預測力 → 用水準, 不用變化。
用途: 反向情緒『偏多/中性/偏空』濾網, 輔助 TXF 日內進出 (非獨立策略)。

⚠ 樣本內揭露: 門檻 (138/115/90/87) 與邊際 (+1.06%/+0.19%) 皆在同一段 2021-26
   資料上以分位數挑出並回報, 屬樣本內, 未做樣本外驗證; 邊際本就溫和 (隔日+0.19%),
   應僅當『方向偏好的輔助』, 切勿單獨據此進場或放大部位。
"""
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# 經 5 年驗證的門檻 (P/C 未平倉比率 %)
PANIC = 138    # ≥ 此 = 極端恐慌 → 強反向偏多
HIGH = 115     # ≥ 此 = 賣權偏多 → 偏多
LOW = 90       # ≤ 此 = 樂觀
EUPHORIA = 87  # ≤ 此 = 極端樂觀 → 偏空/謹慎


def fetch_pcr(months=2):
    """抓最近 months 個月的 P/C 未平倉比率, 回傳 [(date, ratio), ...] 由舊到新。"""
    import calendar
    import requests
    out = {}
    now = time.localtime()
    y, m = now.tm_year, now.tm_mon
    for _ in range(months):
        last = calendar.monthrange(y, m)[1]      # 當月實際最後一天 (避免月底 29-31 漏抓)
        d1 = f"{y}/{m:02d}/01"; d2 = f"{y}/{m:02d}/{last:02d}"
        try:
            r = requests.post("https://www.taifex.com.tw/cht/3/pcRatioDown",
                              data={"queryStartDate": d1, "queryEndDate": d2},
                              headers=HEADERS, timeout=20)
            for ln in r.content.decode("big5", "ignore").splitlines()[1:]:
                f = [x.strip() for x in ln.split(",")]
                if len(f) >= 7 and "/" in f[0]:
                    try:
                        out[f[0].replace("/", "-")] = float(f[6])
                    except ValueError:
                        pass
        except Exception:
            pass
        m -= 1
        if m == 0:
            m = 12; y -= 1
    return sorted(out.items())


def classify(pc):
    if pc >= PANIC:
        return ("🟢 反向偏多（極端恐慌）",
                "賣權大量堆積=市場過度避險; 歷史此時後5日+1.06% 勝率66%。"
                "TXF 偏作多/不追空, 留意反彈。")
    if pc >= HIGH:
        return ("🟢 偏多（賣權偏多）",
                "避險盤偏多, 歷史隔日+0.19% 勝率61%。TXF 多單訊號可較積極。")
    if pc <= EUPHORIA:
        return ("🔴 偏空/謹慎（極端樂觀）",
                "買權偏多=市場過度樂觀; 歷史此時後5日僅+0.23%(偏弱)。"
                "TXF 多單保守、留意回檔。")
    if pc <= LOW:
        return ("🟡 略偏謹慎（樂觀）",
                "情緒偏樂觀, 多單品質一般。")
    return ("⚪ 中性", "情緒中性, 依台指本身訊號操作。")


def report():
    """回傳台指選擇權情緒文字 (供 /txf 內嵌)。失敗回 None。"""
    hist = fetch_pcr(2)
    if not hist:
        return None
    date, pc = hist[-1]
    label, note = classify(pc)
    chg = ""
    if len(hist) >= 6:
        prev = hist[-6][1]
        chg = f"（5日前 {prev:.0f}→今 {pc:.0f}）"
    return (f"🎲 選擇權情緒 P/C未平倉比率 {pc:.0f}{chg}\n"
            f"  {label}\n  {note}")


def week_report():
    """含『近一週』P/C 走勢的情緒文字 (供盤中推播)。失敗回 None。"""
    hist = fetch_pcr(2)
    if not hist:
        return None
    last = hist[-5:]                      # 近 5 個交易日
    date, pc = hist[-1]
    label, note = classify(pc)
    # 近一週趨勢: 比較本週首尾
    if len(last) >= 2:
        diff = last[-1][1] - last[0][1]
        trend = ("恐慌升溫→更偏多" if diff >= 8 else
                 "樂觀升溫→轉謹慎" if diff <= -8 else "大致持平")
    else:
        trend = ""
    wk = "｜".join(f"{d[5:]} {v:.0f}" for d, v in last)
    return (f"🎲 選擇權情緒 P/C未平倉比率（賣權OI/買權OI）\n"
            f"近一週 {wk}\n"
            f"今日 {pc:.0f} {label}（週趨勢: {trend}）\n"
            f"{note}")


def extreme_alert():
    """若最新 P/C 為極端值 (≥PANIC 恐慌 / ≤EUPHORIA 樂觀), 回傳 (date, 推播文字);
    否則 (date_or_None, None)。供盤中監控『極端時單獨推播一次』, 呼叫端以 date 去重。"""
    hist = fetch_pcr(2)
    if not hist:
        return None, None
    date, pc = hist[-1]
    if pc < PANIC and pc > EUPHORIA:
        return date, None
    last = hist[-5:]
    wk = "｜".join(f"{d[5:]} {v:.0f}" for d, v in last)
    label, note = classify(pc)
    head = "🔥 選擇權極端恐慌訊號" if pc >= PANIC else "🔥 選擇權極端樂觀訊號"
    return date, (f"{head}（P/C未平倉比率 {pc:.0f}）\n"
                  f"近一週 {wk}\n{label}\n{note}")


if __name__ == "__main__":
    print(report() or "❌ 無法取得 P/C 資料")
