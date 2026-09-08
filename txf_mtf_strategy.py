"""台指期 多時間框架順勢回檔波段 (txf_mtf) — 即時訊號 (實單參考)。

重用 `txf_mtf.run()` (單一程式碼真相): 抓最近 ^TWII 15 分 K 作『進出場時間框架』,
由 15 分 resample 出每日收盤算『日線 EMA20 趨勢濾網』, 對整段連續 15 分序列跑同一套
狀態機, 回報:
  ‧ 今日日線趨勢方向 (只做順勢);
  ‧ 目前是否持有波段部位 (進場價/停損/目前損益);
  ‧ 若空手: 今日可進場的條件與關鍵價位 (快線=回檔進場基準、慢線=趨勢破壞出場參考)。

⚠️ 判定為『觀察』(見 TXF_MTF_VALIDATION.md): 小時K 3年抗過擬合六檢多數過關, 但
bootstrap 顯著性未過 (t=1.61) 且高度依賴右尾大贏家; 15 分僅 60 天樣本不足。故此訊號
僅供『模擬/觀察累積樣本』, 尚未通過實單閘門, 請勿投入真實資金。
"""
import os

import txf_mtf

POINT_VALUE = txf_mtf.POINT_VALUE


def _recent_15m():
    """抓最近 15 分 K, 回傳 rows=[(dt,o,h,l,c)] 由舊到新 (只留正常盤中時段)。"""
    import txf_data
    raw = txf_data.fetch("15m", "60d") or []
    rows = []
    for dt, o, h, l, c, v in raw:
        hm = dt.split(" ")[1]
        if "09:00" <= hm <= "13:30":
            rows.append((dt, o, h, l, c))
    return rows


def live_report():
    """回傳台指多時間框架順勢回檔策略的今日狀態文字。"""
    rows = _recent_15m()
    if len(rows) < 60:
        return "❌ 台指MTF: 15分資料不足"
    bars, ddates, dcloses = txf_mtf.prepare(rows)
    bias = txf_mtf.daily_bias_map(ddates, dcloses)
    trades, pos = txf_mtf.run(bars, bias)
    last = bars[-1]
    today = last["date"]
    tbias = bias.get(today, 0)
    bias_txt = {1: "多頭 (只做多)", -1: "空頭 (只做空)", 0: "不明確 (觀望)"}[tbias]
    lines = [
        f"📐 台指 多時間框架順勢回檔波段 · {today}",
        f"日線趨勢濾網 EMA{txf_mtf.EMA_DAILY}：{bias_txt}（依昨收 vs 昨日EMA）",
        f"現價 {last['c']:.0f}｜時區快線 EMA{txf_mtf.EMA_FAST} {last['ef']:.0f}／"
        f"慢線 EMA{txf_mtf.EMA_SLOW} {last['es']:.0f}",
        "",
    ]
    if pos is not None:
        side = pos["side"]; entry = pos["entry"]; stop = pos["stop"]
        cur = last["c"]
        pnl = side * (cur - entry)
        lines.append(f"🟢 持有 {'多' if side>0 else '空'} 單（波段, 可留倉）")
        lines.append(f"進場 {entry:.0f}（{pos['entry_dt']}）｜停損 {stop:.0f}｜"
                     f"目前 {pnl:+.0f}點（NT${pnl*POINT_VALUE:+,.0f}）")
        lines.append(f"出場條件：觸停損 {stop:.0f}，或收盤{'跌破' if side>0 else '升破'}"
                     f"慢線 EMA{txf_mtf.EMA_SLOW}（{last['es']:.0f}）")
        equity = float(os.environ.get("TXF_EQUITY", "0") or 0)
        if equity > 0:
            units = max(1, int(equity * 0.01 // (txf_mtf.STOP_PT * POINT_VALUE)))
            lines.append(f"💰 權益 NT${equity:,.0f}·風險1%：建議 {units} 口（停損 "
                         f"{txf_mtf.STOP_PT:.0f}點）")
    else:
        if tbias > 0:
            cond = (f"多頭日：等『回檔到快線下再收回快線上』(快線 {last['ef']:.0f}) 且 "
                    f"快線>慢線 → 做多，停損 進場−{txf_mtf.STOP_PT:.0f}")
        elif tbias < 0:
            cond = (f"空頭日：等『反彈到快線上再收回快線下』(快線 {last['ef']:.0f}) 且 "
                    f"快線<慢線 → 做空，停損 進場+{txf_mtf.STOP_PT:.0f}")
        else:
            cond = "趨勢不明確日：今日不進場，等日線站上/跌破 EMA20 再說"
        lines.append(f"⚪ 目前空手")
        lines.append(cond)
    lines += [
        "",
        "⚠️ 判定=觀察（非實單）：bootstrap 未顯著、依賴右尾大贏家、15分樣本僅60天；",
        "　僅供模擬累積樣本，勿投真實資金。詳見 TXF_MTF_VALIDATION.md。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(live_report())
