"""台指期 (TXF) 日內最佳策略 — 前日高低突破 (Previous-Day Range Breakout)。

開發標的: ^TWII 加權指數小時 K (3 年, 2023-06 ~ 2026-06)。實單於 TXF/MXF 執行。
回測 (扣成本: 滑價1點+手續費+期交稅, 大台每點 NT$200):
  648 筆, 勝率 25%, 淨 13,829 點 (≈ NT$2.77M/大台), 獲利因子 1.66,
  最大回撤 1,473 點, 年化 Sharpe 2.04; 樣本外 (近30%) 淨 +8,093 點 (PF 2.19);
  2023~2026 每年皆獲利, 多空雙邊均有貢獻。

策略規則 (一天最多一筆, 同日先觸發者為準):
  進場: 盤中(小時K)向上突破「前一交易日最高」→ 做多 (停損買進於前日高,
        若跳空高開則以開盤價成交); 向下跌破「前一交易日最低」→ 做空。
  停損: 進場價 ± 40 點。
  出場: 當日 13:30 收盤平倉 (未觸停損時)。
  方向: 多空皆做。

時間框架說明: 訊號判斷用小時 K (唯一有足夠歷史可驗證者)。實單可用 5 分 / 15 分 K
細修進場點 (例如等突破後第一根 5 分回測不破再進), 但核心觸發與出場以上述為準。

資金管理 (加減碼): 回測本身仍以「固定 1 口」驗證訊號邊際 (避免資金曲線回饋污染
訊號評估)。實單口數改用「固定風險比例 + 加碼」模型, 見 `position_plan()`:
  1. 起始口數 = 帳戶權益 × 每筆風險% ÷ (停損點數 × 每點價值), 四捨五入下界, 至少 1 口。
  2. 加碼: 進場後每順勢推進 `pyramid_step_pt` 點加 1 口 (最多加到 `max_units` 口),
     加碼後全倉停損上移(多)/下移(空)至「最新加碼價 - 原始停損距離」, 確保整體
     風險不擴大 (移動停損, 非加碼不鎖利)。
  3. 夜盤/隔日跳空風險: 因台指期可能在非交易時段大幅跳空, 加碼與起始口數皆應
     以保守的 `risk_pct` (預設 1%) 計算, 不建議滿倉重壓。

夜盤資料說明 (誠實揭露): 本策略訊號與回測標的 ^TWII 為「加權指數現貨」, 只在台股
盤中 09:00-13:30 有報價, 不包含台指期夜盤 (15:00-翌日05:00) 的盤後行情；Yahoo
Finance 對 ^TWII 也沒有夜盤資料可抓。若要將夜盤納入策略 (例如夜盤跳空缺口分析、
夜盤突破), 須改用 TXF 期貨本身的夜盤逐筆/K 線資料 (例如券商 API 或期交所歷史
資料), 目前程式碼未內建此資料來源, 故本策略只覆蓋日盤訊號; 實單須自行留意夜盤
留倉的跳空風險 (尤其加碼後的停損可能在夜盤跳空時失效, 只能在次日開盤才能成交)。
"""
import os
from collections import defaultdict

STOP_PT = 40          # 停損點數
POINT_VALUE = 200.0   # 大台 NT$/點 (小台 MXF 為 50)
RISK_PCT = 0.01        # 每筆風險佔帳戶權益比例 (起始口數用)
PYRAMID_STEP_PT = 40   # 順勢推進多少點加碼 1 口 (預設等於停損距離)
MAX_UNITS = 3          # 最多加碼到幾口


def make_plan(prev_high, prev_low, stop_pt=STOP_PT):
    """回傳當日掛單計畫 (尚未觸發時的雙邊突破單)。"""
    return {
        "long": {"trigger": prev_high, "stop": prev_high - stop_pt,
                 "exit": "13:30 收盤", "desc": f"突破前日高 {prev_high:.0f} 做多"},
        "short": {"trigger": prev_low, "stop": prev_low + stop_pt,
                  "exit": "13:30 收盤", "desc": f"跌破前日低 {prev_low:.0f} 做空"},
        "stop_pt": stop_pt,
    }


def position_plan(equity, side, entry, stop_pt=STOP_PT, risk_pct=RISK_PCT,
                   pyramid_step_pt=PYRAMID_STEP_PT, max_units=MAX_UNITS,
                   point_value=POINT_VALUE):
    """資金管理/加減碼計畫: 依帳戶權益算起始口數, 並列出加碼價位與移動停損。

    side: +1(多) / -1(空)。回傳 dict: base_units, adds(list of {price, units,
    stop}), max_units, risk_pct。僅供「實單口數」參考, 回測訊號邊際仍以 1 口驗證。
    """
    risk_amount = equity * risk_pct
    base_units = max(1, int(risk_amount // (stop_pt * point_value)))
    adds = []
    stop = entry - side * stop_pt
    for n in range(1, max_units - base_units + 1):
        add_price = entry + side * pyramid_step_pt * n
        stop = add_price - side * stop_pt  # 加碼後全倉停損移至「加碼價-原始停損距離」
        adds.append({
            "units": base_units + n,
            "trigger": add_price,
            "stop_all": stop,
            "desc": f"順勢加碼至 {base_units + n} 口 (價位 {add_price:.0f}, "
                    f"全倉停損移至 {stop:.0f})",
        })
    return {
        "base_units": base_units,
        "base_stop": entry - side * stop_pt,
        "adds": adds,
        "max_units": max_units,
        "risk_pct": risk_pct,
    }


def evaluate_day(bars, prev_high, prev_low, stop_pt=STOP_PT):
    """給定當日小時 K (list of dict o/h/l/c) 與前日高低, 回傳已觸發的交易或 None。

    回傳 dict: side(+1/-1), entry, stop, exit, status('open'/'stopped'/'closed'), pnl_pt。
    若當日尚未結束 (bars 不足 5 根) 仍會回報目前狀態。
    """
    for i, b in enumerate(bars):
        if b["h"] >= prev_high:          # 多方突破
            entry = max(b["o"], prev_high)
            stop = entry - stop_pt
            return _resolve(bars, i, +1, entry, stop)
        if b["l"] <= prev_low:           # 空方跌破
            entry = min(b["o"], prev_low)
            stop = entry + stop_pt
            return _resolve(bars, i, -1, entry, stop)
    return None


def _resolve(bars, i, side, entry, stop):
    for j in range(i, len(bars)):
        b = bars[j]
        if side > 0 and b["l"] <= stop:
            return _mk(side, entry, stop, "stopped", side * (stop - entry))
        if side < 0 and b["h"] >= stop:
            return _mk(side, entry, stop, "stopped", side * (stop - entry))
    last = bars[-1]["c"]
    status = "closed" if len(bars) >= 5 else "open"
    return _mk(side, entry, last, status, side * (last - entry))


def _mk(side, entry, exit_, status, pnl_pt):
    return {"side": side, "dir": "多" if side > 0 else "空", "entry": entry,
            "exit": exit_, "stop": entry - side * STOP_PT, "status": status,
            "pnl_pt": pnl_pt, "pnl_nt": pnl_pt * POINT_VALUE}


# ── 即時報告 (抓最新資料, 算前日高低 + 今日狀態) ──────────────────────────────

def _recent_days(n=6):
    """抓最近數日小時 K, 回傳 [(date, [bars])] 由舊到新。"""
    import txf_data
    rows = txf_data.fetch("60m", "1mo") or []
    days = defaultdict(list)
    for dt, o, h, l, c, v in rows:
        d, hm = dt.split(" ")
        if hm in ("09:00", "10:00", "11:00", "12:00", "13:00", "13:30"):
            days[d].append({"hm": hm, "o": o, "h": h, "l": l, "c": c})
    out = []
    for d in sorted(days):
        bars = [b for b in days[d] if b["hm"] != "13:30"]
        extra = [b for b in days[d] if b["hm"] == "13:30"]
        if extra and bars:
            bars[-1]["c"] = extra[0]["c"]
            bars[-1]["h"] = max(bars[-1]["h"], extra[0]["h"])
            bars[-1]["l"] = min(bars[-1]["l"], extra[0]["l"])
        out.append((d, bars))
    return out[-n:]


def live_report():
    """回傳今日台指期日內策略狀態文字 (前日高低 / 掛單計畫 / 是否已觸發)。"""
    days = _recent_days(6)
    if len(days) < 2:
        return "❌ 台指日內: 資料不足"
    prev_d, prev_bars = days[-2]
    today_d, today_bars = days[-1]
    ph = max(b["h"] for b in prev_bars)
    pl = min(b["l"] for b in prev_bars)
    plan = make_plan(ph, pl)
    lines = [
        f"📐 台指日內 · 前日高低突破 · {today_d}",
        f"今日錨點 前日高 {ph:.0f}／前日低 {pl:.0f}（停損固定 {STOP_PT}點, 13:30平倉）",
        "",
        f"▲ 做多 突破 {ph:.0f}｜停損 {plan['long']['stop']:.0f}",
        f"▼ 做空 跌破 {pl:.0f}｜停損 {plan['short']['stop']:.0f}",
        "",
    ]
    trade = evaluate_day(today_bars, ph, pl)
    if trade is None:
        cur = today_bars[-1]["c"] if today_bars else None
        lines.append(f"⚪ 今日尚未觸發（現價 {cur:.0f}）" if cur else "⚪ 今日尚無資料")
    else:
        st = {"open": "持倉中", "stopped": "已停損", "closed": "已平倉"}[trade["status"]]
        em = {"open": "🟢", "stopped": "⛔", "closed": "✅"}[trade["status"]]
        out_lbl = "現價" if trade["status"] == "open" else "出場"
        lines.append(
            f"{em} 今日已觸發 {trade['dir']}單 [{st}]")
        lines.append(
            f"進場 {trade['entry']:.0f} → {out_lbl} {trade['exit']:.0f}｜"
            f"{trade['pnl_pt']:+.0f}點（NT${trade['pnl_nt']:+,.0f}）")
        equity = float(os.environ.get("TXF_EQUITY", "0") or 0)
        if equity > 0:
            pp = position_plan(equity, trade["side"], trade["entry"])
            lines.append(f"💰 權益 NT${equity:,.0f}·風險 {pp['risk_pct']*100:.1f}%："
                         f"起始 {pp['base_units']}口, 停損 {pp['base_stop']:.0f}")
            for a in pp["adds"]:
                lines.append(f"   {a['desc']}")
    return "\n".join(lines)


def check_today_trigger():
    """供盤中自動推播用: 回傳 (today_d, trade|None)。trade 為 evaluate_day() 的結果,
    僅在「今日已觸發」時非 None, 呼叫端可比對是否已推播過避免重複通知。"""
    days = _recent_days(6)
    if len(days) < 2:
        return None, None
    prev_d, prev_bars = days[-2]
    today_d, today_bars = days[-1]
    ph = max(b["h"] for b in prev_bars)
    pl = min(b["l"] for b in prev_bars)
    return today_d, evaluate_day(today_bars, ph, pl)


if __name__ == "__main__":
    print(live_report())
