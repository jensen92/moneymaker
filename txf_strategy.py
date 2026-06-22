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
"""
import os
from collections import defaultdict

STOP_PT = 40          # 停損點數
POINT_VALUE = 200.0   # 大台 NT$/點 (小台 MXF 為 50)


def make_plan(prev_high, prev_low, stop_pt=STOP_PT):
    """回傳當日掛單計畫 (尚未觸發時的雙邊突破單)。"""
    return {
        "long": {"trigger": prev_high, "stop": prev_high - stop_pt,
                 "exit": "13:30 收盤", "desc": f"突破前日高 {prev_high:.0f} 做多"},
        "short": {"trigger": prev_low, "stop": prev_low + stop_pt,
                  "exit": "13:30 收盤", "desc": f"跌破前日低 {prev_low:.0f} 做空"},
        "stop_pt": stop_pt,
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
        f"📐 台指期日內策略 (前日高低突破)  {today_d}",
        f"前一交易日 ({prev_d}) 高 {ph:.0f} / 低 {pl:.0f}",
        "",
        f"▲ 做多: 突破 {ph:.0f} 進場, 停損 {plan['long']['stop']:.0f}, 13:30 平倉",
        f"▼ 做空: 跌破 {pl:.0f} 進場, 停損 {plan['short']['stop']:.0f}, 13:30 平倉",
        f"(停損固定 {STOP_PT} 點; 大台每點 NT${POINT_VALUE:.0f})",
        "",
    ]
    trade = evaluate_day(today_bars, ph, pl)
    if trade is None:
        cur = today_bars[-1]["c"] if today_bars else None
        lines.append(f"今日尚未觸發 (現價 {cur:.0f})" if cur else "今日尚無資料")
    else:
        st = {"open": "持倉中", "stopped": "已停損出場", "closed": "已收盤平倉"}[trade["status"]]
        lines.append(
            f"今日已觸發 {trade['dir']}單: 進場 {trade['entry']:.0f} → "
            f"{'現價/出場' if trade['status']=='open' else '出場'} {trade['exit']:.0f}  "
            f"{trade['pnl_pt']:+.0f} 點 (NT${trade['pnl_nt']:+,.0f})  [{st}]")
    return "\n".join(lines)


if __name__ == "__main__":
    print(live_report())
