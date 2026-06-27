"""黃金期貨 (GC=F) 小時線順勢突破 — 每日/即時訊號報告.

抓取最新 GC 小時線, 用 gold_strategies 的順勢突破邏輯判斷:
  - 最新一根是否「剛突破」過去 N 小時高點 → 新進場做多訊號
  - 或目前若已在突破區間上方, 報告參考停損 (ATR 移動停損)

用法:
    python3 gold_signals.py              # 抓最新資料並報告
    python3 gold_signals.py --no-fetch   # 用本地既有 GC_60m.csv (不連網)
"""
import argparse
import csv
import os
import time

import numpy as np

import gold_strategies as gs

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


TW_OFFSET = 8 * 3600   # 台北時間 = UTC+8 (與 txf_data/download_all/telegram_bot 一致)


def fetch_gc_hourly(path=gs.CSV):
    """抓 Yahoo GC=F 近 730 日小時線, 覆寫 path。失敗則回傳 False。

    時間戳一律轉台北時間 (UTC+8) 並『只寫入整點 (:00) 已完成棒』—
    濾掉 Yahoo 尾端附帶的即時快照半成品棒與假日/夏令時的非整點棒,
    避免污染回測/突破視窗 (見 gold_strategies.load_bars 說明)。
    """
    import requests
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"range": "730d", "interval": "1h"},
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]; q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
                v = q["volume"][i]
                if None in (o, h, l, c):
                    continue
                stamp = time.strftime("%Y-%m-%d %H:%M", time.gmtime(t + TW_OFFSET))
                if not stamp.endswith(":00"):      # 跳過即時快照 / 非整點棒
                    continue
                rows.append((stamp, round(o, 2), round(h, 2), round(l, 2),
                             round(c, 2), int(v) if v else 0))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "open", "high", "low", "close", "volume"])
                w.writerows(rows)
            return True
        except Exception:
            if attempt == 2:
                return False
            time.sleep(2 ** attempt)


def fetch_live_price():
    """即時參考價: Yahoo 1 分鐘線最後一筆有效收盤 (近乎即時)。失敗回傳 None。"""
    import requests
    try:
        r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F",
                         params={"range": "1d", "interval": "1m"},
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        closes = res["indicators"]["quote"][0]["close"]
        for v in reversed(closes):
            if v is not None:
                return float(v)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    if not args.no_fetch:
        ok = fetch_gc_hourly()
        if not ok and not os.path.exists(gs.CSV):
            print("❌ 無法抓取 GC 資料且無本地檔。")
            return

    cfg = gs.CONFIG
    dt, o, h, l, c = gs.load_bars()           # 只含已完成整點棒
    a = gs.atr(h, l, c, cfg["atr_n"])
    i = len(c) - 1                            # 最後一根『已完成』棒
    bo = cfg["breakout"]
    bar_close = c[i]
    atr_now = a[i]
    live = fetch_live_price() or bar_close    # 即時價 (抓不到則退回收盤)
    next_level = gs.breakout_level(h, i, bo)  # 形成中下一根要突破的多單參考價 (含本根)

    trades = gs.backtest()
    m = gs.metrics(trades)
    stop_mult = cfg["atr_stop"]

    def risk(entry, stop):
        pts = abs(entry - stop)
        return f"風險{pts:.0f}點/${pts * gs.POINT_VALUE:,.0f}口"

    # ── 行情列 ──
    print(f"🥇 黃金 GC 順勢突破（僅做多）")
    print(f"{dt[i][5:]} 台北 · {bo}H突破/{stop_mult}ATR\n")
    print(f"即時 {live:,.1f}｜收盤 {bar_close:,.1f}｜ATR {atr_now:.1f}")
    print(f"突破門檻 {next_level:,.1f}（距 {next_level - live:+,.1f} / {(next_level - live) / live:+.2%}）")
    pd, pdh, pdl = gs.prev_day_high_low(dt, h, l, i)
    if pdh is not None:
        print(f"今日錨點 前日高 {pdh:,.1f}／前日低 {pdl:,.1f}")

    # ── 目前趨勢狀態 (策略視角: 跟隨突破, 非預判方向) ──
    win = min(120, i)                             # 約 5 個交易日的中期均線
    ma_mid = float(np.mean(c[i - win + 1:i + 1]))
    hi24 = float(h[i - bo + 1:i + 1].max())
    lo24 = float(l[i - bo + 1:i + 1].min())
    if live > ma_mid and live >= hi24 * 0.997:
        trend = "🟢 多頭續勢（近24H高, 站中期均上）— 易出現進場訊號"
    elif live > ma_mid:
        trend = f"🟡 多頭回檔（站中期均上但距24H高 {(hi24-live)/live:+.1%}）— 待再突破"
    elif live > lo24 * 1.003:
        trend = "🟠 盤整/轉弱（跌破中期均, 區間中段）— 多空手等突破"
    else:
        trend = "🔴 弱勢/空頭（近24H低, 中期均下方）— 策略空手休息, 不做空"
    # 目前部位 (讀背景監控狀態; 無則視為空手)
    pos_txt = "空手"
    try:
        sp = os.path.join(gs.HERE, "gold_state.json")
        if os.path.exists(sp):
            import json
            with open(sp) as f:
                st = json.load(f)
            if st.get("pos") == 1 and st.get("entry"):
                pos_txt = f"持有多單 進場{st['entry']:,.1f} 停損{st.get('trail',0):,.1f}"
    except Exception:
        pass
    print(f"趨勢 {trend}")
    print(f"部位 {pos_txt}")

    # ── 訊號區 ──
    sig = gs.signal_breakout(h, l, c, i, cfg, a)
    if sig == 1:
        stop = bar_close - stop_mult * atr_now
        print(f"\n🟢 確認進場（做多）")
        print(f"進場 {bar_close:,.1f}｜停損 {stop:,.1f}（{risk(bar_close, stop)}）")
        print("停利 無 · ATR移動停損鎖利, 跌破即出")
    elif sig == -1:
        stop = bar_close + stop_mult * atr_now
        print(f"\n🔴 確認進場（做空）")
        print(f"進場 {bar_close:,.1f}｜停損 {stop:,.1f}（{risk(bar_close, stop)}）")
        print("停利 無 · ATR移動停損鎖利, 突破即出")
    elif live > next_level:
        stop = live - stop_mult * atr_now
        print(f"\n🟡 即時突破中（待本小時收盤確認）")
        print(f"進場(預估) {live:,.1f}｜停損(預估) {stop:,.1f}（{risk(live, stop)}）")
        print("停利 無 · ATR移動停損；收在門檻上即確認")
    else:
        stop = next_level - stop_mult * atr_now
        print(f"\n⚪ 無進場訊號")
        print(f"進場 突破 {next_level:,.1f} 做多｜停損(預估) {stop:,.1f}（{risk(next_level, stop)}）")
        print("停利 無 · ATR移動停損鎖利")

    # ── 績效 ──
    print(f"\n績效 {m['n']}筆｜PF {m['pf']:.2f}｜勝率 {m['win']:.0%}｜DD ${m['dd']/1000:.0f}k｜MAR {m['mar']:.1f}")
    print(f"逐年 {gs.yearly_line(trades)}")


if __name__ == "__main__":
    main()
