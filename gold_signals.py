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

    print(f"黃金 GC 順勢突破訊號  (最後完成棒 {dt[i]} 台北時間)")
    print(f"參數: 突破窗 {bo} 小時, 移動停損 {cfg['atr_stop']}×ATR{cfg['atr_n']}, "
          f"{'多空雙向' if cfg['allow_short'] else '僅做多'}\n")
    print(f"即時價 {live:,.2f}  |  最後完成棒收盤 {bar_close:,.2f}  "
          f"|  突破參考價(過去{bo}H高, 每根更新) {next_level:,.2f}  |  ATR {atr_now:,.2f}")

    # 今日固定錨點 (前一交易日高/低, 全天不變) — 供盯盤, 不影響滾動訊號
    pd, pdh, pdl = gs.prev_day_high_low(dt, h, l, i)
    if pdh is not None:
        rel = "上方" if live > pdh else "下方"
        print(f"今日固定錨點 (前一交易日 {pd}): 高 {pdh:,.2f} / 低 {pdl:,.2f}  "
              f"→ 即時價在前日高{rel} ({live - pdh:+,.2f})")

    trades = gs.backtest()
    m = gs.metrics(trades)

    stop_mult = cfg["atr_stop"]

    def risk_str(entry, stop):
        pts = abs(entry - stop)
        return f"(風險 {pts:,.2f} 點 ≈ ${pts * gs.POINT_VALUE:,.0f}/口)"

    # (1) 回測一致訊號: 最後『已完成』棒是否突破其前 bo 根高點 → 確認進場
    sig = gs.signal_breakout(h, l, c, i, cfg, a)
    if sig == 1:
        stop = bar_close - stop_mult * atr_now
        print(f"\n🟢 確認進場 (做多): 最後完成棒收盤突破過去{bo}H高點")
        print(f"   進場價: {bar_close:,.2f}")
        print(f"   停損價: {stop:,.2f}  {risk_str(bar_close, stop)}")
        print(f"   停利價: 無固定停利 — 停損價隨每根新K棒以 收盤-{stop_mult}×ATR 上移鎖利, 跌破即出場")
    elif sig == -1:
        stop = bar_close + stop_mult * atr_now
        print(f"\n🔴 確認進場 (做空): 最後完成棒收盤跌破過去{bo}H低點")
        print(f"   進場價: {bar_close:,.2f}")
        print(f"   停損價: {stop:,.2f}  {risk_str(bar_close, stop)}")
        print(f"   停利價: 無固定停利 — 停損價隨每根新K棒以 收盤+{stop_mult}×ATR 下移鎖利, 突破即出場")
    elif live > next_level:
        # (2) 即時價已越過突破參考價, 但本小時尚未收盤 → 待收盤確認
        stop = live - stop_mult * atr_now
        print(f"\n🟡 即時突破中 (待本小時 {bo}H 收盤確認)")
        print(f"   進場價(預估): {live:,.2f}  (即時價已 > 突破參考價 {next_level:,.2f}, 收盤確認後成立)")
        print(f"   停損價(預估): {stop:,.2f}  {risk_str(live, stop)}")
        print(f"   停利價: 無固定停利 — 確認進場後以 ATR 移動停損追蹤")
        print("   (回測規則以小時收盤確認; 此為即時預警, 供提前準備下單)")
    else:
        # (3) 尚未觸發 — 仍給出『預備下單計劃』的進場/停損/停利價
        gap = (next_level - live) / live
        plan_stop = next_level - stop_mult * atr_now
        print(f"\n⚪ 目前無進場訊號 (即時價距突破做多還差 {next_level - live:,.2f} 點 {gap:+.2%})")
        print("   預備計劃 (做多, 待突破才成立):")
        print(f"   進場價: 即時價突破 {next_level:,.2f} (過去{bo}H高) 並站上即進場")
        print(f"   停損價(預估): {plan_stop:,.2f}  {risk_str(next_level, plan_stop)}")
        print(f"   停利價: 無固定停利 — 進場後以 收盤-{stop_mult}×ATR 移動停損追蹤鎖利")
        print("   (持有中部位請依上述 ATR 移動停損規則管理出場)")

    print(f"\n策略歷史績效 ({m['n']}筆, 2024-2026): 勝率 {m['win']:.1%}  PF {m['pf']:.2f}  "
          f"最大回撤 ${m['dd']:,.0f}  MAR {m['mar']:.2f}")
    print(f"逐年損益: {gs.yearly_line(trades)}")
    print("(順勢策略: 趨勢年大賺、盤整/空頭年小賠; 日線26年驗證見 gold_daily_backtest.py)")


if __name__ == "__main__":
    main()
