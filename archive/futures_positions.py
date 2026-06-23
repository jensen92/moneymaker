"""穀物期貨 — 目前各商品進出場 / 持倉狀態.

把 M+D+S 系統從頭跑到最新資料, 列出每個策略在黃豆/小麥/玉米上「現在」的狀態:
  持有中: 進場日/價、口數、現價、未實現損益、停損、出場觸發點與距離
  空手  : 進場觸發條件與目前距離 (還差多少才進場)

用法:
    python3 futures_positions.py
    python3 futures_positions.py --strategies T,D,M,B,S --capital 250000 --risk 0.01
"""
import argparse

import numpy as np

import futures_strategies as fs
from futures_backtest import (INIT_CAPITAL, RISK_PCT, _build_index,
                              collect_signals, load_all, run_strategy)
from futures_data import FUTURES


def _exit_desc(pos, df):
    """回傳持有中部位的出場觸發說明 (文字, 觸發價或 None, 與現價距離%)."""
    row = df.iloc[-1]
    close = row["close"]
    if pos["exit_channel"]:
        n = pos["exit_channel"]
        lvl = df[f"dc_lo{n}"].iloc[-1] if pos["dir"] > 0 else df[f"dc_hi{n}"].iloc[-1]
        gap = (close - lvl) / close if pos["dir"] > 0 else (lvl - close) / close
        return f"反向 {n} 日通道", lvl, gap
    if pos["exit_ma"]:
        f, s = pos["exit_ma"]
        mf, ms = row[f"ma{f}"], row[f"ma{s}"]
        gap = (mf - ms) / ms if pos["dir"] > 0 else (ms - mf) / ms
        return f"MA{f} 跌破 MA{s}", ms, gap
    if pos["exit_mid"]:
        return "回到中軌 MA20", row["ma20"], (row["ma20"] - close) / close
    return "時間/停損", None, None


def _entry_desc(key, df):
    """空手時, 回傳進場觸發說明 (文字, 觸發價或 None, 還差多少% 到觸發)."""
    c = fs.CONFIG[key]
    row = df.iloc[-1]
    close = row["close"]
    if key in ("T", "D"):
        lvl = df[f"dc_hi{c['entry']}"].iloc[-1]
        tr = row[f"ma{c['trend']}"]
        need = max(lvl, tr)
        gap = need / close - 1
        trend_ok = "✓" if close > tr else "✗"
        return (f"收破 {c['entry']} 日高 {lvl:.1f} 且 > MA{c['trend']} "
                f"{tr:.1f}(趨勢{trend_ok})"), need, gap
    if key == "M":
        mf, ms = row[f"ma{c['fast']}"], row[f"ma{c['slow']}"]
        if mf <= ms:
            return f"等 MA{c['fast']} 黃金交叉 MA{c['slow']}", ms, (ms - mf) / mf
        return (f"MA{c['fast']}>MA{c['slow']} 但無新交叉 "
                f"(已多頭排列, 等下次交叉)"), None, None
    if key == "B":
        dn = row["ma20"] - c["std"] * row["std20"]
        return f"跌破布林下軌 {dn:.1f}", dn, (dn - close) / close
    if key == "S":
        m = c["month"]
        return f"每年 {m} 月初進場 (現為 {row['date'].month} 月)", None, None
    return "—", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="M,D,S")
    ap.add_argument("--capital", type=float, default=INIT_CAPITAL)
    ap.add_argument("--risk", type=float, default=RISK_PCT)
    args = ap.parse_args()
    keys = args.strategies.split(",")

    import futures_backtest as fb
    fb.INIT_CAPITAL = args.capital
    fb.RISK_PCT = args.risk

    data = load_all()
    all_dates, date_idx = _build_index(data)
    last_date = max(df["date"].iloc[-1] for df in data.values())
    print(f"資料截至 {last_date.date()}  |  本金 {args.capital:,.0f}  "
          f"風險 {args.risk:.1%}/筆\n")

    total_unreal = 0.0
    held_summary = []

    for key in keys:
        emap = collect_signals(data, key)
        _, _, open_pos = run_strategy(data, emap, key, args.capital,
                                      all_dates, date_idx, return_open=True)
        print(f"── 策略 {key} {fs.STRATEGY_NAMES[key]} "
              f"{'─' * (40 - len(fs.STRATEGY_NAMES[key]))}")
        for m in FUTURES:
            df = data[m]
            close = df["close"].iloc[-1]
            name = FUTURES[m]["name"]
            pos = open_pos.get(m)
            if pos is None:
                desc, lvl, gap = _entry_desc(key, df)
                gtxt = f"  (距觸發 {gap:+.1%})" if gap is not None else ""
                print(f"  {name:<14} 空手   現價 {close:>8.1f}  進場條件: {desc}{gtxt}")
                continue
            pv = FUTURES[m]["point_value"]
            side = "做多 ↑" if pos["dir"] > 0 else "做空 ↓"
            held_days = (len(df) - 1) - pos["entry_idx"]
            unreal = pos["dir"] * (close - pos["entry"]) * pv * pos["contracts"]
            pct = pos["dir"] * (close / pos["entry"] - 1)
            total_unreal += unreal
            stop_gap = (close - pos["stop"]) / close if pos["dir"] > 0 \
                else (pos["stop"] - close) / close
            edesc, elvl, egap = _exit_desc(pos, df)
            etxt = (f"{edesc} @ {elvl:.1f} (距 {egap:+.1%})"
                    if elvl is not None else edesc)
            held_summary.append((key, m, pos["dir"], pos["contracts"], unreal))
            print(f"  {name:<14} {side} {pos['contracts']} 口  "
                  f"進場 {pos['entry_date'].date()} @ {pos['entry']:.1f} "
                  f"(持有 {held_days} 日)")
            print(f"  {'':<14} 現價 {close:.1f}  未實現 {unreal:>+10,.0f} "
                  f"({pct:+.1%})  停損 {pos['stop']:.1f}(距 {stop_gap:+.1%})")
            print(f"  {'':<14} 出場: {etxt}")
        print()

    print("=" * 56)
    if held_summary:
        print(f"目前共 {len(held_summary)} 個持倉,  合計未實現損益 "
              f"{total_unreal:>+,.0f}")
        for key, m, d, n, u in held_summary:
            print(f"  {key} {FUTURES[m]['name']:<12} "
                  f"{'多' if d > 0 else '空'} {n} 口  {u:>+10,.0f}")
    else:
        print("目前全數空手 (無持倉)。")


if __name__ == "__main__":
    main()
