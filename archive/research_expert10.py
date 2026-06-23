"""勝出的 5 個經典飆股篩選法 (專家原始預設參數, 不調參) — 全市場回測.

原始研究比較了 10 個方法 (見 git 歷史 commit c36a4cd), 以同一引擎/成本/風控對比
現行策略 C/D (1x 標準帳, CAGR ~5%)。下列 5 法在 CAGR 上明確勝出, 予以保留:
  1. 52週新高動能  — 學術動能因子 (Jegadeesh; Hwang-George 52w high), CAGR 20.0% / MAR 0.91
  2. ADX 趨勢突破  — Welles Wilder《新趨勢交易系統》(ADX>25), CAGR 16.4%
  3. Turtle System 2 — Curtis Faith《海龜交易法則》(55日突破/20日出場), CAGR 15.2%
  4. CANSLIM 樞紐買點 — William O'Neil《股市致富術》(量>1.4x, 停損7%), CAGR 13.2%
  5. Turtle System 1 — 同海龜 (20日突破/10日出場), CAGR 13.1%
已淘汰 (CAGR 不如 C/D 或虧損): Darvas Box, Weinstein Stage2, 黃金交叉,
布林通道突破 (唯一虧損 PF 0.61), RS90 領漲股 (邊際, 9.9%)。

引擎: 完整複製 backtest.run_sub 之成本/部位/制度濾網 (手續費/稅/滑價/單筆風險
1%/每日最多2檔/最多8檔持倉/回撤熔斷20%/三段式市況/波動目標化), 僅將出場判定
擴充為通用的 ma_exit_col (跌破指定均線) 與 roll_low_exit (跌破N日低) 兩種,
以忠實還原各法原始出場規則 (Minervini 專屬的賣半/三週法則不套用, 避免規則混血)。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402

bt.DATA_DIR = os.environ.get("MM_DATA_DIR",
                             os.path.join(os.path.dirname(__file__), "data_adj"))

MIN_TURNOVER = 50_000_000   # 與現有 A/C/D/PA/PB 一致的流動性下限 (非本研究調參)
DEFAULT_MAX_HOLD = 9999      # 無明確時間出場的方法用的後備上限 (非原方法規則)


# ───────────────────────────── 指標擴充 ─────────────────────────────
def extend(df):
    df = df.copy()
    df["high20p"]  = df["high"].rolling(20).max().shift(1)
    df["high55p"]  = df["high"].rolling(55).max().shift(1)
    df["low10p"]   = df["low"].rolling(10).min().shift(1)
    df["low20p"]   = df["low"].rolling(20).min().shift(1)
    df["atr20"] = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - df["close"].shift()).abs(),
                   (df["low"] - df["close"].shift()).abs())
    ).rolling(20).mean()

    # ADX(14) — Wilder 原始平滑法
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(df["high"] - df["low"],
                     np.maximum((df["high"] - df["close"].shift()).abs(),
                                (df["low"] - df["close"].shift()).abs()))
    atr_w = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["plus_di"] = plus_di.values
    df["minus_di"] = minus_di.values
    df["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values
    return df


# ───────────────────────────── 勝出的 5 個方法 ─────────────────────────────
def sig_canslim(df, i, rs):
    row = df.iloc[i]
    if rs is None or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < MIN_TURNOVER:
        return None
    if not (row["close"] > row["ma50"] > row["ma150"]):     # O'Neil: 多頭排列
        return None
    pivot = row["high20p"]
    if np.isnan(pivot):
        return None
    if not (row["close"] > pivot and row["close"] <= pivot * 1.05):   # 樞紐後5%買進區
        return None
    if not (row["volume"] > row["vol_ma50"] * 1.40):          # O'Neil: 量增40%+
        return None
    # O'Neil 經典出場心法之一: 跌破50日均線賣出 (《股市致富術》第7章停損規則),
    # 取代原先「無出場規則」之 bug (max_hold=9999 會讓贏家永不結算)
    return {"score": rs, "stop_pct": 0.07, "ma_exit_col": "ma50",
            "max_hold": DEFAULT_MAX_HOLD}


def sig_turtle1(df, i, rs):
    row = df.iloc[i]
    if row["close"] < 10 or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    if row["volume"] * row["close"] < MIN_TURNOVER:
        return None
    if np.isnan(row["high20p"]) or row["close"] <= row["high20p"]:
        return None
    return {"score": rs if rs is not None else 0.5,
            "stop_atr20": 2.0, "roll_low_exit": "low10p",
            "max_hold": DEFAULT_MAX_HOLD}


def sig_turtle2(df, i, rs):
    row = df.iloc[i]
    if row["close"] < 10 or np.isnan(row["atr20"]) or row["atr20"] <= 0:
        return None
    if row["volume"] * row["close"] < MIN_TURNOVER:
        return None
    if np.isnan(row["high55p"]) or row["close"] <= row["high55p"]:
        return None
    return {"score": rs if rs is not None else 0.5,
            "stop_atr20": 2.0, "roll_low_exit": "low20p",
            "max_hold": DEFAULT_MAX_HOLD}


def sig_52w_high(df, i, rs):
    row = df.iloc[i]
    if row["close"] < 10 or np.isnan(row["high252"]):
        return None
    if row["volume"] * row["close"] < MIN_TURNOVER:
        return None
    if not (row["close"] >= row["high252"] * 0.98):
        return None
    if not (row["volume"] > row["vol_ma50"] * 1.50):
        return None
    return {"score": rs if rs is not None else 0.5,
            "stop_pct": 0.08, "max_hold": 60}     # 動能文獻常用季度持有期


def sig_adx(df, i, rs):
    row = df.iloc[i]
    if row["close"] < 10 or np.isnan(row["adx14"]) or np.isnan(row["atr14"]):
        return None
    if row["volume"] * row["close"] < MIN_TURNOVER:
        return None
    if i < 20:
        return None
    adx_rising = row["adx14"] > df["adx14"].iloc[i - 5]
    if not (row["adx14"] > 25 and adx_rising and row["plus_di"] > row["minus_di"]):
        return None
    if np.isnan(row["high20p"]) or row["close"] <= row["high20p"]:
        return None
    # Wilder 慣用搭配: 初始 2xATR 停損 + 3xATR 移動停損 (chandelier 衍生,
    # 避免原 max_hold=9999 造成贏家無限期持有不結算)
    return {"score": rs if rs is not None else 0.5,
            "stop_atr": 2.0, "trail_atr": 3.0, "max_hold": DEFAULT_MAX_HOLD}


METHODS = [
    ("52週新高動能",       sig_52w_high,    True),
    ("ADX趨勢突破",        sig_adx,         True),
    ("Turtle System2",    sig_turtle2,     True),
    ("CANSLIM 樞紐買點",   sig_canslim,     True),
    ("Turtle System1",    sig_turtle1,     True),
]


# ───────────────────────────── 通用回測引擎 (run_sub 之擴充版) ─────────────────────────────
def run_generic(data, entry_map, all_dates, date_idx, regime_tiers, vol_scalars,
                 init_eq=2_000_000, leverage=1.0, max_pos=8,
                 dd_pause=0.20, dd_resume=0.12):
    equity = init_eq
    peak_eq = init_eq
    paused = False
    pause_count = 0
    open_pos = []
    trades = []
    curve = []

    for d in all_dates:
        tier = regime_tiers.get(d, 2) if regime_tiers else 2
        vsca = vol_scalars.get(d, 1.0) if vol_scalars else 1.0
        tier_mult = bt.REGIME_RISK_MULT[tier]
        eff_risk = bt.RISK_PCT * tier_mult * vsca

        still_open = []
        for p in open_pos:
            di = date_idx[p["code"]].get(d)
            if di is None or di <= p["entry_idx"]:
                still_open.append(p)
                continue
            df_stock = data[p["code"]]
            row = df_stock.iloc[di]
            exit_price = None

            stop_triggered = row["low"] <= p["stop"]
            if stop_triggered and bt._is_limit_day(df_stock, di) and row["close"] < row["open"]:
                p["expire_idx"] = max(p["expire_idx"], di + 1)
                if di + 1 >= len(df_stock):
                    exit_price = row["close"]
                else:
                    still_open.append(p)
                    continue
            elif stop_triggered:
                exit_price = min(p["stop"], row["open"])
            elif p.get("ma_exit_col") and row["close"] < row[p["ma_exit_col"]]:
                exit_price = row["close"]
            elif p.get("roll_low_col") and row["close"] < row[p["roll_low_col"]]:
                exit_price = row["close"]
            elif di >= p["expire_idx"]:
                exit_price = row["close"]

            if exit_price is None:
                if p.get("trail_atr"):
                    p["high_close"] = max(p["high_close"], row["close"])
                    atr = row["atr14"] if not np.isnan(row["atr14"]) else p["init_atr"]
                    p["stop"] = max(p["stop"], p["high_close"] - p["trail_atr"] * atr)
                still_open.append(p)
                continue

            exit_price *= (1 - bt.SLIP)
            proceeds = p["shares"] * exit_price * (1 - bt.FEE - bt.TAX)
            cost = p["shares"] * p["entry"] * (1 + bt.FEE)
            pnl = proceeds - cost
            equity += pnl
            trades.append({"code": p["code"], "entry_date": p["entry_date"],
                            "exit_date": d, "entry": p["entry"], "exit": exit_price,
                            "pnl": pnl, "r": pnl / p["risk_amt"] if p["risk_amt"] > 0 else 0})
        open_pos = still_open
        peak_eq = max(peak_eq, equity)
        dd = (peak_eq - equity) / peak_eq

        if not paused:
            if dd >= dd_pause:
                paused = True
                pause_count = 0
        else:
            pause_count += 1
            if dd <= dd_resume or pause_count >= bt.PAUSE_COOLDOWN:
                paused = False
                peak_eq = equity

        if not paused and tier > 0 and eff_risk > 0:
            held = {p["code"] for p in open_pos}
            exposure = sum(p["shares"] * p["entry"] for p in open_pos)
            candidates = sorted(entry_map.get(d, []), reverse=True)
            taken = 0
            for score, code, ei, s in candidates:
                if taken >= bt.PICKS_PER_DAY or len(open_pos) >= max_pos:
                    break
                if code in held:
                    continue
                df = data[code]
                if bt._is_limit_day(df, ei):
                    prev_close = df["close"].iloc[ei - 1] if ei > 0 else df["open"].iloc[ei]
                    if df["close"].iloc[ei] > prev_close * 1.08:
                        continue
                entry = df["open"].iloc[ei] * (1 + bt.SLIP)
                atr14 = df["atr14"].iloc[ei - 1]
                atr20 = df["atr20"].iloc[ei - 1] if "atr20" in df.columns else np.nan
                if np.isnan(atr14) or atr14 <= 0:
                    continue
                if "stop_pct" in s:
                    stop = entry * (1 - s["stop_pct"])
                elif "stop_atr20" in s:
                    if np.isnan(atr20) or atr20 <= 0:
                        continue
                    stop = entry - s["stop_atr20"] * atr20
                else:
                    stop = entry - s["stop_atr"] * atr14
                if stop <= 0 or entry <= stop:
                    continue
                risk_per_sh = entry - stop
                risk_amt = equity * eff_risk
                shares = int(risk_amt / risk_per_sh / 1000) * 1000
                if shares <= 0:
                    shares = max(1, int(risk_amt / risk_per_sh))
                notional = shares * entry
                if exposure + notional > equity * leverage:
                    allowed = equity * leverage - exposure
                    shares = int(allowed / entry / 1000) * 1000
                    if shares <= 0:
                        continue
                    notional = shares * entry
                open_pos.append({
                    "code": code, "shares": shares, "entry": entry, "stop": stop,
                    "trail_atr": s.get("trail_atr"),
                    "ma_exit_col": s.get("ma_exit_col"),
                    "roll_low_col": s.get("roll_low_exit"),
                    "high_close": entry, "init_atr": atr14,
                    "entry_idx": ei, "expire_idx": ei + s["max_hold"],
                    "entry_date": d, "risk_amt": shares * risk_per_sh,
                })
                exposure += notional
                held.add(code)
                taken += 1

        curve.append({"date": d, "equity": equity, "drawdown": dd})

    return trades, pd.DataFrame(curve)


def summarize(trades, curve, init_eq, years):
    if not trades or curve.empty:
        return {"n": 0, "wr": 0, "pf": 0, "cagr": 0, "mdd": 0}
    pnl = np.array([t["pnl"] for t in trades])
    win = pnl > 0
    pf = (pnl[win].sum() / -pnl[~win].sum()) if (~win).any() and pnl[~win].sum() < 0 else np.inf
    final_eq = curve["equity"].iloc[-1]
    cagr = (final_eq / init_eq) ** (1 / years) - 1 if final_eq > 0 else -1
    mdd = curve["drawdown"].max()
    return {"n": len(trades), "wr": win.mean(), "pf": pf, "cagr": cagr, "mdd": mdd,
            "total_ret": final_eq / init_eq - 1}


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    data = {c: extend(df) for c, df in data.items()}
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    print(f"{len(data)} 檔股票, {len(all_dates)} 交易日, 跨 {years:.1f} 年\n")

    only = os.environ.get("ONLY_METHODS")
    methods = METHODS
    if only:
        keep = set(only.split(","))
        methods = [m for m in METHODS if m[0] in keep]

    results = []
    for name, fn, needs_rs in methods:
        signals = {}
        for code, df in data.items():
            n = len(df)
            for i in range(210, n - 1):
                d = df["date"].iloc[i]
                rk = None
                if needs_rs:
                    try:
                        rk = rs.at[d, code]
                    except KeyError:
                        rk = None
                    if rk is not None and np.isnan(rk):
                        rk = None
                s = fn(df, i, rk)
                if s:
                    signals.setdefault(d, []).append((s["score"], code, i, s))
        # 多頭市況濾網 (與現行 A/C/D 一致)
        signals = {d: lst for d, lst in signals.items() if d in regime}
        entry_map = {}
        for sd, lst in signals.items():
            for score, code, i, s in lst:
                df = data[code]
                if i + 1 < len(df):
                    ed = df["date"].iloc[i + 1]
                    entry_map.setdefault(ed, []).append((score, code, i + 1, s))

        trades, curve = run_generic(data, entry_map, all_dates, date_idx,
                                     regime_tiers, vol_scalars)
        st = summarize(trades, curve, 2_000_000, years)
        results.append((name, st))
        print(f"  {name:<16} 交易{st['n']:>5}  勝率{st['wr']*100:>5.1f}%  "
              f"PF{st['pf']:>5.2f}  年化{st.get('cagr',0)*100:>6.1f}%  "
              f"回撤{st.get('mdd',0)*100:>5.1f}%  總報酬{st.get('total_ret',0)*100:>7.1f}%")

    print("\n=== 與現行策略比較 (1x 槓桿, 同回測引擎/成本/部位規則) ===")
    print("現行 C : 見 backtest.py --strategies C --leverage 1 之輸出")
    print("現行 D : 見 backtest.py --strategies D --leverage 1 之輸出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
