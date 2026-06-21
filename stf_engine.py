"""個股期貨 (STF) 回測執行核心 — 忠實實作規格書 §0.2 部位控管 / §0.3 出場 / §0.4 成本.

設計: 沿用已驗證的選股/RS/大盤濾網基礎 (backtest.py 的 load_all/compute_rs_rank/
load_regime/build_entry_map), 但用 STF 專屬的執行核心 run_stf 取代股票引擎, 差異:

§0.2 部位控管 (統一公式):
  R = 帳戶權益 × 風險%  ;  D = 進場價 − 停損價  ;  Qty_theory = R / D
  名目市值 MV = min(Qty_theory × 進場價, 單檔上限 100萬)
  口數 Lots = floor(MV / (進場價 × 口股數))   (標準 2000 股/口; 高價股用小型 100 股/口)
  總名目曝險 ≤ 帳戶權益 × 槓桿(預設 3); 同時持倉上限 max_pos.

§0.3 出場 (各策略共用, 進場時只給初始停損 + 結構均線 ma_exit):
  三段式移動停利: 1R→停損移成本; 2R→停損移 1R; 之後 Chandelier = 最高價 − 3×ATR14
  時間停損: 進場後 time_stop_days 日未達 0.5R → 收盤出場
  結構出場: 收盤跌破 ma_exit (MA20/50/60/150) → 出場
  大盤濾網: TAIEX 跌破 MA60 (load_regime risk_on=False) → 停止新進場

§0.4 成本 (STF, 遠低於現股, 無 0.3% 證交稅):
  期交稅 0.002% × 成交值 (買賣各課) + 手續費 (每口固定, 買賣各課) + 滑價 (價格 ×)
"""
import numpy as np
import pandas as pd

import backtest as bt

# ── 帳戶 / 部位參數 (規格 §0.2) ──────────────────────────────────────────────
ACCOUNT_EQUITY = 5_000_000.0    # 單策略帳戶權益
RISK_PCT = 0.01                 # 單筆風險 1%
POS_CAP = 1_000_000.0           # 單檔名目上限 100 萬
LEVERAGE = 3.0                  # 總名目曝險上限倍數
MAX_POS = 8                     # 同時持倉上限
PICKS_PER_DAY = 2               # 每日最多新倉
MARGIN_RATE = 0.135             # 個股期保證金率 (級距1, 僅供報告)

# ── STF 成本 (規格 §0.4) ────────────────────────────────────────────────────
STF_TAX = 0.00002               # 期交稅 0.002% (買賣各課)
STF_COMM_PER_LOT = 20.0         # 手續費 NT$/口 (買賣各課, 保守值)
STF_SLIP = 0.001                # 滑價 (單邊, ≈1-2 檔)

# ── 出場參數 (規格 §0.3) ────────────────────────────────────────────────────
CHANDELIER_ATR = 3.0            # 2R 後 Chandelier 倍數
TIME_STOP_DAYS = 7              # 時間停損窗口 (規格 5-8 日, 取中)
TIME_STOP_MIN_R = 0.5           # 未達 0.5R 即時間停損

TD = 244                        # 台股年交易日


def _lot_shares(entry):
    """口股數: 高價股 (進場×2000 > 100萬) 用小型 100 股/口, 否則標準 2000 股/口."""
    return 100 if entry * 2000 > POS_CAP else 2000


def _size(entry, stop, equity):
    """規格 §0.2 部位公式 → (shares, lots, lot_shares). 不可建倉回 (0,0,ls)."""
    d = entry - stop
    if d <= 0:
        return 0, 0, 2000
    qty_theory = (equity * RISK_PCT) / d
    mv = min(qty_theory * entry, POS_CAP)
    ls = _lot_shares(entry)
    lots = int(mv // (entry * ls))
    return lots * ls, lots, ls


def _costs(shares, price, lots):
    """單邊成本 (稅 + 手續費). 滑價另由價格調整."""
    return shares * price * STF_TAX + lots * STF_COMM_PER_LOT


def run_stf(data, entry_map, strategy_key, all_dates=None, date_idx=None,
            risk_on=None, equity0=ACCOUNT_EQUITY, leverage=LEVERAGE,
            max_pos=MAX_POS, picks_per_day=PICKS_PER_DAY):
    """STF 子帳戶回測. 回傳 (trades, equity_curve_df)."""
    equity = equity0
    open_pos = []
    trades = []
    curve = []
    if all_dates is None or date_idx is None:
        all_dates, date_idx = bt.build_date_index(data)

    for d in all_dates:
        # ── 出場 ──────────────────────────────────────────────────────────
        still_open = []
        for p in open_pos:
            di = date_idx[p["code"]].get(d)
            if di is None or di <= p["entry_idx"]:
                still_open.append(p)
                continue
            df = data[p["code"]]
            row = df.iloc[di]
            R = p["entry"] - p["init_stop"]          # 每股風險 (1R)
            exit_price = None

            # 1) 初始/移動停損觸及 (含跌停延遲, 沿用股票引擎判斷)
            if row["low"] <= p["stop"]:
                if (bt._is_limit_day(df, di) and row["close"] < row["open"]
                        and di + 1 < len(df)):
                    p["expire_idx"] = max(p["expire_idx"], di + 1)
                    still_open.append(p)
                    continue
                exit_price = min(p["stop"], row["open"])
            # 2) 結構出場: 收盤跌破 ma_exit
            elif p["ma_exit"] and not np.isnan(row.get(p["ma_exit"], np.nan)) \
                    and row["close"] < row[p["ma_exit"]]:
                exit_price = row["close"]
            # 3) 時間停損: 持有滿 N 日且未達 0.5R
            elif (di - p["entry_idx"] >= TIME_STOP_DAYS
                  and R > 0 and not p["reached_1r"]
                  and row["close"] < p["entry"] + TIME_STOP_MIN_R * R):
                exit_price = row["close"]
            # 4) 最長持有
            elif di >= p["expire_idx"]:
                exit_price = row["close"]

            if exit_price is None:
                # 尚未出場 → 更新三段式移動停利
                p["high_since"] = max(p["high_since"], row["high"])
                if R > 0:
                    if row["high"] >= p["entry"] + 1 * R:
                        p["reached_1r"] = True
                    if row["high"] >= p["entry"] + 2 * R:
                        p["reached_2r"] = True
                    if p["reached_1r"]:
                        p["stop"] = max(p["stop"], p["entry"])          # 移成本
                    if p["reached_2r"]:
                        atr = row["atr14"] if not np.isnan(row["atr14"]) else p["init_atr"]
                        chand = p["high_since"] - CHANDELIER_ATR * atr
                        p["stop"] = max(p["stop"], p["entry"] + 1 * R, chand)
                still_open.append(p)
                continue

            # 平倉結算 (STF 成本)
            exit_eff = exit_price * (1 - STF_SLIP)
            gross = p["shares"] * (exit_eff - p["entry_eff"])
            cost = _costs(p["shares"], exit_eff, p["lots"]) + p["entry_cost"]
            pnl = gross - cost
            equity += pnl
            trades.append({
                "strategy": strategy_key, "code": p["code"], "name": p["sig_name"],
                "entry_date": p["entry_date"], "exit_date": d,
                "entry": p["entry"], "exit": exit_eff, "lots": p["lots"],
                "shares": p["shares"], "pnl": pnl,
                "r": pnl / p["risk_amt"] if p["risk_amt"] > 0 else 0.0,
            })
        open_pos = still_open

        # ── 新倉 (大盤濾網: risk_on=False → 不進場) ──────────────────────────
        allow_entry = True if risk_on is None else (d in risk_on)
        if allow_entry and len(open_pos) < max_pos:
            held = {p["code"] for p in open_pos}
            exposure = sum(p["shares"] * p["entry"] for p in open_pos)
            taken = 0
            for score, code, ei, s in sorted(entry_map.get(d, []), reverse=True):
                if taken >= picks_per_day or len(open_pos) >= max_pos:
                    break
                if code in held:
                    continue
                df = data[code]
                if ei >= len(df):
                    continue
                # 漲停封板無法合理買入 → 略過
                if bt._is_limit_day(df, ei):
                    prev_close = df["close"].iloc[ei - 1] if ei > 0 else df["open"].iloc[ei]
                    if df["close"].iloc[ei] > prev_close * 1.08:
                        continue
                entry = df["open"].iloc[ei] * (1 + STF_SLIP)
                atr = df["atr14"].iloc[ei - 1]
                if np.isnan(atr) or atr <= 0:
                    continue
                stop = entry * (1 - s["stop_pct"])
                if stop <= 0 or entry <= stop:
                    continue
                shares, lots, ls = _size(entry, stop, equity)
                if shares <= 0:
                    continue
                notional = shares * entry
                if exposure + notional > equity * leverage:
                    continue
                entry_cost = _costs(shares, entry, lots)
                open_pos.append({
                    "code": code, "sig_name": s.get("name", strategy_key),
                    "shares": shares, "lots": lots, "lot_shares": ls,
                    "entry": entry, "entry_eff": entry, "entry_cost": entry_cost,
                    "stop": stop, "init_stop": stop,
                    "ma_exit": s.get("ma_exit"),
                    "high_since": df["high"].iloc[ei],
                    "reached_1r": False, "reached_2r": False,
                    "init_atr": atr, "entry_idx": ei,
                    "expire_idx": ei + s["max_hold"], "entry_date": d,
                    "risk_amt": shares * (entry - stop),
                })
                exposure += notional
                held.add(code)
                taken += 1

        peak = max((c["equity"] for c in curve), default=equity0)
        curve.append({"date": d, "equity": equity,
                      "drawdown": (max(peak, equity) - equity) / max(peak, equity),
                      "n_pos": len(open_pos)})

    return trades, pd.DataFrame(curve)


def metrics(trades, curve, equity0=ACCOUNT_EQUITY):
    """規格 §0.6 驗收指標: 年化 / MaxDD / PF / Sharpe / 期望值 / 勝率."""
    if not trades or curve.empty:
        return dict(trades=0, win=0, pf=0, cagr=0, mdd=0, sharpe=0, expectancy=0,
                    mar=0, final=equity0)
    eq = curve.set_index("date")["equity"]
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    # 日報酬 Sharpe (用權益曲線)
    daily = eq.pct_change().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(TD)) if daily.std() > 0 else 0.0
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] / equity0) ** (1 / yrs) - 1
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    expectancy = sum(t["r"] for t in trades) / n      # 平均 R (期望值)
    return dict(trades=n, win=len(wins) / n, pf=pf, cagr=cagr, mdd=mdd,
                sharpe=sharpe, expectancy=expectancy,
                mar=cagr / mdd if mdd > 0 else 0.0, final=eq.iloc[-1])
