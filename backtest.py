"""組合回測: 200 萬本金, 策略 C / D 各 50% 獨立資金池.

交易成本: 手續費 0.1425% 買賣各一次, 證交稅 0.3% (賣出), 滑價 0.1% 單邊.
進場: 訊號日次日開盤價. 出場: 停損/停利或時間出場.

部位控制 (各子帳戶):
  - 單筆風險 = 當日子帳戶權益 × RISK_PCT × 制度調節因子
  - 名目曝險上限 = 子帳戶權益 × 槓桿
  - 策略 C/D: 最大 8 檔同時持倉, 每日最多 2 檔新倉
  - 回撤保護: 子帳戶從峰值回撤 > DD_PAUSE 時暫停, 縮回 DD_RESUME 以內恢復

上線交易強化 (live-ready):
  1. 漲停/跌停填單模擬: 進場日漲停 → 跳過; 停損日跌停 → 延至次日開盤
  2. 回撤熔斷: 子帳戶 DD > DD_PAUSE 暫停新倉 (預設 20%)
  3. 三段式市況調節: TAIEX vs MA20/MA60 → 強勢/盤整/防禦三級
  4. 波動目標化: TAIEX 20日實現波動率高時自動縮小 RISK_PCT
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from strategies import STRATEGIES, add_indicators

DATA_DIR = os.environ.get(
    "MM_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
FEE = 0.001425
TAX = 0.003
SLIP = 0.001
INIT_CAPITAL = 2_000_000
ALLOC = 0.5          # 每策略分配比例 (各 100 萬)
RISK_PCT = 0.01      # 單筆風險
PICKS_PER_DAY = 2    # 每子帳戶每日最多新倉
# 各策略回撤保護閾值 (B 波動大,靠停損與部位控制,不設 pause 上限)
# 回撤熔斷: DD_PAUSE 為暫停閾值, DD_RESUME 為恢復閾值
# 1.00 = 停用; 上線建議 0.20/0.15 (C/D 波動較小)
DD_PAUSE  = {"A": 0.20, "C": 0.20, "D": 0.20, "E": 0.20}
DD_RESUME = {"A": 0.12, "C": 0.12, "D": 0.12, "E": 0.12}

# 各策略最大同時持倉 (持有期越長需要越多槽)
MAX_POS = {"A": 8, "C": 8, "D": 8, "E": 8}

# 三週法則參數 (策略 D): 突破後 THREE_WEEK_DAYS 日內漲幅 >= THREE_WEEK_GAIN
# 即視為主升段, 取消 +20% 賣半, 讓強勢股自由奔跑 (各訊號可覆寫 three_week_gain)
THREE_WEEK_GAIN = 0.20
THREE_WEEK_DAYS = 15

# 回撤保護暫停後的強制恢復冷卻天數 (交易日). 避免已實現權益凍結造成永久暫停.
PAUSE_COOLDOWN = 60

# 三段式市況調節: TAIEX vs MA20/MA60 → 各階段 RISK_PCT 乘數
REGIME_RISK_MULT = {2: 1.0, 1: 0.6, 0: 0.0}   # 強勢/盤整/防禦
# 波動目標化: TAIEX 20日年化波動率的「基準值」; 超過時縮小 RISK_PCT
VOL_TARGET = 0.15   # 15% 年化波動 = 1× ; 30% → 0.5×
VOL_CAP    = 1.0    # 只縮不放: 低波動時維持原 RISK_PCT, 高波動時縮小

# 漲跌停判定門檻: 日振幅 / 開盤價 < 此值視為限制板
LIMIT_RANGE_THRESH = 0.005


def load_all():
    data = {}
    uni = os.path.join(DATA_DIR, "universe.json")
    names = {}
    if os.path.exists(uni):
        with open(uni) as f:
            names = {s["code"]: s["name"] for s in json.load(f)}
    for fn in os.listdir(DATA_DIR):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(DATA_DIR, fn), parse_dates=["date"])
        df = add_indicators(df).reset_index(drop=True)
        data[code] = df
    return data, names


def load_regime():
    df = pd.read_csv(os.path.join(DATA_DIR, "_TWII.csv"), parse_dates=["date"])
    df["ma60"] = df["close"].rolling(60).mean()
    return set(df.loc[df["close"] > df["ma60"], "date"])


def load_regime_tiers():
    """三段式市況分級: 2=強勢(TAIEX>MA20>MA60), 1=盤整, 0=防禦(TAIEX<MA60且MA60下彎).

    回傳 {date: tier} dict 供 run_sub 動態調整 RISK_PCT 與開倉限制.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "_TWII.csv"), parse_dates=["date"])
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma60_5d_ago"] = df["ma60"].shift(5)
    tiers = {}
    for row in df.itertuples():
        if pd.isna(row.ma20) or pd.isna(row.ma60):
            tiers[row.date] = 2   # 資料不足期預設正常
            continue
        if row.close > row.ma20 and row.ma20 > row.ma60:
            tiers[row.date] = 2   # 強勢
        elif row.close < row.ma60 and (pd.isna(row.ma60_5d_ago)
                                        or row.ma60 < row.ma60_5d_ago):
            tiers[row.date] = 0   # 防禦
        else:
            tiers[row.date] = 1   # 盤整/過渡
    return tiers


def load_vol_scalars():
    """波動目標化: 每日波動標量 = VOL_TARGET / 當日20日年化波動率, 夾在 [0.5, VOL_CAP].

    實現波動率 = 20日日報酬標準差 × √244 (年化).
    波動高時縮小 RISK_PCT, 避免高波動期過度下注.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "_TWII.csv"), parse_dates=["date"])
    df["ret"] = df["close"].pct_change()
    df["vol20"] = df["ret"].rolling(20).std() * (244 ** 0.5)
    scalars = {}
    for row in df.itertuples():
        if pd.isna(row.vol20) or row.vol20 <= 0:
            scalars[row.date] = 1.0
        else:
            s = VOL_TARGET / row.vol20
            scalars[row.date] = max(0.5, min(VOL_CAP, s))
    return scalars


def compute_rs_rank(data):
    """126 日報酬的全市場百分位排名 (DataFrame: date x code)."""
    panel = pd.DataFrame({code: df.set_index("date")["ret126"]
                          for code, df in data.items()})
    return panel.rank(axis=1, pct=True)


def collect_signals(data, strategy_key):
    sig_fn = STRATEGIES[strategy_key]
    signals = defaultdict(list)
    needs_rs = strategy_key in ("A", "C", "D", "E")
    rs = compute_rs_rank(data) if needs_rs else None
    min_i = 210 if needs_rs else 120  # A/C/D 需要 MA200
    for code, df in data.items():
        for i in range(min_i, len(df) - 1):
            if needs_rs:
                d = df["date"].iloc[i]
                try:
                    rank = rs.at[d, code]
                except KeyError:
                    continue
                if np.isnan(rank):
                    continue
                s = sig_fn(df, i, rs_rank=rank)
            else:
                s = sig_fn(df, i)
            if s:
                signals[df["date"].iloc[i]].append((s["score"], code, i, s))
    return signals


def build_entry_map(signals, data):
    entry_map = defaultdict(list)
    for sd, lst in signals.items():
        for score, code, i, s in lst:
            df = data[code]
            if i + 1 < len(df):
                entry_map[df["date"].iloc[i + 1]].append((score, code, i + 1, s))
    return entry_map


def build_date_index(data):
    """預先建立 (所有交易日, 各檔日期→列索引) 供多次回測共用, 避免重建."""
    all_dates = sorted({d for df in data.values() for d in df["date"]})
    date_idx = {code: {d: i for i, d in enumerate(df["date"])}
                for code, df in data.items()}
    return all_dates, date_idx


def _climax_run(df, di, p):
    """衝頂 / 延長訊號: 強勢賣出條件 (§10.2).

    衝頂: 近 15 日漲幅 >= climax_sprint_pct 且總獲利 >= climax_min_gain
    延長: 近 10 日上漲天數 >= extended_updays 且總獲利 >= extended_min_gain
    """
    entry_idx = p["entry_idx"]
    days_held = di - entry_idx
    if days_held < 5:
        return False
    init_entry = p.get("init_entry", p["entry"])
    cur_gain = df["close"].iloc[di] / init_entry - 1.0

    # 衝頂訊號
    sprint_days = min(15, days_held)
    sprint_gain = df["close"].iloc[di] / df["close"].iloc[di - sprint_days] - 1.0
    climax = (sprint_gain >= p.get("climax_sprint_pct", 0.25)
              and cur_gain >= p.get("climax_min_gain", 0.30))

    # 延長訊號
    start_k = max(entry_idx + 1, di - 9)
    up_days = sum(
        1 for k in range(start_k, di + 1)
        if df["close"].iloc[k] > df["close"].iloc[k - 1]
    )
    extended = (up_days >= p.get("extended_updays", 7)
                and cur_gain >= p.get("extended_min_gain", 0.20))

    return climax or extended


def _is_limit_day(df, di):
    """日振幅 < LIMIT_RANGE_THRESH 視為漲停或跌停封板 (台灣 ±10%)."""
    row = df.iloc[di]
    if row["open"] <= 0:
        return False
    return (row["high"] - row["low"]) / row["open"] < LIMIT_RANGE_THRESH


def run_sub(data, entry_map, strategy_key, leverage, init_eq,
            all_dates=None, date_idx=None,
            regime_tiers=None, vol_scalars=None):
    """單一策略子帳戶回測.

    all_dates / date_idx 可預先以 build_date_index 算好傳入 (參數掃描加速).
    regime_tiers  : {date: 0/1/2} 三段式市況; None = 停用 (全部視為 tier 2)
    vol_scalars   : {date: float} 波動標量; None = 停用 (全部 1.0)
    """
    max_pos = MAX_POS[strategy_key]
    dd_pause = DD_PAUSE[strategy_key]
    dd_resume = DD_RESUME[strategy_key]
    equity = init_eq
    peak_eq = init_eq
    paused = False
    pause_count = 0
    open_pos = []
    trades = []
    curve = []
    if all_dates is None or date_idx is None:
        all_dates, date_idx = build_date_index(data)

    for d in all_dates:
        # 三段式市況 + 波動標量 → 有效 RISK_PCT
        tier = regime_tiers.get(d, 2) if regime_tiers else 2
        vsca = vol_scalars.get(d, 1.0) if vol_scalars else 1.0
        tier_mult = REGIME_RISK_MULT[tier]
        eff_risk = RISK_PCT * tier_mult * vsca

        # 出場 (含跌停延遲)
        still_open = []
        for p in open_pos:
            di = date_idx[p["code"]].get(d)
            if di is None or di <= p["entry_idx"]:
                still_open.append(p)
                continue
            row = data[p["code"]].iloc[di]
            df_stock = data[p["code"]]
            exit_price = None

            # 跌停封板: 停損觸及但無法賣出 → 延至次日開盤
            stop_triggered = row["low"] <= p["stop"]
            if stop_triggered and _is_limit_day(df_stock, di) and row["close"] < row["open"]:
                # 跌停, 明日開盤再出場 (expire_idx 往後一天保留部位)
                p["expire_idx"] = max(p["expire_idx"], di + 1)
                p["stop"] = p["stop"]   # 停損價不變, 次日用 open
                # 若已是最後一個交易日則強制以今日收盤出
                if di + 1 >= len(df_stock):
                    exit_price = row["close"]
                else:
                    still_open.append(p)
                    continue
            elif stop_triggered:
                exit_price = min(p["stop"], row["open"])
            elif p["target"] is not None and row["high"] >= p["target"]:
                exit_price = row["open"] if row["open"] >= p["target"] else p["target"]
            elif p.get("climax_exit") and _climax_run(data[p["code"]], di, p):
                exit_price = row["close"]   # 衝頂強勢賣出
            elif p.get("minervini") and row["close"] < row["ma50"]:
                exit_price = row["close"]   # 跌破 50 日線全部出場
            elif di >= p["expire_idx"]:
                exit_price = row["close"]

            if exit_price is None:
                if p.get("trail_atr"):
                    p["high_close"] = max(p["high_close"], row["close"])
                    atr = row["atr14"] if not np.isnan(row["atr14"]) else p["init_atr"]
                    p["stop"] = max(p["stop"],
                                    p["high_close"] - p["trail_atr"] * atr)
                if p.get("minervini"):
                    risk_per_sh = p["entry"] - p["init_stop"]
                    days_held = di - p["entry_idx"]
                    init_entry = p.get("init_entry", p["entry"])
                    tw_gain = p.get("three_week_gain", THREE_WEEK_GAIN)
                    # 三週法則: 突破後 N 日內漲 >= G → 跳過賣半, 讓強勢股自由奔跑
                    if not p["three_week_hold"] and days_held <= THREE_WEEK_DAYS:
                        if row["high"] >= init_entry * (1 + tw_gain):
                            p["three_week_hold"] = True
                            # 不改 expire_idx, 只取消 +20% 賣半動作
                    # +20% 賣出一半 (三週法則啟動時跳過)
                    if (not p["half_sold"] and not p["three_week_hold"]
                            and row["high"] >= init_entry * 1.20):
                        px = max(p["entry"] * 1.20, row["open"]) * (1 - SLIP)
                        half = p["shares"] // 2
                        if half > 0:
                            pnl = half * (px * (1 - FEE - TAX)
                                          - p["entry"] * (1 + FEE))
                            equity += pnl
                            trades.append({
                                "strategy": strategy_key, "code": p["code"],
                                "entry_date": p["entry_date"], "exit_date": d,
                                "entry": p["entry"], "exit": px, "pnl": pnl,
                                "r": pnl / (half * risk_per_sh) if risk_per_sh > 0 else 0,
                            })
                            p["shares"] -= half
                            p["risk_amt"] = p["shares"] * risk_per_sh
                        p["half_sold"] = True
                    # 獲利達 3R 時停損上移至成本價 (保本)
                    if risk_per_sh > 0 and row["close"] >= p["entry"] + 3 * risk_per_sh:
                        p["stop"] = max(p["stop"], p["entry"])
                    # MA50 上穿成本價後改用 MA50 移動停損
                    if row["ma50"] >= p["entry"]:
                        p["stop"] = max(p["stop"], row["ma50"])
                still_open.append(p)
                continue

            exit_price *= (1 - SLIP)
            proceeds = p["shares"] * exit_price * (1 - FEE - TAX)
            cost = p["shares"] * p["entry"] * (1 + FEE)
            pnl = proceeds - cost
            equity += pnl
            trades.append({
                "strategy": strategy_key,
                "code": p["code"],
                "entry_date": p["entry_date"],
                "exit_date": d,
                "entry": p["entry"],
                "exit": exit_price,
                "pnl": pnl,
                "r": pnl / p["risk_amt"] if p["risk_amt"] > 0 else 0,
            })
        open_pos = still_open
        peak_eq = max(peak_eq, equity)
        dd = (peak_eq - equity) / peak_eq

        # 回撤保護: 跌破 dd_pause 暫停新倉. 因權益為「已實現」, 暫停後權益凍結,
        # 單靠 dd 回到 dd_resume 永遠無法恢復 (吸收態), 故以冷卻天數強制恢復:
        # 回撤回到 dd_resume 以內 或 暫停滿 PAUSE_COOLDOWN 日即恢復並重設峰值.
        if not paused:
            if dd >= dd_pause:
                paused = True
                pause_count = 0
        else:
            pause_count += 1
            if dd <= dd_resume or pause_count >= PAUSE_COOLDOWN:
                paused = False
                peak_eq = equity   # 重設峰值, 避免恢復後立即再次觸發

        # pyramid 加碼 (先於新倉，利用當日行情)
        for p in open_pos:
            if not p.get("pyramid_adds"):
                continue
            di = date_idx[p["code"]].get(d)
            if di is None:
                continue
            row = data[p["code"]].iloc[di]
            done = []
            for add in p["pyramid_adds"]:
                if row["high"] >= add["trigger"]:
                    add_px = max(add["trigger"], row["open"]) * (1 + SLIP)
                    add_sh = add["shares"]
                    notional_add = add_sh * add_px
                    exposure_now = sum(q["shares"] * q["entry"] for q in open_pos)
                    if exposure_now + notional_add <= equity * leverage and add_sh > 0:
                        total_sh = p["shares"] + add_sh
                        p["entry"] = (p["shares"] * p["entry"] + add_sh * add_px) / total_sh
                        p["shares"] = total_sh
                        # 停損隨加碼上移
                        if "new_stop_frac" in add:
                            new_stop = p["init_entry"] * add["new_stop_frac"]
                            p["stop"] = max(p["stop"], new_stop)
                        p["risk_amt"] = total_sh * (p["entry"] - p["stop"])
                    done.append(add)
            for a in done:
                p["pyramid_adds"].remove(a)

        # 新倉 (tier 0 防禦模式完全停止新倉)
        if not paused and tier > 0 and eff_risk > 0:
            held = {p["code"] for p in open_pos}
            exposure = sum(p["shares"] * p["entry"] for p in open_pos)
            candidates = sorted(entry_map.get(d, []), reverse=True)
            taken = 0
            for score, code, ei, s in candidates:
                if taken >= PICKS_PER_DAY or len(open_pos) >= max_pos:
                    break
                if code in held:
                    continue
                df = data[code]
                # 漲停封板: 進場日振幅極小且收盤 > 前日收盤 → 無法以合理價格買入
                if _is_limit_day(df, ei):
                    prev_close = df["close"].iloc[ei - 1] if ei > 0 else df["open"].iloc[ei]
                    if df["close"].iloc[ei] > prev_close * 1.08:
                        continue
                entry = df["open"].iloc[ei] * (1 + SLIP)
                atr = df["atr14"].iloc[ei - 1]
                if np.isnan(atr) or atr <= 0:
                    continue
                if "stop_pct" in s:
                    stop = entry * (1 - s["stop_pct"])
                else:
                    stop = entry - s["stop_atr"] * atr
                if stop <= 0 or entry <= stop:
                    continue
                risk_per_sh = entry - stop
                risk_amt = equity * eff_risk
                # pyramid: 計算全倉後初始只建 25%
                full_shares = int(risk_amt / risk_per_sh / 1000) * 1000
                if full_shares <= 0:
                    full_shares = max(1, int(risk_amt / risk_per_sh))
                if s.get("minervini"):
                    cap = int(equity * 0.25 / entry / 1000) * 1000
                    full_shares = min(full_shares, max(cap, 1))
                if s.get("pyramid"):
                    init_shares = max(1, full_shares // 4)
                else:
                    init_shares = full_shares
                shares = init_shares
                notional = shares * entry
                if exposure + notional > equity * leverage:
                    allowed = equity * leverage - exposure
                    shares = int(allowed / entry / 1000) * 1000
                    if shares <= 0:
                        continue
                    notional = shares * entry
                    full_shares = shares
                target = (entry + s["target_r"] * risk_per_sh
                          if "target_r" in s else None)
                # 準備 pyramid 加碼批次
                pyramid_adds = []
                if s.get("pyramid") and full_shares > shares:
                    rem = full_shares - shares
                    add1 = max(1, rem // 3)
                    add2 = rem - add1
                    # 隨加碼上移停損: 加碼後停損升至前一批進場點下方
                    # 確保滿倉後最大虧損 < 初始單批風險
                    pyramid_adds = [
                        {"trigger": entry * 1.02, "shares": add1,
                         "new_stop_frac": 0.995},   # 停損升至略低於突破價
                        {"trigger": entry * 1.04, "shares": add2,
                         "new_stop_frac": 1.015},   # 停損升至突破價 +1.5%
                    ]
                open_pos.append({
                    "code": code,
                    "shares": shares,
                    "entry": entry,
                    "init_entry": entry,   # 用於三週法則，不受加碼均價影響
                    "stop": stop,
                    "init_stop": stop,
                    "target": target,
                    "trail_atr": s.get("trail_atr"),
                    "minervini": s.get("minervini", False),
                    "pyramid_adds": pyramid_adds,
                    "half_sold": False,
                    "three_week_hold": False,  # 三週法則旗標
                    "three_week_gain": s.get("three_week_gain", THREE_WEEK_GAIN),
                    "climax_exit":       s.get("climax_exit", False),
                    "climax_sprint_pct": s.get("climax_sprint_pct", 0.25),
                    "climax_min_gain":   s.get("climax_min_gain", 0.30),
                    "extended_updays":   s.get("extended_updays", 7),
                    "extended_min_gain": s.get("extended_min_gain", 0.20),
                    "high_close": entry,
                    "init_atr": atr,
                    "entry_idx": ei,
                    "expire_idx": ei + s["max_hold"],
                    "entry_date": d,
                    "risk_amt": shares * risk_per_sh,
                })
                exposure += notional
                held.add(code)
                taken += 1

        curve.append({"date": d, "equity": equity, "drawdown": dd,
                      "paused": paused, "n_pos": len(open_pos),
                      "regime_tier": tier, "vol_scalar": round(vsca, 3)})

    return trades, pd.DataFrame(curve)


def run(data, names, leverage=2.0, strategies=("A", "B")):
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars  = load_vol_scalars()
    init_eq = INIT_CAPITAL / len(strategies)

    all_trades = []
    curves = {}
    for k in strategies:
        print(f"  回測策略 {k}...")
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        entry_map = build_entry_map(sigs, data)
        trades, curve = run_sub(data, entry_map, k, leverage, init_eq,
                                regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        all_trades.extend(trades)
        curves[k] = curve

    # 合併權益曲線 (任意數量子帳戶相加)
    combined = None
    for k in strategies:
        c = curves[k][["date", "equity"]].rename(columns={"equity": f"eq_{k}"})
        combined = c if combined is None else combined.merge(c, on="date")
    combined["equity"] = sum(combined[f"eq_{k}"] for k in strategies)
    combined["drawdown"] = ((combined["equity"].cummax() - combined["equity"])
                            / combined["equity"].cummax())

    return all_trades, combined, curves


def report(trades, combined, curves, leverage):
    tdf = pd.DataFrame(trades)
    eq = combined["equity"]
    ret = eq.iloc[-1] / INIT_CAPITAL - 1
    dd_max = combined["drawdown"].max()
    years = len(combined) / 244
    cagr = (eq.iloc[-1] / INIT_CAPITAL) ** (1 / max(years, 0.01)) - 1

    print(f"\n{'='*55}")
    print(f"組合回測  ({' + '.join(curves)} 均分資金,  槓桿 {leverage}x)")
    print(f"{'='*55}")
    print(f"總報酬: {ret:+.1%}   年化: {cagr:+.1%}   最大回撤: {dd_max:.1%}")
    print(f"期末權益: {eq.iloc[-1]:,.0f}   (初始 {INIT_CAPITAL:,.0f})")

    print(f"\n{'策略':<6} {'交易':>5} {'勝率':>6} {'均R':>6} {'獲利因子':>8} "
          f"{'報酬':>7} {'年化':>7} {'最大回撤':>8} {'暫停日':>6}")
    print("-" * 65)
    for k, curve in curves.items():
        sub_init = INIT_CAPITAL / len(curves)
        sub_eq = curve["equity"]
        sub_ret = sub_eq.iloc[-1] / sub_init - 1
        sub_dd = curve["drawdown"].max()
        sub_cagr = (sub_eq.iloc[-1] / sub_init) ** (1 / max(years, 0.01)) - 1
        sub_pause = curve["paused"].sum()
        stdf = pd.DataFrame([t for t in trades if t["strategy"] == k])
        if len(stdf):
            win = (stdf["pnl"] > 0).mean()
            avg_r = stdf["r"].mean()
            pf = (stdf.loc[stdf.pnl > 0, "pnl"].sum()
                  / max(1, -stdf.loc[stdf.pnl < 0, "pnl"].sum()))
        else:
            win = avg_r = pf = 0
        print(f"  {k:<4} {len(stdf):>5} {win:>6.1%} {avg_r:>6.2f} {pf:>8.2f} "
              f"{sub_ret:>7.1%} {sub_cagr:>7.1%} {sub_dd:>8.1%} {sub_pause:>6}")

    # 市況分級統計 (從第一個子帳曲線取)
    first_curve = next(iter(curves.values()))
    if "regime_tier" in first_curve.columns:
        combined = combined.merge(
            first_curve[["date", "regime_tier"]].drop_duplicates("date"), on="date", how="left")
    if "regime_tier" in combined.columns:
        tier_days = combined["regime_tier"].value_counts().sort_index()
        total = len(combined)
        print(f"\n市況分級 (交易日):  "
              f"強勢={tier_days.get(2,0)} ({tier_days.get(2,0)/total:.0%})  "
              f"盤整={tier_days.get(1,0)} ({tier_days.get(1,0)/total:.0%})  "
              f"防禦={tier_days.get(0,0)} ({tier_days.get(0,0)/total:.0%})")

    return tdf, combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--strategies", default="A,C,D",
                    help="逗號分隔, 例如 A,B,C")
    ap.add_argument("--data-dir", default=None,
                    help="覆寫資料目錄 (例如 data_adj)")
    args = ap.parse_args()

    if args.data_dir:
        global DATA_DIR
        # 支援絕對路徑 (地端資料如 /Users/.../adjusted_data) 與相對路徑
        DATA_DIR = (args.data_dir if os.path.isabs(args.data_dir)
                    else os.path.join(os.path.dirname(__file__), args.data_dir))

    strategies = tuple(args.strategies.split(","))
    print("載入資料...")
    data, names = load_all()
    print(f"{len(data)} 檔股票\n")

    trades, combined, curves = run(data, names, leverage=args.leverage,
                                   strategies=strategies)
    tdf, _ = report(trades, combined, curves, args.leverage)
    tdf.to_csv("trades_combined.csv", index=False)
    combined.to_csv("equity_combined.csv", index=False)


if __name__ == "__main__":
    main()
