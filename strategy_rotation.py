"""策略 R — 橫斷面動能輪動 (cross-sectional momentum rotation).

思維: K/D 是事件驅動狙擊 (單筆風險1%, 資金大多閒置), 報酬天花板卡在資金利用率。
R 換 alpha 家族: 每月把資金滿載輪動到全市場 RS 最強的一籃股票, 吃「強者恆強」
的橫斷面動能溢價 (文獻支持最強的異常之一), 大盤走弱 (regime) 時全數退場現金。

規則 (刻意極簡, 全部常規值, 不做網格挑選):
  每 21 個交易日 (約月度) 再平衡:
    合格: 收盤>10元, 成交額>5000萬, 站上MA200, 有126日報酬
    排名: 126日報酬 (RS) 由高到低
    持有: 前 TOP_N (等權); 緩衝: 舊持股仍在前 TOP_N*2 名內則續抱 (降換手)
    大盤: 再平衡日不在 risk_on (大盤濾網) → 全數出清持現金
  執行: 訊號日次日開盤 (含滑價/手續費/證交稅, 與 run_sub 同成本)

驗證結論 (2026-07, data_adj 1977檔15年): ❌ 不採用 —
  總損益 +634萬 (未達 2×K=1,116萬 門檻), MaxDD 67.5% 災難級,
  IS(<2020) 八年僅 +35萬, 獲利集中 2026 單年 (+436萬) — 典型 momentum crash
  風險 + 單年集中, 不可上生產。同場加測: K/D + 金字塔加碼 (引擎原生 pyramid)
  亦不合格 (勝率崩至23-24%, IS/OOS 均R皆轉負, 加碼上移停損在台股被反覆洗出)。
  結論: 事件驅動 (K/D) 仍是此資料上最穩的股票 alpha; 要放大總報酬的已驗證路徑
  是 K+D 並用 (+約1,019萬, 各自獨立驗證) 與風險係數調整 (槓桿, 非新alpha)。
用法: MM_DATA_DIR=data_adj python3 strategy_rotation.py
"""
import os
import time

import numpy as np
import pandas as pd

from backtest import load_all, load_regime, FEE, TAX, SLIP, INIT_CAPITAL

TOP_N = 10
REBAL = 21          # 交易日
BUFFER_MULT = 2     # 舊持股在前 TOP_N*BUFFER 名內續抱
MIN_PRICE = 10.0
MIN_TURNOVER = 50_000_000


def run(top_n=TOP_N, rebal=REBAL, buf=BUFFER_MULT, verbose=True):
    t0 = time.time()
    data, names = load_all()
    risk_on = load_regime()
    if verbose:
        print(f"load_all {len(data)} 檔 ({time.time()-t0:.0f}s)", flush=True)

    # 全市場交易日軸 + 每檔 date→idx
    all_dates = sorted({d for df in data.values() for d in df["date"]})
    idx = {c: {d: i for i, d in enumerate(df["date"])} for c, df in data.items()}

    equity = INIT_CAPITAL
    cash = INIT_CAPITAL
    hold = {}            # code -> {"shares","entry","entry_date"}
    trades = []
    curve = []           # (date, equity)

    def px(c, d, col):
        i = idx[c].get(d)
        return None if i is None else data[c][col].iloc[i]

    def sell(c, d, reason):
        nonlocal cash
        p = px(c, d, "open")
        if p is None or np.isnan(p) or p <= 0:      # 停牌: 用最後收盤估值出
            i = idx[c].get(d)
            p = hold[c]["last_close"]
        p = p * (1 - SLIP)
        proceeds = hold[c]["shares"] * p * (1 - FEE - TAX)
        cost = hold[c]["shares"] * hold[c]["entry"] * (1 + FEE)
        trades.append({"code": c, "entry_date": hold[c]["entry_date"],
                       "exit_date": d, "entry": hold[c]["entry"], "exit": p,
                       "pnl": proceeds - cost})
        cash += hold[c]["shares"] * p * (1 - FEE - TAX)
        del hold[c]

    last_rebal = -10**9
    for di, d in enumerate(all_dates):
        # 逐日估值
        val = cash
        for c, h in hold.items():
            cl = px(c, d, "close")
            if cl is not None and not np.isnan(cl):
                h["last_close"] = cl
            val += h["shares"] * h["last_close"]
        equity = val
        curve.append((d, equity))

        if di - last_rebal < rebal:
            continue
        # ── 再平衡 (以今日資料選股, 次日開盤執行) ──
        if di + 1 >= len(all_dates):
            break
        nd = all_dates[di + 1]
        last_rebal = di

        if d not in risk_on:                       # 大盤濾網: 全出
            for c in list(hold):
                sell(c, nd, "regime")
            continue

        scored = []
        for c, df in data.items():
            i = idx[c].get(d)
            if i is None or i < 210:
                continue
            row = df.iloc[i]
            if (row["close"] <= MIN_PRICE or np.isnan(row["ma200"])
                    or row["close"] < row["ma200"]
                    or row["close"] * row["volume"] < MIN_TURNOVER
                    or np.isnan(row["ret126"])):
                continue
            scored.append((row["ret126"], c))
        scored.sort(reverse=True)
        top = [c for _, c in scored[:top_n]]
        keep_zone = {c for _, c in scored[:top_n * buf]}

        # 賣出: 掉出緩衝區的舊持股
        for c in list(hold):
            if c not in keep_zone:
                sell(c, nd, "rotate")
        # 買進: 新進前段者, 等權配置
        slots = top_n - len(hold)
        buys = [c for c in top if c not in hold][:max(slots, 0)]
        if buys:
            alloc = cash / len(buys)
            for c in buys:
                p = px(c, nd, "open")
                if p is None or np.isnan(p) or p <= 0:
                    continue
                p = p * (1 + SLIP)
                sh = int(alloc / (p * (1 + FEE)) / 1000) * 1000
                if sh <= 0:
                    sh = int(alloc / (p * (1 + FEE)))
                if sh <= 0:
                    continue
                cash -= sh * p * (1 + FEE)
                hold[c] = {"shares": sh, "entry": p, "entry_date": nd,
                           "last_close": p}
    # 期末清算
    if hold:
        dl = all_dates[-1]
        for c in list(hold):
            sell(c, dl, "final")

    eq = pd.Series([e for _, e in curve], index=[d for d, _ in curve])
    return trades, eq


def report(trades, eq):
    p = np.array([t["pnl"] for t in trades])
    years = len(eq) / 252
    cagr = (eq.iloc[-1] / INIT_CAPITAL) ** (1 / years) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    w = p[p > 0]; ls = p[p < 0]
    print(f"交易 {len(p)} 筆  勝率 {(p>0).mean():.0%}  PF {w.sum()/-ls.sum():.2f}")
    print(f"總損益 {p.sum():+,.0f}  期末權益 {eq.iloc[-1]:,.0f}  "
          f"年化 {cagr:+.1%}  MaxDD {dd:.1%}")
    cut = pd.Timestamp(2020, 1, 1)
    for lbl, sel in (("IS(<2020)", [t for t in trades if t["exit_date"] < cut]),
                     ("OOS(>=2020)", [t for t in trades if t["exit_date"] >= cut])):
        pp = np.array([t["pnl"] for t in sel])
        if len(pp):
            print(f"  {lbl}: {len(pp)}筆 勝率{(pp>0).mean():.0%} 損益{pp.sum():+,.0f}")
    by = {}
    for t in trades:
        by.setdefault(t["exit_date"].year, 0)
        by[t["exit_date"].year] += t["pnl"]
    print("  逐年: " + "  ".join(f"{y}:{v/10000:+,.0f}萬" for y, v in sorted(by.items())))


if __name__ == "__main__":
    tr, eq = run()
    report(tr, eq)
    print("DONE", flush=True)
