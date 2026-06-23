"""個股期貨 (STF) 技術面 5 策略回測 — 規格書實證 (一/五/六/七/八).

沿用 backtest.py 已驗證的選股基礎 (load_all / compute_rs_rank / load_regime /
build_entry_map / build_date_index), 用 stf_engine 的規格化執行核心回測, 並依
規格 §0.6 輸出各策略驗收指標對照表 (年化 / MaxDD / PF / Sharpe / 期望值)。

用法:
    python3 stf_backtest.py                 # 全 5 策略, 全期
    python3 stf_backtest.py --strategies S1,S7
    MM_DATA_DIR=data_adj python3 stf_backtest.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_guard
env_guard.apply()

import numpy as np
import pandas as pd

import backtest as bt
import stf_engine as se
import stf_strategies as ss

# 規格 §0.6 單策略驗收門檻
THRESH = dict(mdd=0.25, pf=1.5, sharpe=1.0)


def build_market_ret60():
    """大盤 (TWII) 近60日報酬 {date: ret60}, 供策略七相對強度比較."""
    twii = pd.read_csv(os.path.join(bt.DATA_DIR, "_TWII.csv"), parse_dates=["date"])
    twii = twii.sort_values("date").reset_index(drop=True)
    twii["ret60"] = twii["close"].pct_change(60)
    return dict(zip(twii["date"], twii["ret60"]))


def collect_stf_signals(data, rs, key, all_dates, date_idx):
    """掃全市場逐根套 STF 策略, 回傳 {entry_date: [(score,code,entry_idx,sig)]}."""
    fn = ss.STF_STRATEGIES[key]
    sig_fn_signals = {}
    for code, df in data.items():
        n = len(df)
        for i in range(210, n - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if np.isnan(rank):
                continue
            s = fn(df, i, rs_rank=rank)
            if s:
                sig_fn_signals.setdefault(d, []).append((s["score"], code, i, s))
    return sig_fn_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="S1,S5,S6,S7,S8")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="只取前 N 檔 (快速冒煙測試用; 0=全市場)")
    args = ap.parse_args()
    keys = [k.strip().upper() for k in args.strategies.split(",") if k.strip()]

    print(f"載入全市場資料 ({bt.DATA_DIR}) ...")
    data, names = bt.load_all()
    if args.limit and args.limit < len(data):
        data = {c: data[c] for c in list(data)[:args.limit]}
    # 期間裁切 (選用)
    if args.start or args.end:
        s = pd.Timestamp(args.start) if args.start else None
        e = pd.Timestamp(args.end) if args.end else None
        for code in list(data):
            df = data[code]
            if s is not None:
                df = df[df["date"] >= s]
            if e is not None:
                df = df[df["date"] <= e]
            if len(df) < 220:
                del data[code]
            else:
                data[code] = df.reset_index(drop=True)
    print(f"  {len(data)} 檔有效標的")

    print("計算 RS 排名 / 大盤濾網 / 大盤報酬 ...")
    rs = bt.compute_rs_rank(data)
    risk_on = bt.load_regime()
    ss.set_market(build_market_ret60())
    all_dates, date_idx = bt.build_date_index(data)

    print(f"帳戶 {se.ACCOUNT_EQUITY:,.0f} / 風險 {se.RISK_PCT:.0%} / 單檔上限 "
          f"{se.POS_CAP:,.0f} / 槓桿 {se.LEVERAGE:.0f}x / 持倉 {se.MAX_POS} 檔")
    print("成本: 期交稅 0.002%×2 + 手續費 NT$20/口×2 + 滑價 0.1%")
    print("出場: 1R移成本→2R移1R→Chandelier(高−3ATR) + 時間停損7日<0.5R + 跌破結構均線\n")

    hdr = (f"{'策略':<22}{'交易':>5}{'勝率':>6}{'PF':>6}{'年化':>7}"
           f"{'MaxDD':>7}{'Sharpe':>7}{'期望R':>7}{'MAR':>6}  驗收")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for k in keys:
        sigs = collect_stf_signals(data, rs, k, all_dates, date_idx)
        entry_map = bt.build_entry_map(sigs, data)
        trades, curve = se.run_stf(data, entry_map, k, all_dates, date_idx,
                                   risk_on=risk_on)
        m = se.metrics(trades, curve)
        ok = (m["mdd"] < THRESH["mdd"] and m["pf"] > THRESH["pf"]
              and m["sharpe"] > THRESH["sharpe"])
        verdict = "✅通過" if ok else "✗未達"
        label = f"{k} {ss.STF_NAMES[k]}"
        print(f"{label:<22}{m['trades']:>5}{m['win']:>6.0%}{m['pf']:>6.2f}"
              f"{m['cagr']:>7.1%}{m['mdd']:>7.1%}{m['sharpe']:>7.2f}"
              f"{m['expectancy']:>7.2f}{m['mar']:>6.2f}  {verdict}")
        rows.append((k, m))
    print("-" * len(hdr))
    print("驗收門檻 (規格§0.6): MaxDD<25% 且 PF>1.5 且 Sharpe>1.0 (年化須>大盤另計)")
    return rows


if __name__ == "__main__":
    main()
