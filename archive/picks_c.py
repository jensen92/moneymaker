"""策略 C (Minervini) 本月訊號掃描 — 輸出進場日期與參考價.

用法: python3 picks_c.py [--data-dir data_adj] [--since 2026-06-01]
"""
import argparse
import os
import numpy as np
import pandas as pd

from strategies import add_indicators, signal_c
from backtest import load_regime, INIT_CAPITAL, RISK_PCT, FEE, SLIP

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")


def compute_rs_rank(data):
    panel = pd.DataFrame({code: df.set_index("date")["ret126"]
                          for code, df in data.items()})
    return panel.rank(axis=1, pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--since", default="2026-06-01", help="掃描起始日 YYYY-MM-DD")
    args = ap.parse_args()

    since = pd.Timestamp(args.since)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else \
        os.path.join(os.path.dirname(__file__), args.data_dir)

    print("載入資料...")
    data = {}
    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        code = fn[:-4]
        df = pd.read_csv(os.path.join(data_dir, fn), parse_dates=["date"])
        if len(df) < 215:
            continue
        data[code] = add_indicators(df).reset_index(drop=True)
    print(f"{len(data)} 檔股票")

    print("計算 RS Rank...")
    rs = compute_rs_rank(data)

    risk_on = load_regime()

    print(f"掃描 {args.since} 以來的訊號...\n")
    results = []

    for code, df in data.items():
        for i in range(210, len(df)):
            sig_date = df["date"].iloc[i]
            if sig_date < since:
                continue
            if sig_date not in risk_on:
                continue
            try:
                rank = rs.at[sig_date, code]
            except KeyError:
                continue
            if np.isnan(rank):
                continue
            s = signal_c(df, i, rs_rank=rank)
            if not s:
                continue

            # 進場參考: 次日開盤
            if i + 1 >= len(df):
                entry_date = "次一交易日"
                entry_ref = df["close"].iloc[i]   # 今收盤估
                is_today = True
            else:
                entry_date = df["date"].iloc[i + 1].strftime("%Y-%m-%d")
                entry_ref = df["open"].iloc[i + 1]
                is_today = False

            row = df.iloc[i]
            atr = row["atr14"]
            stop = entry_ref * (1 - s["stop_pct"])
            risk_per_sh = entry_ref - stop
            risk_amt = INIT_CAPITAL / 3 * RISK_PCT   # C 獨立 1/3 資金池
            shares = int(risk_amt / risk_per_sh / 1000) * 1000
            if shares <= 0:
                shares = max(1, int(risk_amt / risk_per_sh))
            # 單股上限 25%
            cap = int(INIT_CAPITAL / 3 * 0.25 / entry_ref / 1000) * 1000
            shares = min(shares, max(cap, 1))

            results.append({
                "訊號日": sig_date.strftime("%Y-%m-%d"),
                "進場日": entry_date,
                "代碼": code,
                "參考進場價": round(entry_ref, 2),
                "停損價": round(stop, 2),
                "停損%": f"-{s['stop_pct']*100:.0f}%",
                "RS排名": f"{rank:.0%}",
                "建議股數": shares,
                "金額估": f"{shares * entry_ref:,.0f}",
                "最長持有": f"{s['max_hold']}日",
                "今日訊號": "★" if is_today else "",
            })

    if not results:
        print("本月無策略 C 訊號")
        return

    df_out = pd.DataFrame(results).sort_values(["訊號日", "RS排名"], ascending=[False, False])
    print(f"共 {len(df_out)} 筆訊號\n")
    print(df_out.to_string(index=False))

    # 今日 / 最新日訊號特別標示
    latest = df_out["訊號日"].max()
    today_sigs = df_out[df_out["訊號日"] == latest]
    if not today_sigs.empty:
        print(f"\n{'='*60}")
        print(f"最新訊號 ({latest}) — 明日進場候選:")
        print(today_sigs.to_string(index=False))

    df_out.to_csv("picks_c_monthly.csv", index=False, encoding="utf-8-sig")
    print("\n結果已存至 picks_c_monthly.csv")


if __name__ == "__main__":
    main()
