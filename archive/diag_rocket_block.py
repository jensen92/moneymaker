import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt
import strategies as st
import numpy as np
import pandas as pd

bt.DATA_DIR = os.path.join(os.path.dirname(__file__), "data_adj")
data, names = bt.load_all()
rs = bt.compute_rs_rank(data)

codes_dates = [
    ("1752", "2024-10-16"), ("3114", "2025-01-17"), ("7780", "2025-10-03"),
    ("5314", "2023-12-15"), ("7717", "2025-11-17"), ("6861", "2026-02-26"),
    ("6144", "2024-05-10"), ("3081", "2026-01-22"), ("6715", "2026-02-11"),
    ("6683", "2026-02-05"), ("6217", "2026-03-02"), ("5386", "2026-02-10"),
]

for code, sd in codes_dates:
    if code not in data:
        print(code, "no data"); continue
    df = data[code]
    sd = pd.Timestamp(sd)
    # find index closest to/just before sd within +-30 days, check features over a 40-day window around start
    mask = (df["date"] >= sd - pd.Timedelta(days=40)) & (df["date"] <= sd + pd.Timedelta(days=10))
    sub = df[mask]
    if sub.empty:
        print(code, "no rows near start"); continue
    print(f"\n{code} {names.get(code,'')}  飆漲起點 ~{sd.date()}")
    rmap = rs.get(code, {})
    for idx in sub.index:
        i = df.index.get_loc(idx)
        if i < 210:
            continue
        d = df["date"].iloc[i]
        feat = st._d_features(df, i, rmap.get(d, np.nan))
        turnover = df["volume"].iloc[i] * df["close"].iloc[i]
        if feat is None:
            print(f"  {d.date()} close={df['close'].iloc[i]:.1f} turnover={turnover/1e6:.0f}M  模板未過(None)")
        else:
            print(f"  {d.date()} close={df['close'].iloc[i]:.1f} rank={feat['rank']} "
                  f"contraction={feat['contraction']:.2f} vol_dryup={feat['vol_dryup']} "
                  f"breakout={feat['breakout']} vol_surge={feat['vol_surge']:.2f} gain={feat['gain']:.2f}")
