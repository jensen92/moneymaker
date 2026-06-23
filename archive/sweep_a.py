"""策略 A 參數掃描."""
import itertools

import numpy as np
import pandas as pd

import strategies
import backtest
from backtest import load_all, run


def make_signal_a(stop_atr, target_r, max_hold, touch, mom_lo, mom_hi):
    def sig(df, i):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if np.isnan(row["ma60"]) or row["close"] < 10:
            return None
        if row["volume"] * row["close"] < 50_000_000:
            return None
        uptrend = (row["close"] > row["ma20"] > row["ma60"]
                   and row["ma20"] > df["ma20"].iloc[i - 5]
                   and row["ma60"] > df["ma60"].iloc[i - 5])
        momentum = mom_lo <= row["ret20"] <= mom_hi
        touched = row["low"] <= row["ma20"] * touch
        reclaimed = row["close"] > row["ma20"] and row["close"] > prev["close"]
        if uptrend and momentum and touched and reclaimed:
            return {"score": row["ret60"], "stop_atr": stop_atr,
                    "target_r": target_r, "max_hold": max_hold}
        return None
    return sig


def main():
    data, names = load_all()
    results = []
    grid = itertools.product(
        [1.5, 2.0, 2.5],        # stop_atr
        [2.0],                  # target_r
        [20, 30],               # max_hold
        [1.0, 1.02],            # touch threshold
        [(0.10, 0.50), (0.15, 0.60)])  # momentum range
    for stop_atr, tr, mh, touch, (lo, hi) in grid:
        strategies.STRATEGIES["A"] = make_signal_a(stop_atr, tr, mh, touch, lo, hi)
        trades, curve = run(data, names, "A", leverage=2.0)
        tdf = pd.DataFrame(trades)
        if not len(tdf):
            continue
        win = (tdf["pnl"] > 0).mean()
        pf = tdf.loc[tdf.pnl > 0, "pnl"].sum() / max(1, -tdf.loc[tdf.pnl < 0, "pnl"].sum())
        eq = curve["equity"]
        ret = eq.iloc[-1] / backtest.INIT_CAPITAL - 1
        dd = ((eq.cummax() - eq) / eq.cummax()).max()
        results.append((stop_atr, mh, touch, lo, hi, len(tdf), win, pf, ret, dd))
        print(f"stop={stop_atr} hold={mh} touch={touch} mom={lo}-{hi}: "
              f"n={len(tdf)} win={win:.1%} pf={pf:.2f} ret={ret:+.1%} dd={dd:.1%}")
    rdf = pd.DataFrame(results, columns=["stop", "hold", "touch", "mlo", "mhi",
                                         "n", "win", "pf", "ret", "dd"])
    rdf.sort_values("pf", ascending=False).to_csv("sweep_a.csv", index=False)


if __name__ == "__main__":
    main()
