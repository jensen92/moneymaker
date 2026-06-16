"""驗證使用者的黃金 GC 1W+1D 波段策略 + 與替代策略對照.

忠實重現使用者規則, 但用 25 年資料 (screen_data/GC=F.csv) + 加成本 + 完整指標
(年化/Sharpe/%最大回撤/樣本內外), 並與「買進持有」「唐奇安趨勢」「集成趨勢」對照。

使用者策略規則:
  週線濾網: 週收 > EMA20、EMA20 較 4 週前上升、MACD 柱 > 0
  日線觸發: 日收 > 唐奇安55日高 且 > SMA200
  進場: 次日開盤; 停損: 進場 - 2.5×ATR20
  出場: 觸 2.5ATR 停損 / 收盤跌破唐奇安20日低 / 週收跌破週EMA20
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "screen_data", "GC=F.csv")
TD = 252
COST = 0.0005          # 單邊成本 (含滑價), 佔名目 0.05%
IS_END = pd.Timestamp("2015-12-31")
OOS_START = pd.Timestamp("2016-01-01")


def load():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").set_index("date")
    return df[["open", "high", "low", "close"]].astype(float)


def weekly_filter(df):
    w = df.resample("W").agg({"open": "first", "high": "max",
                             "low": "min", "close": "last"}).dropna()
    ema = w["close"].ewm(span=20, adjust=False).mean()
    macd = (w["close"].ewm(span=12, adjust=False).mean()
            - w["close"].ewm(span=26, adjust=False).mean())
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    ok = (w["close"] > ema) & (ema > ema.shift(4)) & (hist > 0)
    ok_prev = ok.shift(1).astype(bool)                  # 用前一週(已完成)避免未來函數
    return ok_prev.reindex(df.index, method="ffill").fillna(False).astype(bool)


def user_strategy(df):
    """回傳每日持倉 (0/1) 序列, 忠實重現使用者規則。"""
    wk_ok = weekly_filter(df)
    sma200 = df["close"].rolling(200).mean()
    dh55 = df["high"].rolling(55).max()
    dl20 = df["low"].rolling(20).min()
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    atr = tr.rolling(20).mean()
    c = df["close"]
    pos = pd.Series(0.0, index=df.index)
    in_pos = False
    stop = 0.0
    idx = df.index
    for i in range(2, len(df)):
        if not in_pos:
            entry_ok = (wk_ok.iloc[i - 1]
                        and c.iloc[i - 1] > dh55.iloc[i - 2]
                        and c.iloc[i - 1] > sma200.iloc[i - 1]
                        and not np.isnan(atr.iloc[i - 1]))
            if entry_ok:
                in_pos = True
                stop = df["open"].iloc[i] - 2.5 * atr.iloc[i - 1]
        else:
            exit_now = (df["low"].iloc[i] <= stop
                        or c.iloc[i - 1] < dl20.iloc[i - 2]
                        or not wk_ok.iloc[i - 1])
            if exit_now:
                in_pos = False
        pos.iloc[i] = 1.0 if in_pos else 0.0
    return pos


def metrics(pos, df, label):
    ret = df["close"].pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret - turn * COST
    return _stat(strat, label)


def _stat(strat, label, sub=None):
    s = strat if sub is None else strat[sub]
    eq = (1 + s).cumprod()
    yrs = len(s) / TD
    sharpe = s.mean() / s.std() * np.sqrt(TD) if s.std() > 0 else 0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    expo = (s != 0).mean()
    return {"label": label, "sharpe": sharpe, "cagr": cagr, "dd": dd,
            "mar": cagr / dd if dd > 0 else 0, "expo": expo}


def alt_donchian(df):
    c, h, l = df["close"], df["high"], df["low"]
    hi = h.rolling(55).max().shift(1)
    lo = l.rolling(20).min().shift(1)
    pos = pd.Series(np.where(c > hi, 1.0, np.where(c < lo, 0.0, np.nan)),
                    index=df.index).ffill().fillna(0.0)
    return pos


def alt_ensemble(df):
    c, h, l = df["close"], df["high"], df["low"]
    donch = pd.Series(np.where(c > h.rolling(55).max().shift(1), 1.0,
                      np.where(c < l.rolling(55).min().shift(1), -1.0, np.nan)),
                      index=df.index).ffill().fillna(0.0)
    ma = np.sign(c.rolling(50).mean() - c.rolling(100).mean()).fillna(0.0)
    mom = np.sign(c - c.shift(252)).fillna(0.0)
    return ((donch + ma + mom) / 3).clip(lower=0.0)


def main():
    df = load()
    print(f"黃金 GC 策略驗證  資料 {df.index[0].date()} ~ {df.index[-1].date()} "
          f"({len(df)} 日, ~{len(df)//TD} 年)\n")

    strategies = {
        "使用者 1W+1D 波段": user_strategy(df),
        "唐奇安55/20 趨勢": alt_donchian(df),
        "集成趨勢(突破+均線+動能)": alt_ensemble(df),
        "買進持有 黃金": pd.Series(1.0, index=df.index),
    }
    ret = df["close"].pct_change().fillna(0.0)

    hdr = f"{'策略':<24}{'全期Sh':>7}{'IS Sh':>7}{'OOS Sh':>7}{'年化':>8}{'回撤':>8}{'MAR':>6}{'在場%':>7}"
    print(hdr); print("-" * len(hdr))
    for name, pos in strategies.items():
        turn = pos.diff().abs().fillna(0.0)
        strat = pos.shift(1).fillna(0.0) * ret - turn * COST
        full = _stat(strat, name)
        isw = _stat(strat, name, sub=strat.index <= IS_END)
        oos = _stat(strat, name, sub=strat.index >= OOS_START)
        print(f"{name:<24}{full['sharpe']:>7.2f}{isw['sharpe']:>7.2f}"
              f"{oos['sharpe']:>7.2f}{full['cagr']:>8.1%}{full['dd']:>8.1%}"
              f"{full['mar']:>6.2f}{full['expo']:>7.0%}")
    print("\n註: 以日報酬基準比較 (含 0.05% 單邊成本); 在場% = 平均持倉時間比例。")


if __name__ == "__main__":
    main()
