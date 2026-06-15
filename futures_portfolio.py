"""5 市場真實合約組合回測 + 三輪「結構性」優化 (非參數調整).

5 標的 (每板塊各 1, 流動性 A/B, ~25 年): 黃金 GC / 那斯達克 NQ / 黃豆 ZS /
美2年公債 ZT / 活牛 LE。資料來自 screen_data/ (Yahoo 連續合約)。

回測採固定本金 (不複利, 避免循環依賴), 真實點值換算每口損益, 含往返成本。
依時序: 訊號/口數以 t 之前資訊決定, 套用於 t→t+1 的價格變動。

結構性優化 (每輪改的是邏輯/架構, 不是調參數):
  v0 基準 : 單一唐奇安突破, 僅多, 等名目金額配置 (naive)
  v1 風險平價: 改為波動率目標定量 — 各市場等風險貢獻、組合波動恆定
  v2 多訊號集成: 唐奇安突破 + 雙均線 + 12月時序動能 三訊號投票 (取代單一訊號)
  v3 組合風控: 依近 20 日「組合波動」把整體曝險縮放到目標波動 (turbulent 時自動降槓桿)
              (實測: 200 日趨勢濾網反而砍掉有效多單, 已捨棄, 僅留波動風控)

用法: python3 futures_portfolio.py
"""
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "screen_data")
CAPITAL = 1_000_000.0
PORT_VOL = 0.12                      # 目標年化組合波動
IS_END = pd.Timestamp("2015-12-31")
OOS_START = pd.Timestamp("2016-01-01")
TD = 252

SPECS = {
    "GC=F": {"name": "黃金 Gold",   "pv": 100.0,  "rt": 25.0},
    "NQ=F": {"name": "那斯達克",     "pv": 20.0,   "rt": 15.0},
    "ZS=F": {"name": "黃豆 Soybean", "pv": 50.0,   "rt": 30.0},
    "ZT=F": {"name": "美2年公債",    "pv": 2000.0, "rt": 36.0},
    "LE=F": {"name": "活牛 Cattle",  "pv": 400.0,  "rt": 25.0},
}


def load(sym):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{sym}.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    c, h, l = df["close"], df["high"], df["low"]
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    df["atr"] = tr.rolling(20).mean()
    df["hi55"] = h.rolling(55).max().shift(1)
    df["lo55"] = l.rolling(55).min().shift(1)
    df["ma50"] = c.rolling(50).mean()
    df["ma100"] = c.rolling(100).mean()
    df["ma200"] = c.rolling(200).mean()
    df["mom"] = c - c.shift(252)
    return df.set_index("date")


def signal(df, cfg):
    """回傳期望方向序列 (連續 [-1,1]); 依 cfg 決定用哪些訊號。"""
    c = df["close"]
    parts = []
    if "donch" in cfg["signals"]:
        d = pd.Series(np.where(c > df["hi55"], 1.0,
                               np.where(c < df["lo55"], -1.0, np.nan)),
                      index=df.index).ffill().fillna(0.0)
        parts.append(d)
    if "ma" in cfg["signals"]:
        parts.append(np.sign(df["ma50"] - df["ma100"]).fillna(0.0))
    if "mom" in cfg["signals"]:
        parts.append(np.sign(df["mom"]).fillna(0.0))
    if "season" in cfg["signals"]:              # 穀物季節性: 12-4 月偏多
        mth = c.index.month
        parts.append(pd.Series(np.where(np.isin(mth, [12, 1, 2, 3, 4]), 1.0, 0.0),
                               index=c.index))
    sig = sum(parts) / len(parts)
    if not cfg["allow_short"]:
        sig = sig.clip(lower=0.0)
    if cfg.get("regime"):                       # 200 日趨勢濾網
        sig = sig.where(c > df["ma200"], 0.0) if not cfg["allow_short"] else \
            sig * (np.sign(c - df["ma200"]) == np.sign(sig))
    return sig.fillna(0.0)


def contracts(df, cfg, n_mkt):
    """回傳每日目標口數 (未含方向)。"""
    pv = cfg["pv"]
    if cfg["sizing"] == "equaldollar":          # naive: 各市場等名目金額
        notional = CAPITAL * (1.0 / n_mkt)
        return (notional / (df["close"] * pv)).clip(upper=50)
    # voltarget: 各市場目標日 $ 波動相等 → 風險平價
    pm_daily_vol = CAPITAL * PORT_VOL / np.sqrt(TD) / np.sqrt(n_mkt)
    atr_usd = (df["atr"] * pv).replace(0, np.nan)
    return (pm_daily_vol / atr_usd).clip(upper=200).fillna(0.0)


def backtest(cfg, start=None, end=None, specs=None):
    specs = specs or SPECS
    syms = list(specs)
    pnl_cols = {}
    pos_cols = {}
    for sym in syms:
        df = load(sym)
        sp = specs[sym]
        cfg = {**cfg, "pv": sp["pv"]}
        sig = signal(df, cfg)
        ct = contracts(df, cfg, len(syms))
        pos = (sig * ct).round()                # 整數口數, 含方向
        dpx = df["close"].diff()
        pnl = pos.shift(1) * dpx * sp["pv"]
        turn = pos.diff().abs().fillna(0.0)
        pnl = pnl - turn.shift(1).fillna(0.0) * sp["rt"]
        pnl_cols[sym] = pnl
        pos_cols[sym] = pos
    P = pd.DataFrame(pnl_cols).fillna(0.0)

    # v3 組合風控層: 依近 20 日「組合報酬波動」把整體曝險縮放到目標波動
    if cfg.get("vol_control"):
        gross = P.sum(axis=1) / CAPITAL
        rv = gross.rolling(20).std() * np.sqrt(TD)
        scale = (PORT_VOL / rv).clip(upper=1.5).shift(1).fillna(1.0)
        P = P.mul(scale, axis=0)

    port = P.sum(axis=1)
    if start:
        port = port[port.index >= pd.Timestamp(start)]
    if end:
        port = port[port.index <= pd.Timestamp(end)]
    return port


def metrics(port):
    eq = CAPITAL + port.cumsum()
    r = port / CAPITAL
    yrs = len(port) / TD
    sharpe = r.mean() / r.std() * np.sqrt(TD) if r.std() > 0 else 0
    tot = eq.iloc[-1] / CAPITAL - 1
    cagr = (eq.iloc[-1] / CAPITAL) ** (1 / max(yrs, .01)) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return {"sharpe": sharpe, "tot": tot, "cagr": cagr, "dd": dd,
            "mar": cagr / dd if dd > 0 else 0}


VERSIONS = {
    "v0 基準(單訊號/等金額)": {"signals": ["donch"], "allow_short": False,
                       "sizing": "equaldollar"},
    "v1 +風險平價/波動目標": {"signals": ["donch"], "allow_short": False,
                       "sizing": "voltarget"},
    "v2 +多訊號集成":      {"signals": ["donch", "ma", "mom"], "allow_short": False,
                       "sizing": "voltarget"},
    "v3 +組合波動風控": {"signals": ["donch", "ma", "mom"], "allow_short": False,
                    "sizing": "voltarget", "vol_control": True},
}


def main():
    print(f"5 市場組合回測 (本金 {CAPITAL:,.0f}, 目標波動 {PORT_VOL:.0%})")
    print(f"標的: {', '.join(s['name'] for s in SPECS.values())}\n")
    hdr = f"{'版本':<22}{'全期Sh':>7}{'IS Sh':>7}{'OOS Sh':>7}{'CAGR':>8}{'MaxDD':>8}{'MAR':>6}"
    print(hdr); print("-" * len(hdr))
    for name, cfg in VERSIONS.items():
        mf = metrics(backtest(cfg))
        mi = metrics(backtest(cfg, end=IS_END))
        mo = metrics(backtest(cfg, start=OOS_START))
        print(f"{name:<22}{mf['sharpe']:>7.2f}{mi['sharpe']:>7.2f}"
              f"{mo['sharpe']:>7.2f}{mf['cagr']:>8.1%}{mf['dd']:>8.1%}{mf['mar']:>6.2f}")


if __name__ == "__main__":
    main()
