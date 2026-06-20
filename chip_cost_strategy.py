"""籌碼成本關卡策略 (大戶/外資成本倍數) — 結構不同於 D/B/K/L 突破家族.

使用者想法: 籌碼K線/CMoney 類軟體會秀「大戶成本」「融資成本」等真實籌碼均價線。
站上外資成本 1.04 倍可買進; 1.2/1.4/1.8 倍是關卡; 超過 1.8 倍後改看融資成本,
融資成本若跌破 1.8 倍要減碼。

**資料限制 (必須誠實說明)**: data_adj/data 只有日線 OHLCV, 沒有三大法人/融資餘額等
真實籌碼資料, 無法算出真正的「外資成本」或「融資成本」均價。本檔用最接近的代理:
N日成交量加權均價 (VWAP) 近似「主力/散戶平均持有成本」— 這是市場上常見的籌碼成本
代理算法之一 (籌碼K線本身的『成本線』也是用買賣超數量反推 VWAP, 原理相通, 但其
分子分母用的是「買賣超」不是「總成交量」, 故仍非真值, 僅供概念驗證)。

驗證: 站上 VWAP_N 的 1.04 倍 = 進場訊號, 接 D 已驗證的事件風控引擎 (移動停損/
+20%賣半/市況分級/波動標量), 比照之前 M/N1/N2/N3 的接法。並測試 1.2/1.4/1.8 關卡
作為「追高保護」(超過上限不追價) 與「失守減碼」(跌破已站上的關卡視為轉弱出場)
兩種變化。
門檻: MAR >= D_MAR(0.99) * 1.3 = 1.29 才算「比D高」。
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/user/moneymaker")
import backtest as bt

bt.DATA_DIR = "/home/user/moneymaker/data_adj"


def vwap_ratio_series(df, vwap_n):
    """向量化算 close / 滾動N日VWAP, 整檔股票一次算好 (取代逐日切片加總, 避免 O(n^2))."""
    pv = (df["close"] * df["volume"]).rolling(vwap_n).sum()
    vv = df["volume"].rolling(vwap_n).sum()
    vwap = pv / vv.replace(0, np.nan)
    return df["close"] / vwap


def cc_features(df, i, rs_rank, ratio_col):
    row = df.iloc[i]
    if np.isnan(row.get("ma200", np.nan)) or row["close"] < 10 or i < 1:
        return None
    if row["volume"] * row["close"] < 50_000_000:
        return None
    uptrend = (row["ma50"] > row["ma150"] > row["ma200"]
               and row["ma200"] > df["ma200"].iloc[i - 21]
               and row["close"] > row["ma50"])
    if not uptrend:
        return None
    ratio = ratio_col.iloc[i]
    prev_ratio = ratio_col.iloc[i - 1]
    if np.isnan(ratio) or np.isnan(prev_ratio):
        return None
    return dict(rank=rs_rank, ratio=ratio, prev_ratio=prev_ratio)


def cc_signal(feat, cfg):
    if feat is None or feat["rank"] is None or feat["rank"] < cfg["rs_min"]:
        return None
    r, pr = feat["ratio"], feat["prev_ratio"]
    if np.isnan(pr):
        return None
    cross_up = pr < cfg["enter_ratio"] <= r
    if not cross_up:
        return None
    if cfg.get("cap_ratio") and r > cfg["cap_ratio"]:
        return None  # 追高保護: 超過上限關卡不追價
    return dict(score=feat["rank"], minervini=True, stop_pct=cfg["stop_pct"],
                use_three_week=cfg.get("use_three_week", False),
                three_week_gain=cfg.get("three_week_gain", 0.25), max_hold=9999)


_ratio_cache = {}


def get_ratio_series(data, vwap_n):
    if vwap_n not in _ratio_cache:
        _ratio_cache[vwap_n] = {code: vwap_ratio_series(df, vwap_n)
                                for code, df in data.items()}
    return _ratio_cache[vwap_n]


def eval_cfg(data, rs, regime, regime_tiers, vol_scalars, label, cfg, vwap_n=60):
    ratios = get_ratio_series(data, vwap_n)
    sigs_by_date = {}
    for code, df in data.items():
        ratio_col = ratios[code]
        for i in range(max(210, vwap_n + 5), len(df) - 1):
            d = df["date"].iloc[i]
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if np.isnan(rank):
                continue
            feat = cc_features(df, i, rank, ratio_col)
            sig = cc_signal(feat, cfg)
            if sig:
                sigs_by_date.setdefault(d, []).append((sig["score"], code, i, sig))
    sigs_by_date = {d: lst for d, lst in sigs_by_date.items() if d in regime}
    entry_map = bt.build_entry_map(sigs_by_date, data)
    trades, _ = bt.run_sub(data, entry_map, "D", 1.0, bt.INIT_CAPITAL,
                           regime_tiers=regime_tiers, vol_scalars=vol_scalars)
    n = len(trades)
    if n == 0:
        print(f"{label:<32} 無交易"); return
    wins = [t for t in trades if t["pnl"] > 0]
    pf_num = sum(t["pnl"] for t in wins)
    pf_den = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    pf = pf_num / pf_den if pf_den > 0 else float("inf")
    eq = bt.INIT_CAPITAL
    curve = []
    for t in sorted(trades, key=lambda t: t["exit_date"]):
        eq += t["pnl"]; curve.append(eq)
    eqs = pd.Series(curve)
    peak = eqs.cummax()
    dd = ((peak - eqs) / peak).max()
    yrs = (max(t["exit_date"] for t in trades) - min(t["entry_date"] for t in trades)).days / 365.25
    cagr = (eq / bt.INIT_CAPITAL) ** (1 / max(yrs, 0.5)) - 1
    mar = cagr / dd if dd > 0 else 0
    print(f"{label:<32} {n:>5}  {len(wins)/n:>5.0%}  {pf:>5.2f}  {cagr:>6.1%}  {dd:>6.1%}  {mar:>5.2f}")
    return mar


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()

    print(f"{'設定':<32} {'交易':>5}  {'勝率':>5}  {'PF':>5}  {'CAGR':>6}  {'MaxDD':>6}  {'MAR':>5}")
    print("-" * 78)
    print("D 基準 (突破家族, 已知): 806筆 50% PF1.46 CAGR8.3% MaxDD8.4% MAR0.99")
    print("目標門檻 = 0.99 * 1.3 = 1.29")
    print("-" * 78)

    base = dict(rs_min=0.80, enter_ratio=1.04, stop_pct=0.08)
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC 基準 站上VWAP60*1.04", base)
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC +追高保護上限1.2", {**base, "cap_ratio": 1.2})
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC +追高保護上限1.4", {**base, "cap_ratio": 1.4})
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC +追高保護上限1.8", {**base, "cap_ratio": 1.8})
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC RS0.90", {**base, "rs_min": 0.90})
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC stop0.10", {**base, "stop_pct": 0.10})
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC VWAP30 (短)", base, vwap_n=30)
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC VWAP120 (長)", base, vwap_n=120)
    eval_cfg(data, rs, regime, regime_tiers, vol_scalars, "CC +三週法則", {**base, "use_three_week": True})


if __name__ == "__main__":
    main()
