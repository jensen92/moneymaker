"""策略 M 開發: 與 D/B/K/L (突破動能) 結構相反的「強勢股回檔買進」進場,
但接上 D 已驗證的事件風控引擎 (minervini: 移動 MA50 停損 / +20%賣半 / 讓利潤奔跑 /
市況分級 / 波動標量 / 回撤暫停)。E 策略已有回檔進場, 但用短線 2R 出場 (MAR 僅 0.09);
本實驗把同類「回檔領導股」進場改掛 D 的長線風控引擎, 看能否達 1.3×D。
門檻: MAR >= D_MAR * 1.3 才算數。進場用其原始(回檔策略)參數族, 非 D 參數微調。
"""
import os, sys
import numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, "/home/user/moneymaker")
import backtest as bt
from strategies import _d_features, _d_signal, D_CONFIG

bt.DATA_DIR = "/home/user/moneymaker/data_adj"


def m_features(df, i, rs_rank):
    """強勢股回檔結構特徵 (與 D 突破完全不同的進場家族)."""
    row = df.iloc[i]
    if np.isnan(row["ma200"]) or np.isnan(row["rsi14"]) or row["close"] < 10:
        return None
    if row["volume"] * row["close"] < 50_000_000 or i < 200:
        return None
    uptrend = (row["ma50"] > row["ma150"] > row["ma200"]
               and row["ma200"] > df["ma200"].iloc[i - 21]
               and row["close"] > row["ma200"])
    if not uptrend:
        return None
    hi40 = df["high"].iloc[max(0, i - 39):i + 1].max()
    pullback = (hi40 - row["close"]) / hi40 if hi40 > 0 else 0.0
    lo5 = df["low"].iloc[max(0, i - 4):i + 1].min()
    near_ma20 = abs(lo5 - row["ma20"]) / row["ma20"] if row["ma20"] > 0 else 9.9
    near_ma50 = abs(lo5 - row["ma50"]) / row["ma50"] if row["ma50"] > 0 else 9.9
    rsi_min5 = df["rsi14"].iloc[max(0, i - 4):i + 1].min()
    prev = df.iloc[i - 1]
    reversal = row["close"] > row["open"] and row["close"] > prev["high"]
    vol3 = df["volume"].iloc[max(0, i - 2):i + 1].mean()
    dryup = vol3 < row["vol_ma50"] if row["vol_ma50"] > 0 else False
    ma20_up = i >= 5 and row["ma20"] > df["ma20"].iloc[i - 5]
    prox = row["close"] / row["high252"] if row["high252"] > 0 else 0.0
    return dict(rank=rs_rank, pullback=pullback, near_ma20=near_ma20, near_ma50=near_ma50,
                rsi_min5=rsi_min5, reversal=reversal, dryup=dryup, ma20_up=ma20_up, prox=prox)


def m_signal(feat, cfg):
    if feat is None or feat["rank"] is None or feat["rank"] < cfg["rs_min"]:
        return None
    if not (cfg["pullback_min"] <= feat["pullback"] <= cfg["pullback_max"]):
        return None
    near = feat["near_ma20"] if cfg["support_ma"] == "ma20" else min(feat["near_ma20"], feat["near_ma50"])
    if near > cfg["near_ma_pct"]:
        return None
    if feat["rsi_min5"] >= cfg["rsi_os"]:
        return None
    if cfg["require_reversal"] and not feat["reversal"]:
        return None
    if cfg["require_dryup"] and not feat["dryup"]:
        return None
    if cfg["require_ma20_up"] and not feat["ma20_up"]:
        return None
    if cfg.get("prox_min") and feat["prox"] < cfg["prox_min"]:
        return None
    # 掛 D 的長線事件風控引擎 (與 E 的短線 2R 出場不同)
    return dict(score=feat["rank"], minervini=True, stop_pct=cfg["stop_pct"],
                use_three_week=cfg.get("use_three_week", False),
                three_week_gain=cfg.get("three_week_gain", 0.25), max_hold=9999)


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    print("預算 m_features ...")
    feats = {}
    for code, df in data.items():
        rows = []
        for i in range(210, len(df) - 1):
            d = df["date"].iloc[i]
            try: rank = rs.at[d, code]
            except KeyError: continue
            if pd.isna(rank): continue
            f = m_features(df, i, rank)
            if f is not None: rows.append((i, d, f))
        if rows: feats[code] = rows

    # D 基準 (同引擎跑一次拿 D_MAR)
    dfeats = {}
    for code, df in data.items():
        rows = []
        for i in range(210, len(df) - 1):
            d = df["date"].iloc[i]
            try: rank = rs.at[d, code]
            except KeyError: continue
            if pd.isna(rank): continue
            f = _d_features(df, i, rank)
            if f is not None: rows.append((i, d, f))
        if rows: dfeats[code] = rows

    def metrics(trades, curve):
        eq = curve.set_index("date")["equity"].sort_index()
        if len(eq) < 2: return None
        years = (eq.index[-1]-eq.index[0]).days/365.25
        cagr = (eq.iloc[-1]/eq.iloc[0])**(1/max(years,.01))-1 if eq.iloc[-1]>0 else -1
        dd = ((eq.cummax()-eq)/eq.cummax()).max()
        td = pd.DataFrame(trades); n=len(td)
        if n:
            w=td["pnl"]>0; gl=-td.loc[~w,"pnl"].sum()
            pf=td.loc[w,"pnl"].sum()/gl if gl>0 else np.inf; wr=w.mean()
        else: pf=wr=0
        return dict(n=n,wr=wr,pf=pf,cagr=cagr,dd=dd,mar=cagr/dd if dd>0 else np.nan)

    def run(fset, sigfn, cfg, key, label):
        sigs = defaultdict(list)
        for code, rows in fset.items():
            for i, d, f in rows:
                s = sigfn(f, cfg)
                if s: sigs[d].append((s["score"], code, i, s))
        sigs = {d: lst for d, lst in sigs.items() if d in regime}
        em = bt.build_entry_map(sigs, data)
        tr, cv = bt.run_sub(data, em, key, 1.0, 2_000_000, all_dates=all_dates,
            date_idx=date_idx, regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        m = metrics(tr, cv)
        print(f"  {label:<36}{m['n']:>5}{m['wr']*100:>6.0f}%{m['pf']:>6.2f}"
              f"{m['cagr']*100:>7.1f}%{m['dd']*100:>6.1f}%{m['mar']:>6.2f}", flush=True)
        return m

    print(f"\n  {'設定':<36}{'交易':>5}{'勝率':>6}{'PF':>6}{'CAGR':>7}{'MaxDD':>6}{'MAR':>6}", flush=True)
    print("  "+"-"*72, flush=True)
    dM = run(dfeats, _d_signal, D_CONFIG, "D", "D 基準 (突破家族)")
    THR = dM["mar"] * 1.3
    print(f"  目標門檻 = D_MAR {dM['mar']:.2f} × 1.3 = {THR:.2f}\n", flush=True)

    base = dict(rs_min=0.88, pullback_min=0.05, pullback_max=0.15, near_ma_pct=0.03,
                rsi_os=50.0, support_ma="ma20", require_reversal=True,
                require_dryup=True, require_ma20_up=True)
    variants = {
        "M 回檔基準(+長線引擎)":     {},
        "+三週讓奔跑":               {"use_three_week": True},
        "深回檔可(max0.25)":         {"pullback_max": 0.25},
        "支撐放寬near0.05":          {"near_ma_pct": 0.05},
        "RS最強0.92":                {"rs_min": 0.92},
        "MA50支撐":                  {"support_ma": "ma50", "near_ma_pct": 0.05},
        "免反轉K":                   {"require_reversal": False},
        "緊停損0.05":                {"stop_pct": 0.05},
        "近高prox0.85+三週":         {"prox_min": 0.85, "use_three_week": True},
        "RS0.92+三週+prox0.85":      {"rs_min": 0.92, "use_three_week": True, "prox_min": 0.85},
        "深回檔0.25+三週+RS0.92":    {"pullback_max": 0.25, "use_three_week": True, "rs_min": 0.92},
    }
    best = (None, -9)
    for lab, extra in variants.items():
        cfg = {**base, "stop_pct": 0.08, **extra}
        m = run(feats, m_signal, cfg, "E", lab)
        flag = " <==達標!" if m["mar"] >= THR else ""
        if flag: print(f"      {flag}", flush=True)
        if m["mar"] > best[1]: best = (lab, m["mar"])
    print(f"\n本批最佳: {best[0]}  MAR={best[1]:.2f}  (D={dM['mar']:.2f}, 門檻{THR:.2f})", flush=True)


if __name__ == "__main__":
    main()
