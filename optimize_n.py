"""第2試: 留在動能方向, 但用「不同的突破機制」(各自原生參數), 接 D 的事件風控引擎,
看能否達 1.3×D (MAR>=1.29)。三個機制與 D 的 VCP-ATR收縮突破截然不同:
  N1 新52週高突破 (Darvas/Weinstein 第二階段): 今收 > 昨日 high252 → 創歷史新高才買。
  N2 口袋支點 Pocket Pivot (O'Neil): 上漲日且量 > 近10日最大「下跌日量」, 站上 MA50。
  N3 NR7 壓縮突破: 今日為近7日最窄真實區間後的隔日, 帶量突破近7日高。
全部掛 minervini 長線引擎 (移動MA50停損/讓利潤奔跑/市況/波動/回撤暫停)。
"""
import os, sys
import numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, "/home/user/moneymaker")
import backtest as bt
from strategies import _d_features, _d_signal, D_CONFIG
bt.DATA_DIR = "/home/user/moneymaker/data_adj"


def common_ok(df, i):
    row = df.iloc[i]
    if np.isnan(row["ma200"]) or row["close"] < 10 or i < 210:
        return False
    if row["volume"] * row["close"] < 50_000_000:
        return False
    # 多頭結構 (與所有動能策略一致)
    return (row["ma50"] > row["ma150"] > row["ma200"]
            and row["ma200"] > df["ma200"].iloc[i - 21] and row["close"] > row["ma50"])


def feat_n1(df, i, rank):  # 新52週高突破
    if not common_ok(df, i): return None
    row = df.iloc[i]; prev = df.iloc[i - 1]
    prior_h252 = prev["high252"]
    if np.isnan(prior_h252) or row["close"] <= prior_h252:   # 必須創新高
        return None
    vol_surge = row["volume"] / row["vol_ma50"] if row["vol_ma50"] > 0 else 0
    gain = row["close"] / prev["close"] - 1
    return dict(rank=rank, vol_surge=vol_surge, gain=gain)


def sig_n1(f, cfg):
    if f is None or f["rank"] < cfg["rs_min"]: return None
    if f["vol_surge"] < cfg["vol_mult"]: return None
    if f["gain"] > cfg["gain_cap"]: return None
    return dict(score=f["rank"], minervini=True, stop_pct=cfg["stop_pct"],
                use_three_week=cfg.get("use_three_week", False),
                three_week_gain=0.25, max_hold=9999)


def feat_n2(df, i, rank):  # 口袋支點
    if not common_ok(df, i): return None
    row = df.iloc[i]; prev = df.iloc[i - 1]
    if row["close"] <= prev["close"]: return None          # 須為上漲日
    vols = df["volume"].iloc[i - 10:i].values
    closes = df["close"].iloc[i - 10:i].values
    pcloses = df["close"].iloc[i - 11:i - 1].values
    down_vols = vols[closes < pcloses]
    max_down_vol = down_vols.max() if len(down_vols) else 0
    pivot = row["volume"] > max_down_vol if max_down_vol > 0 else False
    # 緊貼 10日均線 (O'Neil: 支點出現在均線附近)
    near = abs(row["low"] - row["ma20"]) / row["ma20"] if row["ma20"] > 0 else 9.9
    return dict(rank=rank, pivot=pivot, near=near)


def sig_n2(f, cfg):
    if f is None or f["rank"] < cfg["rs_min"]: return None
    if not f["pivot"]: return None
    if f["near"] > cfg["near_max"]: return None
    return dict(score=f["rank"], minervini=True, stop_pct=cfg["stop_pct"],
                use_three_week=cfg.get("use_three_week", False),
                three_week_gain=0.25, max_hold=9999)


def feat_n3(df, i, rank):  # NR7 壓縮突破
    if not common_ok(df, i): return None
    row = df.iloc[i]; prev = df.iloc[i - 1]
    tr = (df["high"] - df["low"]).iloc[i - 7:i].values   # 前7日區間
    is_nr7 = (df["high"].iloc[i - 1] - df["low"].iloc[i - 1]) <= tr.min() + 1e-9
    high7 = df["high"].iloc[i - 7:i].max()
    breakout = row["close"] > high7
    vol_surge = row["volume"] / row["vol_ma50"] if row["vol_ma50"] > 0 else 0
    return dict(rank=rank, is_nr7=is_nr7, breakout=breakout, vol_surge=vol_surge)


def sig_n3(f, cfg):
    if f is None or f["rank"] < cfg["rs_min"]: return None
    if not (f["is_nr7"] and f["breakout"]): return None
    if f["vol_surge"] < cfg["vol_mult"]: return None
    return dict(score=f["rank"], minervini=True, stop_pct=cfg["stop_pct"],
                use_three_week=cfg.get("use_three_week", False),
                three_week_gain=0.25, max_hold=9999)


def main():
    print("載入資料 ...")
    data, _ = bt.load_all()
    rs = bt.compute_rs_rank(data)
    regime = bt.load_regime()
    regime_tiers = bt.load_regime_tiers()
    vol_scalars = bt.load_vol_scalars()
    all_dates, date_idx = bt.build_date_index(data)

    def precompute(ff):
        out = {}
        for code, df in data.items():
            rows = []
            for i in range(210, len(df) - 1):
                d = df["date"].iloc[i]
                try: rank = rs.at[d, code]
                except KeyError: continue
                if pd.isna(rank): continue
                f = ff(df, i, rank)
                if f is not None: rows.append((i, d, f))
            if rows: out[code] = rows
        return out

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

    def run(fset, sigfn, cfg, label):
        sigs = defaultdict(list)
        for code, rows in fset.items():
            for i, d, f in rows:
                s = sigfn(f, cfg)
                if s: sigs[d].append((s["score"], code, i, s))
        sigs = {d: lst for d, lst in sigs.items() if d in regime}
        em = bt.build_entry_map(sigs, data)
        tr, cv = bt.run_sub(data, em, "D", 1.0, 2_000_000, all_dates=all_dates,
            date_idx=date_idx, regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        m = metrics(tr, cv)
        flag = " <==達標!" if m["mar"] >= 1.29 else (" <超0.99" if m["mar"]>0.99 else "")
        print(f"  {label:<32}{m['n']:>5}{m['wr']*100:>6.0f}%{m['pf']:>6.2f}"
              f"{m['cagr']*100:>7.1f}%{m['dd']*100:>6.1f}%{m['mar']:>6.2f}{flag}", flush=True)
        return m

    print(f"\n  {'設定':<32}{'交易':>5}{'勝率':>6}{'PF':>6}{'CAGR':>7}{'MaxDD':>6}{'MAR':>6}", flush=True)
    print("  "+"-"*72, flush=True)

    print("  預算 N1 新52週高 ...", flush=True)
    f1 = precompute(feat_n1)
    for vm in [1.5, 2.0]:
        for gc in [0.06, 0.10]:
            for tw in [False, True]:
                lab = f"N1新高 vm{vm} g{gc}{' +三週' if tw else ''}"
                run(f1, sig_n1, dict(rs_min=0.85, vol_mult=vm, gain_cap=gc, stop_pct=0.08, use_three_week=tw), lab)

    print("  預算 N2 口袋支點 ...", flush=True)
    f2 = precompute(feat_n2)
    for nm in [0.05, 0.10]:
        for tw in [False, True]:
            lab = f"N2支點 near{nm}{' +三週' if tw else ''}"
            run(f2, sig_n2, dict(rs_min=0.85, near_max=nm, stop_pct=0.08, use_three_week=tw), lab)

    print("  預算 N3 NR7壓縮 ...", flush=True)
    f3 = precompute(feat_n3)
    for vm in [1.3, 1.8]:
        for tw in [False, True]:
            lab = f"N3 NR7 vm{vm}{' +三週' if tw else ''}"
            run(f3, sig_n3, dict(rs_min=0.85, vol_mult=vm, stop_pct=0.08, use_three_week=tw), lab)

    print("\n門檻 1.29 = D(0.99)×1.3", flush=True)


if __name__ == "__main__":
    main()
