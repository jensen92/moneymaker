"""K/D 選股策略在『修正後引擎』上的重新優化掃描.

動機: K/D 參數當年是在『高價股被壓成 1 股』的部位 bug 下挑選的 (損益失真),
修正引擎後必須重新驗證/選擇。非過擬合 — 是用正確的度量重做選擇。

方法: _d_features 與參數無關 → 全市場預算一次; 每組 config 僅套 _d_signal
門檻 + 真實引擎 run_sub。紀律: 樣本內 IS(出場<2020) 選參數, 樣本外 OOS(>=2020)
驗證; 只採「IS 與 OOS 皆不劣於現行」且落在鄰域平台上的組合。

用法: MM_DATA_DIR=data_adj python3 optimize_kd.py   (結果寫 optimize_kd_results.csv)
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

import backtest as bt
from backtest import (load_all, load_regime, load_regime_tiers, load_vol_scalars,
                      run_sub, build_date_index, INIT_CAPITAL)
from strategies import _d_features, _d_signal, K_CONFIG, D_CONFIG

CUT = pd.Timestamp(2020, 1, 1)


def precompute():
    t0 = time.time()
    data, names = load_all()
    print(f"load_all: {len(data)} 檔 ({time.time()-t0:.0f}s)", flush=True)
    rs = bt.compute_rs_rank(data)
    risk_on = load_regime()
    cands = []   # (entry_date=i+1日期, code, i+1, rank, feat)
    for code, df in data.items():
        dates = df["date"]
        n = len(df)
        for i in range(210, n - 1):
            d = dates.iloc[i]
            if d not in risk_on:
                continue
            try:
                rank = rs.at[d, code]
            except KeyError:
                continue
            if rank is None or (isinstance(rank, float) and np.isnan(rank)):
                continue
            feat = _d_features(df, i, rank)
            if feat is not None:
                cands.append((dates.iloc[i + 1], code, i + 1, rank, feat))
    print(f"候選特徵: {len(cands)} 筆 ({time.time()-t0:.0f}s)", flush=True)
    ad, di = build_date_index(data)
    return data, cands, ad, di, load_regime_tiers(), load_vol_scalars()


def eval_cfg(cfg, ctx, skey):
    data, cands, ad, di, rt, vs = ctx
    emap = {}
    for d_next, code, ei, rank, feat in cands:
        sig = _d_signal(feat, cfg)
        if sig:
            emap.setdefault(d_next, []).append((rank, code, ei, sig))
    tr, curve = run_sub(data, emap, skey, 1.0, INIT_CAPITAL, ad, di,
                        regime_tiers=rt, vol_scalars=vs)
    if not tr:
        return None
    p = np.array([t["pnl"] for t in tr]); r = np.array([t["r"] for t in tr])
    eq = curve["equity"]
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    w = p[p > 0]; ls = p[p < 0]
    isel = np.array([t["exit_date"] < CUT for t in tr])
    def seg(mask):
        rr = r[mask]; pp = p[mask]
        if len(rr) == 0:
            return (0, 0.0, 0.0, 0.0)
        return (len(rr), float(rr.mean()), float(rr.sum()), float(pp.sum()))
    is_n, is_avg, is_tot, is_pnl = seg(isel)
    oos_n, oos_avg, oos_tot, oos_pnl = seg(~isel)
    return {"n": len(p), "win": float((p > 0).mean()),
            "pf": float(w.sum() / -ls.sum()) if len(ls) else 99.0,
            "pnl": float(p.sum()), "dd": float(dd),
            "is_n": is_n, "is_avgR": is_avg, "is_totR": is_tot, "is_pnl": is_pnl,
            "oos_n": oos_n, "oos_avgR": oos_avg, "oos_totR": oos_tot, "oos_pnl": oos_pnl}


def main():
    ctx = precompute()
    out = open("optimize_kd_results.csv", "w", newline="")
    wr = csv.writer(out)
    wr.writerow(["tag", "rs", "contr", "gain", "vol", "stop", "prox", "tw",
                 "n", "win", "pf", "pnl", "dd",
                 "is_n", "is_avgR", "is_totR", "oos_n", "oos_avgR", "oos_totR"])

    def run(tag, cfg):
        skey = "K" if tag.startswith("K") else "D"
        m = eval_cfg(cfg, ctx, skey)
        if m is None:
            print(f"{tag}: 無交易", flush=True); return
        wr.writerow([tag, cfg["rs_min"], cfg["contraction"], cfg["gain_cap"],
                     cfg["vol_mult"], cfg["stop_pct"], cfg.get("prox_min"),
                     cfg.get("use_three_week"),
                     m["n"], f"{m['win']:.3f}", f"{m['pf']:.2f}", f"{m['pnl']:.0f}",
                     f"{m['dd']:.4f}", m["is_n"], f"{m['is_avgR']:.3f}",
                     f"{m['is_totR']:.1f}", m["oos_n"], f"{m['oos_avgR']:.3f}",
                     f"{m['oos_totR']:.1f}"])
        out.flush()
        print(f"{tag:<26} n{m['n']:>4} win{m['win']:.0%} PF{m['pf']:.2f} "
              f"pnl{m['pnl']:>+12,.0f} DD{m['dd']:.1%} | "
              f"IS n{m['is_n']} avg{m['is_avgR']:+.2f} tot{m['is_totR']:+.0f} | "
              f"OOS n{m['oos_n']} avg{m['oos_avgR']:+.2f} tot{m['oos_totR']:+.0f}",
              flush=True)

    # 基準
    run("K(現行)", dict(K_CONFIG))
    run("D(現行)", dict(D_CONFIG))
    # 逐軸消融 (以 K 為中心)
    for rs in (0.80, 0.90):
        run(f"K rs{rs}", {**K_CONFIG, "rs_min": rs})
    for ct in (0.70, 0.80, 1.00):
        if ct != K_CONFIG["contraction"]:
            run(f"K contr{ct}", {**K_CONFIG, "contraction": ct})
    for gc in (0.04, 0.08):
        run(f"K gain{gc}", {**K_CONFIG, "gain_cap": gc})
    for vm in (1.5, 2.5):
        run(f"K vol{vm}", {**K_CONFIG, "vol_mult": vm})
    for sp in (0.08, 0.12):
        run(f"K stop{sp}", {**K_CONFIG, "stop_pct": sp})
    run("K prox0.85", {**K_CONFIG, "prox_min": 0.85})
    run("K prox無", {**{k: v for k, v in K_CONFIG.items() if k != "prox_min"}})
    run("K 賣半(tw關)", {**K_CONFIG, "use_three_week": False})
    # D 軸消融 (D 交易多, 是總R主力)
    for ct in (0.80, 1.20):
        run(f"D contr{ct}", {**D_CONFIG, "contraction": ct})
    for gc in (0.06, 0.10):
        run(f"D gain{gc}", {**D_CONFIG, "gain_cap": gc})
    for vm in (2.0,):
        run(f"D vol{vm}", {**D_CONFIG, "vol_mult": vm})
    for sp in (0.10, 0.12):
        run(f"D stop{sp}", {**D_CONFIG, "stop_pct": sp})
    run("D tw開", {**D_CONFIG, "use_three_week": True})
    run("D prox0.80", {**D_CONFIG, "prox_min": 0.80})
    out.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
