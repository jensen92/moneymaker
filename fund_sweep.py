"""基本面門檻 + 財報避險的參數掃描驗證 (一次載入資料/收訊號, 共用給全網格).

回答兩件事, 給 production 上線決策用:
  1. 選股門檻: 在現行 PA/PB/K/L/D 技術訊號後, 疊「EPS 年增 >= e 且 ROE >= r」
     (point-in-time) 對風險報酬的影響。掃 (e, r) 小網格 + 無資料處置 fail/pass。
  2. 部位管理: 在「寬鬆版」門檻 (EPS>0, ROE>15%) 上, 疊「財報發布前 D 日內且帳上
     獲利 < G → 先出場」, 掃 D ∈ {3,7,14} 觀察是否規避財報跳空、改善 MAR。

與 production 同引擎 (run_sub), 共用一次 _d_features / 一次 RS rank / 一份 PIT DB,
故全網格只載入一次資料、只收一次訊號, 公平且快。
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest as bt
bt.DATA_DIR = os.path.join(HERE, "data_adj")
from backtest import (load_all, load_regime, load_regime_tiers, load_vol_scalars,
                      compute_rs_rank, build_date_index)
from fundamentals import FundamentalDB
from fund_backtest import (build_signals, gate_signals, run_set, ptable, ORDER)


def _portfolio(rows):
    """五策略合計的投組層級彙總 (交易數 / 總R / 加權年化近似 / 最大MAR代理)。"""
    n = sum(rows[k]["n"] for k in ORDER)
    totR = sum(rows[k]["totR"] for k in ORDER)
    # 以五子帳戶等權, 取平均年化與平均回撤做投組概況
    cagr = sum(rows[k]["cagr"] for k in ORDER) / len(ORDER)
    dd = sum(rows[k]["dd"] for k in ORDER) / len(ORDER)
    pf_num = sum(rows[k]["pf"] * rows[k]["n"] for k in ORDER)
    pf = pf_num / max(1, n)
    return {"n": n, "totR": totR, "cagr": cagr, "dd": dd,
            "mar": cagr / dd if dd > 0 else 0, "pf": pf}


def _pline(tag, p):
    print(f"{tag:>26} | 交易 {p['n']:>5} | 加權PF {p['pf']:>5.2f} | "
          f"均年化 {p['cagr']:>6.1%} | 均回撤 {p['dd']:>6.1%} | "
          f"均MAR {p['mar']:>5.2f} | 總R {p['totR']:>+7.0f}")


def main():
    print("載入 data_adj...")
    data, names = load_all()
    print(f"{len(data)} 檔, RS rank...")
    rs = compute_rs_rank(data)
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    all_dates, date_idx = build_date_index(data)
    print("收集訊號 (五策略共用 _d_features)...")
    sigs = build_signals(data, rs, risk_on)
    tot_sig = sum(len(lst) for k in sigs for lst in sigs[k].values())
    print(f"  原始技術訊號數 (五策略合計, 去市況外): {tot_sig}")

    print("載入基本面 PIT 資料庫...")
    db = FundamentalDB()
    print(f"  快取標的數: {len(db.q)}")

    def run(label, gated_sigs, earn_days=0, earn_gain=0.10):
        rows, yR = run_set(data, gated_sigs, all_dates, date_idx,
                           regime_tiers, vol_scalars,
                           earnings_db=(db if earn_days > 0 else None),
                           earn_days=earn_days, earn_gain=earn_gain)
        return label, rows, yR

    # ── baseline ──
    print("\n回測 baseline...")
    _, base_rows, base_yR = run("baseline", sigs)

    # ── 選股門檻網格 (missing=fail 保守) ──
    print("回測選股門檻網格...")
    GRID = [(0.0, 0.15), (0.0, 0.20), (0.20, 0.15), (0.20, 0.20)]
    gate_runs = {}
    keep_rates = {}
    for e, r in GRID:
        gated, kr = gate_signals(sigs, db, e, r, "fail")
        keep_rates[(e, r)] = kr
        gate_runs[(e, r)] = run(f"EPS>={e:.0%} ROE>={r:.0%}", gated)

    # 寬鬆版另跑 missing=pass (無基本面資料不擋), 觀察覆蓋率影響
    gated_loose_pass, kr_pass = gate_signals(sigs, db, 0.0, 0.15, "pass")
    pass_run = run("EPS>=0% ROE>=15% (無資料放行)", gated_loose_pass)

    # ── 財報避險 (疊在寬鬆版 fail 門檻上) ──
    print("回測財報避險 (寬鬆版 + 財報前避險)...")
    gated_loose, _ = gate_signals(sigs, db, 0.0, 0.15, "fail")
    earn_runs = {}
    for d_ in (3, 7, 14):
        earn_runs[d_] = run(f"寬鬆門檻+財報避險{d_}日(<10%獲利)",
                            gated_loose, earn_days=d_, earn_gain=0.10)

    # ════════════ 報表 ════════════
    print(f"\n{'='*92}")
    print("基本面門檻 + 財報避險 掃描 (2011-2026, data_adj 全市場, PA/PB/K/L/D 等權子帳戶)")
    print('='*92)

    print("\n— 選股門檻訊號通過率 (PIT, 無資料=剔除) —")
    for e, r in GRID:
        print(f"  EPS>={e:>4.0%} ROE>={r:>4.0%} : 通過 {keep_rates[(e,r)]:>5.1%} 技術訊號")
    print(f"  EPS>= 0% ROE>= 15% (無資料放行): 通過 {kr_pass:>5.1%}")

    print("\n— 投組層級概況 (五策略合計) —")
    _pline("baseline 無門檻", _portfolio(base_rows))
    for e, r in GRID:
        _pline(gate_runs[(e, r)][0], _portfolio(gate_runs[(e, r)][1]))
    _pline(pass_run[0], _portfolio(pass_run[1]))
    print("  " + "-" * 88)
    _pline("寬鬆門檻 (再附避險基準)", _portfolio(gate_runs[(0.0, 0.15)][1]))
    for d_ in (3, 7, 14):
        _pline(earn_runs[d_][0], _portfolio(earn_runs[d_][1]))

    # 逐策略明細: baseline vs 寬鬆門檻 vs 寬鬆門檻+7日避險
    ptable("【baseline 無基本面門檻】", base_rows)
    ptable("【寬鬆門檻 EPS>=0% ROE>=15% (無資料剔除)】", gate_runs[(0.0, 0.15)][1])
    ptable("【寬鬆門檻 + 財報避險 7日/<10%獲利】", earn_runs[7][1])

    # 逐年總R 對照
    years = sorted({y for k in ORDER for y in base_yR[k]})
    g_yR = gate_runs[(0.0, 0.15)][2]
    e_yR = earn_runs[7][2]
    print("\n逐年總R 對照 (五策略合計):")
    print(f"{'年':>6} {'baseline':>9} {'寬鬆門檻':>9} {'門檻+避險7':>11} "
          f"{'門檻-base':>10}")
    for y in years:
        b = sum(base_yR[k].get(y, 0) for k in ORDER)
        g = sum(g_yR[k].get(y, 0) for k in ORDER)
        ev = sum(e_yR[k].get(y, 0) for k in ORDER)
        print(f"{y:>6} {b:>+9.1f} {g:>+9.1f} {ev:>+11.1f} {g-b:>+10.1f}")


if __name__ == "__main__":
    main()
