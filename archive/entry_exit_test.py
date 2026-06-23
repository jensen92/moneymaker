"""對付「高檔急殺洗盤把突破單掃停損」的兩條路線, 各自回測驗證 (重用 backtest 引擎):

路線 1 — 進場確認 (entry confirmation):
  現況: 突破訊號日的「次日開盤」直接追進。
  改法: 突破後不馬上追, 等價格「回測樞軸 (前20日高) 不破再進」。
        對每筆訊號, 往後最多 W 個交易日尋找確認:
          - 若某日 low 回測到樞軸附近 (<= pivot×(1+tol)) 且 close 仍站在樞軸之上
            → 確認成立, 次日開盤進場;
          - 若中途 close 跌破樞軸 (< pivot×(1-break)) → 放棄該訊號 (假突破不追);
          - W 日內未回測確認 → 放棄 (強勢未拉回者錯過, 屬此法的代價)。
  取捨: 砍掉「追高後被洗」的單, 但也會錯過「不回頭直噴」的大贏家 → 淨效益待回測。

路線 2 — 停損/出場調整 (stop adjustment):
  現況: 初始停損 stop_pct (D=7%, C=8%, PA/PB 各自), 之後 3R 保本 + MA50 移動停損。
  改法: 全策略改用更寬初始停損 (掃 7%~15%), 減少高檔洗盤在初始停損就被掃出。
        (3R 保本與 MA50 移動停損維持不變, 由引擎處理。)

兩者各自與 baseline 對照整體 + 逐年 (確認非單一年度的事後諸葛)。

用法: python3 entry_exit_test.py    (預設 data/ 設計 universe; 與 breadth_filter_test 同源)
限制: data/ 490 檔, 絕對金額僅供方向性比較 (同 breadth_filter_test 說明)。

═══════════════════════════════════════════════════════════════════════════
結論 (2023-08~2026-06, data/ 490 檔):

  設定                  交易  勝率   回撤   PF    總R   2024R  2025R  2026R
  baseline             120  31%   1.4%  0.92  +55   -21.9  +41.5  +34.9
  進場確認 W=8日         58  34%   0.5%  1.05  +30    -9.5  +28.7  +11.1
  停損 15%             117  35%   0.9%  0.92  +29   -16.1  +22.5  +22.4

  路線1 (進場確認/回測不破才進): 交易數砍半 (120→50~58, 約六成突破不回測直噴
    或直接失敗), 勝率與回撤明顯改善 (2026 勝率 43%→67%, 回撤 1.4%→0.5%),
    但**總報酬也砍半** (+55→+30) —— 因為錯過「不回頭直噴」的大贏家 (本系統靠
    右尾肥尾獲利)。風險調整後 (PF/MAR) 約略持平、小幅變好。即「更少更乾淨的單、
    曲線更平滑, 但賺更少」, 非免費的獲利提升。⚠ 樣本變小 (2026 僅 10~12 筆),
    勝率數字統計上不穩, 僅供參考。

  路線2 (放寬初始停損 10~15%): 只小幅減輕 2024 盤整年虧損 (-21.9→-16.1R),
    卻讓 2025/2026 趨勢年**報酬下降更多** (2025 +41.5→+22.5, 2026 +34.9→+22.4),
    總R +55→+29、PF 0.92→0.86~0.92 → **淨負面**。寬停損讓每筆 R 縮水 (初始風險
    變大) 又沒擋住趨勢年的回檔, 不划算。回撤變小主因是部位變小 (寬停損=同樣1%
    風險下股數變少), 非真正抗洗盤。

  整體: 兩條路線都「降低報酬」, 沒有免費午餐。近期虧損很大程度是動能突破系統在
  高檔洗盤期的必要成本。若你重視「曲線平滑/少被洗」勝於「最大報酬」, 進場確認
  (W=8) 是相對較有道理的取捨 (回撤砍 2/3、勝率升、風險調整後持平); 放寬停損則
  不建議。baseline 在此精簡 universe 呈現負 CAGR 是 490 檔子集低估了真實贏家
  (對照正式 /year 2026 約 +15 萬), 故重點看「相對 baseline 的變化」而非絕對值。
═══════════════════════════════════════════════════════════════════════════
"""
import copy
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import backtest as bt
bt.DATA_DIR = os.environ.get("MM_DATA_DIR", "").strip() or os.path.join(HERE, "data")
from backtest import (load_all, load_regime, load_regime_tiers, load_vol_scalars,
                      collect_signals, build_entry_map, run_sub, build_date_index,
                      INIT_CAPITAL)

STRATS = ["PA", "PB", "K", "L", "D", "C"]
PIVOT_LOOKBACK = 20   # 樞軸 = 前 20 日高 (與各策略 breakout 定義一致)


def confirmed_entry_map(signals, data, window, tol=0.01, brk=0.02):
    """路線1: 把每筆訊號改為「回測樞軸不破才進」的延後確認進場。"""
    em = defaultdict(list)
    for sd, lst in signals.items():
        for score, code, i, s in lst:
            df = data[code]
            if i < PIVOT_LOOKBACK:
                continue
            pivot = df["high"].iloc[i - PIVOT_LOOKBACK:i].max()
            entry_j = None
            for j in range(i + 1, min(i + 1 + window, len(df) - 1)):
                c = df["close"].iloc[j]
                lo = df["low"].iloc[j]
                if c < pivot * (1 - brk):        # 跌破樞軸 → 假突破, 放棄
                    break
                if lo <= pivot * (1 + tol) and c >= pivot:   # 回測不破 → 確認
                    entry_j = j + 1
                    break
            if entry_j is not None and entry_j < len(df):
                em[df["date"].iloc[entry_j]].append(
                    (score, code, entry_j, s))
    return em


def override_stop(signals, stop_pct):
    """路線2: 複製訊號並改寫 stop_pct (其餘規則不變)。"""
    out = defaultdict(list)
    for sd, lst in signals.items():
        for score, code, i, s in lst:
            s2 = copy.copy(s)
            s2["stop_pct"] = stop_pct
            out[sd].append((score, code, i, s2))
    return out


def year_metrics(trades):
    by = defaultdict(list)
    for t in trades:
        by[t["exit_date"].year].append(t)
    rows = {}
    for y, ts in sorted(by.items()):
        rows[y] = (len(ts), sum(1 for t in ts if t["pnl"] > 0) / len(ts),
                   sum(t["r"] for t in ts), sum(t["pnl"] for t in ts))
    return rows


def portfolio(data, em_by_strat, regime_tiers, vol_scalars, all_dates, date_idx):
    init_eq = INIT_CAPITAL / len(STRATS)
    all_tr = []
    eqs = None
    for k in STRATS:
        trades, curve = run_sub(data, em_by_strat[k], k, 1.0, init_eq,
                                all_dates=all_dates, date_idx=date_idx,
                                regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        all_tr.extend(trades)
        c = curve[["date", "equity"]].rename(columns={"equity": f"eq_{k}"})
        eqs = c if eqs is None else eqs.merge(c, on="date")
    eqs["equity"] = sum(eqs[f"eq_{k}"] for k in STRATS)
    eqs["dd"] = (eqs["equity"].cummax() - eqs["equity"]) / eqs["equity"].cummax()
    eq = eqs["equity"]
    years = len(eqs) / 244
    cagr = (eq.iloc[-1] / INIT_CAPITAL) ** (1 / max(years, .01)) - 1
    maxdd = eqs["dd"].max()
    wins = [t for t in all_tr if t["pnl"] > 0]
    pf = (sum(t["pnl"] for t in wins)
          / max(1, -sum(t["pnl"] for t in all_tr if t["pnl"] < 0)))
    return {"n": len(all_tr), "win": len(wins) / max(1, len(all_tr)),
            "totR": sum(t["r"] for t in all_tr), "cagr": cagr, "maxdd": maxdd,
            "mar": cagr / maxdd if maxdd > 0 else 0, "pf": pf, "trades": all_tr}


def show(label, m):
    print(f"\n=== {label} ===")
    print(f"  交易 {m['n']}  勝率 {m['win']:.0%}  總R {m['totR']:+.0f}  "
          f"年化 {m['cagr']:+.1%}  最大回撤 {m['maxdd']:.1%}  "
          f"MAR {m['mar']:.2f}  PF {m['pf']:.2f}")
    for y, (n, w, r, pnl) in year_metrics(m["trades"]).items():
        print(f"     {y}  {n:>4}筆  勝率{w:>4.0%}  總R{r:>+7.1f}  損益{pnl:>+11,.0f}")


def main():
    print("載入資料 + 各策略原始訊號 (一次)...")
    data, names = load_all()
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    all_dates, date_idx = build_date_index(data)

    raw_sigs = {}
    for k in STRATS:
        s = collect_signals(data, k)
        raw_sigs[k] = {d: lst for d, lst in s.items() if d in risk_on}

    base_em = {k: build_entry_map(raw_sigs[k], data) for k in STRATS}
    results = []

    def run_cfg(label, em):
        m = portfolio(data, em, regime_tiers, vol_scalars, all_dates, date_idx)
        show(label, m)
        results.append((label, m))

    run_cfg("baseline (突破次日追進, 原停損)", base_em)

    print("\n\n######## 路線1: 進場確認 (回測樞軸不破才進) ########")
    for W in (3, 5, 8):
        em = {k: confirmed_entry_map(raw_sigs[k], data, window=W) for k in STRATS}
        run_cfg(f"進場確認 window={W}日", em)

    print("\n\n######## 路線2: 放寬初始停損 ########")
    for sp in (0.10, 0.12, 0.15):
        em = {k: build_entry_map(override_stop(raw_sigs[k], sp), data) for k in STRATS}
        run_cfg(f"停損 {sp:.0%}", em)

    print("\n\n### 總結對照 ###")
    print(f"{'設定':<26} {'交易':>5} {'勝率':>5} {'年化':>7} {'回撤':>6} "
          f"{'MAR':>6} {'PF':>5} {'總R':>6}")
    for label, m in results:
        print(f"{label:<26} {m['n']:>5} {m['win']:>5.0%} {m['cagr']:>7.1%} "
              f"{m['maxdd']:>6.1%} {m['mar']:>6.2f} {m['pf']:>5.2f} {m['totR']:>+6.0f}")


if __name__ == "__main__":
    main()
