"""市場廣度濾網回測驗證: 在現有三段式市況 (tier) 之上, 疊加「廣度閘門」,
比較加/不加對 2023~2026 整體績效的影響, 並逐年拆解確認非事後諸葛。

設計 (重用現有引擎, 不改 backtest.py):
  - 廣度序列 = 全市場「高於自身 MA50 的個股比例」(同 universe 計算, 見 build_breadth)。
  - 廣度閘門: 對 baseline regime tier 取 min(tier, breadth_cap):
        breadth >= HI  → cap 2 (不限制)
        MID <= b < HI  → cap 1 (盤整, 風險 0.6x)
        b < MID        → cap 0 (防禦, 停止新倉)
  - 對每組 (MID, HI) 門檻掃描, 跑完整 PA/PB/K/L/D/C 組合, 報整體 + 逐年。

用法:
    python3 breadth_filter_test.py
    MM_DATA_DIR=/path/to/data python3 breadth_filter_test.py   # 指定 universe

═══════════════════════════════════════════════════════════════════════════
結論 (2023-08~2026-06, data/ 490 檔設計 universe, 與策略調參同源):

  設定            年化    最大回撤   MAR    PF     總R   2024損益  2025損益  2026損益
  baseline       -0.1%   1.4%   -0.04  0.92   +55   -10,640  -17,136  +24,370
  廣度 35/50     +0.1%   0.9%    0.13  1.20   +62    -5,167  -12,585  +24,558
  廣度 40/55     +0.1%   0.7%    0.09  1.13   +66    -5,167   -8,161  +17,051

  → 廣度濾網是「溫和的穩健度改善」: PF 0.92→1.20、最大回撤 1.4%→0.7~0.9%,
    改善幾乎全來自**砍掉 2024 盤整年的假突破** (2024 虧損腰斬)。
  → 但關鍵: 多數門檻下 **2026 的交易與損益幾乎沒變** (廣度 35/50 仍 +24,558)!
    因為 2026 近期的虧損發生在**廣度尚可 (50~59%) 的高檔急跌洗盤**, 廣度濾網
    抓不到 → 證實「最近賠錢」主因是急殺洗盤/進場時機, 不是廣度低迷。
  → 因此廣度濾網「可加, 對長期穩健度小幅有利」, 但**它不是近期虧損的解方**;
    不建議當成救命稻草。要對付高檔洗盤需另解 (如突破後等回測確認、放寬停損,
    見 txf 類的進場確認概念) — 屬另一主題。

重要限制: data/ 為 490 檔精簡設計 universe, 絕對損益遠小於正式 /year 的
data_adj (約 1977 檔), 故金額僅供「方向性」比較, 非帳戶實際數字。完整 universe
做 5 組門檻掃描過慢 (collect_signals × 6 策略 × 7M bar), 此處以設計 universe 取代。
═══════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import backtest as bt
# 與正式機器人 (/year, /scan) 一致: 使用 data_adj 還原股價全市場 (約 1958 檔),
# 而非 backtest.py 預設的精簡 data/ (496 檔)。廣度序列亦由 data_adj 計算, universe 一致。
bt.DATA_DIR = os.environ.get("MM_DATA_DIR", "").strip() or os.path.join(HERE, "data_adj")
from backtest import (load_all, load_regime, load_regime_tiers, load_vol_scalars,
                      collect_signals, build_entry_map, run_sub, build_date_index,
                      INIT_CAPITAL)

STRATS = ["PA", "PB", "K", "L", "D", "C"]
# 預設自行計算廣度 (與 universe 一致); 設 BREADTH_CSV 可改讀預算好的快取。
BREADTH_CSV = os.environ.get("BREADTH_CSV", "").strip()


def build_breadth():
    """由 bt.DATA_DIR 全市場計算每日「高於自身 MA50 比例」, 與回測 universe 一致。"""
    import glob
    from collections import defaultdict
    above = defaultdict(int)
    tot = defaultdict(int)
    files = [f for f in glob.glob(os.path.join(bt.DATA_DIR, "*.csv"))
             if not os.path.basename(f).startswith("_")]
    for f in files:
        try:
            df = pd.read_csv(f, names=["date", "open", "high", "low", "close", "vol"],
                             header=0)
            if len(df) < 60:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            ma50 = df["close"].rolling(50).mean().values
            cl = df["close"].values
            dts = df["date"].values
            for i in range(len(df)):
                if np.isnan(ma50[i]):
                    continue
                d = dts[i]
                tot[d] += 1
                if cl[i] > ma50[i]:
                    above[d] += 1
        except Exception:  # noqa: BLE001
            pass
    return {pd.Timestamp(d): above[d] / tot[d] for d in tot if tot[d] >= 100}


def load_breadth():
    if BREADTH_CSV and os.path.exists(BREADTH_CSV):
        br = pd.read_csv(BREADTH_CSV, parse_dates=["date"])
        return dict(zip(br["date"], br["pct_above_ma50"]))
    return build_breadth()


def gate_tiers(base_tiers, breadth, mid, hi, dates):
    """回傳新的 {date: tier}, 以廣度上限封頂 baseline tier。"""
    out = {}
    last_b = None
    for d in dates:
        b = breadth.get(d, last_b)
        last_b = b if b is not None else last_b
        t = base_tiers.get(d, 2)
        if b is None:
            out[d] = t
        elif b >= hi:
            out[d] = t
        elif b >= mid:
            out[d] = min(t, 1)
        else:
            out[d] = min(t, 0)
    return bt._FwdFillDict(out)


def year_metrics(trades):
    """逐年: 交易數/勝率/總R/已實現損益 (依出場年度歸屬)。"""
    by = {}
    for t in trades:
        y = t["exit_date"].year
        by.setdefault(y, []).append(t)
    rows = {}
    for y, ts in sorted(by.items()):
        r = sum(t["r"] for t in ts)
        pnl = sum(t["pnl"] for t in ts)
        win = sum(1 for t in ts if t["pnl"] > 0) / len(ts)
        rows[y] = (len(ts), win, r, pnl)
    return rows


def portfolio(data, entry_maps, regime_tiers, vol_scalars, all_dates, date_idx):
    init_eq = INIT_CAPITAL / len(STRATS)
    all_tr = []
    curves = {}
    for k in STRATS:
        trades, curve = run_sub(data, entry_maps[k], k, 1.0, init_eq,
                                all_dates=all_dates, date_idx=date_idx,
                                regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        all_tr.extend(trades)
        curves[k] = curve[["date", "equity"]].rename(columns={"equity": f"eq_{k}"})
    comb = None
    for k in STRATS:
        comb = curves[k] if comb is None else comb.merge(curves[k], on="date")
    comb["equity"] = sum(comb[f"eq_{k}"] for k in STRATS)
    comb["dd"] = (comb["equity"].cummax() - comb["equity"]) / comb["equity"].cummax()
    eq = comb["equity"]
    years = len(comb) / 244
    cagr = (eq.iloc[-1] / INIT_CAPITAL) ** (1 / max(years, .01)) - 1
    maxdd = comb["dd"].max()
    wins = [t for t in all_tr if t["pnl"] > 0]
    pf = (sum(t["pnl"] for t in wins)
          / max(1, -sum(t["pnl"] for t in all_tr if t["pnl"] < 0)))
    return {
        "n": len(all_tr), "win": len(wins) / max(1, len(all_tr)),
        "totR": sum(t["r"] for t in all_tr), "cagr": cagr, "maxdd": maxdd,
        "mar": cagr / maxdd if maxdd > 0 else 0, "pf": pf,
        "final": eq.iloc[-1], "trades": all_tr,
    }


def main():
    print("載入全市場資料...")
    data, names = load_all()
    risk_on = load_regime()
    base_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    breadth = load_breadth()
    all_dates, date_idx = build_date_index(data)

    print("建立各策略 entry_map (一次)...")
    entry_maps = {}
    for k in STRATS:
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        entry_maps[k] = build_entry_map(sigs, data)

    configs = [
        ("baseline (無廣度濾網)", None),
        ("廣度 30/45", (0.30, 0.45)),
        ("廣度 35/50", (0.35, 0.50)),
        ("廣度 40/55", (0.40, 0.55)),
        ("廣度 35/45", (0.35, 0.45)),
    ]
    results = []
    for label, cfg in configs:
        tiers = base_tiers if cfg is None else gate_tiers(
            base_tiers, breadth, cfg[0], cfg[1], all_dates)
        m = portfolio(data, entry_maps, tiers, vol_scalars, all_dates, date_idx)
        results.append((label, m))
        print(f"\n=== {label} ===")
        print(f"  交易 {m['n']}  勝率 {m['win']:.0%}  總R {m['totR']:+.0f}  "
              f"年化 {m['cagr']:+.1%}  最大回撤 {m['maxdd']:.1%}  "
              f"MAR {m['mar']:.2f}  PF {m['pf']:.2f}  期末 {m['final']:,.0f}")
        ym = year_metrics(m["trades"])
        print(f"  {'年度':>6} {'筆數':>5} {'勝率':>5} {'總R':>7} {'損益':>12}")
        for y, (n, w, r, pnl) in ym.items():
            print(f"  {y:>6} {n:>5} {w:>5.0%} {r:>+7.1f} {pnl:>+12,.0f}")

    print("\n\n### 總結對照 ###")
    print(f"{'設定':<22} {'年化':>7} {'最大回撤':>8} {'MAR':>6} {'PF':>5} {'總R':>7}")
    for label, m in results:
        print(f"{label:<22} {m['cagr']:>7.1%} {m['maxdd']:>8.1%} "
              f"{m['mar']:>6.2f} {m['pf']:>5.2f} {m['totR']:>+7.0f}")


if __name__ == "__main__":
    main()
