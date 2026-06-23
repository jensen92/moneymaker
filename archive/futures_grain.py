"""黃豆 / 小麥 / 玉米 專屬策略 — 套用 5 市場版的最佳架構 + 穀物季節性.

把 futures_portfolio.py 驗證有效的結構 (風險平價波動目標 + 多訊號集成 + 組合風控)
專門套用到三大穀物, 並加入穀物獨有的「季節性偏多」訊號 (12-4 月), 做出專屬的
最佳穀物組合策略。僅做多 (穀物做空長期虧)。

逐輪 (結構性, 非調參數):
  g0 基準   : 單一唐奇安突破, 等名目金額
  g1 +風險平價/波動目標
  g2 +多訊號集成 (突破+均線+動能)
  g3 +穀物季節性訊號 (季節投票)
  g4 +組合波動風控

用法: python3 futures_grain.py
"""
from futures_portfolio import backtest, metrics, IS_END, OOS_START

GRAIN_SPECS = {
    "ZS=F": {"name": "黃豆 Soybean", "pv": 50.0, "rt": 30.0},
    "ZW=F": {"name": "小麥 Wheat",   "pv": 50.0, "rt": 30.0},
    "ZC=F": {"name": "玉米 Corn",    "pv": 50.0, "rt": 30.0},
}

VERSIONS = {
    "g0 基準(單訊號/等金額)": {"signals": ["donch"], "allow_short": False,
                       "sizing": "equaldollar"},
    "g1 +風險平價/波動目標": {"signals": ["donch"], "allow_short": False,
                       "sizing": "voltarget"},
    "g2 +多訊號集成":      {"signals": ["donch", "ma", "mom"], "allow_short": False,
                       "sizing": "voltarget"},
    "g3 +穀物季節訊號":     {"signals": ["donch", "ma", "mom", "season"],
                       "allow_short": False, "sizing": "voltarget"},
    "g4 +組合波動風控":     {"signals": ["donch", "ma", "mom", "season"],
                       "allow_short": False, "sizing": "voltarget",
                       "vol_control": True},
}


def run(cfg, start=None, end=None):
    return metrics(backtest(cfg, start=start, end=end, specs=GRAIN_SPECS))


def main():
    print("黃豆 / 小麥 / 玉米 專屬策略 (本金 100 萬, 僅做多)")
    print(f"標的: {', '.join(s['name'] for s in GRAIN_SPECS.values())}\n")
    hdr = (f"{'版本':<22}{'全期Sh':>7}{'IS Sh':>7}{'OOS Sh':>7}"
           f"{'CAGR':>8}{'MaxDD':>8}{'MAR':>6}")
    print(hdr); print("-" * len(hdr))
    for name, cfg in VERSIONS.items():
        mf = run(cfg)
        mi = run(cfg, end=IS_END)
        mo = run(cfg, start=OOS_START)
        print(f"{name:<22}{mf['sharpe']:>7.2f}{mi['sharpe']:>7.2f}"
              f"{mo['sharpe']:>7.2f}{mf['cagr']:>8.1%}{mf['dd']:>8.1%}"
              f"{mf['mar']:>6.2f}")


if __name__ == "__main__":
    main()
