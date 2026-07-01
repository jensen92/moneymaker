"""季節交易策略有效性驗證 — 統計各商品逐年『報酬百分比』.

對 grain_signals (黃豆/玉米) 與 energy_signals (天然氣/原油) 的季節窗,
逐年重放『進場價→出場價』的 % 報酬 (與策略同一進出邏輯: 進場月第一交易日買進,
持有 N 月或觸 3×ATR 停損), 統計:
  勝率 (正報酬年占比) / 平均年報酬 / 中位 / 最佳最差 / 平均÷標準差 / 樣本內外對切。

用途: 用直觀的 % 而非 R 值, 驗證季節性是否『多數年份為正、非單一離群年撐起』。

用法: python3 season_validate.py
"""
import numpy as np

import grain_signals as g
import energy_signals as e


def yearly_pct(mod, key):
    """回傳 [(年, 進場日, 出場日, 報酬%)], 每年一筆季節交易。"""
    cfg = mod.CONFIG[key]
    dt, o, h, l, c = mod._load(key)
    a = mod._atr(h, l, c, 20)
    n = len(c); hold = int(21 * cfg["hold_mo"]); st = cfg["stop_atr"]; em = cfg["month"]
    out = []; ydone = set()
    for i in range(21, n):
        m = int(dt[i][5:7]); pm = int(dt[i - 1][5:7]); yr = dt[i][:4]
        if m == em and pm != em and yr not in ydone and not np.isnan(a[i]):
            ydone.add(yr)
            entry = c[i]; stop = entry - st * a[i]
            j = min(i + hold, n - 1); ex = c[j]; exd = dt[j]
            for k in range(i + 1, j + 1):
                if c[k] <= stop:
                    ex = c[k]; exd = dt[k]; break
            out.append((yr, dt[i], exd, (ex - entry) / entry * 100.0))
    return out


def report(name, rows):
    pcts = np.array([r[3] for r in rows])
    n = len(pcts); pos = int((pcts > 0).sum())
    half = n // 2
    print(f"\n{'=' * 60}\n{name}  (共 {n} 年)")
    for yr, ed, xd, p in rows:
        bar = "█" * int(abs(p) / 2)
        print(f"  {yr}: {p:>+6.1f}%  {ed[5:]}→{xd[5:]}  {'+' if p >= 0 else '-'}{bar}")
    verdict = "✅ 有效" if (pos / n >= 0.5 and np.median(pcts) > 0) else "⚠️ 存疑"
    print(f"── 驗證統計  {verdict} ──")
    print(f"  勝率(正報酬年): {pos}/{n} = {pos/n:.0%}")
    print(f"  平均 {pcts.mean():+.1f}%  中位 {np.median(pcts):+.1f}%  "
          f"最佳 {pcts.max():+.1f}%  最差 {pcts.min():+.1f}%")
    print(f"  標準差 {pcts.std():.1f}%  平均/標準差 {pcts.mean()/pcts.std():.2f}")
    print(f"  前半 {pcts[:half].mean():+.1f}% ({(pcts[:half]>0).mean():.0%}勝)  "
          f"後半 {pcts[half:].mean():+.1f}% ({(pcts[half:]>0).mean():.0%}勝)")
    return {"name": name, "n": n, "win": pos / n, "avg": float(pcts.mean()),
            "median": float(np.median(pcts))}


def main():
    print("季節交易有效性驗證 — 逐年報酬% (判定: 勝率≥50% 且 中位>0 = ✅有效)")
    summ = []
    for key in ("ZS", "ZC"):
        cfg = g.CONFIG[key]
        summ.append(report(f"穀物 {cfg['name']} ({cfg['month']}月進/持{cfg['hold_mo']}月)",
                            yearly_pct(g, key)))
    for key in ("NG", "CL"):
        cfg = e.CONFIG[key]
        summ.append(report(f"能源 {cfg['name']} ({cfg['month']}月進/持{cfg['hold_mo']}月)",
                            yearly_pct(e, key)))
    print(f"\n{'=' * 60}\n總表 (勝率 / 平均 / 中位):")
    for s in summ:
        v = "✅" if (s["win"] >= 0.5 and s["median"] > 0) else "⚠️"
        print(f"  {v} {s['name']:<28} 勝率{s['win']:.0%}  平均{s['avg']:+.1f}%  中位{s['median']:+.1f}%")


if __name__ == "__main__":
    main()
