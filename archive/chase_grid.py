"""追漲網格實證: 當日漲幅 >= 門檻(5%~9.5%) 時, 隔日開盤進場, 持有 H 天(1~5)後
隔日開盤賣出。掃描 門檻 × 持有天數, 看哪個組合「淨報酬/勝率/期望值」最好。

口徑:
  - 進場: 訊號日(t)收盤漲幅>=門檻 → t+1 開盤買入 (可實作, 不偷看當日收盤後資訊)
  - 出場: t+1+H 開盤賣出 (持有 H 個交易日)
  - 成本: 來回 0.785% (手續費x2+稅+滑價) 從毛報酬扣除
  - 母體: 價>=10、前一日有量、有完整未來H+1日
報告每格: 樣本數 n、毛均報酬、淨均報酬、勝率、年化期望(淨/持有天數*252)。
"""
import os
import sys
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data_adj")
THRS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.095]
HOLDS = [1, 2, 3, 4, 5]
COST = 0.00785


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".csv") and not f.startswith("_"))
    print(f"掃描 {len(files)} 檔 ...")
    # 累積每格的報酬list
    rec = {(t, h): [] for t in THRS for h in HOLDS}
    for k, fn in enumerate(files):
        df = pd.read_csv(os.path.join(DATA, fn))
        o = df["open"].values; c = df["close"].values; v = df["volume"].values
        n = len(c)
        if n < 10:
            continue
        gain = np.empty(n); gain[0] = 0
        gain[1:] = c[1:] / c[:-1] - 1.0
        for i in range(1, n - max(HOLDS) - 1):
            if c[i] < 10 or v[i] <= 0 or c[i - 1] <= 0:
                continue
            g = gain[i]
            if g < THRS[0]:
                continue
            entry = o[i + 1]
            if entry <= 0:
                continue
            for t in THRS:
                if g >= t:
                    for h in HOLDS:
                        exit_px = o[i + 1 + h]
                        if exit_px > 0:
                            rec[(t, h)].append(exit_px / entry - 1.0)
        if (k + 1) % 500 == 0:
            print(f"  ... {k+1}/{len(files)}")

    print("\n=== 追漲門檻 × 持有天數: 淨均報酬 (扣0.785%來回) ===")
    print("  持有天數 ->")
    print("  門檻\\H " + "".join(f"{h:>9}d" for h in HOLDS))
    for t in THRS:
        row = f"  >={t*100:>3.0f}% "
        for h in HOLDS:
            arr = np.array(rec[(t, h)])
            net = arr.mean() - COST if len(arr) else float("nan")
            row += f"{net*100:>9.2f}%"
        print(row)

    print("\n=== 勝率 (淨報酬>0 比例) ===")
    print("  門檻\\H " + "".join(f"{h:>9}d" for h in HOLDS))
    for t in THRS:
        row = f"  >={t*100:>3.0f}% "
        for h in HOLDS:
            arr = np.array(rec[(t, h)])
            wr = (arr - COST > 0).mean() if len(arr) else float("nan")
            row += f"{wr*100:>9.1f}%"
        print(row)

    print("\n=== 年化期望 (淨均報酬 / 持有天數 * 252, 粗略可比性) ===")
    print("  門檻\\H " + "".join(f"{h:>9}d" for h in HOLDS))
    best = (None, -999)
    for t in THRS:
        row = f"  >={t*100:>3.0f}% "
        for h in HOLDS:
            arr = np.array(rec[(t, h)])
            if len(arr):
                net = arr.mean() - COST
                ann = net / h * 252
                row += f"{ann*100:>8.0f}%"
                if ann > best[1]:
                    best = (f">={t*100:.0f}% 持有{h}天", ann)
            else:
                row += f"{'na':>9}"
        print(row)

    print("\n=== 各格樣本數 ===")
    print("  門檻\\H " + "".join(f"{h:>9}d" for h in HOLDS))
    for t in THRS:
        row = f"  >={t*100:>3.0f}% "
        for h in HOLDS:
            row += f"{len(rec[(t,h)]):>9}"
        print(row)

    print(f"\n年化期望最佳格: {best[0]}  年化≈{best[1]*100:.0f}%")
    print("(注: 年化期望未計每年可進場次數限制與資金容納度, 僅供格間相對比較; "
          "淨均報酬與勝率才是單筆真實值)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
