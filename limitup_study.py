"""漲停前特徵統計 + 篩選建構 + 與現行 K/D 策略對比 (limitup_study.py)

目的
----
現行生產策略 (K/D, Minervini VCP 突破變體) 的目標是「騎乘數週的主升段」,
進場當日刻意把漲幅上限壓在 6% (`gain_cap`) 並跳過漲停 (漲停不追高)。
本腳本回答一個『不同的問題』: 一檔股票在『隔日漲停 (+10%)』之前, 前一日
收盤時長什麼樣子? 把這些前兆特徵量化, 組成一個「隔日漲停機率」篩選,
再拿它和現行 K/D 訊號比對, 說明兩者結構差異。

方法
----
事件定義: 台股每日漲跌幅上限 10%; 因跳動單位取整, 用 `LIMIT_THRESH=0.095`
判定某日為『漲停日』(當日收盤 / 昨收 - 1 >= 9.5%)。可選 `--sealed` 進一步要求
『鎖死』(收在當日最高附近, 收盤>=最高*0.995), 更貼近真正封板。

對『可交易宇宙』(收盤>=10, 成交值>=門檻, MA200 有效) 的每一個交易日 T,
在僅使用『到 T 收盤為止』的資訊下抽取一組前兆特徵, 並標記 label =
『T+1 是否漲停』。全市場彙整後:
  1. 基準漲停率 (base rate)。
  2. 每個特徵切成十分位, 算各分位的『隔日漲停條件機率』與相對基準的 lift。
  3. 前兆日 (label=1) 的特徵均值 vs 全體均值。
  4. 由前兆訊號最強的特徵組一個可解釋的 `LU_SCREEN`, 報 precision / recall /
     每日候選數。
  5. 與現行 K / D 訊號比對: K/D 訊號隔日漲停命中率、全體漲停被 K/D 事前
     覆蓋率、候選重疊 (Jaccard)。

用法
----
    python3 limitup_study.py                 # 用 data/ (未還原除權息)
    MM_DATA_DIR=data_adj python3 limitup_study.py --sealed
    python3 limitup_study.py --data-dir data_adj --min-turnover 50e6
    python3 limitup_study.py --sealed --dump-md   # 另存 markdown 報告 (limitup_report.md)

需先執行 download_all.py 備妥日線資料。輸出印到 stdout, 並把十分位 lift 表
存成 limitup_lift.csv 供進一步查閱; 加 --dump-md [PATH] 會把完整四段結果
另存成一份 markdown 報告 (預設 limitup_report.md, 可直接貼上或分享)。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest as bt  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

LIMIT_THRESH = 0.095   # 漲停判定 (10% 上限, 容跳動取整)

# 拿來做十分位 lift 分析與 LU_SCREEN 的候選特徵 (皆為 T 收盤可得)
FEATURES = [
    "rs_rank",      # 126日報酬全市場百分位 (相對強度)
    "prox_52wh",    # 收盤 / 52週高 (越接近前高越易點火)
    "ret5", "ret20", "ret60",   # 多框架動能
    "trend_score",  # 0~6 趨勢模板得分
    "contraction",  # ATR5/ATR14 波動收縮 (越小越壓縮)
    "vr5",          # 近5日均量 / 50日均量 (量能翻升)
    "vol_dryup",    # 昨量 < 20日均量 (爆發前量縮)
    "day_range",    # (高-低)/收 當日振幅 (越小越壓縮)
    "near20h",      # 收盤 / 20日高 (突破就緒度)
    "ext20",        # 收盤 / MA20 - 1 (乖離, 太大代表已延伸)
    "rsi14",
    "upstreak",     # 連續上漲天數
    "gain_today",   # 當日漲幅 (動能點火常有前一日中紅)
]


def build_panel(data, min_turnover, min_price):
    """對每檔股票向量化抽取前兆特徵 + label(T+1漲停), 彙整成單一 DataFrame。"""
    rs = bt.compute_rs_rank(data)   # date x code 百分位
    frames = []
    for code, df in data.items():
        if len(df) < 260:
            continue
        d = df.copy()
        c = d["close"]
        prev_c = c.shift(1)
        gain = c / prev_c - 1.0
        # label: 隔日 (T+1) 是否漲停
        next_gain = gain.shift(-1)
        label = (next_gain >= LIMIT_THRESH).astype(float)
        # 若要求鎖死, 由呼叫端另外過濾 (見 --sealed)
        next_close = c.shift(-1)
        next_high = d["high"].shift(-1)
        sealed = (next_close >= next_high * 0.995).astype(float)

        ma200 = d["ma200"]
        d_feat = pd.DataFrame({
            "code": code,
            "date": d["date"],
            "label": label,
            "sealed": sealed,
            "rs_rank": rs[code].reindex(d["date"]).values if code in rs.columns else np.nan,
            "prox_52wh": c / d["high252"],
            "ret5": c.pct_change(5),
            "ret20": d["ret20"],
            "ret60": d["ret60"],
            "trend_score": (
                (c > d["ma20"]).astype(int)
                + (d["ma20"] > d["ma50"]).astype(int)
                + (d["ma50"] > d["ma150"]).astype(int)
                + (d["ma150"] > d["ma200"]).astype(int)
                + (c > d["ma50"]).astype(int)
                + (ma200 > ma200.shift(21)).astype(int)
            ),
            "contraction": d["atr5"] / d["atr14"],
            "vr5": d["volume"].rolling(5).mean() / d["vol_ma50"],
            "vol_dryup": (d["volume"].shift(1) < d["vol_ma20"].shift(1)).astype(float),
            "day_range": (d["high"] - d["low"]) / c,
            "near20h": c / d["high20"],
            "ext20": c / d["ma20"] - 1.0,
            "rsi14": d["rsi14"],
            "upstreak": _up_streak(gain),
            "gain_today": gain,
            "turnover": c * d["volume"],
        })
        # 可交易宇宙門檻 + label 必須存在 (最後一列 T+1 未知 → 丟)
        valid = (
            (c >= min_price) & (d_feat["turnover"] >= min_turnover)
            & ma200.notna() & next_gain.notna()
        )
        frames.append(d_feat[valid])
    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(subset=["rs_rank"])


def _up_streak(gain):
    """連續上漲天數 (含當日): 當日跌則歸零。"""
    up = (gain > 0).astype(int).values
    out = np.zeros(len(up))
    run = 0
    for k in range(len(up)):
        run = run + 1 if up[k] else 0
        out[k] = run
    return out


def lift_table(panel, feature, bins=10):
    """把 feature 切十分位, 算各分位隔日漲停率與相對基準 lift。"""
    base = panel["label"].mean()
    try:
        q = pd.qcut(panel[feature], bins, duplicates="drop")
    except ValueError:
        return None
    g = panel.groupby(q, observed=True)["label"].agg(["mean", "count"])
    g["lift"] = g["mean"] / base
    g.index = [f"[{iv.left:.3g}, {iv.right:.3g}]" for iv in g.index]
    return g


def evaluate_screen(panel, mask, name):
    """對一個布林篩選 mask 算 precision / recall / 每日候選數, 回傳結果 dict 並印出。"""
    n_sig = int(mask.sum())
    total_events = int(panel["label"].sum())
    if n_sig == 0:
        print(f"  {name}: 無候選")
        return {"name": name, "n_sig": 0}
    hits = int(panel.loc[mask, "label"].sum())
    precision = hits / n_sig
    recall = hits / total_events if total_events else 0.0
    ndays = panel["date"].nunique()
    base = panel["label"].mean()
    print(f"  {name}: 候選 {n_sig} 筆 ({n_sig / ndays:.1f}/日)  "
          f"命中隔日漲停 {hits}  precision {precision:.2%} "
          f"(基準 {base:.2%}, lift {precision / base:.1f}x)  recall {recall:.1%}")
    return {"name": name, "n_sig": n_sig, "per_day": n_sig / ndays, "hits": hits,
            "precision": precision, "base": base, "lift": precision / base,
            "recall": recall}


def build_lu_screen(panel):
    """可解釋的漲停前兆篩選 (與 K/D 對照用)。

    綜合前兆最強的維度: 高相對強度 + 貼近52週高 + 波動/振幅壓縮 + 量縮蓄勢 +
    多頭排列 + 就緒於20日高附近但尚未過度延伸。門檻取自各特徵 lift 表的高分位起點,
    刻意用『固定可讀規則』而非過擬合的權重, 方便和 K/D 直接比較結構差異。
    """
    return (
        (panel["rs_rank"] >= 0.80)
        & (panel["prox_52wh"] >= 0.90)
        & (panel["trend_score"] >= 5)
        & (panel["contraction"] <= 0.90)
        & (panel["day_range"] <= 0.05)
        & (panel["vol_dryup"] > 0.5)
        & (panel["near20h"] >= 0.95)
        & (panel["ext20"] <= 0.12)
    )


def compare_kd(data, panel):
    """與現行 K / D 訊號比對: 訊號隔日漲停命中率 + 漲停事前覆蓋率 + 候選重疊。

    回傳 list of dict (每個策略一筆), 供 markdown 報告重用。
    """
    print("\n" + "=" * 72)
    print("與現行 K / D 策略對比")
    print("=" * 72)

    # 全體漲停事件 (T+1 漲停 → 事件發生在 T+1)。以 (code, T日) 標記前兆日。
    lu_days = set(zip(panel.loc[panel["label"] > 0.5, "code"],
                      panel.loc[panel["label"] > 0.5, "date"]))
    lu_mask = build_lu_screen(panel)
    lu_screen_days = set(zip(panel.loc[lu_mask, "code"], panel.loc[lu_mask, "date"]))
    base = panel["label"].mean()

    results = []
    for key in ("K", "D"):
        sigs = bt.collect_signals(data, key)
        # 訊號日 (code, 訊號日T) → 檢查該股 T+1 是否漲停
        sig_days = set()
        hit = 0
        for sd, lst in sigs.items():
            for score, code, i, s in lst:
                df = data[code]
                sig_days.add((code, sd))
                if i + 1 < len(df):
                    g = df["close"].iloc[i + 1] / df["close"].iloc[i] - 1.0
                    if g >= LIMIT_THRESH:
                        hit += 1
        n = sum(len(v) for v in sigs.values())
        covered = len(lu_days & sig_days)
        inter = len(lu_screen_days & sig_days)
        union = len(lu_screen_days | sig_days)
        res = {
            "key": key, "n_signals": n, "base": base,
            "next_day_lu_rate": (hit / n) if n else None,
            "covered": covered, "total_lu": len(lu_days),
            "coverage": (covered / len(lu_days)) if lu_days else None,
            "jaccard_inter": inter, "jaccard_union": union,
            "jaccard": (inter / union) if union else None,
        }
        results.append(res)
        print(f"\n策略 {key}: 訊號 {n} 筆")
        if n:
            print(f"  訊號隔日自身漲停率: {hit / n:.2%}  (基準 {base:.2%})")
        if lu_days:
            print(f"  漲停前兆日被策略 {key} 覆蓋: "
                  f"{covered}/{len(lu_days)} = {covered / len(lu_days):.1%}")
        if union:
            print(f"  與 LU_SCREEN 候選 Jaccard 重疊: {inter}/{union} = {inter / union:.1%}")
    return results


# ─────────────────────────────────────────────────────────────
# Markdown 報告 (--dump-md)
# ─────────────────────────────────────────────────────────────

def _md_table(headers, rows):
    """由表頭與資料列組出 markdown 表格字串。"""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_markdown(meta, mean_rows, lift_tables, screen_results, kd_results):
    """把四段結果組成完整 markdown 報告字串。"""
    L = []
    L.append("# 漲停前特徵統計報告\n")
    L.append(f"- 產生時間: {meta['now']}")
    L.append(f"- 資料夾: `{meta['data_dir']}`  ·  股票數: {meta['n_stocks']}")
    L.append(f"- 事件定義: 隔日漲停 (+10%, `gain>= {LIMIT_THRESH}`)"
             f"{'  ·  只計封板鎖死' if meta['sealed'] else ''}")
    L.append(f"- 篩選門檻: 收盤≥{meta['min_price']:g}, 日成交值≥{meta['min_turnover']:g}")
    L.append(f"- 可交易宇宙樣本: **{meta['n_rows']:,}** 個股票日  ·  "
             f"漲停事件: **{meta['n_events']:,}**  ·  "
             f"基準隔日漲停率: **{meta['base']:.3%}**\n")

    L.append("## 1. 前兆特徵均值 (漲停前一日 vs 全體)\n")
    L.append("差異倍率 = 漲停前一日均值 / 全體均值; 明顯偏離 1 者為候選前兆。\n")
    L.append(_md_table(
        ["特徵", "漲停前一日", "全體", "差異倍率"],
        [(f, f"{a:.4f}", f"{b:.4f}", f"{r:.2f}") for f, a, b, r in mean_rows]))
    L.append("")

    L.append("## 2. 各特徵十分位『隔日漲停』條件機率與 lift\n")
    L.append("lift = 該分位隔日漲停率 / 基準; lift>1 且隨分位單調上升者為有效前兆。\n")
    for f, t in lift_tables:
        top, bot = t.iloc[-1], t.iloc[0]
        L.append(f"### {f}  (最高分位 lift {top['lift']:.2f}x · 最低分位 lift {bot['lift']:.2f}x)\n")
        L.append(_md_table(
            ["分位", "P(隔日漲停)", "lift", "n"],
            [(idx, f"{r['mean']:.3%}", f"{r['lift']:.2f}x", int(r["count"]))
             for idx, r in t.iterrows()]))
        L.append("")

    L.append("## 3. 漲停前兆篩選 LU_SCREEN 評估\n")
    L.append("precision = 候選中隔日真漲停比例; recall = 全體漲停被涵蓋比例。\n")
    srows = []
    for s in screen_results:
        if s.get("n_sig", 0) == 0:
            srows.append((s["name"], 0, "-", "-", "-", "-", "-"))
        else:
            srows.append((s["name"], s["n_sig"], f"{s['per_day']:.1f}", s["hits"],
                          f"{s['precision']:.2%}", f"{s['lift']:.1f}x",
                          f"{s['recall']:.1%}"))
    L.append(_md_table(
        ["篩選", "候選數", "每日", "命中", "precision", "lift", "recall"], srows))
    L.append("")

    L.append("## 4. 與現行 K / D 策略對比\n")
    L.append("『訊號隔日漲停率』= K/D 進場訊號日的隔日該股自身漲停比例; "
             "『漲停事前覆蓋』= 全體漲停在前一日被 K/D 訊號命中的比例; "
             "『Jaccard』= K/D 候選與 LU_SCREEN 候選 (code,日) 的重疊度。\n")
    krows = []
    for r in kd_results:
        krows.append((
            f"策略 {r['key']}", r["n_signals"],
            f"{r['next_day_lu_rate']:.2%}" if r["next_day_lu_rate"] is not None else "-",
            f"{r['covered']}/{r['total_lu']} = {r['coverage']:.1%}" if r["coverage"] is not None else "-",
            f"{r['jaccard_inter']}/{r['jaccard_union']} = {r['jaccard']:.1%}" if r["jaccard"] is not None else "-",
        ))
    L.append(_md_table(
        ["策略", "訊號數", "訊號隔日漲停率", "漲停事前覆蓋", "與 LU_SCREEN Jaccard"], krows))
    L.append(f"\n> 基準隔日漲停率 {meta['base']:.3%}。K/D 訊號隔日漲停率高於基準代表 K/D "
             "進場點附近確有動能, 但覆蓋率/Jaccard 若偏低則印證兩者是低重疊的不同獵物。\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="資料夾 (預設 data/, 或 MM_DATA_DIR)")
    ap.add_argument("--min-turnover", type=float, default=50e6, help="最低日成交值")
    ap.add_argument("--min-price", type=float, default=10.0, help="最低股價")
    ap.add_argument("--sealed", action="store_true", help="只計鎖死漲停 (收在最高附近)")
    ap.add_argument("--dump-md", nargs="?", const="limitup_report.md", default=None,
                    metavar="PATH", help="把完整結果另存為 markdown 報告 (預設 limitup_report.md)")
    args = ap.parse_args()
    if args.data_dir:
        bt.DATA_DIR = (args.data_dir if os.path.isabs(args.data_dir)
                       else os.path.join(os.path.dirname(__file__), args.data_dir))

    print(f"載入資料 ({bt.DATA_DIR}) ...")
    data, names = bt.load_all()
    print(f"{len(data)} 檔股票")

    panel = build_panel(data, args.min_turnover, args.min_price)
    if args.sealed:
        # 鎖死: 隔日既漲停又收在最高附近才算事件
        panel["label"] = panel["label"] * panel["sealed"]
    base = panel["label"].mean()
    n_events = int(panel["label"].sum())
    print(f"\n可交易宇宙樣本: {len(panel):,} 個股票日, "
          f"漲停{'(鎖死)' if args.sealed else ''}事件: {n_events:,} "
          f"→ 基準隔日漲停率 {base:.3%}")

    # ── 前兆日 vs 全體特徵均值 ──
    print("\n" + "=" * 72)
    print("前兆特徵均值: 漲停前一日 (label=1)  vs  全體")
    print("=" * 72)
    print(f"{'特徵':<14}{'漲停前一日':>14}{'全體':>12}{'差異倍率':>12}")
    pre = panel[panel["label"] > 0.5]
    mean_rows = []
    for f in FEATURES:
        a, b = pre[f].mean(), panel[f].mean()
        ratio = a / b if b else float("nan")
        mean_rows.append((f, a, b, ratio))
        print(f"{f:<14}{a:>14.4f}{b:>12.4f}{ratio:>12.2f}")

    # ── 各特徵十分位 lift ──
    print("\n" + "=" * 72)
    print("各特徵十分位『隔日漲停』條件機率與 lift (找單調前兆)")
    print("=" * 72)
    lift_tables = []
    all_lift = []
    for f in FEATURES:
        t = lift_table(panel, f)
        if t is None:
            continue
        lift_tables.append((f, t))
        top = t.iloc[-1]
        bot = t.iloc[0]
        print(f"\n[{f}]  最高分位 lift {top['lift']:.2f}x  最低分位 lift {bot['lift']:.2f}x")
        for idx, r in t.iterrows():
            bar = "█" * int(round(r["lift"] * 10))
            print(f"    {idx:<22} P={r['mean']:.3%} lift={r['lift']:.2f}x n={int(r['count'])} {bar}")
        tt = t.reset_index().rename(columns={"index": "bucket"})
        tt.insert(0, "feature", f)
        all_lift.append(tt)
    if all_lift:
        pd.concat(all_lift, ignore_index=True).to_csv("limitup_lift.csv", index=False)
        print("\n(十分位 lift 表已存 limitup_lift.csv)")

    # ── LU_SCREEN 篩選評估 ──
    print("\n" + "=" * 72)
    print("漲停前兆篩選 LU_SCREEN 評估 (可解釋固定規則)")
    print("=" * 72)
    mask = build_lu_screen(panel)
    screen_results = [
        evaluate_screen(panel, mask, "LU_SCREEN"),
        evaluate_screen(panel, panel["rs_rank"] >= 0.90, "僅 RS>=0.90"),
        evaluate_screen(panel, panel["prox_52wh"] >= 0.95, "僅 貼52週高>=0.95"),
        evaluate_screen(panel, (panel["contraction"] <= 0.8) & (panel["day_range"] <= 0.04),
                        "僅 波動壓縮"),
    ]

    # ── 與現行 K/D 對比 ──
    kd_results = compare_kd(data, panel)

    # ── 選用: 輸出 markdown 報告 ──
    if args.dump_md:
        import datetime
        meta = {
            "now": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data_dir": bt.DATA_DIR, "n_stocks": len(data),
            "sealed": args.sealed, "min_price": args.min_price,
            "min_turnover": args.min_turnover, "n_rows": len(panel),
            "n_events": n_events, "base": base,
        }
        md = build_markdown(meta, mean_rows, lift_tables, screen_results, kd_results)
        with open(args.dump_md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"\nmarkdown 報告已輸出 → {args.dump_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
