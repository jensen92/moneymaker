"""Telegram 策略機器人: 在 Telegram 直接下指令跑選股 / 回測 / 個股分析.

設計: 純 HTTP 長輪詢 (long-polling), 只需 requests, 不用 webhook / 伺服器.
重活在背景 thread 序列化執行, 不阻塞指令接收.

啟動前設定環境變數 (token 切勿寫進程式或 commit):
    export TELEGRAM_BOT_TOKEN="<BotFather 給的 token>"
    export TELEGRAM_CHAT_ID="<你的 chat id>"
    export MM_DATA_DIR="/Users/jensenchen/money maker/adjusted_data"

指令:
    /picks              今日選股 (PA/PB/D/C)
    /analyze 2330       分析個股現況 + 進出場價格 (自動更新資料)
    /backtest [C,D]     組合回測 (慢, 數分鐘)
    /status             機器人 / 資料狀態
    /help               指令說明
"""
import csv
import io
import os
import sys
import subprocess
import threading
import time

import requests

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
DATA_DIR = os.environ.get("MM_DATA_DIR", "").strip()
HERE = os.path.dirname(os.path.abspath(__file__))
API = f"https://api.telegram.org/bot{TOKEN}"
YF_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

_job_lock = threading.Lock()


def send(chat_id, text):
    for i in range(0, len(text), 3900):
        chunk = text[i:i + 3900]
        try:
            requests.post(f"{API}/sendMessage",
                          data={"chat_id": chat_id, "text": chunk},
                          timeout=30)
        except requests.RequestException as e:
            print("send error:", e)


def send_get_id(chat_id, text):
    """送訊息並回傳 message_id (供後續 edit 更新進度)."""
    try:
        r = requests.post(f"{API}/sendMessage",
                          data={"chat_id": chat_id, "text": text},
                          timeout=30)
        return r.json().get("result", {}).get("message_id")
    except (requests.RequestException, ValueError) as e:
        print("send error:", e)
        return None


def edit(chat_id, message_id, text):
    """更新既有訊息 (進度回報用, 不洗版)."""
    if message_id is None:
        return
    try:
        requests.post(f"{API}/editMessageText",
                      data={"chat_id": chat_id, "message_id": message_id,
                            "text": text}, timeout=30)
    except requests.RequestException as e:
        print("edit error:", e)


def run_script(args):
    env = dict(os.environ)
    if DATA_DIR:
        env["MM_DATA_DIR"] = DATA_DIR
    try:
        p = subprocess.run(["python3", *args], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=3600)
        out = p.stdout or ""
        if p.returncode != 0:
            out += f"\n[錯誤 returncode={p.returncode}]\n{p.stderr[-1500:]}"
        return out.strip() or "(無輸出)"
    except subprocess.TimeoutExpired:
        return "[逾時] 執行超過 60 分鐘已中止"
    except Exception as e:  # noqa: BLE001
        return f"[例外] {e}"


def heavy_job(chat_id, label, args):
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        send(chat_id, f"⏳ {label} 執行中, 請稍候...")
        t0 = time.time()
        out = run_script(args)
        mins = (time.time() - t0) / 60
        send(chat_id, f"✅ {label} 完成 (耗時 {mins:.1f} 分)\n\n{out}")
    finally:
        _job_lock.release()


# ── 個股分析 ───────────────────────────────────────────────────────────────

def _fetch_yf(code, range_="1y"):
    """從 Yahoo Finance 抓取個股日線, 回傳 list of (date,o,h,l,c,v)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.TW"
    for attempt in range(3):
        try:
            r = requests.get(url, params={"range": range_, "interval": "1d"},
                             headers=YF_HEADERS, timeout=20)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp", [])
            q = res["indicators"]["quote"][0]
            rows = []
            for i, t in enumerate(ts):
                o, h, l, c, v = (q["open"][i], q["high"][i], q["low"][i],
                                  q["close"][i], q["volume"][i])
                if None in (o, h, l, c):
                    continue
                date = time.strftime("%Y-%m-%d", time.gmtime(t + 8 * 3600))
                rows.append((date, round(o, 2), round(h, 2), round(l, 2),
                              round(c, 2), int(v or 0)))
            return rows
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return None


def _update_stock_data(code):
    """下載/更新個股 CSV (只補新資料)。回傳 (csv_path, msg)."""
    if not DATA_DIR:
        return None, "未設定 MM_DATA_DIR"
    path = os.path.join(DATA_DIR, f"{code}.csv")

    # 讀現有最後日期
    last_date = ""
    if os.path.exists(path):
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        if len(lines) > 1:
            last_date = lines[-1].split(",")[0]

    # 決定抓取範圍
    range_ = "max" if not last_date else "3mo"
    rows = _fetch_yf(code, range_)
    if rows is None:
        return path if os.path.exists(path) else None, "下載失敗, 使用舊資料"

    new_rows = [r for r in rows if r[0] > last_date]
    if not new_rows:
        return path, "資料已是最新"

    if not last_date:
        # 全新檔案
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            w.writerows(new_rows)
        msg = f"新建 {len(new_rows)} 筆"
    else:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(new_rows)
        msg = f"補 {len(new_rows)} 筆新資料"
    return path, msg


def _analyze_stock(code, progress=None):
    """下載最新資料, 用策略 C/D 分析現況, 回傳文字報告.

    progress: 可選 callback(text), 用於即時回報掃描進度.
    """
    def note(msg):
        if progress:
            progress(msg)

    # 把 HERE 加進 sys.path 讓 import 找到 strategies / backtest
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    import importlib
    import numpy as np
    import pandas as pd

    # 動態 import (避免模組快取過時)
    for mod in ["strategies", "backtest"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

    from strategies import add_indicators, STRATEGIES
    from backtest import (DATA_DIR as BT_DATA_DIR,
                          load_regime_tiers, load_vol_scalars,
                          RISK_PCT, INIT_CAPITAL)

    # 1. 更新個股資料
    note(f"⏳ {code}: 下載最新資料中...")
    path, dl_msg = _update_stock_data(code)
    if path is None or not os.path.exists(path):
        return f"❌ {code}: {dl_msg}"

    # 2. 載入個股
    df = pd.read_csv(path, parse_dates=["date"])
    df = add_indicators(df).reset_index(drop=True)
    if len(df) < 210:
        return f"❌ {code}: 資料不足 210 日 ({len(df)} 日), 無法分析"

    latest = df.iloc[-1]
    d = latest["date"]
    close = latest["close"]

    # 3. 嘗試計算 RS rank (需要所有股票; 若 DATA_DIR 無效則跳過)
    rs_rank = None
    data_dir = DATA_DIR or BT_DATA_DIR
    try:
        import pickle
        from concurrent.futures import ThreadPoolExecutor

        files = sorted([fn for fn in os.listdir(data_dir)
                        if fn.endswith(".csv") and not fn.startswith("_")])
        total = len(files)

        # 快取 126 日報酬矩陣 (date x code). 判斷新舊時「排除目標股票自己」,
        # 因為每次分析都會更新目標 CSV, 否則快取永遠失效. 其它檔只在批次更新時變動.
        cache_path = os.path.join(data_dir, "_ret_cache.pkl")
        cache_valid = False
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            others = [fn for fn in files if fn[:-4] != code]
            newest_csv = max((os.path.getmtime(os.path.join(data_dir, fn))
                              for fn in others), default=0)
            cache_valid = cache_mtime >= newest_csv

        if cache_valid:
            note(f"⏳ {code}: 讀取 RS 快取...")
            with open(cache_path, "rb") as f:
                ret_panel = pickle.load(f)
        else:
            # 並行讀 close 列 (只需 126 日報酬, 不用完整 indicators)
            def load_close(fn):
                try:
                    df_c = pd.read_csv(os.path.join(data_dir, fn),
                                       usecols=["date", "close"],
                                       parse_dates=["date"])
                    return fn[:-4], df_c.set_index("date")["close"]
                except Exception:  # noqa: BLE001
                    return fn[:-4], None

            note(f"⏳ {code}: 並行讀取 {total} 檔...")
            closes = {}
            with ThreadPoolExecutor(max_workers=16) as ex:
                futs = [ex.submit(load_close, fn) for fn in files]
                done = 0
                last_pct = -1
                for fut in futs:
                    c_code, series = fut.result()
                    if series is not None:
                        closes[c_code] = series
                    done += 1
                    pct = int(done / total * 100)
                    if pct >= last_pct + 20:
                        last_pct = pct
                        note(f"⏳ {code}: 讀檔 {pct}% ({done}/{total})")

            note(f"⏳ {code}: 計算報酬矩陣...")
            ret_panel = pd.DataFrame(closes).pct_change(126)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(ret_panel, f)
            except Exception:  # noqa: BLE001
                pass

        # 只對需要的「那一天」即時排名 (一列, 極快); 用目標股票最新報酬即時計算,
        # 不依賴快取是否含目標當日資料 → 快取可跨日重用其它股票
        try:
            target_ret = (df["close"].iloc[-1] / df["close"].iloc[-127] - 1
                          if len(df) > 127 else None)
            if target_ret is not None and d in ret_panel.index:
                row = ret_panel.loc[d].dropna()
            elif target_ret is not None and len(ret_panel):
                row = ret_panel.iloc[-1].dropna()  # 快取最後一日近似
            else:
                row = None
            if row is not None and len(row):
                row = row.drop(labels=[code], errors="ignore")
                rs_rank = float((row < target_ret).mean())
        except Exception:  # noqa: BLE001
            rs_rank = None
    except Exception:  # noqa: BLE001
        rs_rank = None

    # 4. 跑策略訊號
    i = len(df) - 1
    results = {}
    for key in ("C", "D"):
        try:
            sig = STRATEGIES[key](df, i, rs_rank=rs_rank)
            results[key] = sig
        except Exception:  # noqa: BLE001
            results[key] = None

    # 5. 趨勢模板檢查
    ma50  = latest.get("ma50",  float("nan"))
    ma150 = latest.get("ma150", float("nan"))
    ma200 = latest.get("ma200", float("nan"))

    def chk(cond): return "✅" if cond else "❌"

    hi52 = df["high"].iloc[max(0, i - 252):i + 1].max()
    lo52 = df["low"].iloc[max(0, i - 252):i + 1].min()

    trend_lines = [
        f"{chk(close > ma50)}  收盤 {close:.1f} > MA50 {ma50:.1f}",
        f"{chk(ma50 > ma150)}  MA50 {ma50:.1f} > MA150 {ma150:.1f}",
        f"{chk(ma150 > ma200)}  MA150 {ma150:.1f} > MA200 {ma200:.1f}",
        f"{chk(close > ma200)}  收盤 > MA200",
        f"{chk(close >= hi52 * 0.75)}  距 52W 高 {close/hi52:.0%}",
        f"{chk(close >= lo52 * 1.25)}  距 52W 低 {close/lo52:.0%}",
        f"   RS排名: {rs_rank:.0%}" if rs_rank is not None else "   RS排名: (計算中)",
    ]

    # 6. 大盤市況
    try:
        regime_tiers = load_regime_tiers()
        tier = regime_tiers.get(d, 2)
        tier_str = {2: "強勢 (全力開倉)", 1: "盤整 (縮手 40%)", 0: "防禦 (停止進場)"}[tier]
        vol_scalars = load_vol_scalars()
        vsca = vol_scalars.get(d, 1.0)
        eff_risk = RISK_PCT * {2: 1.0, 1: 0.6, 0: 0.0}[tier] * vsca
    except Exception:  # noqa: BLE001
        tier_str = "(無法判斷)"
        eff_risk = RISK_PCT

    # 7. 進出場參考
    signal_lines = []
    for key, sig in results.items():
        if sig:
            stop_pct = sig.get("stop_pct", 0.08)
            ref = close          # 以今日收盤估算明日進場
            stop = ref * (1 - stop_pct)
            risk_amt = INIT_CAPITAL * eff_risk
            risk_per_sh = ref - stop
            shares = int(risk_amt / risk_per_sh / 1000) * 1000 if risk_per_sh > 0 else 0
            if shares <= 0:
                shares = max(1, int(risk_amt / risk_per_sh)) if risk_per_sh > 0 else 0
            signal_lines.append(
                f"  策略{key}: ✅ 訊號觸發\n"
                f"    參考進場: {ref:.1f}  停損: {stop:.1f} (-{stop_pct:.0%})\n"
                f"    建議股數: {shares:,}  最長持有: {sig['max_hold']} 日"
            )
        else:
            signal_lines.append(f"  策略{key}: ❌ 條件不符")

    lines = [
        f"📊 {code}  {d:%Y-%m-%d}  收盤 {close:.1f}",
        "",
        "── 趨勢模板 ──",
        *trend_lines,
        "",
        f"── 大盤市況: {tier_str} ──",
        f"   有效風險係數: {eff_risk:.2%}",
        "",
        "── 策略訊號 ──",
        *signal_lines,
        "",
        f"(資料更新: {dl_msg})",
    ]
    return "\n".join(lines)


def analyze_job(chat_id, code):
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        msg_id = send_get_id(chat_id, f"⏳ 分析 {code} 中...")
        # 進度節流: 至少間隔 1.5 秒才 edit 一次, 避免觸發 Telegram 限流
        state = {"last": 0.0}

        def progress(text):
            now = time.time()
            if now - state["last"] >= 1.5:
                state["last"] = now
                edit(chat_id, msg_id, text)

        result = _analyze_stock(code, progress=progress)
        edit(chat_id, msg_id, f"✅ {code} 分析完成")
        send(chat_id, result)
    except Exception as e:  # noqa: BLE001
        send(chat_id, f"❌ 分析失敗: {e}")
    finally:
        _job_lock.release()


# ── 全市場掃描 (符合 C / D 的清單) ──────────────────────────────────────────

SCAN_KEYS = ["PA", "PB", "D", "C"]   # 與 GitHub Actions 每日報告一致


def _scan_all(progress=None, keys=None):
    """掃全市場, 回傳符合任一策略的清單 (標注符合的策略 + 進出場點位)."""
    keys = keys or SCAN_KEYS

    def note(msg):
        if progress:
            progress(msg)

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import importlib
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    for mod in ["strategies", "backtest"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    from strategies import add_indicators, STRATEGIES
    from backtest import (DATA_DIR as BT_DATA_DIR, load_regime_tiers,
                          load_vol_scalars, RISK_PCT, INIT_CAPITAL)

    data_dir = DATA_DIR or BT_DATA_DIR
    names = {}
    uni = os.path.join(data_dir, "universe.json")
    if os.path.exists(uni):
        import json
        with open(uni) as f:
            names = {s["code"]: s["name"] for s in json.load(f)}

    files = sorted([fn for fn in os.listdir(data_dir)
                    if fn.endswith(".csv") and not fn.startswith("_")])
    total = len(files)

    # 並行讀取 + 計算指標 (一次完成, 供 RS 與訊號共用)
    def load_one(fn):
        try:
            df = pd.read_csv(os.path.join(data_dir, fn), parse_dates=["date"])
            df = add_indicators(df).reset_index(drop=True)
            return fn[:-4], df if len(df) >= 210 else None
        except Exception:  # noqa: BLE001
            return fn[:-4], None

    note(f"⏳ 讀取全市場 {total} 檔...")
    data = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(load_one, fn) for fn in files]
        done = 0
        last_pct = -1
        for fut in futs:
            c, df = fut.result()
            if df is not None:
                data[c] = df
            done += 1
            pct = int(done / total * 100)
            if pct >= last_pct + 20:
                last_pct = pct
                note(f"⏳ 讀取全市場 {pct}% ({done}/{total})")

    # RS 排名: 取各檔最新 126 日報酬橫斷面排序
    note("⏳ 計算 RS 排名...")
    ret_last = {}
    for c, df in data.items():
        if len(df) > 127:
            ret_last[c] = df["close"].iloc[-1] / df["close"].iloc[-127] - 1
    ranks = pd.Series(ret_last).rank(pct=True)

    # 大盤市況 → 有效風險係數 (用於估算建議股數)
    try:
        latest_d = max(df["date"].iloc[-1] for df in data.values())
        tier = load_regime_tiers().get(latest_d, 2)
        vsca = load_vol_scalars().get(latest_d, 1.0)
        eff_risk = RISK_PCT * {2: 1.0, 1: 0.6, 0: 0.0}[tier] * vsca
        tier_str = {2: "強勢(全力)", 1: "盤整(縮手)", 0: "防禦(停進)"}[tier]
    except Exception:  # noqa: BLE001
        eff_risk = RISK_PCT
        tier_str = "未知"

    note(f"⏳ 套用策略 {'/'.join(keys)}...")
    matches = []
    for c, df in data.items():
        i = len(df) - 1
        rk = ranks.get(c)
        if rk is None or np.isnan(rk):
            continue
        hit = [k for k in keys if STRATEGIES[k](df, i, rs_rank=rk)]
        if not hit:
            continue
        # 取最嚴格策略的訊號 (keys 依嚴格度排序, 第一個命中即最嚴)
        sig = next(STRATEGIES[k](df, i, rs_rank=rk) for k in keys if k in hit)
        close = df["close"].iloc[-1]
        stop_pct = sig.get("stop_pct", 0.08)
        stop = close * (1 - stop_pct)
        risk_per_sh = close - stop
        risk_amt = INIT_CAPITAL * eff_risk
        shares = (int(risk_amt / risk_per_sh / 1000) * 1000
                  if risk_per_sh > 0 else 0)
        matches.append({
            "code": c, "name": names.get(c, ""), "tag": "+".join(hit),
            "close": close, "stop": stop, "stop_pct": stop_pct,
            "shares": shares, "max_hold": sig["max_hold"], "rs": rk,
            "nhit": len(hit),
        })

    # 排序: 命中策略數多者優先, 再依 RS 由高到低
    matches.sort(key=lambda m: (-m["nhit"], -m["rs"]))

    # 格式化
    if not matches:
        return (f"📋 全市場掃描 ({latest_d:%Y-%m-%d})\n大盤: {tier_str}\n\n"
                f"今日無符合 {'/'.join(keys)} 的標的")

    lines = [
        f"📋 全市場掃描 ({latest_d:%Y-%m-%d})  大盤: {tier_str}",
        f"策略: {'/'.join(keys)}   符合標的: {len(matches)} 檔",
        f"出場規則: 觸停損 或 跌破MA50 或 +20%賣半",
        "",
    ]
    for m in matches:
        nm = f" {m['name']}" if m["name"] else ""
        lines.append(
            f"[{m['tag']}] {m['code']}{nm}  RS{m['rs']:.0%}\n"
            f"   進場 {m['close']:.1f}  停損 {m['stop']:.1f}"
            f"(-{m['stop_pct']:.0%})  {m['shares']:,}股")
    return "\n".join(lines)


def scan_job(chat_id):
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        msg_id = send_get_id(chat_id, "⏳ 全市場掃描中...")
        state = {"last": 0.0}

        def progress(text):
            now = time.time()
            if now - state["last"] >= 1.5:
                state["last"] = now
                edit(chat_id, msg_id, text)

        result = _scan_all(progress=progress)
        edit(chat_id, msg_id, "✅ 掃描完成")
        send(chat_id, result)
    except Exception as e:  # noqa: BLE001
        send(chat_id, f"❌ 掃描失敗: {e}")
    finally:
        _job_lock.release()


# ── 本年度進出清單 ─────────────────────────────────────────────────────────

def _year_trades(progress=None, keys=None):
    """回測 PA/PB/D, 列出「今年進場」的交易 (含已出場與持有中)."""
    keys = keys or SCAN_KEYS

    def note(msg):
        if progress:
            progress(msg)

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import importlib
    import datetime as dt
    import pandas as pd
    for mod in ["strategies", "backtest"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    import backtest as bt
    if DATA_DIR:
        bt.DATA_DIR = DATA_DIR
    from backtest import load_all, run

    note("⏳ 載入全市場資料...")
    data, names = load_all()

    note(f"⏳ 回測 {'/'.join(keys)} (約 1-3 分)...")
    all_trades, _, _ = run(data, names, leverage=1.0, strategies=tuple(keys))

    year = dt.date.today().year
    rows = [t for t in all_trades
            if pd.Timestamp(t["entry_date"]).year == year]
    rows.sort(key=lambda t: pd.Timestamp(t["entry_date"]))

    if not rows:
        return f"📅 {year} 年進出清單\n\n今年無任何進場交易 ({'/'.join(keys)})"

    # 同一檔同一進場可能因分批賣出有多筆 trade, 彙總成「一個部位」
    wins = sum(1 for t in rows if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in rows)
    lines = [
        f"📅 {year} 年進出清單  ({'/'.join(keys)})",
        f"交易筆數: {len(rows)}   勝率: {wins/len(rows):.0%}   "
        f"損益合計: {total_pnl:+,.0f}",
        "",
    ]
    for t in rows:
        ed = pd.Timestamp(t["entry_date"]).strftime("%m/%d")
        xd = pd.Timestamp(t["exit_date"]).strftime("%m/%d")
        nm = names.get(t["code"], "")
        ret_pct = (t["exit"] / t["entry"] - 1)
        emoji = "🟢" if t["pnl"] > 0 else "🔴"
        lines.append(
            f"{emoji} {t['code']} {nm} [{t['strategy']}]\n"
            f"   進 {ed} {t['entry']:.1f} → 出 {xd} {t['exit']:.1f}  "
            f"({ret_pct:+.1%}, {t['r']:+.1f}R)")
    lines.append("")
    lines.append("註: 回測重現的歷史進出, 非即時持倉。+20%會自動賣半故同檔可能多筆。")
    return "\n".join(lines)


def year_trades_job(chat_id):
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        msg_id = send_get_id(chat_id, "⏳ 計算本年度進出清單中...")
        state = {"last": 0.0}

        def progress(text):
            now = time.time()
            if now - state["last"] >= 1.5:
                state["last"] = now
                edit(chat_id, msg_id, text)

        result = _year_trades(progress=progress)
        edit(chat_id, msg_id, "✅ 本年度進出清單完成")
        send(chat_id, result)
    except Exception as e:  # noqa: BLE001
        send(chat_id, f"❌ 計算失敗: {e}")
    finally:
        _job_lock.release()


# ── 策略設計說明 ───────────────────────────────────────────────────────────

STRATEGY_DOC = (
    "📖 策略設計方式 (PA / PB / D)\n"
    "三者同源 Minervini SEPA 趨勢突破引擎, 差在「進場嚴格度」與「出場」。\n"
    "\n"
    "【核心邏輯 — 怎麼抓波段】\n"
    "1. 趨勢模板 (第二階段): 收盤 > MA50 > MA150 > MA200, 且 MA200 向上,\n"
    "   股價距 52 週高點 25% 內、距低點 25% 以上 → 只做「已經在主升段」的強勢股。\n"
    "2. 相對強度 RS: 取近 126 日報酬全市場排名, 只挑前段班 (PA/PB RS≥85,\n"
    "   D RS≥80) → 自動過濾出當下最強的族群 (AI / 記憶體 / 被動元件輪到誰強就抓誰)。\n"
    "3. 量縮回檔 (VCP): 基底波動逐段收斂 + 量能枯竭, 等「洗盤結束」。\n"
    "4. 樞紐突破進場: 帶量突破整理區上緣才進 → 不左側猜底, 只右側順勢。\n"
    "\n"
    "【進出場怎麼設計 — 不會抱上又抱下】\n"
    "• 進場: 突破當下價, PA/PB 限定「距樞紐 6% 內」不追高 (gain_cap),\n"
    "  避免追在波段尾端被巴。\n"
    "• 初始停損: PA/PB/D 約 -8% (隨 VCP 末段低點), 一旦破線立刻認賠出場。\n"
    "• 鎖利: +20% 自動「賣一半」落袋, 停損同步移到成本價 (之後最差打平)。\n"
    "• 移動停損: 獲利 3R 後停損上移到成本; 之後改用 MA50 移動停損,\n"
    "  讓利潤奔跑但「跌破 MA50 就全出」→ 這就是避免抱上又抱下的關鍵。\n"
    "\n"
    "【三檔差異】\n"
    "• PA 最嚴 (PF3.5/勝率57%): 訊號少、最精準, 適合資金大、求穩。\n"
    "• PB 平衡 (PF2.5/勝率51%): 訊號適中, 報酬與頻率折衷。\n"
    "• D  原版 (PF2.0/勝率53%): 訊號最多, 機會多但需嚴守紀律。\n"
    "\n"
    "【大盤濾網】加權指數 < MA60 時全部停止進場, 空手避開系統性下跌。\n"
)


def scheduler_loop():
    """每天 08:00 (本機時間) 自動全市場掃描並推播給授權使用者."""
    if not ALLOWED_CHAT:
        print("未設定 TELEGRAM_CHAT_ID, 略過每日排程")
        return
    import datetime as dt
    while True:
        now = dt.datetime.now()
        nxt = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += dt.timedelta(days=1)
        wait = (nxt - now).total_seconds()
        print(f"下次自動掃描: {nxt:%Y-%m-%d %H:%M}")
        time.sleep(wait)
        try:
            send(ALLOWED_CHAT, "🌅 每日 08:00 自動掃描")
            scan_job(ALLOWED_CHAT)
        except Exception as e:  # noqa: BLE001
            print("排程錯誤:", e)
        time.sleep(60)  # 避免同一分鐘重複觸發


# ── 指令路由 ───────────────────────────────────────────────────────────────

HELP = (
    "📈 策略機器人 — 點下方按鈕直接執行:\n"
    "🔍 全市場掃描     PA/PB/D 符合清單 + 進出場點位\n"
    "📅 本年度進出清單  今年回測進出交易 + 損益\n"
    "📖 策略設計方式    PA/PB/D 設計與進出場邏輯\n"
    "\n"
    "也可打字: /analyze 2330 (個股分析)、/menu (叫出選單)\n"
    "(每日 08:00 自動全市場掃描並推播)"
)

# Telegram inline keyboard: callback_data 對應 handle_callback 的分支
MENU_KEYBOARD = {
    "inline_keyboard": [
        [{"text": "🔍 全市場掃描", "callback_data": "scan"}],
        [{"text": "📅 本年度進出清單", "callback_data": "year"}],
        [{"text": "📖 策略設計方式", "callback_data": "doc"}],
    ]
}


def send_menu(chat_id, text="請選擇功能 👇"):
    try:
        requests.post(f"{API}/sendMessage",
                      json={"chat_id": chat_id, "text": text,
                            "reply_markup": MENU_KEYBOARD}, timeout=30)
    except requests.RequestException as e:
        print("send_menu error:", e)


def answer_callback(callback_id, text=""):
    """回應 callback 讓 Telegram 按鈕的轉圈消失."""
    try:
        requests.post(f"{API}/answerCallbackQuery",
                      data={"callback_query_id": callback_id, "text": text},
                      timeout=15)
    except requests.RequestException:
        pass


def handle_callback(chat_id, data):
    """處理按鈕點擊 (callback_data)."""
    if data == "scan":
        threading.Thread(target=scan_job, args=(chat_id,), daemon=True).start()
    elif data == "year":
        threading.Thread(target=year_trades_job, args=(chat_id,),
                         daemon=True).start()
    elif data == "doc":
        send(chat_id, STRATEGY_DOC)
        send_menu(chat_id)


def handle(chat_id, text):
    parts = text.strip().split()
    cmd = parts[0].lower().lstrip("/").split("@")[0] if parts else ""
    args = parts[1:]

    if cmd in ("start", "help", "menu"):
        send(chat_id, f"你的 chat id: {chat_id}\n\n{HELP}")
        send_menu(chat_id)
    elif cmd == "year":
        threading.Thread(target=year_trades_job, args=(chat_id,),
                         daemon=True).start()
    elif cmd == "doc":
        send(chat_id, STRATEGY_DOC)
    elif cmd == "status":
        dd = DATA_DIR or "(未設定 MM_DATA_DIR)"
        n = (len([f for f in os.listdir(DATA_DIR)
                  if f.endswith(".csv") and not f.startswith("_")])
             if DATA_DIR and os.path.isdir(DATA_DIR) else "?")
        send(chat_id, f"🟢 機器人運作中\n資料夾: {dd}\n股票數: {n}")
    elif cmd == "picks":
        threading.Thread(target=scan_job, args=(chat_id,),
                         daemon=True).start()
    elif cmd == "backtest":
        strats = args[0] if args else "C,D"
        threading.Thread(
            target=heavy_job,
            args=(chat_id, f"回測 {strats}",
                  ["backtest.py", "--strategies", strats,
                   "--data-dir", DATA_DIR or "data", "--leverage", "1"]),
            daemon=True).start()
    elif cmd == "scan":
        threading.Thread(target=scan_job, args=(chat_id,),
                         daemon=True).start()
    elif cmd == "analyze":
        if not args:
            send(chat_id, "用法: /analyze 2330")
            return
        code = args[0].strip().upper()
        threading.Thread(target=analyze_job, args=(chat_id, code),
                         daemon=True).start()
    else:
        send(chat_id, f"未知指令: {text}\n\n{HELP}")


# ── 主迴圈 ─────────────────────────────────────────────────────────────────

def ensure_data():
    """雲端首次啟動時資料夾為空, 自動下載全市場資料 (含 _TWII)."""
    data_dir = DATA_DIR or os.path.join(HERE, "data_adj")
    csvs = ([f for f in os.listdir(data_dir)
             if f.endswith(".csv") and not f.startswith("_")]
            if os.path.isdir(data_dir) else [])
    if len(csvs) >= 100:
        print(f"資料已存在 ({len(csvs)} 檔), 略過下載")
        return
    print("資料夾為空, 開始下載全市場資料 (首次約數分鐘)...")
    env = dict(os.environ)
    env.setdefault("MM_DATA_DIR", data_dir)
    subprocess.run(["python3", "download_all.py"], cwd=HERE, env=env)


def main():
    if not TOKEN:
        raise SystemExit("請先設定環境變數 TELEGRAM_BOT_TOKEN")
    # download_all.py 預設寫入 ./data_adj; 雲端未設 MM_DATA_DIR 時也指向它
    if not DATA_DIR:
        os.environ["MM_DATA_DIR"] = os.path.join(HERE, "data_adj")
    ensure_data()
    print("策略機器人啟動, 等待指令... (Ctrl+C 結束)")
    # 每日 08:00 自動掃描預設關閉 (GitHub Actions 已負責每日推播, 避免重複)。
    # 若要改由本機器人推播, 設環境變數 BOT_DAILY_SCAN=1
    if os.environ.get("BOT_DAILY_SCAN", "").strip() == "1":
        threading.Thread(target=scheduler_loop, daemon=True).start()
    try:
        r0 = requests.get(f"{API}/getUpdates", params={"offset": -1}, timeout=10)
        updates0 = r0.json().get("result", [])
        offset = updates0[-1]["update_id"] + 1 if updates0 else None
    except Exception:  # noqa: BLE001
        offset = None
    while True:
        try:
            r = requests.get(f"{API}/getUpdates",
                             params={"timeout": 30, "offset": offset},
                             timeout=40)
            updates = r.json().get("result", [])
        except (requests.RequestException, ValueError) as e:
            print("poll error:", e)
            time.sleep(3)
            continue

        for u in updates:
            offset = u["update_id"] + 1

            # 按鈕點擊 (callback_query)
            cq = u.get("callback_query")
            if cq:
                chat_id = str(cq["message"]["chat"]["id"])
                answer_callback(cq["id"])
                if ALLOWED_CHAT and chat_id != ALLOWED_CHAT:
                    send(chat_id, "⛔ 未授權的使用者")
                    continue
                print(f"按鈕 from {chat_id}: {cq.get('data')}")
                handle_callback(chat_id, cq.get("data", ""))
                continue

            msg = u.get("message") or u.get("edited_message")
            if not msg or "text" not in msg:
                continue
            chat_id = str(msg["chat"]["id"])
            if ALLOWED_CHAT and chat_id != ALLOWED_CHAT:
                send(chat_id, "⛔ 未授權的使用者")
                print(f"拒絕未授權 chat id: {chat_id}")
                continue
            print(f"指令 from {chat_id}: {msg['text']}")
            handle(chat_id, msg["text"])


if __name__ == "__main__":
    main()
