"""Telegram 策略機器人: 在 Telegram 直接下指令 / 點按鈕跑選股 / 回測 / 個股分析.

設計: 純 HTTP 長輪詢 (long-polling), 只需 requests, 不用 webhook / 伺服器.
重活在背景 thread 序列化執行, 不阻塞指令接收.

啟動前設定環境變數 (token 切勿寫進程式或 commit):
    export TELEGRAM_BOT_TOKEN="<BotFather 給的 token>"
    export TELEGRAM_CHAT_ID="<你的 chat id>"
    export MM_DATA_DIR="/path/to/data_adj"

指令 (亦可用 /menu 叫出按鈕選單, 點一下直接執行):
    /menu               叫出按鈕選單
    /scan               全市場掃描 PA/PB/K/L/D/C 清單 + 進出場點位
    /year [C,D]         本年度進出清單 (已平倉交易 + 績效摘要)
    /info               策略設計說明
    /picks              今日選股
    /analyze 2330       分析個股現況 + 進出場價格 (自動更新資料)
    /backtest [C,D]     組合回測 (慢, 數分鐘)
    /chart              取得圖像化儀表板連結 (權益曲線/月損益/R分布/交易清單)
    /futures [M,D,S]    期貨每日訊號 (穀物期貨)
    /refresh            手動更新股價歷史資料 (增量下載)
    /update             git pull 雲端最新策略並重啟
    /c <指令>           叫 Claude Code 依文字指令分析/優化策略/改程式碼 (需本機已裝 claude CLI)
    /status             機器人 / 資料狀態
    /help               指令說明

圖像化儀表板 (webapp.py): 另開終端執行 `python3 webapp.py`, 瀏覽器開
http://<機器IP>:8800 即可看圖表化分析。設定環境變數 MM_WEB_URL 可讓
/chart 直接回傳可點擊連結 (例如內網位址或 ngrok 網址)。

自動同步/排程 (雲端部署用):
    export BOT_AUTO_UPDATE=1   每小時自動 git pull 並重啟套用更新 (預設開啟)
    export BOT_DAILY_SCAN=1    機器人自行於每日 08:00 推播全市場掃描 (預設關閉,
                                GitHub Actions 已負責每日報告時免開)
"""
import csv
import datetime as _dt
import io
import json
import os
import sys
import subprocess
import threading
import time

# 在任何子模組 (strategies/backtest) 匯入 pandas 之前, 先停用與 numpy 2.x 不相容的
# 選用加速套件 (numexpr/bottleneck), 避免掃描時噴 "compiled using NumPy 1.x" 一大串.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_guard
env_guard.apply(verbose=True)

import requests

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
DATA_DIR = os.environ.get("MM_DATA_DIR", "").strip()
WEB_URL = os.environ.get("MM_WEB_URL", "").strip()
WEB_PORT = os.environ.get("MM_WEB_PORT", "8800").strip()
HERE = os.path.dirname(os.path.abspath(__file__))
API = f"https://api.telegram.org/bot{TOKEN}"
YF_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# 與每日報告 (github_scan.py) 一致的掃描策略組合
SCAN_KEYS = ["PA", "PB", "K", "L", "D", "C"]

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


def send_keyboard(chat_id, text, rows):
    """送出帶 inline 按鈕的訊息. rows: list[list[(label, data)]].

    data 以 http(s):// 開頭視為連結按鈕 (url), 否則視為 callback_data。
    """
    def _btn(lbl, data):
        if data.startswith("http://") or data.startswith("https://"):
            return {"text": lbl, "url": data}
        return {"text": lbl, "callback_data": data}
    keyboard = [[_btn(lbl, cb) for lbl, cb in row] for row in rows]
    try:
        requests.post(f"{API}/sendMessage",
                      data={"chat_id": chat_id, "text": text,
                            "reply_markup": json.dumps(
                                {"inline_keyboard": keyboard})},
                      timeout=30)
    except requests.RequestException as e:
        print("send_keyboard error:", e)


def answer_callback(callback_id, text=""):
    """回應 callback query (消除按鈕上的轉圈圈)."""
    try:
        requests.post(f"{API}/answerCallbackQuery",
                      data={"callback_query_id": callback_id, "text": text},
                      timeout=30)
    except requests.RequestException as e:
        print("answer_callback error:", e)


def _web_url():
    return WEB_URL or f"http://localhost:{WEB_PORT}"


def send_menu(chat_id):
    rows = [
        [("📋 全市場掃描", "scan")],
        [("📈 本年度進出清單", "year")],
        [("📊 圖像化儀表板", "chart")],
        [("🌽 期貨每日訊號", "futures")],
        [("🧠 策略設計說明", "info")],
        [("📥 更新股價資料", "refresh")],
        [("🔄 同步最新策略", "update")],
        [("📊 機器人狀態", "status")],
    ]
    send_keyboard(chat_id, "📊 策略機器人選單 — 點選功能直接執行:", rows)


def chart_text():
    url = _web_url()
    note = "" if WEB_URL else "\n(未設定 MM_WEB_URL, 此為本機網址, 手機需用內網IP或ngrok才能開)"
    return (f"📊 圖像化儀表板\n{url}\n\n"
            f"頁面含: 權益曲線 / 月度損益 / R值分布 / 交易清單, 可切換策略組合 (例如 C,D 或 K,L)。\n"
            f"伺服器啟動: python3 webapp.py\n"
            f"首次需先建快取 (整組回測約十餘分鐘): python3 webapp.py --build C,D K,L\n"
            f"之後網頁秒開; 建議每日盤後排程重建快取。{note}")


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
    """下載最新資料, 用策略 PA/PB/K/L/D/C 分析現況, 回傳文字報告.

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
                          RISK_PCT, INIT_CAPITAL, suggested_position)

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

    # 4. 跑策略訊號 (與 /scan 一致的現行輪動)
    i = len(df) - 1
    results = {}
    for key in SCAN_KEYS:
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
    except Exception:  # noqa: BLE001
        tier_str = "(無法判斷)"
        tier, vsca = 2, 1.0

    # 7. 進出場參考 (與 run_sub 一致: 市況×波動有效風險 + minervini 25% 上限)
    signal_lines = []
    for key, sig in results.items():
        if sig:
            ref = close          # 以今日收盤估算明日進場
            pos = suggested_position(INIT_CAPITAL, ref, sig,
                                     tier=tier, vol_scalar=vsca)
            signal_lines.append(
                f"  策略{key}: ✅ 訊號觸發\n"
                f"    參考進場: {ref:.1f}  停損: {pos['stop']:.1f} "
                f"(-{pos['stop_pct']:.0%})\n"
                f"    建議股數: {pos['shares']:,}  最長持有: {sig['max_hold']} 日"
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

def _scan_all(progress=None, keys=None):
    """掃全市場, 回傳符合 keys 任一策略的清單 (標注複合命中 + 進出場點位).

    keys 預設 = SCAN_KEYS (PA/PB/K/L/D/C), 與每日報告一致.
    """
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
    from strategies import add_indicators, STRATEGIES, _d_features
    from backtest import (DATA_DIR as BT_DATA_DIR, load_regime_tiers,
                          load_vol_scalars, RISK_PCT, INIT_CAPITAL,
                          suggested_position)

    data_dir = DATA_DIR or BT_DATA_DIR
    names = {}
    for uni_name in ("_universe_all.json", "universe.json"):
        uni = os.path.join(data_dir, uni_name)
        if os.path.exists(uni):
            with open(uni) as f:
                names = {s["code"]: s.get("name", "") for s in json.load(f)}
            break

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

    # 大盤市況 → 有效風險係數 (用於估算建議股數, 與 run_sub 一致)
    latest_d = max(df["date"].iloc[-1] for df in data.values())
    try:
        tier = load_regime_tiers().get(latest_d, 2)
        vsca = load_vol_scalars().get(latest_d, 1.0)
        tier_str = {2: "強勢(全力)", 1: "盤整(縮手)", 0: "防禦(停進)"}[tier]
    except Exception:  # noqa: BLE001
        tier, vsca = 2, 1.0
        tier_str = "未知"

    note(f"⏳ 套用策略 {'/'.join(keys)}...")
    # 觀察名單門檻: 過趨勢模板 + RS>0.75, 但量縮收斂尚未達標 (差臨門一腳)
    WATCH_RS = 0.75
    WATCH_CONTRACTION = 0.85
    matches = []
    watch = []
    for c, df in data.items():
        i = len(df) - 1
        rk = ranks.get(c)
        if rk is None or np.isnan(rk):
            continue
        sigs = {k: STRATEGIES[k](df, i, rs_rank=rk) for k in keys}
        hit = [k for k in keys if sigs[k]]
        if not hit:
            # 無正式訊號 → 評估是否列入近觸發觀察名單
            if rk >= WATCH_RS:
                feat = _d_features(df, i, rk)
                if feat is not None and feat["contraction"] <= WATCH_CONTRACTION:
                    row = df.iloc[i]
                    prior_high20 = df["high"].iloc[i - 20: i].max()
                    near_bo = row["close"] >= prior_high20 * 0.95
                    watch.append({
                        "code": c, "name": names.get(c, ""), "rs": rk,
                        "contraction": feat["contraction"],
                        "vol_surge": feat["vol_surge"],
                        "breakout": feat["breakout"], "near_bo": near_bo,
                    })
            continue
        sig = sigs[hit[0]]   # keys 已按嚴格度排序, 取最嚴者的停損/持有規則
        close = df["close"].iloc[-1]
        # 與 run_sub 一致: 有效風險(市況×波動) + minervini 25% 上限
        pos = suggested_position(INIT_CAPITAL, close, sig,
                                 tier=tier, vol_scalar=vsca)
        matches.append({
            "code": c, "name": names.get(c, ""), "tag": "+".join(hit),
            "nhit": len(hit), "close": close, "stop": pos["stop"],
            "stop_pct": pos["stop_pct"], "shares": pos["shares"],
            "max_hold": sig["max_hold"], "rs": rk,
        })

    # 排序: 複合命中越多越優先, 再依 RS 由高到低
    matches.sort(key=lambda m: (-m["nhit"], -m["rs"]))
    # 觀察名單: 接近突破者優先, 再依 RS 由高到低, 取前 8 檔
    watch.sort(key=lambda w: (-int(w["near_bo"]), -w["rs"]))

    def _watch_lines():
        out = ["", "▍近觸發觀察名單 (RS>0.75 模板✅ 收縮未達)"]
        if not watch:
            out.append("  (今日無近觸發候選)")
            return out
        for w in watch[:8]:
            nm = f" {w['name']}" if w["name"] else ""
            out.append(
                f"  {w['code']}{nm}  RS{w['rs']:.0%} 收縮{w['contraction']:.2f} "
                f"量{w['vol_surge']:.1f}x 突破{'✅' if w['breakout'] else '❌'}")
        return out

    # 格式化
    if not matches:
        lines = [
            f"📋 全市場掃描 ({latest_d:%Y-%m-%d})  大盤: {tier_str}",
            f"今日無符合 {'/'.join(keys)} 的標的",
        ]
        return "\n".join(lines + _watch_lines())

    lines = [
        f"📋 全市場掃描 ({latest_d:%Y-%m-%d})  大盤: {tier_str}",
        f"策略: {'/'.join(keys)}  符合標的: {len(matches)} 檔 (多命中優先)",
        f"出場規則: 觸停損 或 跌破MA50 或 滿{matches[0]['max_hold']}日",
        "",
    ]
    for m in matches:
        nm = f" {m['name']}" if m["name"] else ""
        lines.append(
            f"[{m['tag']}] {m['code']}{nm}  RS{m['rs']:.0%}\n"
            f"   進場 {m['close']:.1f}  停損 {m['stop']:.1f}"
            f"(-{m['stop_pct']:.0%})  {m['shares']:,}股")
    return "\n".join(lines + _watch_lines())


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


def claude_job(chat_id, prompt):
    """把 /c 後面的文字交給 Claude Code CLI (headless) 執行, 完成後回報結果.

    在本機 repo 目錄下以無人值守模式跑 `claude -p`, 讓它可以直接讀寫程式碼、
    跑回測、commit。因為是無人值守, 用 --dangerously-skip-permissions 跳過互動式
    權限詢問 (否則指令會卡死等不到人按確認) —— 等同把整台機器/repo 的寫入權限
    交給 Telegram 訊息發送者, 務必只在 ALLOWED_CHAT 限制下使用。
    """
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        send(chat_id, f"🤖 Claude Code 執行中:\n{prompt}\n\n(可能需數分鐘~數十分鐘, 完成後會回報結果)")
        t0 = time.time()
        try:
            p = subprocess.run(
                ["claude", "-p", prompt, "--dangerously-skip-permissions"],
                cwd=HERE, env=dict(os.environ),
                capture_output=True, text=True, timeout=1800)
            out = (p.stdout or "").strip()
            if p.returncode != 0:
                out += f"\n[錯誤 returncode={p.returncode}]\n{(p.stderr or '')[-1500:]}"
        except FileNotFoundError:
            out = "[錯誤] 找不到 claude CLI, 請先在此機器安裝 Claude Code 並確保在 PATH 中"
        except subprocess.TimeoutExpired:
            out = "[逾時] 執行超過 30 分鐘已中止"
        mins = (time.time() - t0) / 60
        out = out.strip() or "(無輸出)"
        send(chat_id, f"✅ Claude Code 完成 (耗時 {mins:.1f} 分)\n\n{out[-3500:]}")
    except Exception as e:  # noqa: BLE001
        send(chat_id, f"❌ Claude Code 執行失敗: {e}")
    finally:
        _job_lock.release()


def data_update_job(chat_id):
    """手動觸發股價歷史資料增量更新 (download_all.py)."""
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        send(chat_id, "⏳ 更新股價歷史資料中 (增量, 約數十秒~數分鐘)...")
        out = run_script(["download_all.py"])
        send(chat_id, f"✅ 資料更新完成\n{out[-1000:] if out else ''}")
    except Exception as e:  # noqa: BLE001
        send(chat_id, f"❌ 資料更新失敗: {e}")
    finally:
        _job_lock.release()


# ── 本年度進出清單 (回測本年已平倉交易) ─────────────────────────────────────

def _year_trades(keys=("C", "D"), progress=None):
    """回測指定策略, 回傳本年度 (進場或出場落在今年) 的已平倉交易清單 + 績效."""
    def note(msg):
        if progress:
            progress(msg)

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import importlib
    import numpy as np  # noqa: F401
    import pandas as pd
    for mod in ["strategies", "backtest"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    import backtest as btmod
    if DATA_DIR:
        btmod.DATA_DIR = DATA_DIR
    from backtest import (load_all, load_regime, load_regime_tiers,
                          load_vol_scalars, collect_signals, build_entry_map,
                          run_sub, INIT_CAPITAL)

    note("⏳ 載入全市場資料...")
    data, names = load_all()
    risk_on = load_regime()
    regime_tiers = load_regime_tiers()
    vol_scalars = load_vol_scalars()
    init_eq = INIT_CAPITAL / len(keys)

    year = _dt.date.today().year
    ystart = pd.Timestamp(year, 1, 1)

    all_tr = []
    for k in keys:
        note(f"⏳ 回測策略 {k} (掃描全市場訊號)...")
        sigs = collect_signals(data, k)
        sigs = {d: lst for d, lst in sigs.items() if d in risk_on}
        entry_map = build_entry_map(sigs, data)
        trades, _ = run_sub(data, entry_map, k, 1.0, init_eq,
                            regime_tiers=regime_tiers, vol_scalars=vol_scalars)
        all_tr.extend(trades)

    yt = [t for t in all_tr
          if t["entry_date"] >= ystart or t["exit_date"] >= ystart]
    yt.sort(key=lambda t: t["entry_date"])

    if not yt:
        return (f"📈 {year} 年進出清單 (策略 {'/'.join(keys)})\n\n"
                f"本年度尚無已平倉交易 (持倉中部位請用 /scan 查看)")

    wins = [t for t in yt if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in yt)
    total_r = sum(t["r"] for t in yt)
    win_rate = len(wins) / len(yt)
    avg_r = total_r / len(yt)

    lines = [
        f"📈 {year} 年進出清單 (策略 {'/'.join(keys)})",
        f"已平倉 {len(yt)} 筆  勝率 {win_rate:.0%}  均R {avg_r:+.2f}",
        f"總R {total_r:+.1f}  已實現損益 {total_pnl:+,.0f} 元",
        "(每筆風險固定 1% 本金; 賣半鎖利會列為獨立一筆)",
        "",
    ]
    for t in yt:
        nm = names.get(t["code"], "")
        nm = f" {nm}" if nm else ""
        emoji = "🟢" if t["pnl"] > 0 else "🔴"
        lines.append(
            f"{emoji} [{t['strategy']}] {t['code']}{nm}\n"
            f"   進 {t['entry_date']:%m/%d} {t['entry']:.1f} → "
            f"出 {t['exit_date']:%m/%d} {t['exit']:.1f}  "
            f"{t['r']:+.1f}R ({t['pnl']:+,.0f})")
    return "\n".join(lines)


def year_job(chat_id, strats="C,D"):
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        keys = [k.strip().upper() for k in strats.split(",") if k.strip()]
        msg_id = send_get_id(chat_id, f"⏳ 回測本年度進出 ({'/'.join(keys)})...")
        state = {"last": 0.0}

        def progress(text):
            now = time.time()
            if now - state["last"] >= 1.5:
                state["last"] = now
                edit(chat_id, msg_id, text)

        result = _year_trades(keys=keys, progress=progress)
        edit(chat_id, msg_id, "✅ 本年度進出清單完成")
        send(chat_id, result)
    except Exception as e:  # noqa: BLE001
        send(chat_id, f"❌ 本年度進出失敗: {e}")
    finally:
        _job_lock.release()


# alias for backward-compat callback/menu wiring from origin/main
year_trades_job = year_job


# ── 策略設計說明 ─────────────────────────────────────────────────────────────

STRATEGY_INFO = (
    "🧠 策略設計方式 (Minervini SEPA 量化版)\n"
    "\n"
    "【核心邏輯】只買「第二階段上升趨勢 + 強相對強度 + 波動收縮後突破」的股票，"
    "讓贏家奔跑、用緊停損砍輸家。\n"
    "\n"
    "【進場三道關卡】\n"
    "1. 趨勢模板: 收盤 > MA50 > MA150 > MA200，且 MA200 向上、股價距 52 週高 25% 內、"
    "距 52 週低 25% 以上。\n"
    "2. 相對強度 RS: 近 126 日報酬在全市場排名前段 (PA/PB/D≧85%、C≧80%)。\n"
    "3. 波動收縮突破: ATR5/ATR14 收縮 (能量壓縮)、量縮後帶量突破前 20 日高點。\n"
    "\n"
    "【四個版本 (由嚴到寬)】\n"
    "• PA 最嚴 — 收縮門檻最緊、追高限制最多，訊號最少但獲利因子最高 (PF≈3.4)。\n"
    "• PB 平衡 — 略放寬收縮 (PF≈2.5)。\n"
    "• D  原版 — 收緊突破日漲幅、寬停損 (PF≈2.0)。\n"
    "• C  寬鬆 — RS 門檻較低、不限突破日漲幅，訊號最多 (PF≈1.8)。\n"
    "\n"
    "【出場心法】\n"
    "• 初始停損 8~10% (固定 1% 本金風險換算股數)。\n"
    "• 獲利達 3R → 停損移到成本價 (保本)。\n"
    "• 站上後改用 MA50 移動停損，跌破 MA50 全出。\n"
    "• +20% 賣出一半鎖利 (強勢股可由三週法則跳過)。\n"
    "• 最長持有約 60~90 日。\n"
    "\n"
    "【大盤濾網】三段式市況 (強勢全力 / 盤整縮手 40% / 防禦停止進場)，"
    "搭配波動標量調整曝險。\n"
    "\n"
    "📌 AI / 記憶體 / 被動元件這類強勢股，通常會在「整理收縮後帶量突破」時被 PA~C "
    "捕捉到；緊停損 + MA50 移動停損確保不會抱上又抱下。"
)


def _status_text():
    dd = DATA_DIR or "(未設定 MM_DATA_DIR)"
    n = (len([f for f in os.listdir(DATA_DIR)
              if f.endswith(".csv") and not f.startswith("_")])
         if DATA_DIR and os.path.isdir(DATA_DIR) else "?")
    return f"🟢 機器人運作中\n資料夾: {dd}\n股票數: {n}"


def _refresh_data():
    """增量更新全市場股價資料 (供每日掃描使用); 僅在 MM_AUTO_DOWNLOAD=1 時呼叫."""
    try:
        subprocess.run(["python3", "download_all.py"], cwd=HERE,
                       env=dict(os.environ), capture_output=True,
                       text=True, timeout=3600)
    except Exception as e:  # noqa: BLE001
        print("資料更新失敗:", e)


# alias for backward-compat callback/menu wiring from origin/main (doc command)
STRATEGY_DOC = STRATEGY_INFO


def scheduler_loop():
    """每天 08:00 (本機時間) 自動全市場掃描並推播給授權使用者."""
    if not ALLOWED_CHAT:
        print("未設定 TELEGRAM_CHAT_ID, 略過每日排程")
        return
    import datetime as dt
    auto_dl = os.environ.get("MM_AUTO_DOWNLOAD") == "1"
    while True:
        now = dt.datetime.now()
        nxt = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += dt.timedelta(days=1)
        wait = (nxt - now).total_seconds()
        print(f"下次自動掃描: {nxt:%Y-%m-%d %H:%M}")
        time.sleep(wait)
        try:
            if auto_dl:
                send(ALLOWED_CHAT, "🌅 每日更新股價資料中 (增量)...")
                _refresh_data()
            send(ALLOWED_CHAT, "🌅 每日 08:00 自動掃描")
            scan_job(ALLOWED_CHAT)
        except Exception as e:  # noqa: BLE001
            print("排程錯誤:", e)
        time.sleep(60)  # 避免同一分鐘重複觸發


# ── 指令路由 ───────────────────────────────────────────────────────────────

HELP = (
    "📈 策略機器人指令 (或輸入 /menu 用按鈕):\n"
    "/menu               叫出按鈕選單\n"
    "/scan               全市場掃描 PA/PB/K/L/D/C 清單 + 進出場點位\n"
    "/year [C,D]         本年度進出清單 + 績效摘要\n"
    "/info               策略設計說明\n"
    "/picks              今日選股\n"
    "/analyze 2330       個股分析 + 進出場價格\n"
    "/backtest [C,D]     組合回測 (慢, 數分鐘)\n"
    "/chart              圖像化儀表板連結 (權益曲線/月損益/R分布)\n"
    "/futures [M,D,S]    期貨每日訊號 (穀物期貨)\n"
    "/refresh            更新股價歷史資料 (增量下載)\n"
    "/update             git pull 雲端最新策略並重啟\n"
    "/c <指令>           叫 Claude Code 依指令分析/優化/改程式碼 (例: /c 優化策略)\n"
    "/status             資料 / 機器人狀態\n"
    "/help               顯示此說明\n"
    "(每日 08:00 自動全市場掃描並推播)"
)


def handle(chat_id, text):
    parts = text.strip().split()
    cmd = parts[0].lower().lstrip("/").split("@")[0] if parts else ""
    args = parts[1:]

    if cmd in ("start", "help", "menu"):
        if cmd == "start":
            send(chat_id, f"你的 chat id: {chat_id}\n\n{HELP}")
        send_menu(chat_id)
    elif cmd == "status":
        send(chat_id, _status_text())
    elif cmd in ("info", "doc"):
        send(chat_id, STRATEGY_INFO)
    elif cmd == "year":
        strats = args[0] if args else "C,D"
        threading.Thread(target=year_job, args=(chat_id, strats),
                         daemon=True).start()
    elif cmd == "picks":
        threading.Thread(target=heavy_job,
                         args=(chat_id, "今日選股", ["daily_picks.py"]),
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
    elif cmd == "chart":
        send_keyboard(chat_id, chart_text(), [[("🔗 開啟儀表板", _web_url())]])
    elif cmd == "futures":
        strats = args[0] if args else "M,D,S"
        threading.Thread(target=futures_job, args=(chat_id, strats),
                         daemon=True).start()
    elif cmd == "refresh":
        threading.Thread(target=data_update_job, args=(chat_id,),
                         daemon=True).start()
    elif cmd == "update":
        out = _git_pull()
        send(chat_id, f"🔄 git pull:\n{out[:1000]}")
        if "up to date" not in out.lower():
            send(chat_id, "♻️ 套用更新並重啟中...")
            _reexec()
    elif cmd == "analyze":
        if not args:
            send(chat_id, "用法: /analyze 2330")
            return
        code = args[0].strip().upper()
        threading.Thread(target=analyze_job, args=(chat_id, code),
                         daemon=True).start()
    elif cmd == "c":
        prompt = " ".join(args).strip()
        if not prompt:
            send(chat_id, "用法: /c <你要 Claude Code 做的事>\n例如: /c 優化策略 D 的停損參數")
            return
        threading.Thread(target=claude_job, args=(chat_id, prompt),
                         daemon=True).start()
    else:
        send(chat_id, f"未知指令: {text}\n\n{HELP}")


def handle_callback(chat_id, data):
    """處理 inline 按鈕點擊."""
    if data == "scan":
        threading.Thread(target=scan_job, args=(chat_id,), daemon=True).start()
    elif data == "year":
        threading.Thread(target=year_job, args=(chat_id, "C,D"),
                         daemon=True).start()
    elif data == "chart":
        send_keyboard(chat_id, chart_text(), [[("🔗 開啟儀表板", _web_url())]])
    elif data == "futures":
        threading.Thread(target=futures_job, args=(chat_id,),
                         daemon=True).start()
    elif data in ("info", "doc"):
        send(chat_id, STRATEGY_INFO)
    elif data == "refresh":
        threading.Thread(target=data_update_job, args=(chat_id,),
                         daemon=True).start()
    elif data == "update":
        out = _git_pull()
        send(chat_id, f"🔄 git pull:\n{out[:1000]}")
        if "up to date" not in out.lower():
            send(chat_id, "♻️ 套用更新並重啟中...")
            _reexec()
    elif data == "status":
        send(chat_id, _status_text())
    elif data == "menu":
        send_menu(chat_id)
    else:
        send(chat_id, f"未知按鈕: {data}")


# ── 期貨每日訊號 ─────────────────────────────────────────────────────────────

def futures_job(chat_id, strats="M,D,S"):
    if not _job_lock.acquire(blocking=False):
        send(chat_id, "⏳ 已有任務在執行中, 請待其完成後再試")
        return
    try:
        send(chat_id, "⏳ 更新期貨資料 + 掃描訊號中...")
        run_script(["futures_data.py"])
        out = run_script(["futures_signals.py", "--strategies", strats])
        send(chat_id, f"📈 期貨每日訊號 ({strats})\n\n{out}")
    finally:
        _job_lock.release()


# ── 自動同步雲端最新程式碼 ───────────────────────────────────────────────────

def _git_pull():
    """git pull 最新 main; 回傳輸出文字 (含錯誤)."""
    try:
        out = subprocess.run(["git", "pull", "--rebase", "origin", "main"],
                             cwd=HERE, capture_output=True, text=True,
                             timeout=120)
        return ((out.stdout or "") + (out.stderr or "")).strip() or "(無輸出)"
    except Exception as e:  # noqa: BLE001
        return f"git pull 失敗: {e}"


def _reexec():
    """用最新程式碼重啟自己 (套用 telegram_bot.py / 策略檔更新)."""
    os.execv(sys.executable, [sys.executable] + sys.argv)


def auto_update_loop(interval=3600):
    """每隔 interval 秒自動 git pull; 偵測到更新且無任務在跑時重啟套用."""
    while True:
        time.sleep(interval)
        msg = _git_pull()
        if "up to date" in msg.lower() or "已經是最新" in msg:
            continue
        # 有更新 → 趁沒有任務在跑時重啟 (策略檔每次指令會自動 reload, 重啟確保完整)
        if _job_lock.acquire(blocking=False):
            _job_lock.release()
            if ALLOWED_CHAT:
                send(ALLOWED_CHAT, "🔄 偵測到雲端策略更新，已同步並重啟")
            print("[auto-update] 偵測到更新, 重啟中...\n" + msg[:300])
            _reexec()


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
    # 自動同步雲端最新策略 (預設開啟; 設 BOT_AUTO_UPDATE=0 可關閉)
    if os.environ.get("BOT_AUTO_UPDATE", "1").strip() == "1" \
            and os.path.isdir(os.path.join(HERE, ".git")):
        threading.Thread(target=auto_update_loop, daemon=True).start()
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

            # 按鈕點擊 (callback query)
            cq = u.get("callback_query")
            if cq:
                chat_id = str(cq["message"]["chat"]["id"])
                if ALLOWED_CHAT and chat_id != ALLOWED_CHAT:
                    answer_callback(cq["id"], "⛔ 未授權")
                    continue
                answer_callback(cq["id"])
                data = cq.get("data", "")
                print(f"按鈕 from {chat_id}: {data}")
                handle_callback(chat_id, data)
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
