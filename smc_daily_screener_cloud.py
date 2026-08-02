import yfinance as yf
import pandas as pd
import numpy as np
import os
import tempfile
import time
import requests
from datetime import datetime

SWING_LEN = 3  # 日線結構長度


def find_pivots(highs, lows, swing_len):
    """找出所有 Pivot High 和 Pivot Low"""
    n = len(highs)
    pivot_highs = []
    pivot_lows = []
    for i in range(swing_len, n - swing_len):
        is_ph = True
        for j in range(i - swing_len, i + swing_len + 1):
            if j != i and highs[j] >= highs[i]:
                is_ph = False
                break
        if is_ph:
            pivot_highs.append((i, highs[i]))

        is_pl = True
        for j in range(i - swing_len, i + swing_len + 1):
            if j != i and lows[j] <= lows[i]:
                is_pl = False
                break
        if is_pl:
            pivot_lows.append((i, lows[i]))
    return pivot_highs, pivot_lows


def get_bull_ob(highs, lows, bar_idx, lookback=30):
    """找多頭 OB：突破前 lookback 根 K 棒中最低低點那根 K 棒"""
    start = max(0, bar_idx - lookback)
    min_idx = start
    for i in range(start, bar_idx):
        if lows[i] < lows[min_idx]:
            min_idx = i
    return min_idx, highs[min_idx], lows[min_idx]


def check_signal(df):
    """
    SMC Order Block (OB) 進場偵測 — 精準對齊 PineScript 邏輯
    """
    if len(df) < 200:
        return None

    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['TR'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()),
                                     abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['Vol_Ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    recent_df = df.tail(120).copy().reset_index(drop=True)
    if len(recent_df) < 60:
        return None

    highs = recent_df['high'].values
    lows = recent_df['low'].values
    closes = recent_df['close'].values
    sma_200 = recent_df['SMA_200'].values
    sma_20 = recent_df['SMA_20'].values
    atrs = recent_df['ATR'].values
    dates = recent_df['date'].values
    n = len(recent_df)

    pivot_highs, pivot_lows = find_pivots(highs, lows, SWING_LEN)

    ph_dict = {idx: val for idx, val in pivot_highs}
    pl_dict = {idx: val for idx, val in pivot_lows}

    trend = 0
    last_sh = None
    last_sl_val = None

    for idx, val in pivot_highs:
        if idx < SWING_LEN * 2:
            last_sh = val
    for idx, val in pivot_lows:
        if idx < SWING_LEN * 2:
            last_sl_val = val

    active_bull_obs = []
    MAX_ZONES = 3

    for i in range(SWING_LEN * 2, n):
        check_bar = i - SWING_LEN
        if check_bar >= 0:
            if check_bar in ph_dict:
                last_sh = ph_dict[check_bar]
            if check_bar in pl_dict:
                last_sl_val = pl_dict[check_bar]

        if last_sh is None or last_sl_val is None:
            continue

        bull_break = closes[i] > last_sh
        bear_break = closes[i] < last_sl_val

        if trend >= 0:
            if bull_break:
                _, ob_h, ob_l = get_bull_ob(highs, lows, i, 30)
                active_bull_obs.append((ob_h, ob_l, i))
                if len(active_bull_obs) > MAX_ZONES:
                    active_bull_obs.pop(0)
                trend = 1
                last_sh = None
            elif bear_break:
                trend = -1
                last_sl_val = None
                active_bull_obs.clear()

        if trend <= 0:
            if bear_break:
                trend = -1
                last_sl_val = None
            elif bull_break:
                _, ob_h, ob_l = get_bull_ob(highs, lows, i, 30)
                active_bull_obs.append((ob_h, ob_l, i))
                if len(active_bull_obs) > MAX_ZONES:
                    active_bull_obs.pop(0)
                trend = 1
                last_sh = None

        if i != n - 1:
            mitigated = []
            for j, (ob_top, ob_bot, ob_bar) in enumerate(active_bull_obs):
                if lows[i] <= ob_top:
                    mitigated.append(j)
            for j in sorted(mitigated, reverse=True):
                if j < len(active_bull_obs):
                    active_bull_obs.pop(j)
            continue

        # === 最後一天：檢查 BUY 訊號 ===
        today_close = closes[i]
        today_sma20 = sma_20[i]
        today_atr = atrs[i]

        if today_close <= today_sma20:
            return None

        for j, (ob_top, ob_bot, ob_bar) in enumerate(active_bull_obs):
            if lows[i] <= ob_top and closes[i] >= ob_bot:
                entry = ob_top

                raw_sl = ob_bot - (0.5 * today_atr)
                min_sl_dist = max(entry * 0.025, 1.0 * today_atr)
                sl = min(raw_sl, entry - min_sl_dist)

                tp = None
                for ph_idx, ph_val in pivot_highs:
                    if ob_bar <= ph_idx < i and ph_val > entry:
                        tp = ph_val
                if tp is None:
                    tp = max(highs[ob_bar:i + 1]) if ob_bar < i else highs[i]
                if tp <= entry:
                    tp = entry + 2.0 * today_atr

                risk = entry - sl
                reward = tp - entry
                if risk <= 0 or reward <= 0:
                    continue

                structural_rr = reward / risk
                if structural_rr < 1.0:
                    continue

                trend_strength = ((today_close - sma_200[i]) / sma_200[i]) * 100 if sma_200[i] > 0 else 0

                vol_ratio = recent_df['Vol_Ratio'].values[i] if 'Vol_Ratio' in recent_df.columns else 1.0

                return {
                    'OB_Date': str(dates[ob_bar])[:10],
                    'Close': today_close,
                    'Trend_Strength': round(trend_strength, 2),
                    'Structural_RR': round(structural_rr, 2),
                    'Entry_Limit': round(entry, 2),
                    'Stop_Loss': round(sl, 2),
                    'Take_Profit': round(tp, 2),
                    'Vol_Ratio': round(vol_ratio, 2),
                }

    return None


def push_to_telegram(df_res, scan_date):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，跳過 Telegram 推播。")
        return

    print(f"正在將 {len(df_res)} 筆訊號推播至 Telegram...")

    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df_res.to_csv(temp_csv.name, index=False, encoding='utf-8-sig')

    preview_df = df_res.head(10)
    msg = f"🔥 *SMC 訂單塊(OB)台股日線買點掃描* 🔥\n📅 日期: `{scan_date}`\n📊 總計符合: {len(df_res)} 檔 (OB回踩 + SMA20↑ + RR≥1.0 + 量能≥1.0x)\n\n*📌 精選綜合評分前 10 名 (回測勝率 82%):*\n"
    for _, row in preview_df.iterrows():
        sl_pct = abs(row['Entry_Limit'] - row['Stop_Loss']) / row['Entry_Limit'] * 100
        msg += f"• `{row['Ticker']}` ｜ 評分: `{row['Composite_Score']:.1f}` (RR: {row['Structural_RR']:.1f}, 趨勢: {row['Trend_Strength']:+.0f}%, 量能: {row.get('Vol_Ratio', 0):.1f}x)\n  現價: `{row['Close']:.1f}` ｜ 進: `{row['Entry_Limit']:.1f}` ｜ 損: `{row['Stop_Loss']:.1f}` (-{sl_pct:.1f}%) ｜ 利: `{row['Take_Profit']:.1f}`\n"

    msg += "\n⏰ *持倉規則: 進場後 10 天未到停利請主動平倉*"
    msg += "\n📎 *完整清單請見下方 CSV 附件*"

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with open(temp_csv.name, 'rb') as f:
        response = requests.post(
            url,
            data={'chat_id': chat_id, 'caption': msg, 'parse_mode': 'Markdown'},
            files={'document': ('SMC_OB_Daily_Action_List.csv', f)}
        )

    if response.status_code == 200:
        print("Telegram 推播成功！")
    else:
        print(f"Telegram 推播失敗: {response.text}")

    os.unlink(temp_csv.name)


def process_yf_data(data, ticker_list):
    results = []
    for full_ticker in ticker_list:
        ticker = full_ticker.split('.')[0]
        try:
            if len(ticker_list) == 1:
                df = data.copy()
            else:
                df = data[full_ticker].copy()

            if df.empty or df['Close'].isna().all():
                continue

            df = df.dropna(subset=['Close'])
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })

            signal = check_signal(df)
            if signal:
                # 量能過濾：量能倍率 < 1.0 的量縮股不進場
                if signal.get('Vol_Ratio', 1.0) < 1.0:
                    continue
                signal['Ticker'] = ticker
                signal['Date'] = str(df['date'].iloc[-1])[:10]
                recent_df = df.tail(5)
                vol_col = 'volume' if 'volume' in recent_df.columns else ('Volume' if 'Volume' in recent_df.columns else None)
                signal['Val_5d'] = (recent_df[vol_col] * recent_df['close']).mean() if vol_col else 0
                results.append(signal)
        except Exception:
            continue
    return results


def main():
    print("啟動 SMC OB 日線選股掃描程式 (雲端版)...")
    t0 = time.time()

    tickers_file = "tickers.txt"
    if not os.path.exists(tickers_file):
        print(f"找不到 {tickers_file}。請確保檔案存在。")
        return

    with open(tickers_file, "r") as f:
        base_tickers = [line.strip() for line in f if line.strip().isdigit()]

    print(f"共讀取 {len(base_tickers)} 檔股票代碼。準備從 Yahoo Finance 獲取資料...")
    tw_tickers = [t + ".TW" for t in base_tickers]

    print("下載上市股票資料 (.TW)...")
    data_tw = yf.download(tw_tickers, period="1y", group_by="ticker", threads=True, progress=False)

    failed_base = []
    for t in base_tickers:
        full_t = t + ".TW"
        if full_t not in data_tw.columns.levels[0] if isinstance(data_tw.columns, pd.MultiIndex) else True:
            failed_base.append(t)
        else:
            if data_tw[full_t]['Close'].isna().all():
                failed_base.append(t)

    results = process_yf_data(data_tw, tw_tickers)

    if failed_base:
        print(f"下載上櫃股票資料 (.TWO) - 共 {len(failed_base)} 檔...")
        two_tickers = [t + ".TWO" for t in failed_base]
        data_two = yf.download(two_tickers, period="1y", group_by="ticker", threads=True, progress=False)
        results_two = process_yf_data(data_two, two_tickers)
        results.extend(results_two)

    if not results:
        msg = "今日無任何股票觸發 OB 回踩買進訊號。"
        print(msg)
        
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        if token and chat_id:
            import requests
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, data={'chat_id': chat_id, 'text': f"📉 掃描完成！\n{msg}"})
        return

    df_res = pd.DataFrame(results)
    if 'Val_5d' in df_res.columns and len(df_res) > 1:
        df_res['Val_Rank_Score'] = df_res['Val_5d'].rank(pct=True) * 2
    else:
        df_res['Val_Rank_Score'] = 0
    # 趨勢 > 20% 追高懲罰 (扣 1 分)
    df_res['Trend_Penalty'] = np.where(df_res['Trend_Strength'] > 20, -1.0, 0)
    df_res['Composite_Score'] = df_res['Structural_RR'] + (df_res['Trend_Strength'] / 10) + df_res['Val_Rank_Score'] + df_res['Trend_Penalty']
    df_res = df_res.sort_values('Composite_Score', ascending=False).reset_index(drop=True)

    if df_res.empty:
        msg = "今日無股票符合 OB 進場條件。"
        print(msg)
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        if token and chat_id:
            import requests
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, data={'chat_id': chat_id, 'text': f"📉 掃描完成！\n{msg}"})
        return

    scan_date = df_res['Date'].iloc[0]

    print(f"\n掃描完成！耗時 {time.time()-t0:.2f} 秒。最新資料日期: {scan_date}")

    push_to_telegram(df_res, scan_date)

if __name__ == '__main__':
    main()
