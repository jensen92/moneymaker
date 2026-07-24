import yfinance as yf
import pandas as pd
import numpy as np
import os
import tempfile
import time
import requests
from datetime import datetime

def check_signal(df):
    if len(df) < 200:
        return None
        
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['TR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    recent_df = df.tail(60).copy().reset_index(drop=True)
    if recent_df.empty:
        return None
        
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    closes = recent_df['close'].values
    sma_200 = recent_df['SMA_200'].values
    atrs = recent_df['ATR'].values
    dates = recent_df['date'].values
    
    active_fvg_top = None
    active_fvg_bot = None
    fvg_date = None
    highest_since_fvg = None
    
    for i in range(2, len(recent_df)):
        if lows[i] > highs[i-2]:
            active_fvg_top = lows[i]
            active_fvg_bot = highs[i-2]
            fvg_date = dates[i]
            highest_since_fvg = highs[i]
            
        if active_fvg_top is not None:
            if lows[i] < active_fvg_bot:
                active_fvg_top = None
                active_fvg_bot = None
                highest_since_fvg = None
            else:
                if highs[i] > highest_since_fvg:
                    highest_since_fvg = highs[i]
                
    last_idx = len(recent_df) - 1
    today_low = lows[last_idx]
    today_close = closes[last_idx]
    today_sma = sma_200[last_idx]
    
    if active_fvg_top is not None and today_close > today_sma:
        if today_low <= active_fvg_top:
            entry = active_fvg_top
            sl = active_fvg_bot
            tp = highest_since_fvg
            
            risk = entry - sl
            reward = tp - entry
            if risk <= 0: return None
            
            rr_ratio = reward / risk
            if rr_ratio < 1.2:
                return None
                
            trend_strength = ((today_close - today_sma) / today_sma) * 100
            return {
                'FVG_Date': str(fvg_date)[:10],
                'Close': today_close,
                'Trend_Strength': round(trend_strength, 2),
                'Entry_Limit': entry,
                'Stop_Loss': sl,
                'Take_Profit': tp
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
    msg = f"🔥 *SMC 台股日線買點掃描 (甜蜜區)* 🔥\n📅 日期: `{scan_date}`\n📊 總計符合: {len(df_res)} 檔 (趨勢 20~60%)\n\n*📌 精選最強趨勢前 10 名:*\n"
    for _, row in preview_df.iterrows():
        sl_pct = abs(row['Entry_Limit'] - row['Stop_Loss']) / row['Entry_Limit'] * 100
        msg += f"• `{row['Ticker']}` ｜ 趨勢: `+{row['Trend_Strength']}%`\n  現價: `{row['Close']:.1f}` ｜ 進: `{row['Entry_Limit']:.1f}` ｜ 損: `{row['Stop_Loss']:.1f}` (-{sl_pct:.1f}%) ｜ 利: `{row['Take_Profit']:.1f}`\n"
        
    msg += "\n📎 *完整清單請見下方 CSV 附件*"
    
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with open(temp_csv.name, 'rb') as f:
        response = requests.post(
            url, 
            data={'chat_id': chat_id, 'caption': msg, 'parse_mode': 'Markdown'},
            files={'document': ('SMC_Daily_Action_List.csv', f)}
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
        # yf.download() returns a multi-index DataFrame if >1 tickers, or single if 1 ticker.
        # But we group_by="ticker", so columns are (Ticker, PriceType)
        try:
            if len(ticker_list) == 1:
                df = data.copy()
            else:
                df = data[full_ticker].copy()
                
            if df.empty or df['Close'].isna().all():
                continue
                
            df = df.dropna(subset=['Close'])
            df = df.reset_index()
            # Rename columns
            df = df.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            
            signal = check_signal(df)
            if signal:
                signal['Ticker'] = ticker
                signal['Date'] = str(df['date'].iloc[-1])[:10]
                results.append(signal)
        except Exception as e:
            continue
    return results

def main():
    print("啟動 SMC 日線選股掃描程式 (雲端版)...")
    t0 = time.time()
    
    # Read tickers
    tickers_file = "tickers.txt"
    if not os.path.exists(tickers_file):
        print(f"找不到 {tickers_file}。請確保檔案存在。")
        return
        
    with open(tickers_file, "r") as f:
        base_tickers = [line.strip() for line in f if line.strip().isdigit()]
        
    print(f"共讀取 {len(base_tickers)} 檔股票代碼。準備從 Yahoo Finance 獲取資料...")
    tw_tickers = [t + ".TW" for t in base_tickers]
    
    # Download .TW
    print("下載上市股票資料 (.TW)...")
    data_tw = yf.download(tw_tickers, period="1y", group_by="ticker", threads=True, progress=False)
    
    # Identify failures (likely OTC)
    failed_base = []
    for t in base_tickers:
        full_t = t + ".TW"
        if full_t not in data_tw.columns.levels[0] if isinstance(data_tw.columns, pd.MultiIndex) else True:
            failed_base.append(t)
        else:
            if data_tw[full_t]['Close'].isna().all():
                failed_base.append(t)
                
    results = process_yf_data(data_tw, tw_tickers)
    
    # Download .TWO
    if failed_base:
        print(f"下載上櫃股票資料 (.TWO) - 共 {len(failed_base)} 檔...")
        two_tickers = [t + ".TWO" for t in failed_base]
        data_two = yf.download(two_tickers, period="1y", group_by="ticker", threads=True, progress=False)
        results_two = process_yf_data(data_two, two_tickers)
        results.extend(results_two)
        
    if not results:
        print("今日無任何股票符合 SMC 買進訊號 (回踩 FVG)。")
        return
        
    df_res = pd.DataFrame(results)
    # 過濾趨勢強度 20~60% 甜蜜區 (回測實證勝率從 54.8% → 59.3%, EV +21%)
    df_res = df_res[(df_res['Trend_Strength'] >= 20) & (df_res['Trend_Strength'] <= 60)]
    # 依照 Trend_Strength (趨勢強度，即乖離率) 由大到小排序，選出最強勢的股票
    df_res = df_res.sort_values('Trend_Strength', ascending=False).reset_index(drop=True)
    
    if df_res.empty:
        print("今日無股票落在趨勢強度 20~60% 甜蜜區內。")
        return
    
    scan_date = df_res['Date'].iloc[0]
    
    print(f"\n掃描完成！耗時 {time.time()-t0:.2f} 秒。最新資料日期: {scan_date}")
    
    push_to_telegram(df_res, scan_date)

if __name__ == '__main__':
    main()
