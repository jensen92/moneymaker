import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import time
import requests
import tempfile

def check_signal(df):
    if len(df) < 200:
        return None
        
    # 計算 200 日均線與 ATR
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['TR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 只取最後 60 天來尋找近期有效的 FVG
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
    
    for i in range(2, len(recent_df)):
        # 尋找看多 FVG (Bullish FVG)
        if lows[i] > highs[i-2]:
            active_fvg_top = lows[i]
            active_fvg_bot = highs[i-2]
            fvg_date = dates[i]
            
        # 若 FVG 被完全向下跌破 (低點低於 FVG 底部)，則失效
        if active_fvg_top is not None:
            if lows[i] < active_fvg_bot:
                active_fvg_top = None
                active_fvg_bot = None
                
    # 檢查最後一天 (最新交易日) 是否觸發訊號
    last_idx = len(recent_df) - 1
    today_low = lows[last_idx]
    today_close = closes[last_idx]
    today_sma = sma_200[last_idx]
    today_atr = atrs[last_idx]
    
    # 條件 1: 目前有存活的 FVG
    # 條件 2: 股價在年線之上 (多頭趨勢)
    if active_fvg_top is not None and today_close > today_sma:
        # 條件 3: 今天的低點剛好踩進 FVG 區域 (<= FVG_Top)
        if today_low <= active_fvg_top:
            entry = active_fvg_top
            sl = entry - (today_atr * 2.5)
            tp = entry + (today_atr * 5.0)
            return {
                'FVG_Date': str(fvg_date)[:10],
                'Close': today_close,
                'Entry_Limit': entry,
                'Stop_Loss': sl,
                'Take_Profit': tp
            }
    return None

def process_file(file_path):
    ticker = os.path.basename(file_path).replace('.csv', '')
    if not ticker.isdigit(): return None  # 排除非個股檔案如 _TWII.csv
    
    try:
        df = pd.read_csv(file_path)
        if len(df) < 200: return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        signal = check_signal(df)
        if signal:
            signal['Ticker'] = ticker
            signal['Date'] = str(df['date'].iloc[-1])[:10]
            return signal
    except Exception as e:
        return None
    return None

def push_to_telegram(df_res, scan_date):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，跳過 Telegram 推播。")
        return
        
    print(f"正在將 {len(df_res)} 筆訊號推播至 Telegram...")
    
    # 建立 CSV 檔案
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df_res.to_csv(temp_csv.name, index=False, encoding='utf-8-sig')
    
    # 取前 10 檔作為預覽訊息
    preview_df = df_res.head(10)
    msg = f"🔥 *SMC 台股日線買點掃描* 🔥\n📅 日期: `{scan_date}`\n📊 總計符合: {len(df_res)} 檔\n\n*📌 精選前 10 檔預覽:*\n"
    for _, row in preview_df.iterrows():
        msg += f"• `{row['Ticker']}` ｜ 進場: `{row['Entry_Limit']:.1f}` ｜ 停損: `{row['Stop_Loss']:.1f}`\n"
        
    msg += "\n📎 *完整清單請見下方 CSV 附件*"
    
    # 傳送文件與訊息
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

def main():
    data_dir = "/Users/jensenchen/money maker/adjusted_data"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print("啟動 SMC 日線選股掃描程式...")
    print(f"正在分析 {len(files)} 檔股票的最新技術型態 (多進程平行運算中)...")
    
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_file, files):
            if res is not None:
                results.append(res)
                
    if not results:
        print(f"掃描完成！耗時 {time.time()-t0:.2f} 秒。")
        print("今日無任何股票符合 SMC 買進訊號 (回踩 FVG)。")
        return
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('Ticker').reset_index(drop=True)
    scan_date = df_res['Date'].iloc[0]
    
    print(f"\n掃描完成！耗時 {time.time()-t0:.2f} 秒。最新資料日期: {scan_date}")
    print("\n" + "="*75)
    print("🔥 今日觸發 SMC 買點的飆股候選清單 🔥")
    print("="*75)
    print(f"{'代號':<8} | {'最新收盤價':<10} | {'建議進場(FVG)':<12} | {'停損(2.5ATR)':<12} | {'停利(5ATR)':<12}")
    print("-" * 75)
    
    for _, row in df_res.iterrows():
        print(f"{row['Ticker']:<10} | {row['Close']:<15.2f} | {row['Entry_Limit']:<17.2f} | {row['Stop_Loss']:<16.2f} | {row['Take_Profit']:<12.2f}")
        
    print("-" * 75)
    print(f"總計掃描出 {len(df_res)} 檔具備潛在飆股型態的標的。")
    print("👉 建議操作：明日開盤前，可於「建議進場價(FVG)」掛限價買單等待價格回踩成交。")
    
    # 推播到 Telegram
    push_to_telegram(df_res, scan_date)

if __name__ == '__main__':
    main()
