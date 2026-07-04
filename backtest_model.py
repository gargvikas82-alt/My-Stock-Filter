import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime

# Raw list of 75 high-momentum broad market stocks
RAW_STOCKS = [
    "ADANIENT", "ADANIPORTS", "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA",
    "DIVISLAB", "DLF", "DRREDDY", "EICHERMOT", "GAIL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK",
    "INFY", "IOB", "IRFC", "JINDALSTEL", "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "LTIM", "LT",
    "LUPIN", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", "PIDILITIND", "PFC", "POWERGRID",
    "PNB", "RELIANCE", "RECL", "SBICARD", "SBILIFE", "SBIN", "SUNPHARMA", "SUNTV", "TATACHEM",
    "TATACOMM", "TATAELXSI", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNITDSPR", "VBL", "VEDL", "WIPRO", "ZOMATO", "ZYDUSLIFE"
]

TEST_UNIVERSE = sorted(list(set(RAW_STOCKS)))
yf_symbols = [f"{ticker}.NS" for ticker in TEST_UNIVERSE]

def run_historical_backtest():
    results = []
    
    # GitHub Actions के इनपुट बॉक्स से तारीख उठाना
    target_date_str = os.getenv('TARGET_DATE', '').strip()
    
    if not yf_symbols:
        print("Error: Stock list is empty!")
        return

    try:
        print("📥 Requesting historical data from Yahoo Finance...")
        bulk_data = yf.download(yf_symbols, period="1y", interval="1d", group_by='ticker', threads=True)
        print("✅ Bulk download completed!")
    except Exception as e:
        print(f"Critical error during bulk download: {e}")
        return
    
    print(f"\n🔎 SCANNING FOR FILTER MATCHES...")
    if target_date_str:
        print(f"📅 Target Backtest Date selected by User: {target_date_str}")
    else:
        print("📅 No date entered. Using Latest Available Market Data (Today).")
    print("-" * 75)

    for sym_nse in TEST_UNIVERSE:
        sym_yf = f"{sym_nse}.NS"
        
        if sym_yf not in bulk_data.columns.levels[0]:
            continue
            
        hist = bulk_data[sym_yf].dropna(subset=['Close'])
        
        if len(hist) < 90:
            continue
            
        try:
            # अगर तारीख दी गई है, तो केवल उस दिन तक का ही डेटा प्रोसेस करना
            if target_date_str:
                target_date = pd.to_datetime(target_date_str)
                hist_filtered = hist[hist.index <= target_date]
            else:
                hist_filtered = hist.copy()
                
            if len(hist_filtered) < 65:
                continue
                
            # उस चुनी हुई तारीख का क्लोजिंग प्राइस
            current_price = hist_filtered['Close'].iloc[-1]
            current_date_str = hist_filtered.index[-1].strftime('%Y-%m-%d')
            
            # उस तारीख से ठीक 3 महीने (60 ट्रेडिंग दिन) पहले जाना
            entry_idx = -60  
            entry_price = hist_filtered['Close'].iloc[entry_idx]
            entry_date = hist_filtered.index[entry_idx].strftime('%Y-%m-%d')
            
            # एंट्री के दिन का 50 SMA कैलकुलेट करना
            slice_up_to_entry = hist_filtered.iloc[:len(hist_filtered) + entry_idx]
            if slice_up_to_entry.empty:
                continue
                
            sma_50_at_entry = slice_up_to_entry['Close'].rolling(window=50).mean().iloc[-1]
            if pd.isna(sma_50_at_entry):
                sma_50_at_entry = entry_price
                
            # फ़िल्टर कंडीशन (क्या प्राइस 50 SMA के ऊपर था?)
            is_above_50sma = entry_price >= sma_50_at_entry
            
            if is_above_50sma:
                stock_return_pct = ((current_price - entry_price) / entry_price) * 100
                
                print(f"👉 {sym_nse:<12} | एंट्री: {entry_date} | रिटर्न: {stock_return_pct:.2f}%")
                
                results.append({
                    "Stock": sym_nse,
                    "Entry_Date": entry_date,
                    "Price_At_Entry": round(entry_price, 2),
                    "Price_At_Target_Date": round(current_price, 2),
                    "Strategy_Return_%": round(stock_return_pct, 2),
                    "Above_50SMA_At_Entry": "YES"
                })
        except Exception as e:
            continue

    print("-" * 75)

    df_backtest = pd.DataFrame(results)
    if not df_backtest.empty:
        df_backtest = df_backtest.sort_values(by="Strategy_Return_%", ascending=False)
        df_backtest.to_csv("backtest_results.csv", index=False)
        print(f"\n💾 Success! {len(df_backtest)} matched stocks saved to 'backtest_results.csv'!")
    else:
        print("\n❌ कोई भी शेयर फ़िल्टर कंडीशन पर खरा नहीं उतरा।")

if __name__ == "__main__":
    run_historical_backtest()
