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

def run_historical_backtest():
    results = []
    
    # GitHub Actions से इनपुट डेट रीड करना
    target_date_str = os.getenv('TARGET_DATE', '').strip()
    
    print(f"\n🔎 STARTING SCAN...")
    if target_date_str:
        print(f"📅 Target Date Selected: {target_date_str}")
    else:
        print("📅 No date entered. Using Latest Available Data (Today).")
    print("-" * 75)

    # डेटा मिसमैच से बचने के लिए हम हर स्टॉक का शुद्ध डेटा अलग से डाउनलोड करेंगे (100% Accurate)
    for sym_nse in TEST_UNIVERSE:
        sym_yf = f"{sym_nse}.NS"
        
        try:
            # 1 साल का डेटा डाउनलोड करना बिना किसी ग्रुपिंग गड़बड़ के
            ticker_obj = yf.Ticker(sym_yf)
            hist = ticker_obj.history(period="1y", interval="1d")
            
            if hist.empty or len(hist) < 90:
                continue
            
            # इंडेक्स को नॉर्मल डेट में बदलना ताकि फ़िल्टर सही काम करे
            hist.index = pd.to_datetime(hist.index).date
                
            if target_date_str:
                target_date = pd.to_datetime(target_date_str).date()
                hist_filtered = hist[hist.index <= target_date]
            else:
                hist_filtered = hist.copy()
                
            if len(hist_filtered) < 65:
                continue
                
            # सटीक लाइव/टारगेट क्लोजिंग प्राइस उठाना
            current_price = hist_filtered['Close'].iloc[-1]
            current_date_str = hist_filtered.index[-1].strftime('%Y-%m-%d')
            
            # ठीक 3 महीने (60 ट्रेडिंग दिन) पहले की रो पर जाना
            entry_idx = -60  
            entry_price = hist_filtered['Close'].iloc[entry_idx]
            entry_date = hist_filtered.index[entry_idx].strftime('%Y-%m-%d')
            
            # एंट्री के समय का 50 SMA कैलकुलेट करना (केवल एंट्री डेट तक का डेटा लेकर)
            slice_up_to_entry = hist_filtered.iloc[:len(hist_filtered) + entry_idx]
            if slice_up_to_entry.empty:
                continue
                
            sma_50_at_entry = slice_up_to_entry['Close'].rolling(window=50).mean().iloc[-1]
            if pd.isna(sma_50_at_entry):
                sma_50_at_entry = entry_price
                
            # फ़िल्टर कंडीशन
            is_above_50sma = entry_price >= sma_50_at_entry
            
            if is_above_50sma:
                stock_return_pct = ((current_price - entry_price) / entry_price) * 100
                
                print(f"✅ Match: {sym_nse:<12} | Entry Date: {entry_date} | Return: {stock_return_pct:.2f}%")
                
                results.append({
                    "Stock": sym_nse,
                    "Entry_Date": entry_date,
                    "Price_At_Entry": round(float(entry_price), 2),
                    "Price_At_Target_Date": round(float(current_price), 2),
                    "Strategy_Return_%": round(float(stock_return_pct), 2),
                    "Above_50SMA_At_Entry": "YES"
                })
        except Exception as e:
            # किसी स्टॉक में एरर आने पर स्किप करें
            continue

    print("-" * 75)

    df_backtest = pd.DataFrame(results)
    if not df_backtest.empty:
        df_backtest = df_backtest.sort_values(by="Strategy_Return_%", ascending=False)
        df_backtest.to_csv("backtest_results.csv", index=False)
        print(f"\n💾 Success! {len(df_backtest)} stocks saved accurately to 'backtest_results.csv'!")
    else:
        print("\n❌ No stocks matched the criteria.")

if __name__ == "__main__":
    run_historical_backtest()
