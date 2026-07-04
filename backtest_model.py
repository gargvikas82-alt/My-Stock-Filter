import os
import pandas as pd
import yfinance as yf
import numpy as np

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
    
    # 🎯 यहाँ हमने सीधे तारीख फिक्स कर दी है ताकि GitHub Actions की कोई सेटिंग इसे रोक न सके!
    target_date_str = "2026-06-30"
    
    print(f"\n🔎 RUNNING FIXED BACKTEST FOR TUESDAY: {target_date_str}")
    print("-" * 75)

    for sym_nse in TEST_UNIVERSE:
        sym_yf = f"{sym_nse}.NS"
        
        try:
            ticker_obj = yf.Ticker(sym_yf)
            hist = ticker_obj.history(period="1y", interval="1d", auto_adjust=False, actions=False)
            
            if hist.empty:
                continue
            
            hist.index = pd.to_datetime(hist.index).date
            
            # तारीख के हिसाब से फिल्टर
            target_date = pd.to_datetime(target_date_str).date()
            hist_filtered = hist[hist.index <= target_date]
                
            if len(hist_filtered) < 65:
                continue
                
            # 30 जून का असली क्लोज प्राइस
            current_price = hist_filtered['Close'].iloc[-1]
            
            # ठीक 60 ट्रेडिंग दिन पहले जाना (मार्च के आखरी या अप्रैल की शुरुआत की डेट आएगी)
            entry_idx = -60  
            entry_price = hist_filtered['Close'].iloc[entry_idx]
            entry_date = hist_filtered.index[entry_idx].strftime('%Y-%m-%d')
            
            # 50 SMA कैलकुलेशन
            slice_up_to_entry = hist_filtered.iloc[:len(hist_filtered) + entry_idx]
            if slice_up_to_entry.empty:
                continue
                
            sma_50_at_entry = slice_up_to_entry['Close'].rolling(window=50).mean().iloc[-1]
            if pd.isna(sma_50_at_entry):
                sma_50_at_entry = entry_price
                
            if entry_price >= sma_50_at_entry:
                stock_return_pct = ((current_price - entry_price) / entry_price) * 100
                
                results.append({
                    "Stock": sym_nse,
                    "Entry_Date": entry_date,
                    "Price_At_Entry": round(float(entry_price), 2),
                    "Price_At_Target_Date": round(float(current_price), 2),
                    "Strategy_Return_%": round(float(stock_return_pct), 2),
                    "Above_50SMA_At_Entry": "YES"
                })
        except Exception as e:
            continue

    df_backtest = pd.DataFrame(results)
    if not df_backtest.empty:
        df_backtest = df_backtest.sort_values(by="Strategy_Return_%", ascending=False)
        df_backtest.to_csv("backtest_results.csv", index=False)
        print(f"\n💾 Results successfully updated for {target_date_str}!")
    else:
        print("\n❌ No stocks matched.")

if __name__ == "__main__":
    run_historical_backtest()
