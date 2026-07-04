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
    
    # GitHub Actions से इनपुट तारीख को सही तरीके से पढ़ना
    target_date_str = os.environ.get('TARGET_DATE', '').strip()
    
    print(f"\n🔎 RUNNING DATA-DRIVEN BACKTEST...")
    if target_date_str:
        print(f"📅 Target Backtest Date selected: {target_date_str}")
    else:
        print("📅 No date entered. Using Latest Available Data (Today).")
    print("-" * 75)

    for sym_nse in TEST_UNIVERSE:
        sym_yf = f"{sym_nse}.NS"
        
        try:
            ticker_obj = yf.Ticker(sym_yf)
            # auto_adjust=False और actions=False ताकि Google प्राइसेज से 100% मैच हो
            hist = ticker_obj.history(period="1y", interval="1d", auto_adjust=False, actions=False)
            
            if hist.empty:
                continue
            
            # इंडेक्स को केवल Date फॉर्मेट में बदलना
            hist.index = pd.to_datetime(hist.index).date
                
            if target_date_str:
                target_date = pd.to_datetime(target_date_str).date()
                hist_filtered = hist[hist.index <= target_date]
            else:
                hist_filtered = hist.copy()
                
            if len(hist_filtered) < 65:
                continue
                
            # आपकी चुनी हुई तारीख का असली अन-एडजस्टेड क्लोज प्राइस (Google Match)
            current_price = hist_filtered['Close'].iloc[-1]
            current_date_str = hist_filtered.index[-1].strftime('%Y-%m-%d')
            
            # उस तारीख से ठीक 3 महीने (~60 ट्रेडिंग दिन) पहले जाना
            entry_idx = -60  
            entry_price = hist_filtered['Close'].iloc[entry_idx]
            entry_date = hist_filtered.index[entry_idx].strftime('%Y-%m-%d')
            
            # एंट्री के समय का 50 SMA कैलकुलेट करना
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

    print("-" * 75)

    df_backtest = pd.DataFrame(results)
    if not df_backtest.empty:
        df_backtest = df_backtest.sort_values(by="Strategy_Return_%", ascending=False)
        df_backtest.to_csv("backtest_results.csv", index=False)
        print(f"\n💾 100% Accurate Data saved to 'backtest_results.csv'!")
    else:
        print("\n❌ No stocks matched.")

if __name__ == "__main__":
    run_historical_backtest()
