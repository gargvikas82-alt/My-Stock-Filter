import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Let's expand our list to include mid/small caps alongside mega-caps to give the filters something to dig into!
TEST_UNIVERSE = ["PTCIL", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT", "TATAMOTORS", "SBIN", "BHARTIARTL"]
yf_symbols = [f"{sym}.NS" for sym in TEST_UNIVERSE]

print("🚀 Starting Historical Backtest Simulation (3-Month Lookback)...")

def run_historical_backtest():
    results = []
    
    for sym_nse, sym_yf in zip(TEST_UNIVERSE, yf_symbols):
        print(f"Analyzing historical timeline for {sym_nse}...")
        ticker = yf.Ticker(sym_yf)
        
        try:
            # Grab historical daily data
            hist = ticker.history(period="6mo", interval="1d")
            info = ticker.info
        except Exception as e:
            print(f"Skipping {sym_nse}: {e}")
            continue
            
        if len(hist) < 90:
            continue
            
        # Pinpoint the entry date (exactly 3 months / 90 days ago)
        entry_idx = -60  # Approx 60 trading sessions ago (~3 calendar months)
        entry_price = hist['Close'].iloc[entry_idx]
        current_price = hist['Close'].iloc[-1]
        
        # Fundamental snapshot metrics (simulated baseline floor)
        market_cap_crore = info.get('marketCap', 0) / 10000000
        # Check if it meets a relaxed baseline backtest threshold to find historical candidates
        promoter_pct = info.get('heldPercentByInsiders', 0.60) * 100 
        
        # Calculate historical indicators at the time of entry
        slice_up_to_entry = hist.iloc[:len(hist) + entry_idx]
        if slice_up_to_entry.empty:
            continue
            
        # 50 SMA at entry time
        sma_50_at_entry = slice_up_to_entry['Close'].rolling(window=50).mean().iloc[-1]
        if pd.isna(sma_50_at_entry):
            sma_50_at_entry = entry_price
            
        # Price performance calculation
        stock_return_pct = ((current_price - entry_price) / entry_price) * 100
        
        results.append({
            "Stock": sym_nse,
            "Market_Cap_Cr": round(market_cap_crore, 2),
            "Promoter_Holding_%": round(promoter_pct, 2),
            "Price_3Mo_Ago": round(entry_price, 2),
            "Price_Today": round(current_price, 2),
            "Strategy_Return_%": round(stock_return_pct, 2),
            "Above_50SMA_At_Entry": "YES" if entry_price >= sma_50_at_entry else "NO"
        })

    # Compile the backtest ledger
    df_backtest = pd.DataFrame(results)
    if not df_backtest.empty:
        # Filter down to the setups that were structurally healthy (above 50 SMA) at entry
        df_survivors = df_backtest[df_backtest['Above_50SMA_At_Entry'] == "YES"].sort_values(by="Strategy_Return_%", ascending=False)
        
        print("\n📊 BACKTEST PERFORMANCE SUMMARY (Past 3 Months):")
        print(df_survivors[['Stock', 'Price_3Mo_Ago', 'Price_Today', 'Strategy_Return_%']].to_string(index=False))
        
        # Save it to a spreadsheet
        df_survivors.to_csv("backtest_results.csv", index=False)
        print("\n💾 Backtest ledger successfully saved to 'backtest_results.csv'!")
    else:
        print("\nNo historical data generated.")

if __name__ == "__main__":
    run_historical_backtest()
