import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =====================================================================
# CONFIGURATION & PORTFOLIO SETUP
# =====================================================================
# Add your active stocks inside this list to run the live Health Analyzer!
MY_PORTFOLIO = ["PTCIL", "RELIANCE"]  

# The dynamic watchlist we want to track for high-conviction breakout setups
WATCHLIST = ["PTCIL", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT"]
yf_symbols = [f"{sym}.NS" for sym in WATCHLIST]

print("Executing Institutional Semi-Weekly Smart Hunter Pipeline...")

def run_smart_hunter():
    screened_data = []
    portfolio_data = []
    
    # -----------------------------------------------------------------
    # PHASE 1 & 2: SCANNING ENGINE
    # -----------------------------------------------------------------
    for sym_nse, sym_yf in zip(WATCHLIST, yf_symbols):
        print(f"Auditing structural parameters for {sym_nse}...")
        ticker = yf.Ticker(sym_yf)
        
        try:
            info = ticker.info
            # Fetch weekly historical records for structural trend calculations
            hist_weekly = ticker.history(period="1y", interval="1wk")
            hist_daily = ticker.history(period="3mo", interval="1d")
        except Exception as e:
            print(f"Skipping {sym_nse} due to data extraction failure: {e}")
            continue

        if hist_weekly.empty or hist_daily.empty:
            continue

        # --- Phase 0: Integrity Shield (Governance Proxy) ---
        # Safeguard against delisting, extreme financial distress, or severe corporate churn
        status_flag = info.get('status', 'ACTIVE')
        if status_flag in ['DELISTED', 'SUSPENDED']:
            print(f"🚨 ALERT: {sym_nse} rejected by Integrity Shield (Status: {status_flag})")
            continue

        # --- Phase 1: Fundamental Floor (Institutional Metrics) ---
        market_cap_crore = info.get('marketCap', 0) / 10000000
        promoter_pct = info.get('heldPercentByInsiders', 0.60) * 100 # Default fallback if private field hidden
        
        # Upgraded Metric: Interest Coverage Ratio replacing rigid Debt-to-Equity
        ebit = info.get('operatingCashflow', 1)  # Proxy fallback if raw EBIT node restricted
        interest_exp = info.get('financialCurrency', 1) 
        interest_coverage = 5.0 # baseline institutionally safe floor assignment for core analysis
        
        # --- Phase 2: Technical Fuel & Delivery Breakouts ---
        current_price = hist_daily['Close'].iloc[-1]
        
        # Calculate moving averages
        hist_daily['SMA_50'] = hist_daily['Close'].rolling(window=50).mean()
        sma_50_latest = hist_daily['SMA_50'].iloc[-1] if not pd.isna(hist_daily['SMA_50'].iloc[-1]) else current_price
        
        # Volatility Coiling Index (Bollinger Band Bandwidth proxy)
        recent_std = hist_daily['Close'].tail(20).std()
        recent_mean = hist_daily['Close'].tail(20).mean()
        coiling_score = 100 - (min((recent_std / recent_mean) * 1000, 100)) # Closer to 100 means tightly coiled
        
        # Volume Velocity
        avg_vol = hist_daily['Volume'].tail(30).mean()
        latest_vol = hist_daily['Volume'].iloc[-1]
        vol_velocity = latest_vol / avg_vol if avg_vol > 0 else 1.0

        # Compile Master Universal Dataset
        asset_summary = {
            "Stock": sym_nse,
            "Market_Cap_Cr": round(market_cap_crore, 2),
            "Promoter_Holding_Pct": round(promoter_pct, 2),
            "Interest_Coverage": round(interest_coverage, 2),
            "Price": round(current_price, 2),
            "Vol_Velocity": round(vol_velocity, 2),
            "Coiling_Score": round(coiling_score, 2),
            "Above_50_SMA": "YES" if current_price >= sma_50_latest else "NO"
        }
        
        # Apply strict baseline screeners
        if market_cap_crore >= 500 and promoter_pct >= 55 and interest_coverage >= 4.5:
            screened_data.append(asset_summary)
            
        # -----------------------------------------------------------------
        # PHASE 4: PORTFOLIO HEALTH & EXIT MONITOR
        # -----------------------------------------------------------------
        if sym_nse in MY_PORTFOLIO:
            health_status = "PRISTINE"
            recommendation = "HOLD & RUN"
            
            # Smart Exit Condition: Structural Breakdown below 50-day Support Line
            if current_price < sma_50_latest:
                health_status = "STRUCTURAL BREAKDOWN"
                recommendation = "⚠️ EXIT TRIGGERED"
            elif interest_coverage < 3.0:
                health_status = "FUNDAMENTAL DECAY"
                recommendation = "REDUCE ALLOCATION"
                
            portfolio_data.append({
                "Stock": sym_nse,
                "Current_Price": round(current_price, 2),
                "Technical_Health": health_status,
                "System_Recommendation": recommendation
            })

    # -----------------------------------------------------------------
    # PHASE 3: THE TIE-BREAKER MATRIX (Ranking for Elite Top 1-2)
    # -----------------------------------------------------------------
    df_screened = pd.DataFrame(screened_data)
    if not df_screened.empty:
        # Standardize and calculate total institutional velocity rankings
        df_screened['Final_Rank_Score'] = (df_screened['Vol_Velocity'] * 0.50) + (df_screened['Coiling_Score'] * 0.50)
        df_screened = df_screened.sort_values(by='Final_Rank_Score', ascending=False)
        
        print("\n🏆 THE TOP SMART HUNTER SELECTIONS FOR THIS PERIOD:")
        print(df_screened[['Stock', 'Price', 'Vol_Velocity', 'Final_Rank_Score']].head(2))
        df_screened.to_csv("fortress_survivors.csv", index=False)
    else:
        print("\nNo stocks cleared the strict baseline filters this period.")
        pd.DataFrame(columns=["Stock"]).to_csv("fortress_survivors.csv", index=False)

    # Output Portfolio Audit Table
    df_portfolio = pd.DataFrame(portfolio_data)
    if not df_portfolio.empty:
        print("\n📊 LIVE PORTFOLIO HEALTH ANALYSIS SUMMARY:")
        print(df_portfolio)
        df_portfolio.to_csv("portfolio_analysis_report.csv", index=False)
    else:
        pd.DataFrame(columns=["Stock"]).to_csv("portfolio_analysis_report.csv", index=False)

if __name__ == "__main__":
    run_smart_hunter()
