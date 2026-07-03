import yfinance as yf
import pandas as pd
from nselib import capital_market
from datetime import datetime, timedelta

# 1. Stocks we want to check (We can add more later)
nse_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT"]
yf_symbols = [f"{sym}.NS" for sym in nse_symbols]

def run_forensic_audit(symbol_yf, symbol_nse):
    stock = yf.Ticker(symbol_yf)
    
    try:
        info = stock.info
    except:
        info = {}
    
    print(f"Auditing {symbol_nse}...")

    # --- THE FUNDAMENTAL AUDIT ---
    try:
        cf = stock.cashflow.iloc[0].get('Free Cash Flow', 0)
        ni = stock.financials.iloc[0].get('Net Income', 1)
        quality_score = cf / ni
    except:
        quality_score = 0
        
    ebitda = info.get('ebitda', 0) if info.get('ebitda') is not None else 0
    interest = info.get('interestExpense', 1) if info.get('interestExpense') is not None else 1
    interest_coverage = ebitda / interest if interest > 0 else 999
    pe_ratio = info.get('trailingPE', 999) if info.get('trailingPE') is not None else 999

    # --- THE OPERATOR / MANIPULATION AUDIT ---
    try:
        end_date = datetime.today().strftime('%d-%m-%Y')
        start_date = (datetime.today() - timedelta(days=10)).strftime('%d-%m-%Y')
        
        # Pull real NSE delivery data
        delivery_data = capital_market.price_volume_and_deliverable_position_data(
            symbol=symbol_nse, from_date=start_date, to_date=end_date
        )
        delivery_data['% Dly Qt to Traded Qty'] = pd.to_numeric(
            delivery_data['% Dly Qt to Traded Qty'].astype(str).str.replace(' -', '0').str.replace('%', ''), 
            errors='coerce'
        ).fillna(0)
        avg_delivery_pct = delivery_data['% Dly Qt to Traded Qty'].mean()
    except:
        avg_delivery_pct = 0

    return {
        "Stock": symbol_nse,
        "Cashflow_Quality (>0.8 is Safe)": round(quality_score, 2),
        "Interest_Coverage (>3 is Safe)": round(interest_coverage, 2),
        "P/E_Ratio": round(pe_ratio, 2),
        "Delivery_Pct (>40% is Safe)": round(avg_delivery_pct, 2)
    }

# Run the Engine
results = []
for nse, yf_sym in zip(nse_symbols, yf_symbols):
    try:
        results.append(run_forensic_audit(yf_sym, nse))
    except Exception as e:
        print(f"Skipping {nse} due to data lag.")

df = pd.DataFrame(results)

# The Sieve Filter (Only keeping the robust stocks)
safe_stocks = df[
    (df['Cashflow_Quality (>0.8 is Safe)'] > 0.8) & 
    (df['Interest_Coverage (>3 is Safe)'] > 3.0) & 
    (df['Delivery_Pct (>40% is Safe)'] > 40.0) 
]

print("\n=== THE FORTRESS SURVIVORS ===")
print(safe_stocks)

# Save the output to a clean sheet file
df.to_csv("all_audited_stocks.csv", index=False)
safe_stocks.to_csv("fortress_survivors.csv", index=False)
