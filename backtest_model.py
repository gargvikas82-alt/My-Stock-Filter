"""
Stock Hunter Model - Historical Momentum Backtest Engine
==========================================================
Given a target date, scans the NIFTY Total Market universe (~750 stocks)
and identifies which were in a healthy uptrend (above 50-SMA) 60 trading
days prior, ranking them by return since then.

Fixes vs previous version:
  1. TARGET_DATE is now actually read from the environment (was hardcoded).
  2. Stock universe is read from nifty_total_market.csv (was a hardcoded
     list of 75 names, never the real NIFTY Total Market index).
  3. 50-SMA at entry now correctly INCLUDES the entry date's own close
     (previous version excluded it, silently skewing the trend check).
  4. Removed the NaN->entry_price fallback that made the SMA check
     trivially pass when there wasn't enough history - now those stocks
     are correctly skipped and logged.
  5. Batched yfinance downloads (chunks of 50) instead of 750 individual
     requests - faster and less likely to hit rate limits.
  6. Failures are logged to skipped_stocks.csv with a reason, so nothing
     silently disappears - required for a "foolproof" system.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

UNIVERSE_FILE = "nifty_total_market.csv"
CHUNK_SIZE = 50
LOOKBACK_TRADING_DAYS = 60
MIN_HISTORY_ROWS = LOOKBACK_TRADING_DAYS + 50  # need enough history to compute a 50-SMA at entry too

def get_target_date():
    """Read TARGET_DATE from environment (set by GitHub Actions workflow_dispatch input).
    Falls back to today's date only if not provided (e.g. local manual run)."""
    env_date = os.environ.get("TARGET_DATE", "").strip()
    if env_date:
        try:
            return pd.to_datetime(env_date).date()
        except Exception:
            print(f"WARNING: TARGET_DATE='{env_date}' could not be parsed. Falling back to today.")
    return datetime.today().date()

def load_universe():
    if not os.path.exists(UNIVERSE_FILE):
        print(f"FATAL: {UNIVERSE_FILE} not found in repo root. Cannot build stock universe.")
        sys.exit(1)
    df = pd.read_csv(UNIVERSE_FILE)
    symbols = sorted(df["Symbol"].dropna().unique().tolist())
    return symbols

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def run_historical_backtest():
    target_date = get_target_date()
    print(f"\nRUNNING BACKTEST FOR TARGET DATE: {target_date}")
    print("-" * 75)

    symbols = load_universe()
    print(f"Universe loaded: {len(symbols)} stocks from {UNIVERSE_FILE}")

    results = []
    skipped = []

    total_chunks = (len(symbols) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_num, chunk in enumerate(chunk_list(symbols, CHUNK_SIZE), start=1):
        yf_tickers = [f"{s}.NS" for s in chunk]
        print(f"[{chunk_num}/{total_chunks}] Fetching {len(chunk)} tickers...")

        try:
            data = yf.download(
                tickers=yf_tickers,
                period="1y",
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            for s in chunk:
                skipped.append({"Stock": s, "Reason": f"Chunk download failed: {e}"})
            continue

        for sym_nse, sym_yf in zip(chunk, yf_tickers):
            try:
                if len(yf_tickers) == 1:
                    hist = data
                else:
                    if sym_yf not in data.columns.get_level_values(0):
                        skipped.append({"Stock": sym_nse, "Reason": "No data returned"})
                        continue
                    hist = data[sym_yf]

                hist = hist.dropna(how="all")
                if hist.empty:
                    skipped.append({"Stock": sym_nse, "Reason": "Empty history"})
                    continue

                hist.index = pd.to_datetime(hist.index).date
                hist_filtered = hist[hist.index <= target_date]

                if len(hist_filtered) < MIN_HISTORY_ROWS:
                    skipped.append({"Stock": sym_nse, "Reason": f"Insufficient history ({len(hist_filtered)} rows)"})
                    continue

                # Target date price = most recent close on/before target date
                current_price = hist_filtered["Close"].iloc[-1]

                # Entry date = exactly LOOKBACK_TRADING_DAYS trading sessions prior
                entry_idx = -LOOKBACK_TRADING_DAYS
                entry_price = hist_filtered["Close"].iloc[entry_idx]
                entry_date = hist_filtered.index[entry_idx]

                # 50-SMA AS OF the entry date - must INCLUDE the entry date's own close.
                # Slice from start up to and including entry_idx (fixes previous off-by-one bug).
                slice_end = len(hist_filtered) + entry_idx + 1
                slice_up_to_entry = hist_filtered.iloc[:slice_end]

                if len(slice_up_to_entry) < 50:
                    skipped.append({"Stock": sym_nse, "Reason": "Not enough history to compute 50-SMA at entry"})
                    continue

                sma_50_at_entry = slice_up_to_entry["Close"].rolling(window=50).mean().iloc[-1]

                if pd.isna(sma_50_at_entry):
                    skipped.append({"Stock": sym_nse, "Reason": "50-SMA calculation returned NaN"})
                    continue

                if entry_price >= sma_50_at_entry:
                    stock_return_pct = ((current_price - entry_price) / entry_price) * 100
                    results.append({
                        "Stock": sym_nse,
                        "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                        "Price_At_Entry": round(float(entry_price), 2),
                        "Price_At_Target_Date": round(float(current_price), 2),
                        "Strategy_Return_%": round(float(stock_return_pct), 2),
                        "Above_50SMA_At_Entry": "YES",
                    })
                # If below 50 SMA at entry, it's correctly excluded (not an error, just doesn't qualify)

            except Exception as e:
                skipped.append({"Stock": sym_nse, "Reason": f"Unexpected error: {e}"})
                continue

        time.sleep(1)  # be polite to Yahoo's servers between chunks

    df_backtest = pd.DataFrame(results)
    if not df_backtest.empty:
        df_backtest = df_backtest.sort_values(by="Strategy_Return_%", ascending=False)
        df_backtest.to_csv("backtest_results.csv", index=False)
        print(f"\nRESULTS: {len(df_backtest)} stocks qualified. Saved to backtest_results.csv")
    else:
        pd.DataFrame(columns=[
            "Stock", "Entry_Date", "Price_At_Entry", "Price_At_Target_Date",
            "Strategy_Return_%", "Above_50SMA_At_Entry"
        ]).to_csv("backtest_results.csv", index=False)
        print("\nNo stocks qualified for this target date.")

    df_skipped = pd.DataFrame(skipped)
    df_skipped.to_csv("skipped_stocks.csv", index=False)
    print(f"Skipped/failed: {len(skipped)} stocks (see skipped_stocks.csv for reasons)")

if __name__ == "__main__":
    run_historical_backtest()
