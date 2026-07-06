"""
Sends today's qualified picks (backtest_results.csv) to a Google Sheet, via a
free Google Apps Script Web App acting as a webhook - no Google API key or
service account needed, keeping this zero-budget.
"""
import os
import sys
import pandas as pd
import requests


def main():
    webhook_url = os.environ.get("SHEET_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("SHEET_WEBHOOK_URL secret not set - skipping sheet sync.")
        sys.exit(0)

    if not os.path.exists("backtest_results.csv"):
        print("backtest_results.csv not found - nothing to sync.")
        sys.exit(0)

    df = pd.read_csv("backtest_results.csv")
    if df.empty or "Stock" not in df.columns or df["Stock"].isna().all():
        print("No qualified stocks today - nothing to sync.")
        sys.exit(0)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "Stock": row.get("Stock"),
            "Pick_Date": row.get("Pick_Date"),
            "Price_At_Pick": row.get("Price_At_Pick"),
            "Stop_Loss_Price": row.get("Stop_Loss_Price"),
            "Stop_Loss_%": row.get("Stop_Loss_%"),
            "Suggested_Shares": row.get("Suggested_Shares"),
            "Capital_Allocated_Rs": row.get("Capital_Allocated_Rs"),
            "Conviction_Score": row.get("Conviction_Score"),
            "Liquidity_Tier": row.get("Liquidity_Tier"),
        })

    try:
        resp = requests.post(webhook_url, json={"rows": rows}, timeout=30)
        print(f"Sheet sync response: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"Sheet sync failed (not failing the whole workflow for this): {e}")


if __name__ == "__main__":
    main()
