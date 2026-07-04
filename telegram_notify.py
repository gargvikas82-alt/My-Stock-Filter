"""
Sends a formatted summary of backtest_results.csv to Telegram after each run.
Requires two GitHub repo secrets: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
(setup instructions provided separately - see TELEGRAM_SETUP.md).
"""

import os
import sys
import pandas as pd
import requests

TOP_N = 10

def send_telegram_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    })
    if resp.status_code != 200:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")
        return False
    return True

def build_message():
    if not os.path.exists("backtest_results.csv"):
        return "Stock Hunter run finished, but backtest_results.csv was not found."

    df = pd.read_csv("backtest_results.csv")
    if df.empty:
        return "<b>Stock Hunter</b>\nNo stocks qualified in this run."

    skipped_count = 0
    if os.path.exists("skipped_stocks.csv"):
        try:
            skipped_count = len(pd.read_csv("skipped_stocks.csv"))
        except Exception:
            pass

    top = df.head(TOP_N)
    lines = [f"<b>Stock Hunter - Top {min(TOP_N, len(df))} of {len(df)} qualified</b>\n"]
    for _, row in top.iterrows():
        lines.append(
            f"{row['Stock']}: {row['Strategy_Return_%']:+.1f}% "
            f"(Entry {row['Entry_Date']} @ {row['Price_At_Entry']})"
        )
    lines.append(f"\nTotal qualified: {len(df)} | Skipped/excluded: {skipped_count}")
    lines.append("Full results in the GitHub repo.")
    return "\n".join(lines)

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set - skipping notification.")
        sys.exit(0)  # don't fail the whole workflow just because notification isn't configured yet

    message = build_message()
    success = send_telegram_message(token, chat_id, message)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
