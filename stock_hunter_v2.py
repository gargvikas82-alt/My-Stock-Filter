"""
Stock Hunter Model v2 - Early-Stage Momentum Screener
========================================================
Redesigned from v1 based on a core correction: v1 found stocks that had
ALREADY run up over the past 3 months (late-stage, already extended).
This version finds stocks showing FRESH signs of institutional accumulation -
recently crossed above their 50-day average, not yet extended, with rising
volume but modest price movement so far (the "quiet accumulation" signature).

Methodology (adapted from Mark Minervini's Trend Template + Volatility
Contraction Pattern concepts, and standard momentum-factor literature):

PHASE B - Technical (fully free, yfinance only):
  1. Stage 2 uptrend confirmation:
     - Price > 150-day MA and > 200-day MA
     - 50-day MA > 150-day MA > 200-day MA (proper alignment)
     - 200-day MA itself trending up over the last ~20 trading days
     - Price at least 25% above its 52-week low (confirms it left the base)
  2. Freshness filter (the key fix vs v1):
     - Price within 15% of its 50-day MA (NOT extended/already-flown)
     - Price crossed above its 50-day MA within the last 15 trading days
       (a RECENT crossover, not one that happened months ago)
  3. Quiet accumulation signature:
     - Average volume (last 10 days) at least 30% above average volume
       (prior 40 days) - rising interest
     - But price move over that same 10-day window is modest (-5% to +10%)
       - volume rising without price having run away yet
  4. Liquidity: average daily turnover (Volume x Close) over last 20 days
     >= Rs 1 crore (minimum), flagged "Strong" if >= Rs 5 crore

PHASE A - Fundamentals (best-effort only, not blocking, not faked):
  Only run once for the (small) list of stocks that already passed Phase B,
  to keep runtime sane. Attempted via yfinance .info. If a field is missing,
  it is marked "UNVERIFIED" - never defaulted to a fake safe-looking number.
  This is a LIVE snapshot only; free point-in-time historical fundamentals
  do not exist, so this section is skipped entirely for historical backtest
  dates (From_Date in the past) and only shown for live/today runs.

BACKTEST METHODOLOGY (walk-forward mode):
  You give a FROM_DATE and a TO_DATE.
  Every Tuesday and Friday between them is treated as its own scan date -
  simulating what the live twice-weekly system would actually have found on
  each scheduled run, not just a single point-in-time snapshot.
  On each scan date, the full Phase B filter is run using ONLY data available
  up to that date (point-in-time, no lookahead). Whichever stocks pass become
  "picks" for that date. Each pick's return is then measured from its own
  pick date through to the final TO_DATE.
  The same stock may appear on multiple scan dates if it stayed fresh across
  several scans - that's expected, not a duplicate bug.

Corporate action guard (kept from v1): any stock with a >=20% single-day
move between FROM_DATE and TO_DATE is excluded and logged, since that's a
near-certain demerger/bonus/split artifact, not real momentum.

RISK MANAGEMENT:
  Every qualifying stock gets an ATR(14)-based stop loss:
    Stop_Loss_Price = Price_At_Pick - (2 x ATR_14)
  This scales the stop to each stock's own volatility instead of using a
  flat percentage - a calmer stock gets a tighter stop, a wilder one gets
  more room. Position size is then derived from that stop so every trade
  risks roughly the same amount of capital (1% of TOTAL_CAPITAL by default,
  both overridable via environment variables TOTAL_CAPITAL and
  RISK_PCT_PER_TRADE):
    Suggested_Shares = (TOTAL_CAPITAL x RISK_PCT_PER_TRADE%) / (Price_At_Pick - Stop_Loss_Price)
  This is a suggestion for a fresh entry at Pick_Date, not a live trailing
  stop - an actual trailing exit for positions you already hold belongs in
  a separate portfolio-tracking script (planned next), since that needs to
  persist state (your real entry price/date) across runs, which a
  stateless universe-wide screener like this one is not built to do.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

UNIVERSE_FILE = "nifty_total_market.csv"
CHUNK_SIZE = 50
FETCH_PERIOD = "2y"  # need buffer for 200-day MA trend check + 52-week low/high

MIN_HISTORY_ROWS = 260          # ~1 year of trading days, buffer for 200MA + 52w checks
MA200_TREND_LOOKBACK = 20       # trading days back, to confirm 200MA is rising
FRESH_CROSSOVER_WINDOW = 15     # trading days - crossover must be within this window
EXTENDED_CAP_PCT = 15           # price must be within this % of its 50MA (not extended)
MIN_ABOVE_52W_LOW_PCT = 25      # price must be at least this % above 52-week low
VOL_SURGE_MIN_RATIO = 1.3       # recent 10-day avg volume vs prior 40-day avg volume
PRICE_MOVE_MIN_PCT = -5         # over the same 10-day window
PRICE_MOVE_MAX_PCT = 10
MIN_TURNOVER_CR = 1.0           # Rs 1 crore minimum average daily turnover
STRONG_TURNOVER_CR = 5.0        # Rs 5 crore = "Strong" liquidity tier
CORPORATE_ACTION_THRESHOLD_PCT = 20

ATR_PERIOD = 14                 # standard ATR lookback
ATR_STOP_MULTIPLE = 2.0         # stop = entry - (2 x ATR) - standard institutional default
DEFAULT_TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL", "500000"))  # override via env var
RISK_PCT_PER_TRADE = float(os.environ.get("RISK_PCT_PER_TRADE", "1.0"))  # % of capital risked per trade

CRORE = 10_000_000


def get_dates():
    """FROM_DATE and TO_DATE from environment. TO_DATE defaults to today if blank."""
    from_env = os.environ.get("FROM_DATE", "").strip()
    to_env = os.environ.get("TO_DATE", "").strip()

    if not from_env:
        print("FATAL: FROM_DATE is required (the point-in-time date to run the screener on).")
        sys.exit(1)

    try:
        from_date = pd.to_datetime(from_env).date()
    except Exception:
        print(f"FATAL: FROM_DATE='{from_env}' could not be parsed.")
        sys.exit(1)

    if to_env:
        try:
            to_date = pd.to_datetime(to_env).date()
        except Exception:
            print(f"WARNING: TO_DATE='{to_env}' could not be parsed. Using today instead.")
            to_date = datetime.today().date()
    else:
        to_date = datetime.today().date()

    if to_date <= from_date:
        print(f"FATAL: TO_DATE ({to_date}) must be after FROM_DATE ({from_date}).")
        sys.exit(1)

    return from_date, to_date


def load_universe():
    if not os.path.exists(UNIVERSE_FILE):
        print(f"FATAL: {UNIVERSE_FILE} not found.")
        sys.exit(1)
    df = pd.read_csv(UNIVERSE_FILE)
    return sorted(df["Symbol"].dropna().unique().tolist())


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def compute_atr(hist_pit, period=ATR_PERIOD):
    """Standard Average True Range calculation.
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)"""
    high = hist_pit["High"]
    low = hist_pit["Low"]
    close = hist_pit["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window=period).mean()
    return atr.iloc[-1]


def evaluate_stock(hist, from_date, to_date, sym_nse):
    """Run the full Phase B filter as of from_date using only data up to from_date.
    Returns a result dict if the stock qualifies, or (None, reason) if it doesn't."""

    hist = hist.dropna(how="all")
    if hist.empty:
        return None, "Empty history"

    hist.index = pd.to_datetime(hist.index).date
    hist_pit = hist[hist.index <= from_date]  # point-in-time: only data up to FROM_DATE

    if len(hist_pit) < MIN_HISTORY_ROWS:
        return None, f"Insufficient history as of FROM_DATE ({len(hist_pit)} rows)"

    close = hist_pit["Close"]
    volume = hist_pit["Volume"]

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    if pd.isna(ma200.iloc[-1]) or pd.isna(ma150.iloc[-1]) or pd.isna(ma50.iloc[-1]):
        return None, "Moving averages not computable (insufficient history)"

    price_now = close.iloc[-1]
    ma50_now, ma150_now, ma200_now = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]

    if not (price_now > ma150_now and price_now > ma200_now):
        return None, "Not above 150MA/200MA"
    if not (ma50_now > ma150_now > ma200_now):
        return None, "MA alignment failed (50>150>200)"

    if len(ma200) <= MA200_TREND_LOOKBACK or pd.isna(ma200.iloc[-1 - MA200_TREND_LOOKBACK]):
        return None, "Not enough history to confirm 200MA trend"
    if not (ma200_now > ma200.iloc[-1 - MA200_TREND_LOOKBACK]):
        return None, "200MA not trending up"

    fifty_two_week_low = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    if not (price_now >= fifty_two_week_low * (1 + MIN_ABOVE_52W_LOW_PCT / 100)):
        return None, "Not enough distance above 52-week low"

    pct_above_50ma = ((price_now - ma50_now) / ma50_now) * 100
    if pct_above_50ma > EXTENDED_CAP_PCT:
        return None, f"Too extended above 50MA ({pct_above_50ma:.1f}%) - already flown, not fresh"

    window = min(FRESH_CROSSOVER_WINDOW, len(close) - 1)
    recent_close = close.iloc[-window:]
    recent_ma50 = ma50.iloc[-window:]
    was_below_recently = (recent_close < recent_ma50).any()
    if not was_below_recently:
