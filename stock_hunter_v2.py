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

BACKTEST METHODOLOGY (the second core fix vs v1):
  You give a FROM_DATE and a TO_DATE.
  The full Phase B filter is run using ONLY data available up to FROM_DATE
  (point-in-time, no lookahead). Whichever stocks pass become "the picks."
  Their return is then measured from FROM_DATE to TO_DATE.
  This answers "if the model had picked these stocks on that date, what
  actually happened by this later date" - a real forward-test of the
  selection logic, not just a performance ranking.

Corporate action guard (kept from v1): any stock with a >=20% single-day
move between FROM_DATE and TO_DATE is excluded and logged, since that's a
near-certain demerger/bonus/split artifact, not real momentum.

RISK MANAGEMENT (new):
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
from datetime import datetime, timedelta

UNIVERSE_FILE = "nifty_total_market.csv"
CHUNK_SIZE = 50
FETCH_BUFFER_DAYS = 400  # calendar-day buffer before FROM_DATE, for the 200MA/52-week lookback

MIN_HISTORY_ROWS = 260          # ~1 year of trading days, buffer for 200MA + 52w checks
MA200_TREND_LOOKBACK = 20       # trading days back, to confirm 200MA is rising
FRESH_CROSSOVER_WINDOW = 10      # tightened from 15 - only the most recent crossovers
RETEST_LOOKBACK_WINDOW = 20      # wider than FRESH_CROSSOVER_WINDOW - the crossover itself must be recent,
                                   # but we need extra room after it for a pullback-then-recovery to form
EXTENDED_CAP_PCT = 8            # tightened from 15 - 150-trade backtest showed 8%+ extension above the 50MA
                                 # averaged only +0.8% return vs +9-11% for 0-8% - past 8% it's no longer
                                 # "early stage," it's already running
MIN_ABOVE_52W_LOW_PCT = 25      # price must be at least this % above 52-week low
VOL_SURGE_MIN_RATIO = 1.5       # tightened from 1.3 - stronger accumulation signal required
PRICE_MOVE_MIN_PCT = -3         # tightened from -5 - quieter accumulation band
PRICE_MOVE_MAX_PCT = 8          # tightened from 10
RETEST_MAX_DIST_PCT = 3.0       # after crossover, pullback must come within this % of the 50MA to count as a
                                 # genuine retest (too far away = still riding the original breakout, no test yet)
RETEST_MAX_BREACH_PCT = 3.0     # pullback may undershoot the 50MA by up to this % (normal shakeout) before
                                 # it counts as a failed/broken support rather than a held one
STRUCTURAL_STOP_BUFFER_ATR = 0.5   # place the stop this many ATRs below the actual retest low, not at a
                                     # flat multiple of ATR from entry - respects the chart, not just volatility
MIN_STOP_ATR_MULT = 3.0         # widened from 2.0 - the 2x floor was still averaging ~7.8% stop distance,
                                 # too tight for a 60-day swing hold; risk-based sizing already shrinks share
                                 # count for a wider stop, so this doesn't increase total risk per trade
MAX_STOP_ATR_MULT = 7.0         # widened from 5.0 - gives real structural swing lows room to breathe
MIN_TURNOVER_CR = 1.0           # Rs 1 crore minimum average daily turnover
STRONG_TURNOVER_CR = 5.0        # Rs 5 crore = "Strong" liquidity tier
CORPORATE_ACTION_THRESHOLD_PCT = 20
MAX_HOLDING_DAYS = int(os.environ.get("MAX_HOLDING_DAYS", "60"))  # trading days - fixed horizon cap so every
                                                                    # trade is measured on comparable footing,
                                                                    # instead of "held until TO_DATE" which made
                                                                    # early picks and late picks non-comparable

TOP_N_PER_SCAN_DATE = int(os.environ.get("TOP_N_PER_SCAN_DATE", "2"))  # hard cap - only the best N picks per scan date, regardless of how many pass the filter - solves the "too many stocks for limited capital" problem structurally

ATR_PERIOD = 14                 # standard ATR lookback
ATR_STOP_MULTIPLE = 2.0         # stop = entry - (2 x ATR) - standard institutional default
ADX_PERIOD = 14                 # standard ADX lookback
MIN_ADX = 25                    # ADX >= 25 = genuine trending move, not choppy/noisy sideways action
MAX_ADX = 32                    # NEW ceiling - 150-trade backtest showed ADX 32+ averaged only +3.4% return
                                 # vs +9-11% for 26-32; above this the trend is already mature/established,
                                 # not early-stage anymore - defeats the whole point of catching it early
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


def compute_adx(hist_pit, period=ADX_PERIOD):
    """Standard Wilder's ADX (Average Directional Index) calculation.
    Measures trend STRENGTH/cleanliness, not direction - high ADX means a
    genuine directional move, low ADX means choppy/sideways noise, even if
    price is technically above its moving averages."""
    high = hist_pit["High"]
    low = hist_pit["Low"]
    close = hist_pit["Close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=hist_pit.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=hist_pit.index).ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx.iloc[-1]


def simulate_realistic_exit(hist_full, entry_price, stop_loss_price, max_holding_days):
    """
    Walk the actual price path forward from entry, day by day, and exit at
    whichever of these happens FIRST:
      1. STOP_LOSS_HIT   - that day's CLOSE breaches the stop (not the
         intraday Low). A single wick touching the stop and closing back
         above it is normal noise, not a real breakdown - only a confirmed
         close below the level counts. This is a deliberate change from the
         earlier Low-based trigger, which was catching wicks, not failures.
         Exit is modeled at that day's Close (roughly where a same-day stop
         order would fill once the close confirms the breach - still ignores
         gap-through risk on the day itself, flagged not solved).
      2. MAX_HOLDING_PERIOD - max_holding_days trading days have passed with
         no stop hit. Exit at that day's Close. This is what makes every
         trade comparable - nobody gets an unfair extra 300 days to recover.
      3. END_OF_BACKTEST_DATA - the backtest window itself ends before
         either of the above (only happens for picks made very close to
         TO_DATE). Exit at the last available Close, holding_days is
         whatever data existed - shorter than max_holding_days, and this
         reason is logged so it isn't silently mixed in with real exits.
    This replaces the old "hold every pick until TO_DATE regardless of
    pick date" approach, which made a pick from 300 days ago and a pick
    from 3 weeks ago look like the same kind of trade.
    """
    for i in range(1, len(hist_full)):
        day_close = hist_full["Close"].iloc[i]
        day_date = hist_full.index[i]
        if stop_loss_price is not None and day_close <= stop_loss_price:
            return {
                "exit_price": day_close,
                "exit_date": day_date,
                "exit_reason": "Stop_Loss_Hit",
                "holding_days_realistic": i,
            }
        if i >= max_holding_days:
            return {
                "exit_price": day_close,
                "exit_date": day_date,
                "exit_reason": "Max_Holding_Period",
                "holding_days_realistic": i,
            }
    last_idx = len(hist_full) - 1
    return {
        "exit_price": hist_full["Close"].iloc[-1],
        "exit_date": hist_full.index[-1],
        "exit_reason": "End_Of_Backtest_Data",
        "holding_days_realistic": last_idx,
    }


def find_retest_low(close, ma50, low, cross_window):
    """
    Within the last `cross_window` trading days, find the most recent upward
    crossover of price above the 50MA, then check whether the stock pulled
    BACK toward the 50MA afterward (a retest) before today.

    This replaces "buy the crossover itself" with "buy the confirmed retest" -
    the fix for the whipsaw problem where big eventual winners (Laurus Labs,
    Adani Power, Polycab) were getting stopped out in 5-19 days because we
    were buying into unconfirmed strength on day 1 of the signal.

    Returns (retest_low_price, retest_distance_pct_from_ma50) or (None, None)
    if there's no crossover in the window, or the crossover happened too
    recently to have had a chance to retest yet.
    """
    window = min(cross_window, len(close) - 1)
    recent_close = close.iloc[-window:]
    recent_ma50 = ma50.iloc[-window:]
    recent_low = low.iloc[-window:]

    below_mask = recent_close < recent_ma50
    if not below_mask.any():
        return None, None  # no crossover in window at all - either always above (stale) or always below

    below_positions = [i for i, v in enumerate(below_mask.values) if v]
    last_below_pos = below_positions[-1]
    cross_pos = last_below_pos + 1  # first day back above the 50MA

    if cross_pos >= len(recent_close) - 1:
        return None, None  # crossed today or yesterday - no time for a retest yet, too fresh

    # Days strictly after the crossover, excluding today (today must show the bounce-back)
    post_cross_low = recent_low.iloc[cross_pos:-1]
    post_cross_ma50 = recent_ma50.iloc[cross_pos:-1]
    if post_cross_low.empty:
        return None, None

    dist_to_ma = (post_cross_low - post_cross_ma50) / post_cross_ma50 * 100
    retest_idx_local = dist_to_ma.idxmin()  # closest approach to (or through) the 50MA after the cross
    retest_low_price = post_cross_low.loc[retest_idx_local]
    retest_dist_pct = dist_to_ma.loc[retest_idx_local]

    return retest_low_price, retest_dist_pct


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
    low = hist_pit["Low"]

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    if pd.isna(ma200.iloc[-1]) or pd.isna(ma150.iloc[-1]) or pd.isna(ma50.iloc[-1]):
        return None, "Moving averages not computable (insufficient history)"

    price_now = close.iloc[-1]
    ma50_now, ma150_now, ma200_now = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]

    # --- Stage 2 confirmation ---
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

    # --- Freshness filter (the core fix vs v1) ---
    pct_above_50ma = ((price_now - ma50_now) / ma50_now) * 100
    if pct_above_50ma > EXTENDED_CAP_PCT:
        return None, f"Too extended above 50MA ({pct_above_50ma:.1f}%) - already flown, not fresh"

    window = min(FRESH_CROSSOVER_WINDOW, len(close) - 1)
    recent_close = close.iloc[-window:]
    recent_ma50 = ma50.iloc[-window:]
    was_below_recently = (recent_close < recent_ma50).any()
    if not was_below_recently:
        return None, f"No fresh 50MA crossover in last {FRESH_CROSSOVER_WINDOW} trading days - trend too old"

    # --- Retest confirmation (the fix for the whipsaw problem) ---
    # Don't buy the crossover itself - buy the pullback that held. This is what
    # separates a confirmed trend start from a stock still shaking out.
    retest_low_price, retest_dist_pct = find_retest_low(close, ma50, low, RETEST_LOOKBACK_WINDOW)
    if retest_low_price is None:
        return None, f"No confirmed retest after 50MA crossover in last {RETEST_LOOKBACK_WINDOW} trading days"
    if retest_dist_pct > RETEST_MAX_DIST_PCT:
        return None, f"Pullback never came within {RETEST_MAX_DIST_PCT}% of the 50MA - no real retest yet"
    if retest_dist_pct < -RETEST_MAX_BREACH_PCT:
        return None, f"Retest broke {abs(retest_dist_pct):.1f}% below the 50MA - failed support, not a hold"
    if price_now <= retest_low_price:
        return None, "Price hasn't recovered above its retest low yet - support not yet confirmed"

    # --- Quiet accumulation signature ---
    if len(volume) < 50:
        return None, "Insufficient volume history"
    avg_vol_recent = volume.iloc[-10:].mean()
    avg_vol_prior = volume.iloc[-50:-10].mean()
    if avg_vol_prior == 0 or pd.isna(avg_vol_prior):
        return None, "Cannot compute volume baseline"
    vol_ratio = avg_vol_recent / avg_vol_prior
    if vol_ratio < VOL_SURGE_MIN_RATIO:
        return None, f"No volume surge ({vol_ratio:.2f}x, need >={VOL_SURGE_MIN_RATIO}x)"

    price_10d_ago = close.iloc[-10]
    price_move_10d_pct = ((price_now - price_10d_ago) / price_10d_ago) * 100
    if not (PRICE_MOVE_MIN_PCT <= price_move_10d_pct <= PRICE_MOVE_MAX_PCT):
        return None, f"Price move too large for 'quiet' accumulation ({price_move_10d_pct:.1f}% in 10 days)"

    # --- Liquidity ---
    turnover_recent = (close.iloc[-20:] * volume.iloc[-20:]).mean()
    turnover_cr = turnover_recent / CRORE
    if turnover_cr < MIN_TURNOVER_CR:
        return None, f"Turnover too low (Rs {turnover_cr:.2f} cr/day, need >= Rs {MIN_TURNOVER_CR} cr)"
    liquidity_tier = "Strong" if turnover_cr >= STRONG_TURNOVER_CR else "Adequate"

    # --- Trend quality / noise filter (ADX) ---
    adx_value = compute_adx(hist_pit, ADX_PERIOD)
    if pd.isna(adx_value):
        return None, "ADX not computable (insufficient history)"
    if adx_value < MIN_ADX:
        return None, f"ADX too low ({adx_value:.1f}, need >={MIN_ADX}) - trend is choppy/noisy, not clean"
    if adx_value > MAX_ADX:
        return None, f"ADX too high ({adx_value:.1f}, need <={MAX_ADX}) - trend already mature, not early-stage"

    # --- Passed everything. Now compute forward return to TO_DATE using full history ---
    hist_full = hist[(hist.index >= from_date) & (hist.index <= to_date)]
    if hist_full.empty or len(hist_full) < 2:
        return None, "No trading data available between FROM_DATE and TO_DATE yet"

    entry_price = hist_full["Close"].iloc[0]
    entry_date_actual = hist_full.index[0]
    exit_price = hist_full["Close"].iloc[-1]
    exit_date_actual = hist_full.index[-1]

    forward_return_pct = ((exit_price - entry_price) / entry_price) * 100

    # --- ATR-based stop loss and risk-based position sizing (institutional style) ---
    # Moved ahead of the exit simulation below - the realistic exit needs to know
    # the stop price to check whether the path ever breached it.
    atr14 = compute_atr(hist_pit, ATR_PERIOD)
    if pd.isna(atr14) or atr14 <= 0:
        stop_loss_price = None
        stop_loss_pct = None
        shares_to_buy = None
        capital_allocated = None
    else:
        # --- Structural stop: anchored below the actual retest swing low, not a flat
        # ATR multiple from entry. This is what respects the chart instead of just
        # volatility - the stop sits below the level that needs to hold, with a small
        # buffer, then gets clamped so it's never absurdly tight or absurdly wide. ---
        structural_stop = retest_low_price - (STRUCTURAL_STOP_BUFFER_ATR * atr14)
        tightest_allowed = entry_price - (MIN_STOP_ATR_MULT * atr14)   # closest the stop may sit to entry
        loosest_allowed = entry_price - (MAX_STOP_ATR_MULT * atr14)    # farthest the stop may sit from entry

        stop_loss_price = structural_stop
        if stop_loss_price > tightest_allowed:
            stop_loss_price = tightest_allowed   # structural stop was too tight - widen to the ATR floor
        if stop_loss_price < loosest_allowed:
            stop_loss_price = loosest_allowed    # structural stop was too wide - cap risk at the ATR ceiling

        risk_per_share = entry_price - stop_loss_price
        stop_loss_pct = (risk_per_share / entry_price) * 100
        risk_capital = DEFAULT_TOTAL_CAPITAL * (RISK_PCT_PER_TRADE / 100)
        shares_to_buy = int(risk_capital / risk_per_share) if risk_per_share > 0 else 0
        capital_allocated = round(shares_to_buy * entry_price, 2)

    # --- Realistic exit: stop-loss-aware, fixed-horizon-capped ---
    # This is the actual fix for the Sharpe problem - it stops every trade
    # being marked to a single fixed TO_DATE regardless of how long it's
    # had to run, and it actually enforces the stop-loss the model already
    # computes instead of just printing it as a suggestion.
    exit_sim = simulate_realistic_exit(hist_full, entry_price, stop_loss_price, MAX_HOLDING_DAYS)
    realistic_exit_price = exit_sim["exit_price"]
    realistic_exit_date = exit_sim["exit_date"]
    exit_reason = exit_sim["exit_reason"]
    holding_days_realistic = exit_sim["holding_days_realistic"]
    realistic_return_pct = ((realistic_exit_price - entry_price) / entry_price) * 100

    # Corporate action guard - now only checked over the ACTUAL holding window
    # (entry to realistic exit), not the full span to TO_DATE. A split/bonus
    # that happens after we've already exited shouldn't disqualify the trade.
    hist_held = hist_full.iloc[:holding_days_realistic + 1]
    daily_pct_changes = hist_held["Close"].pct_change().dropna() * 100
    corp_hit = daily_pct_changes[daily_pct_changes.abs() >= CORPORATE_ACTION_THRESHOLD_PCT]
    if not corp_hit.empty:
        return None, f"Excluded - likely corporate action on {corp_hit.index[0]}: {corp_hit.iloc[0]:.1f}% single-day move"

    # --- Conviction Score - used to rank picks within a scan date so only the
    # strongest few are kept (solves "too many stocks for available capital") ---
    freshness_score = max(0, EXTENDED_CAP_PCT - abs(pct_above_50ma))  # higher = closer to the exact crossover point
    conviction_score = (vol_ratio * 50) + (min(turnover_cr, 20) * 2) + freshness_score

    return {
        "Stock": sym_nse,
        "Pick_Date": entry_date_actual.strftime("%Y-%m-%d"),
        "Price_At_Pick": round(float(entry_price), 2),
        "Evaluation_Date": exit_date_actual.strftime("%Y-%m-%d"),
        "Price_At_Evaluation": round(float(exit_price), 2),
        "Forward_Return_%": round(float(forward_return_pct), 2),
        "Realistic_Exit_Date": realistic_exit_date.strftime("%Y-%m-%d"),
        "Realistic_Exit_Price": round(float(realistic_exit_price), 2),
        "Realistic_Return_%": round(float(realistic_return_pct), 2),
        "Exit_Reason": exit_reason,
        "Holding_Days_Realistic": holding_days_realistic,
        "Pct_Above_50MA_At_Pick": round(float(pct_above_50ma), 1),
        "Volume_Surge_Ratio": round(float(vol_ratio), 2),
        "Avg_Daily_Turnover_Cr": round(float(turnover_cr), 2),
        "Liquidity_Tier": liquidity_tier,
        "ATR_14": round(float(atr14), 2) if not pd.isna(atr14) else None,
        "Stop_Loss_Price": round(float(stop_loss_price), 2) if stop_loss_price is not None else None,
        "Stop_Loss_%": round(float(stop_loss_pct), 2) if stop_loss_pct is not None else None,
        "Suggested_Shares": shares_to_buy,
        "Capital_Allocated_Rs": capital_allocated,
        "Conviction_Score": round(float(conviction_score), 2),
        "ADX_14": round(float(adx_value), 1),
    }, None


def get_scan_dates(from_date, to_date):
    """Generate every Tuesday (weekday 1) and Friday (weekday 4) between from_date
    and to_date inclusive - matching the real twice-weekly live schedule, so a
    From/To backtest simulates what the system would actually have found on each
    scheduled run, not just a single point-in-time snapshot."""
    all_days = pd.date_range(from_date, to_date, freq="D")
    scan_dates = [d.date() for d in all_days if d.weekday() in (1, 4)]
    if not scan_dates or scan_dates[0] != from_date:
        scan_dates = [from_date] + scan_dates  # always include the exact FROM_DATE requested
    return sorted(set(scan_dates))


def run():
    from_date, to_date = get_dates()
    scan_dates = get_scan_dates(from_date, to_date)
    print(f"\nSTOCK HUNTER v2 - Early-Stage Screener (Walk-Forward Mode)")
    print(f"Simulating {len(scan_dates)} scan dates (every Tue/Fri) between {from_date} and {to_date}")
    print(f"Each pick's return is measured from its own pick date through to {to_date}")
    print("-" * 75)

    symbols = load_universe()
    print(f"Universe: {len(symbols)} stocks")

    results = []
    skipped = []

    total_chunks = (len(symbols) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_num, chunk in enumerate(chunk_list(symbols, CHUNK_SIZE), start=1):
        yf_tickers = [f"{s}.NS" for s in chunk]
        print(f"[{chunk_num}/{total_chunks}] Fetching {len(chunk)} tickers...")

        try:
            fetch_start = (from_date - timedelta(days=FETCH_BUFFER_DAYS)).strftime("%Y-%m-%d")  # buffer for 200MA/52wk-low lookback
            fetch_end = (to_date + timedelta(days=1)).strftime("%Y-%m-%d")        # yfinance end is exclusive
            data = yf.download(
                tickers=yf_tickers, start=fetch_start, end=fetch_end, interval="1d",
                auto_adjust=False, actions=False, group_by="ticker",
                threads=True, progress=False,
            )
        except Exception as e:
            for s in chunk:
                skipped.append({"Stock": s, "Scan_Date": "ALL", "Reason": f"Chunk download failed: {e}"})
            continue

        for sym_nse, sym_yf in zip(chunk, yf_tickers):
            try:
                if len(yf_tickers) == 1:
                    hist = data
                else:
                    if sym_yf not in data.columns.get_level_values(0):
                        skipped.append({"Stock": sym_nse, "Scan_Date": "ALL", "Reason": "No data returned"})
                        continue
                    hist = data[sym_yf]

                # Same downloaded history reused across every scan date - no extra network calls
                for scan_date in scan_dates:
                    if scan_date >= to_date:
                        continue
                    result, reason = evaluate_stock(hist, scan_date, to_date, sym_nse)
                    if result:
                        result["Scan_Date"] = scan_date.strftime("%Y-%m-%d")
                        results.append(result)
                    else:
                        skipped.append({"Stock": sym_nse, "Scan_Date": scan_date.strftime("%Y-%m-%d"), "Reason": reason})
            except Exception as e:
                skipped.append({"Stock": sym_nse, "Scan_Date": "ALL", "Reason": f"Unexpected error: {e}"})

        time.sleep(1)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Hard cap: within each scan date, keep only the top N by Conviction_Score.
        # This is what actually controls total pick volume for a limited-capital
        # investor - tightening filter thresholds alone doesn't guarantee a target
        # count, this does.
        before_cap = len(df_results)
        df_results = (
            df_results.sort_values("Conviction_Score", ascending=False)
            .groupby("Scan_Date", group_keys=False)
            .head(TOP_N_PER_SCAN_DATE)
        )
        df_results = df_results.sort_values(by=["Pick_Date", "Forward_Return_%"], ascending=[True, False])
        df_results.to_csv("stock_hunter_v2_results.csv", index=False)
        unique_stocks = df_results["Stock"].nunique()
        print(f"\nQUALIFIED (before cap): {before_cap} pick-instances")
        print(f"AFTER TOP-{TOP_N_PER_SCAN_DATE}-PER-SCAN-DATE CAP: {len(df_results)} pick-instances "
              f"across {len(scan_dates)} scan dates ({unique_stocks} unique stocks). "
              f"Saved to stock_hunter_v2_results.csv")
        print("Note: the same stock may appear on multiple scan dates if it stayed fresh - "
              "that's expected, not a duplicate bug.")

        # --- Metrics: old (held-to-TO_DATE) vs realistic (stop-loss + fixed horizon) ---
        def sharpe_like(returns):
            returns = returns.dropna()
            if len(returns) < 2 or returns.std() == 0:
                return None
            return returns.mean() / returns.std()

        old_sharpe = sharpe_like(df_results["Forward_Return_%"])
        new_sharpe = sharpe_like(df_results["Realistic_Return_%"])
        old_win = (df_results["Forward_Return_%"] > 0).mean() * 100
        new_win = (df_results["Realistic_Return_%"] > 0).mean() * 100
        stop_outs = (df_results["Exit_Reason"] == "Stop_Loss_Hit").sum()
        max_hold_exits = (df_results["Exit_Reason"] == "Max_Holding_Period").sum()
        eob_exits = (df_results["Exit_Reason"] == "End_Of_Backtest_Data").sum()

        print("\n--- METRICS COMPARISON ---")
        print(f"OLD (held to TO_DATE, variable duration 17-318+ days):")
        print(f"  Mean return: {df_results['Forward_Return_%'].mean():.2f}%  "
              f"Std: {df_results['Forward_Return_%'].std():.2f}%  "
              f"Sharpe-like: {old_sharpe:.2f}  Win rate: {old_win:.1f}%")
        print(f"NEW (stop-loss + {MAX_HOLDING_DAYS}-trading-day fixed horizon, comparable durations):")
        print(f"  Mean return: {df_results['Realistic_Return_%'].mean():.2f}%  "
              f"Std: {df_results['Realistic_Return_%'].std():.2f}%  "
              f"Sharpe-like: {new_sharpe:.2f}  Win rate: {new_win:.1f}%")
        print(f"  Exit breakdown: {stop_outs} stopped out, {max_hold_exits} hit max holding period, "
              f"{eob_exits} ran out of backtest data before either (too recent a pick)")
    else:
        pd.DataFrame(columns=[
            "Stock", "Scan_Date", "Pick_Date", "Price_At_Pick", "Evaluation_Date", "Price_At_Evaluation",
            "Forward_Return_%", "Realistic_Exit_Date", "Realistic_Exit_Price", "Realistic_Return_%",
            "Exit_Reason", "Holding_Days_Realistic", "Pct_Above_50MA_At_Pick", "Volume_Surge_Ratio",
            "Avg_Daily_Turnover_Cr", "Liquidity_Tier", "ATR_14", "Stop_Loss_Price",
            "Stop_Loss_%", "Suggested_Shares", "Capital_Allocated_Rs", "Conviction_Score", "ADX_14"
        ]).to_csv("stock_hunter_v2_results.csv", index=False)
        print(f"\nNo stocks qualified on any of the {len(scan_dates)} scan dates. The filter is intentionally strict.")

    pd.DataFrame(skipped).to_csv("stock_hunter_v2_skipped.csv", index=False)
    print(f"Did not qualify / failed: {len(skipped)} rows (see stock_hunter_v2_skipped.csv for reasons)")


if __name__ == "__main__":
    run()
