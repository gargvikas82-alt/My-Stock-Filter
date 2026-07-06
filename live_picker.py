"""
Stock Hunter v2 - LIVE PICKER MODE
====================================
Runs the SAME Phase B filter logic as stock_hunter_v2.py (imports its constants
and helper functions directly - compute_atr, compute_adx, find_retest_low,
load_universe - so there is zero drift between what was backtested and what
runs live). The only difference: this stops before the forward-return/exit-
simulation section, because a live pick made TODAY has no future price data
yet to measure a return against.

Output file is still named backtest_results.csv (and skipped_stocks.csv) so the
existing workflow's git-commit step and telegram_notify.py don't need renaming.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

import stock_hunter_v2 as sh2


def screen_stock_live(hist, as_of_date, sym_nse, nifty_hist=None):
    """Same checks as evaluate_stock() in stock_hunter_v2.py, minus the
    forward-return/exit-simulation part (needs future data that doesn't exist
    yet for a fresh live pick)."""
    hist = hist.dropna(how="all")
    if hist.empty:
        return None, "Empty history"

    hist.index = pd.to_datetime(hist.index).date
    hist_pit = hist[hist.index <= as_of_date]

    if len(hist_pit) < sh2.MIN_HISTORY_ROWS:
        return None, f"Insufficient history ({len(hist_pit)} rows)"

    close = hist_pit["Close"]
    volume = hist_pit["Volume"]
    low = hist_pit["Low"]

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    if pd.isna(ma200.iloc[-1]) or pd.isna(ma150.iloc[-1]) or pd.isna(ma50.iloc[-1]):
        return None, "Moving averages not computable"

    price_now = close.iloc[-1]
    ma50_now, ma150_now, ma200_now = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]

    if not (price_now > ma150_now and price_now > ma200_now):
        return None, "Not above 150MA/200MA"
    if not (ma50_now > ma150_now > ma200_now):
        return None, "MA alignment failed (50>150>200)"

    if len(ma200) <= sh2.MA200_TREND_LOOKBACK or pd.isna(ma200.iloc[-1 - sh2.MA200_TREND_LOOKBACK]):
        return None, "Not enough history to confirm 200MA trend"
    if not (ma200_now > ma200.iloc[-1 - sh2.MA200_TREND_LOOKBACK]):
        return None, "200MA not trending up"

    fifty_two_week_low = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    if not (price_now >= fifty_two_week_low * (1 + sh2.MIN_ABOVE_52W_LOW_PCT / 100)):
        return None, "Not enough distance above 52-week low"

    pct_above_50ma = ((price_now - ma50_now) / ma50_now) * 100
    if pct_above_50ma > sh2.EXTENDED_CAP_PCT:
        return None, f"Too extended above 50MA ({pct_above_50ma:.1f}%)"

    window = min(sh2.FRESH_CROSSOVER_WINDOW, len(close) - 1)
    recent_close = close.iloc[-window:]
    recent_ma50 = ma50.iloc[-window:]
    was_below_recently = (recent_close < recent_ma50).any()
    if not was_below_recently:
        return None, f"No fresh 50MA crossover in last {sh2.FRESH_CROSSOVER_WINDOW} days"

    retest_low_price, retest_dist_pct = sh2.find_retest_low(close, ma50, low, sh2.RETEST_LOOKBACK_WINDOW)
    if retest_low_price is None:
        return None, "No confirmed retest yet"
    if retest_dist_pct > sh2.RETEST_MAX_DIST_PCT:
        return None, "Pullback never came within range of the 50MA"
    if retest_dist_pct < -sh2.RETEST_MAX_BREACH_PCT:
        return None, "Retest broke below the 50MA - failed support"
    if price_now <= retest_low_price:
        return None, "Price hasn't recovered above its retest low yet"

    if len(volume) < 50:
        return None, "Insufficient volume history"
    avg_vol_recent = volume.iloc[-10:].mean()
    avg_vol_prior = volume.iloc[-50:-10].mean()
    if avg_vol_prior == 0 or pd.isna(avg_vol_prior):
        return None, "Cannot compute volume baseline"
    vol_ratio = avg_vol_recent / avg_vol_prior
    if vol_ratio < sh2.VOL_SURGE_MIN_RATIO:
        return None, f"No volume surge ({vol_ratio:.2f}x)"

    price_10d_ago = close.iloc[-10]
    price_move_10d_pct = ((price_now - price_10d_ago) / price_10d_ago) * 100
    if not (sh2.PRICE_MOVE_MIN_PCT <= price_move_10d_pct <= sh2.PRICE_MOVE_MAX_PCT):
        return None, f"Price move too large for quiet accumulation ({price_move_10d_pct:.1f}%)"

    turnover_recent = (close.iloc[-20:] * volume.iloc[-20:]).mean()
    turnover_cr = turnover_recent / sh2.CRORE
    if turnover_cr < sh2.MIN_TURNOVER_CR:
        return None, f"Turnover too low (Rs {turnover_cr:.2f} cr/day)"
    liquidity_tier = "Strong" if turnover_cr >= sh2.STRONG_TURNOVER_CR else "Adequate"

    adx_value = sh2.compute_adx(hist_pit, sh2.ADX_PERIOD)
    if pd.isna(adx_value):
        return None, "ADX not computable"
    if adx_value < sh2.MIN_ADX:
        return None, f"ADX too low ({adx_value:.1f}) - choppy/noisy"
    if adx_value > sh2.MAX_ADX:
        return None, f"ADX too high ({adx_value:.1f}) - trend already mature"

    # Market regime filter (Nifty above its own 200 EMA) - same as backtest
    if nifty_hist is not None and not nifty_hist.empty:
        nifty_pit = nifty_hist[nifty_hist.index <= as_of_date]
        if len(nifty_pit) >= 200:
            nifty_ema200 = nifty_pit["Close"].ewm(span=200, adjust=False).mean()
            if nifty_pit["Close"].iloc[-1] <= nifty_ema200.iloc[-1]:
                return None, "Market regime filter: Nifty below its 200 EMA"

    rs_now = None
    if nifty_hist is not None and not nifty_hist.empty:
        nifty_pit = nifty_hist[nifty_hist.index <= as_of_date]
        if len(nifty_pit) >= sh2.RS_MA_PERIOD:
            aligned = pd.DataFrame({"stock": close, "nifty": nifty_pit["Close"]}).dropna()
            if len(aligned) >= sh2.RS_MA_PERIOD:
                rs_ratio = aligned["stock"] / aligned["nifty"]
                if not rs_ratio.empty:
                    rs_now = rs_ratio.iloc[-1]

    atr14 = sh2.compute_atr(hist_pit, sh2.ATR_PERIOD)
    if pd.isna(atr14) or atr14 <= 0:
        stop_loss_price = None
        stop_loss_pct = None
        shares_to_buy = None
        capital_allocated = None
    else:
        structural_stop = retest_low_price - (sh2.STRUCTURAL_STOP_BUFFER_ATR * atr14)
        tightest_allowed = price_now - (sh2.MIN_STOP_ATR_MULT * atr14)
        loosest_allowed = price_now - (sh2.MAX_STOP_ATR_MULT * atr14)

        stop_loss_price = structural_stop
        if stop_loss_price > tightest_allowed:
            stop_loss_price = tightest_allowed
        if stop_loss_price < loosest_allowed:
            stop_loss_price = loosest_allowed

        risk_per_share = price_now - stop_loss_price
        stop_loss_pct = (risk_per_share / price_now) * 100
        shares_to_buy = int(sh2.FIXED_CAPITAL_PER_TRADE / price_now) if price_now > 0 else 0
        capital_allocated = round(shares_to_buy * price_now, 2)

    freshness_score = max(0, sh2.EXTENDED_CAP_PCT - abs(pct_above_50ma))
    conviction_score = (vol_ratio * 50) + (min(turnover_cr, 20) * 2) + freshness_score

    return {
        "Stock": sym_nse,
        "Pick_Date": str(as_of_date),
        "Entry_Date": str(as_of_date),
        "Price_At_Pick": round(float(price_now), 2),
        "Price_At_Entry": round(float(price_now), 2),
        "Stop_Loss_Price": round(float(stop_loss_price), 2) if stop_loss_price is not None else None,
        "Stop_Loss_%": round(float(stop_loss_pct), 2) if stop_loss_pct is not None else None,
        "Suggested_Shares": shares_to_buy,
        "Capital_Allocated_Rs": capital_allocated,
        "Liquidity_Tier": liquidity_tier,
        "ADX_14": round(float(adx_value), 1),
        "Volume_Surge_Ratio": round(float(vol_ratio), 2),
        "Avg_Daily_Turnover_Cr": round(float(turnover_cr), 2),
        "Pct_Above_50MA_At_Pick": round(float(pct_above_50ma), 1),
        "Conviction_Score": round(float(conviction_score), 2),
        "RS_Vs_Nifty": round(float(rs_now), 4) if rs_now is not None else None,
    }, None


def run_live():
    as_of_date = datetime.today().date()
    print(f"Stock Hunter v2 LIVE PICKER - screening as of {as_of_date}")

    universe = sh2.load_universe()
    print(f"Universe: {len(universe)} stocks")

    fetch_start = as_of_date - timedelta(days=sh2.FETCH_BUFFER_DAYS)
    fetch_end = as_of_date + timedelta(days=1)

    print("Downloading NIFTY history for regime filter + relative strength...")
    nifty_hist = yf.download(sh2.NIFTY_TICKER, start=fetch_start, end=fetch_end, progress=False)
    if isinstance(nifty_hist.columns, pd.MultiIndex):
        nifty_hist.columns = nifty_hist.columns.get_level_values(0)
    nifty_hist.index = pd.to_datetime(nifty_hist.index).date

    results = []
    skipped = []

    for chunk in sh2.chunk_list(universe, sh2.CHUNK_SIZE):
        tickers = [f"{s}.NS" for s in chunk]
        data = yf.download(tickers, start=fetch_start, end=fetch_end,
                            group_by="ticker", progress=False, threads=True)
        for sym_nse in chunk:
            ticker = f"{sym_nse}.NS"
            try:
                hist = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
            except Exception:
                skipped.append({"Stock": sym_nse, "Reason": "No data returned"})
                continue
            result, reason = screen_stock_live(hist, as_of_date, sym_nse, nifty_hist)
            if result:
                results.append(result)
            else:
                skipped.append({"Stock": sym_nse, "Reason": reason})
        time.sleep(1)

    if results:
        df_results = pd.DataFrame(results).sort_values("Conviction_Score", ascending=False)
        df_results = df_results.head(sh2.TOP_N_PER_SCAN_DATE)
        df_results.to_csv("backtest_results.csv", index=False)
        print(f"{len(df_results)} stock(s) qualified today "
              f"(capped at TOP_N_PER_SCAN_DATE={sh2.TOP_N_PER_SCAN_DATE}).")
    else:
        pd.DataFrame(columns=["Stock"]).to_csv("backtest_results.csv", index=False)
        print("No stocks qualified today.")

    pd.DataFrame(skipped).to_csv("skipped_stocks.csv", index=False)


if __name__ == "__main__":
    run_live()
