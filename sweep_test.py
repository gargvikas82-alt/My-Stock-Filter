"""
Stock Hunter v2 - Parameter Sweep Test
========================================
Runs SEVERAL filter configurations against the SAME downloaded price data in
one execution, so isolating which change actually matters doesn't cost a
separate GitHub Actions run (and a separate multi-minute yfinance download)
per variant. Downloads once, evaluates N times, prints one comparison table.

Configurations tested (see CONFIGS below):
  A) ORIGINAL       - your pre-loosening thresholds (the untouched main-branch
                       values), included as the reference/sanity-check point.
  B) LOOSENED_ALL    - everything loosened together (current filter-experiments
                       state) - this is the run already reported on.
  C) LOOSENED_MINUS_ADX     - all loosening kept, EXCEPT MAX_ADX restored to
                       the original 32 ceiling. Isolates what the ADX widening
                       (32->40) alone is contributing.
  D) LOOSENED_MINUS_RETEST  - all loosening kept, EXCEPT retest tolerances
                       restored to the original 3%/3%. Isolates what the
                       retest widening (3%->5%) alone is contributing.

Position sizing (TOTAL_CAPITAL / RISK_PCT_PER_TRADE / MAX_POSITION_PCT) is
held constant across all four configs at the already-agreed values (Rs 10L
kitty, 0.5% risk/trade, 20% position cap) - sizing isn't what's in question
here, only the entry filter thresholds are being isolated.

This script is intentionally self-contained (does not import stock_hunter_v2)
so that changes here can never accidentally affect the already-tested
production filter logic in stock_hunter_v2.py or live_picker.py.
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

UNIVERSE_FILE = "nifty_total_market.csv"
NIFTY_TICKER = "^NSEI"
CHUNK_SIZE = 50
FETCH_BUFFER_DAYS = 400

MIN_HISTORY_ROWS = 260
MA200_TREND_LOOKBACK = 20
RETEST_LOOKBACK_WINDOW = 20
MIN_ABOVE_52W_LOW_PCT = 25
STRUCTURAL_STOP_BUFFER_ATR = 0.5
MIN_STOP_ATR_MULT = 3.0
MAX_STOP_ATR_MULT = 7.0
MIN_TURNOVER_CR = 1.0
STRONG_TURNOVER_CR = 5.0
CORPORATE_ACTION_THRESHOLD_PCT = 20
MAX_HOLDING_DAYS = int(os.environ.get("MAX_HOLDING_DAYS", "60"))
TOP_N_PER_SCAN_DATE = int(os.environ.get("TOP_N_PER_SCAN_DATE", "2"))
ATR_PERIOD = 14
ADX_PERIOD = 14
MIN_ADX = 25
RS_MA_PERIOD = 50
TOTAL_CAPITAL = float(os.environ.get("TOTAL_CAPITAL", "1000000"))
RISK_PCT_PER_TRADE = float(os.environ.get("RISK_PCT_PER_TRADE", "0.5"))
MAX_POSITION_PCT_OF_CAPITAL = float(os.environ.get("MAX_POSITION_PCT_OF_CAPITAL", "20"))
ROUND_TRIP_COST_PCT = float(os.environ.get("ROUND_TRIP_COST_PCT", "0.7"))
CRORE = 10_000_000

# --- The four configurations under test ---
CONFIGS = {
    "A_ORIGINAL": {
        "FRESH_CROSSOVER_WINDOW": 10, "EXTENDED_CAP_PCT": 8,
        "VOL_SURGE_MIN_RATIO": 1.5, "PRICE_MOVE_MIN_PCT": -3, "PRICE_MOVE_MAX_PCT": 8,
        "RETEST_MAX_DIST_PCT": 3.0, "RETEST_MAX_BREACH_PCT": 3.0, "MAX_ADX": 32,
    },
    "B_LOOSENED_ALL": {
        "FRESH_CROSSOVER_WINDOW": 15, "EXTENDED_CAP_PCT": 12,
        "VOL_SURGE_MIN_RATIO": 1.3, "PRICE_MOVE_MIN_PCT": -5, "PRICE_MOVE_MAX_PCT": 10,
        "RETEST_MAX_DIST_PCT": 5.0, "RETEST_MAX_BREACH_PCT": 5.0, "MAX_ADX": 40,
    },
    "C_LOOSENED_MINUS_ADX": {
        "FRESH_CROSSOVER_WINDOW": 15, "EXTENDED_CAP_PCT": 12,
        "VOL_SURGE_MIN_RATIO": 1.3, "PRICE_MOVE_MIN_PCT": -5, "PRICE_MOVE_MAX_PCT": 10,
        "RETEST_MAX_DIST_PCT": 5.0, "RETEST_MAX_BREACH_PCT": 5.0, "MAX_ADX": 32,  # restored
    },
    "D_LOOSENED_MINUS_RETEST": {
        "FRESH_CROSSOVER_WINDOW": 15, "EXTENDED_CAP_PCT": 12,
        "VOL_SURGE_MIN_RATIO": 1.3, "PRICE_MOVE_MIN_PCT": -5, "PRICE_MOVE_MAX_PCT": 10,
        "RETEST_MAX_DIST_PCT": 3.0, "RETEST_MAX_BREACH_PCT": 3.0,  # restored
        "MAX_ADX": 40,
    },
}


def get_dates():
    from_env = os.environ.get("FROM_DATE", "").strip()
    to_env = os.environ.get("TO_DATE", "").strip()
    if not from_env:
        print("FATAL: FROM_DATE is required.")
        sys.exit(1)
    from_date = pd.to_datetime(from_env).date()
    to_date = pd.to_datetime(to_env).date() if to_env else datetime.today().date()
    if to_date <= from_date:
        print(f"FATAL: TO_DATE ({to_date}) must be after FROM_DATE ({from_date}).")
        sys.exit(1)
    return from_date, to_date


def load_universe():
    df = pd.read_csv(UNIVERSE_FILE)
    return sorted(df["Symbol"].dropna().unique().tolist())


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def compute_atr(hist_pit, period=ATR_PERIOD):
    high, low, close = hist_pit["High"], hist_pit["Low"], hist_pit["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().iloc[-1]


def compute_adx(hist_pit, period=ADX_PERIOD):
    high, low, close = hist_pit["High"], hist_pit["Low"], hist_pit["Close"]
    prev_close, prev_high, prev_low = close.shift(1), high.shift(1), low.shift(1)
    up_move, down_move = high - prev_high, prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=hist_pit.index).ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=hist_pit.index).ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]


def simulate_realistic_exit(hist_full, entry_price, stop_loss_price, max_holding_days):
    for i in range(1, len(hist_full)):
        day_close = hist_full["Close"].iloc[i]
        day_date = hist_full.index[i]
        if stop_loss_price is not None and day_close <= stop_loss_price:
            return {"exit_price": day_close, "exit_date": day_date, "exit_reason": "Stop_Loss_Hit", "holding_days_realistic": i}
        if i >= max_holding_days:
            return {"exit_price": day_close, "exit_date": day_date, "exit_reason": "Max_Holding_Period", "holding_days_realistic": i}
    return {"exit_price": hist_full["Close"].iloc[-1], "exit_date": hist_full.index[-1],
            "exit_reason": "End_Of_Backtest_Data", "holding_days_realistic": len(hist_full) - 1}


def find_retest_low(close, ma50, low, cross_window):
    window = min(cross_window, len(close) - 1)
    recent_close, recent_ma50, recent_low = close.iloc[-window:], ma50.iloc[-window:], low.iloc[-window:]
    below_mask = recent_close < recent_ma50
    if not below_mask.any():
        return None, None
    below_positions = [i for i, v in enumerate(below_mask.values) if v]
    cross_pos = below_positions[-1] + 1
    if cross_pos >= len(recent_close) - 1:
        return None, None
    post_cross_low = recent_low.iloc[cross_pos:-1]
    post_cross_ma50 = recent_ma50.iloc[cross_pos:-1]
    if post_cross_low.empty:
        return None, None
    dist_to_ma = (post_cross_low - post_cross_ma50) / post_cross_ma50 * 100
    idx = dist_to_ma.idxmin()
    return post_cross_low.loc[idx], dist_to_ma.loc[idx]


def evaluate_stock(hist, from_date, to_date, sym_nse, nifty_hist, cfg):
    """Same logic as stock_hunter_v2.evaluate_stock, but every threshold comes
    from cfg instead of module constants - lets one process evaluate the same
    stock/date under N different configurations without re-fetching data."""
    hist = hist.dropna(how="all")
    if hist.empty:
        return None
    hist.index = pd.to_datetime(hist.index).date
    hist_pit = hist[hist.index <= from_date]
    if len(hist_pit) < MIN_HISTORY_ROWS:
        return None

    close, volume, low = hist_pit["Close"], hist_pit["Volume"], hist_pit["Low"]
    ma50, ma150, ma200 = close.rolling(50).mean(), close.rolling(150).mean(), close.rolling(200).mean()
    if pd.isna(ma200.iloc[-1]) or pd.isna(ma150.iloc[-1]) or pd.isna(ma50.iloc[-1]):
        return None
    price_now, ma50_now, ma150_now, ma200_now = close.iloc[-1], ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]

    if not (price_now > ma150_now and price_now > ma200_now):
        return None
    if not (ma50_now > ma150_now > ma200_now):
        return None
    if len(ma200) <= MA200_TREND_LOOKBACK or pd.isna(ma200.iloc[-1 - MA200_TREND_LOOKBACK]):
        return None
    if not (ma200_now > ma200.iloc[-1 - MA200_TREND_LOOKBACK]):
        return None
    fifty_two_week_low = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    if not (price_now >= fifty_two_week_low * (1 + MIN_ABOVE_52W_LOW_PCT / 100)):
        return None

    pct_above_50ma = ((price_now - ma50_now) / ma50_now) * 100
    if pct_above_50ma > cfg["EXTENDED_CAP_PCT"]:
        return None
    window = min(cfg["FRESH_CROSSOVER_WINDOW"], len(close) - 1)
    if not (close.iloc[-window:] < ma50.iloc[-window:]).any():
        return None

    retest_low_price, retest_dist_pct = find_retest_low(close, ma50, low, RETEST_LOOKBACK_WINDOW)
    if retest_low_price is None:
        return None
    if retest_dist_pct > cfg["RETEST_MAX_DIST_PCT"]:
        return None
    if retest_dist_pct < -cfg["RETEST_MAX_BREACH_PCT"]:
        return None
    if price_now <= retest_low_price:
        return None

    if len(volume) < 50:
        return None
    avg_vol_recent, avg_vol_prior = volume.iloc[-10:].mean(), volume.iloc[-50:-10].mean()
    if avg_vol_prior == 0 or pd.isna(avg_vol_prior):
        return None
    vol_ratio = avg_vol_recent / avg_vol_prior
    if vol_ratio < cfg["VOL_SURGE_MIN_RATIO"]:
        return None
    price_move_10d_pct = ((price_now - close.iloc[-10]) / close.iloc[-10]) * 100
    if not (cfg["PRICE_MOVE_MIN_PCT"] <= price_move_10d_pct <= cfg["PRICE_MOVE_MAX_PCT"]):
        return None

    turnover_cr = (close.iloc[-20:] * volume.iloc[-20:]).mean() / CRORE
    if turnover_cr < MIN_TURNOVER_CR:
        return None

    adx_value = compute_adx(hist_pit, ADX_PERIOD)
    if pd.isna(adx_value) or adx_value < MIN_ADX or adx_value > cfg["MAX_ADX"]:
        return None

    if nifty_hist is not None and not nifty_hist.empty:
        nifty_pit = nifty_hist[nifty_hist.index <= from_date]
        if len(nifty_pit) >= 200:
            nifty_ema200 = nifty_pit["Close"].ewm(span=200, adjust=False).mean()
            if nifty_pit["Close"].iloc[-1] <= nifty_ema200.iloc[-1]:
                return None

    hist_from_signal = hist[(hist.index >= from_date) & (hist.index <= to_date)]
    if hist_from_signal.empty or len(hist_from_signal) < 2:
        return None
    hist_full = hist_from_signal.iloc[1:]
    if hist_full.empty:
        return None
    entry_price, entry_date_actual = hist_full["Open"].iloc[0], hist_full.index[0]

    atr14 = compute_atr(hist_pit, ATR_PERIOD)
    if pd.isna(atr14) or atr14 <= 0:
        return None
    structural_stop = retest_low_price - (STRUCTURAL_STOP_BUFFER_ATR * atr14)
    tightest_allowed = entry_price - (MIN_STOP_ATR_MULT * atr14)
    loosest_allowed = entry_price - (MAX_STOP_ATR_MULT * atr14)
    stop_loss_price = structural_stop
    if stop_loss_price > tightest_allowed:
        stop_loss_price = tightest_allowed
    if stop_loss_price < loosest_allowed:
        stop_loss_price = loosest_allowed
    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        return None

    risk_amount_rs = TOTAL_CAPITAL * (RISK_PCT_PER_TRADE / 100)
    shares_by_risk = int(risk_amount_rs / risk_per_share)
    capital_by_risk = shares_by_risk * entry_price
    max_capital_allowed = TOTAL_CAPITAL * (MAX_POSITION_PCT_OF_CAPITAL / 100)
    shares_to_buy = int(max_capital_allowed / entry_price) if capital_by_risk > max_capital_allowed else shares_by_risk
    if shares_to_buy <= 0:
        return None
    capital_allocated = round(shares_to_buy * entry_price, 2)

    exit_sim = simulate_realistic_exit(hist_full, entry_price, stop_loss_price, MAX_HOLDING_DAYS)
    realistic_return_pct = ((exit_sim["exit_price"] - entry_price) / entry_price) * 100
    realistic_return_net_pct = realistic_return_pct - ROUND_TRIP_COST_PCT

    hist_held = hist_full.iloc[:exit_sim["holding_days_realistic"] + 1]
    daily_pct_changes = hist_held["Close"].pct_change().dropna() * 100
    if not daily_pct_changes[daily_pct_changes.abs() >= CORPORATE_ACTION_THRESHOLD_PCT].empty:
        return None

    freshness_score = max(0, cfg["EXTENDED_CAP_PCT"] - abs(pct_above_50ma))
    conviction_score = (vol_ratio * 50) + (min(turnover_cr, 20) * 2) + freshness_score

    return {
        "Stock": sym_nse,
        "Pick_Date": entry_date_actual.strftime("%Y-%m-%d"),
        "Capital_Allocated_Rs": capital_allocated,
        "Realistic_Return_Net_%": round(float(realistic_return_net_pct), 2),
        "Exit_Reason": exit_sim["exit_reason"],
        "Conviction_Score": round(float(conviction_score), 2),
    }


def get_scan_dates(from_date, to_date):
    weekday_str = os.environ.get("SCAN_WEEKDAYS", "1,4")
    scan_weekdays = tuple(int(x.strip()) for x in weekday_str.split(",") if x.strip() != "")
    all_days = pd.date_range(from_date, to_date, freq="D")
    scan_dates = [d.date() for d in all_days if d.weekday() in scan_weekdays]
    if not scan_dates or scan_dates[0] != from_date:
        scan_dates = [from_date] + scan_dates
    return sorted(set(scan_dates))


def sharpe_like(returns):
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return None
    return returns.mean() / returns.std()


def run():
    from_date, to_date = get_dates()
    scan_dates = get_scan_dates(from_date, to_date)
    print(f"SWEEP TEST - {len(CONFIGS)} configurations, {len(scan_dates)} scan dates, "
          f"{from_date} to {to_date}")
    print("=" * 75)

    symbols = load_universe()
    print(f"Universe: {len(symbols)} stocks - downloading price data ONCE for all configs...")

    fetch_start = (from_date - timedelta(days=FETCH_BUFFER_DAYS)).strftime("%Y-%m-%d")
    fetch_end = (to_date + timedelta(days=1)).strftime("%Y-%m-%d")

    nifty_hist = yf.download(NIFTY_TICKER, start=fetch_start, end=fetch_end, interval="1d",
                              auto_adjust=False, actions=False, progress=False)
    if isinstance(nifty_hist.columns, pd.MultiIndex):
        nifty_hist.columns = nifty_hist.columns.get_level_values(0)
    nifty_hist.index = pd.to_datetime(nifty_hist.index).date

    # --- Download every stock's history ONCE, keep it all in memory ---
    all_hist = {}
    total_chunks = (len(symbols) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_num, chunk in enumerate(chunk_list(symbols, CHUNK_SIZE), start=1):
        yf_tickers = [f"{s}.NS" for s in chunk]
        print(f"[{chunk_num}/{total_chunks}] Fetching {len(chunk)} tickers...")
        try:
            data = yf.download(tickers=yf_tickers, start=fetch_start, end=fetch_end, interval="1d",
                                auto_adjust=False, actions=False, group_by="ticker", threads=True, progress=False)
        except Exception:
            continue
        for sym_nse, sym_yf in zip(chunk, yf_tickers):
            try:
                hist = data if len(yf_tickers) == 1 else (
                    data[sym_yf] if sym_yf in data.columns.get_level_values(0) else None)
                if hist is not None:
                    all_hist[sym_nse] = hist
            except Exception:
                continue

    print(f"\nData download complete: {len(all_hist)} stocks with usable history.")
    print("Now evaluating each configuration against this same dataset...\n")

    summary_rows = []
    for config_name, cfg in CONFIGS.items():
        results = []
        for sym_nse, hist in all_hist.items():
            for scan_date in scan_dates:
                if scan_date >= to_date:
                    continue
                result = evaluate_stock(hist, scan_date, to_date, sym_nse, nifty_hist, cfg)
                if result:
                    result["Scan_Date"] = scan_date.strftime("%Y-%m-%d")
                    results.append(result)

        df_results = pd.DataFrame(results)
        if df_results.empty:
            summary_rows.append({
                "Config": config_name, "MAX_ADX": cfg["MAX_ADX"],
                "Retest_Dist_%": cfg["RETEST_MAX_DIST_PCT"], "Retest_Breach_%": cfg["RETEST_MAX_BREACH_PCT"],
                "N_Picks_Before_Cap": 0, "N_Picks_After_Cap": 0, "Unique_Stocks": 0,
                "Win_Rate_%": None, "Mean_Return_Net_%": None, "Sharpe_Like": None, "Stop_Hit_Rate_%": None,
            })
            print(f"{config_name}: NO QUALIFYING PICKS on any scan date.\n")
            continue

        before_cap = len(df_results)
        df_results = (df_results.sort_values("Conviction_Score", ascending=False)
                      .groupby("Scan_Date", group_keys=False).head(TOP_N_PER_SCAN_DATE))
        win_rate = (df_results["Realistic_Return_Net_%"] > 0).mean() * 100
        mean_ret = df_results["Realistic_Return_Net_%"].mean()
        sharpe = sharpe_like(df_results["Realistic_Return_Net_%"])
        stop_rate = (df_results["Exit_Reason"] == "Stop_Loss_Hit").mean() * 100

        summary_rows.append({
            "Config": config_name, "MAX_ADX": cfg["MAX_ADX"],
            "Retest_Dist_%": cfg["RETEST_MAX_DIST_PCT"], "Retest_Breach_%": cfg["RETEST_MAX_BREACH_PCT"],
            "N_Picks_Before_Cap": before_cap, "N_Picks_After_Cap": len(df_results),
            "Unique_Stocks": df_results["Stock"].nunique(),
            "Win_Rate_%": round(win_rate, 1), "Mean_Return_Net_%": round(mean_ret, 2),
            "Sharpe_Like": round(sharpe, 3) if sharpe is not None else None,
            "Stop_Hit_Rate_%": round(stop_rate, 1),
        })
        df_results.to_csv(f"sweep_results_{config_name}.csv", index=False)
        print(f"{config_name}: {before_cap} before cap -> {len(df_results)} after cap "
              f"({df_results['Stock'].nunique()} unique stocks)")
        print(f"  Win rate: {win_rate:.1f}%  Mean return net: {mean_ret:.2f}%  "
              f"Sharpe-like: {sharpe:.3f}  Stop-hit rate: {stop_rate:.1f}%\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("sweep_summary.csv", index=False)

    print("=" * 75)
    print("SWEEP SUMMARY (side-by-side comparison)")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print(f"\nFull summary saved to sweep_summary.csv. Per-config picks saved to "
          f"sweep_results_<CONFIG_NAME>.csv")
    print("\nHow to read this:")
    print("  A vs B tells you the total effect of loosening everything together.")
    print("  B vs C tells you what the ADX widening (32->40) alone is contributing")
    print("    (C has ADX restored to 32, everything else stays loosened).")
    print("  B vs D tells you what the retest widening (3%->5%) alone is contributing")
    print("    (D has retest restored to 3%/3%, everything else stays loosened).")


if __name__ == "__main__":
    run()
