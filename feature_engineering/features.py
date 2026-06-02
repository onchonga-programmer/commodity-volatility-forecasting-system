"""
features.py — Feature Engineering for Commodity Volatility Forecasting
=======================================================================
Author: (your name)
Week 2 deliverable.

Design principles enforced here:
  1. NO lookahead bias — every feature is shifted(1) before it can be used as input.
     The target (forward RV) legitimately looks ahead; features must NOT.
  2. All RV values are annualized (multiplied by sqrt(252)).
  3. Functions are pure — they take a DataFrame, return a DataFrame. No side effects.
  4. A single entry point `build_feature_store()` orchestrates everything.

Glossary for the reader new to finance:
  - Log return: ln(P_t / P_{t-1}). Measures % price change in log space.
  - Realized Volatility (RV): rolling std of log returns, annualized.
  - OHLC: Open, High, Low, Close — the four standard price columns per day.
  - DXY: US Dollar Index. Measures USD strength vs basket of major currencies.
  - ATR: Average True Range — measures daily price range including overnight gaps.
  - Garman-Klass: a volatility estimator that uses full OHLC, more efficient than
    close-to-close std.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252  # Standard in finance — equities and most commodities

# Rolling window sizes (in trading days)
WINDOWS = [5, 10, 21]  # ~1 week, 2 weeks, 1 month

# Forward horizons for our TARGET variables
FORECAST_HORIZONS = [5, 10]

# Paths — adjust if your folder layout differs
DATA_DIR = Path("data/silver")       # cleaned/validated data lives here
FEATURE_DIR = Path("data/gold")     # engineered features go here

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOG RETURNS
# ---------------------------------------------------------------------------

def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Compute log returns from a price series.

    Log return at time t = ln(P_t / P_{t-1})

    Why log and not simple returns?
      - Log returns are time-additive: a 3-day log return = sum of 3 daily log returns.
      - Simple returns are NOT additive, which causes compounding errors.
      - Log returns are approximately symmetric: +10% and -10% are equal magnitude.

    The first row will be NaN because there's no previous price — that's correct.
    """
    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


# ---------------------------------------------------------------------------
# 2. REALIZED VOLATILITY — TARGET AND FEATURES
# ---------------------------------------------------------------------------

def compute_realized_volatility(
    log_returns: pd.Series,
    window: int,
    annualize: bool = True
) -> pd.Series:
    """
    Rolling standard deviation of log returns, optionally annualized.

    RV_t = std(log_returns over last `window` days) * sqrt(252)

    Annualizing: a daily std of 0.01 means 1% daily moves. To express as an annual
    figure (comparable across commodities and timeframes), multiply by sqrt(252).
    This is the convention used everywhere in finance.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log returns.
    window : int
        Look-back window in trading days.
    annualize : bool
        If True, multiply by sqrt(252). Default True.

    Returns
    -------
    pd.Series
        Realized volatility series (same index as input).
    """
    rv = log_returns.rolling(window=window, min_periods=window).std()
    if annualize:
        rv = rv * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rv


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FORWARD realized volatility targets — what we want to PREDICT.

    For horizon H:
      target_rv_{H}d at time t = RV computed over days [t+1, t+H]

    Implementation trick: compute RV normally, then shift BACKWARD by H days.
      - RV with a 5-day window at time t covers [t-4, t].
      - Shifting that series by -5 moves it so it aligns with 5 days earlier.
      - That earlier row now has the "future" RV as its target. Correct.

    WARNING: These columns contain future information. They are ONLY valid as
    training targets. Never use them as input features.
    """
    df = df.copy()
    for h in FORECAST_HORIZONS:
        # Compute rolling RV (looks backward), then shift backward (makes it forward)
        # shift(-h) means: "the value at row t gets the RV that was computed at row t+h"
        rv_series = compute_realized_volatility(df["log_return"], window=h)
        df[f"target_rv_{h}d"] = rv_series.shift(-h)
        logger.debug(f"Added target_rv_{h}d")
    return df


def add_lagged_rv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged realized volatility as INPUT features.

    These are the most important features in the model. Volatility clustering
    means high volatility tends to persist — yesterday's RV is the best single
    predictor of tomorrow's RV.

    CRITICAL: We shift(1) every feature. This ensures that at time t, the model
    only sees RV computed up to time t-1. Without shift(1), the model would see
    the current day's RV when predicting today's forward RV — that's lookahead bias.
    """
    df = df.copy()
    for w in WINDOWS:
        rv = compute_realized_volatility(df["log_return"], window=w)
        # .shift(1) → use yesterday's value. This is the lookahead bias prevention.
        df[f"rv_{w}d"] = rv.shift(1)
        logger.debug(f"Added rv_{w}d feature")
    return df


def add_volatility_of_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility of Volatility (VoV) — the instability of the volatility regime.

    Computed as: rolling std of the RV series itself.

    High VoV means the market is switching between calm and turbulent states.
    For a risk manager, high VoV is dangerous because it signals regime uncertainty.
    For our model, it provides context that a single RV number misses.

    We use the 21d RV series as input (most stable) and compute a 10d std over it.
    Both are shifted(1) for lookahead safety.
    """
    df = df.copy()
    # First compute the 21d RV series (unshifted, intermediate)
    rv_21d_raw = compute_realized_volatility(df["log_return"], window=21)
    # Then compute rolling std of that series — shift(1) at the end
    df["vol_of_vol_10d"] = rv_21d_raw.rolling(window=10).std().shift(1)
    return df


# ---------------------------------------------------------------------------
# 3. PRICE MOMENTUM
# ---------------------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Price momentum: cumulative log return over N days.

    momentum_{N}d at time t = ln(P_{t-1} / P_{t-N-1})

    Why does momentum matter for volatility?
    - Large directional moves (high momentum) often precede or accompany high volatility.
    - A commodity that moved +15% in 10 days is in a different regime than one that
      moved +0.5%. The model needs to know this.
    - We use log returns (sum of daily log returns = cumulative log return).

    All shifted(1) — we use price available up to yesterday.
    """
    df = df.copy()
    for w in WINDOWS:
        # Sum of last w log returns = cumulative return over w days
        momentum = df["log_return"].rolling(window=w).sum()
        df[f"momentum_{w}d"] = momentum.shift(1)
    return df


# ---------------------------------------------------------------------------
# 4. GARMAN-KLASS VOLATILITY ESTIMATOR
# ---------------------------------------------------------------------------

def add_garman_klass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garman-Klass (1980) volatility estimator — uses OHLC data.

    Formula:
        GK_t = 0.5 * [ln(H_t/L_t)]^2 - (2*ln2 - 1) * [ln(C_t/O_t)]^2

    Why is this better than close-to-close std?
    - Close-to-close only uses 1 data point per day (the closing price).
    - GK uses the full range (High-Low) plus the open-to-close move.
    - It's statistically 7x more efficient — same information, less noise.
    - Particularly valuable for commodities where intraday swings are large.

    We compute a rolling 5-day average of daily GK estimates, then shift(1).

    Requires columns: open, high, low, close (lowercase).
    If your data has different column names, rename before calling this.
    """
    df = df.copy()
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found — skipping Garman-Klass")
            return df

    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    # Daily GK estimate (not yet annualized — we'll annualize the rolling version)
    gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    # Rolling 5-day mean of daily GK, then sqrt to get volatility units, annualize
    gk_rolling = np.sqrt(gk_daily.rolling(window=5).mean() * TRADING_DAYS_PER_YEAR)
    df["garman_klass_5d"] = gk_rolling.shift(1)
    return df


# ---------------------------------------------------------------------------
# 5. AVERAGE TRUE RANGE (ATR)
# ---------------------------------------------------------------------------

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average True Range (ATR) — classic technical volatility indicator.

    True Range at time t = max of:
      1. High_t - Low_t                   (today's intraday range)
      2. |High_t - Close_{t-1}|           (gap up + today's high)
      3. |Low_t  - Close_{t-1}|           (gap down + today's low)

    Taking the max captures overnight gaps (news that hits after market close).
    ATR = rolling mean of True Range over `window` days.

    ATR is in price units (e.g., dollars per barrel for WTI). To normalize it
    across different price levels, we divide by the closing price — giving a
    percentage measure that's comparable over time and across commodities.

    Shifted(1) for lookahead safety.
    """
    df = df.copy()
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found — skipping ATR")
            return df

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)

    # Normalize by price so it's a % measure, not dollar amount
    atr = tr.rolling(window=window).mean() / df["close"]
    df[f"atr_{window}d"] = atr.shift(1)
    return df


# ---------------------------------------------------------------------------
# 6. MACRO FEATURES
# ---------------------------------------------------------------------------

def add_macro_features(
    df: pd.DataFrame,
    dxy: pd.Series,
    treasury_yield: pd.Series
) -> pd.DataFrame:
    """
    Add macroeconomic features: DXY changes and treasury yield changes.

    DXY (US Dollar Index):
      - When USD strengthens, commodity prices (priced in USD) tend to fall.
      - Sharp DXY moves signal macro regime shifts that ripple into commodity vol.
      - We use the 1-day change (log return of DXY) and 5-day change.

    10-Year Treasury Yield:
      - The risk-free benchmark rate. When it spikes (e.g., rate hike expectations),
        it signals tightening financial conditions → affects all risk assets including
        commodities.
      - We use the day-over-day change in yield (not log return — yields can be negative,
        and convention in rates is to use absolute changes, not log changes).

    Both are lagged: we use yesterday's macro value as a feature for today's prediction.
    This is both correct (no lookahead) and realistic (macro data has publication lags).

    Parameters
    ----------
    df : pd.DataFrame
        Commodity feature DataFrame with a DatetimeIndex.
    dxy : pd.Series
        DXY level series, DatetimeIndex aligned.
    treasury_yield : pd.Series
        10Y treasury yield series, DatetimeIndex aligned.
    """
    df = df.copy()

    # Align macro series to commodity index (forward-fill gaps for non-trading days)
    dxy_aligned = dxy.reindex(df.index).ffill()
    yield_aligned = treasury_yield.reindex(df.index).ffill()

    # DXY log return (1d and 5d)
    dxy_ret_1d = np.log(dxy_aligned / dxy_aligned.shift(1))
    dxy_ret_5d = np.log(dxy_aligned / dxy_aligned.shift(5))
    df["dxy_change_1d"] = dxy_ret_1d.shift(1)
    df["dxy_change_5d"] = dxy_ret_5d.shift(1)

    # DXY level (normalized — divide by its 252-day mean to make it stationary-ish)
    dxy_norm = dxy_aligned / dxy_aligned.rolling(252).mean()
    df["dxy_level_norm"] = dxy_norm.shift(1)

    # Treasury yield absolute change (1d and 5d)
    yield_chg_1d = yield_aligned.diff(1)
    yield_chg_5d = yield_aligned.diff(5)
    df["treasury_chg_1d"] = yield_chg_1d.shift(1)
    df["treasury_chg_5d"] = yield_chg_5d.shift(1)

    # Yield level (useful as a regime indicator)
    df["treasury_yield_level"] = yield_aligned.shift(1)

    return df


# ---------------------------------------------------------------------------
# 7. CROSS-COMMODITY FEATURES
# ---------------------------------------------------------------------------

def add_cross_commodity_features(
    df: pd.DataFrame,
    other_rv: pd.Series,
    other_name: str
) -> pd.DataFrame:
    """
    Add another commodity's realized volatility as a feature.

    Example: WTI crude oil volatility as a feature for the natural gas model.
    Energy markets are correlated — a crude oil spike often precedes or accompanies
    a natural gas spike.

    This is a simple but effective feature that captures cross-market information.
    A single-commodity model fundamentally cannot learn this relationship.

    Parameters
    ----------
    df : pd.DataFrame
        Target commodity's feature DataFrame.
    other_rv : pd.Series
        The other commodity's 5-day RV series (already computed, DatetimeIndex).
    other_name : str
        Short name for column labeling, e.g. "wti" or "natgas".
    """
    df = df.copy()
    # Align to target's index and shift(1) for lookahead safety
    aligned = other_rv.reindex(df.index).ffill().shift(1)
    df[f"cross_{other_name}_rv5d"] = aligned
    return df


# ---------------------------------------------------------------------------
# 8. CALENDAR FEATURES
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar / time-based features.

    Why do calendar features matter in commodity markets?

    1. Futures expiry: Commodity prices are futures contracts, not spot prices.
       Each contract has a fixed expiry date. In the ~5 days before expiry,
       trading concentrates in the expiring contract, causing atypical volatility.
       We approximate expiry proximity with a crude heuristic (last 5 trading days
       of each month — real expiry dates vary, but this captures most of the effect).

    2. Day of week: Monday opens often see higher volatility (weekend news, geopolitical
       events that happened while markets were closed). Friday can see position-squaring.

    3. Month: Energy markets are seasonal — winter demand for heating fuel, summer for
       agricultural commodities during planting/harvest.

    4. Quarter end: Portfolio rebalancing at quarter end creates unusual flows.
    """
    df = df.copy()
    idx = df.index

    # Day of week (0=Monday, 4=Friday)
    df["day_of_week"] = idx.dayofweek
    df["is_monday"] = (idx.dayofweek == 0).astype(int)
    df["is_friday"] = (idx.dayofweek == 4).astype(int)

    # Month (1–12) — captures seasonality
    df["month"] = idx.month

    # Quarter (1–4)
    df["quarter"] = idx.quarter
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)

    # Approximate futures expiry proximity
    # Heuristic: last 5 trading days of month are "expiry week"
    # A more precise version would use actual CME expiry calendars
    days_in_month = idx.days_in_month
    day_of_month = idx.day
    df["days_to_month_end"] = days_in_month - day_of_month
    df["near_expiry"] = (df["days_to_month_end"] <= 5).astype(int)

    # Calendar features don't need shifting — they're observable at time t
    # (you know what day it is today)
    return df


# ---------------------------------------------------------------------------
# 9. MASTER PIPELINE
# ---------------------------------------------------------------------------

def build_feature_store(
    commodity_df: pd.DataFrame,
    dxy: pd.Series = None,
    treasury_yield: pd.Series = None,
    cross_rv: pd.Series = None,
    cross_name: str = "other",
    commodity_name: str = "commodity"
) -> pd.DataFrame:
    """
    Master function that applies all feature engineering steps in order.

    Steps:
      1. Compute log returns (foundation of everything)
      2. Add forward RV targets (what we want to predict)
      3. Add lagged RV features (most important inputs)
      4. Add volatility-of-volatility
      5. Add price momentum
      6. Add Garman-Klass volatility (requires OHLC)
      7. Add ATR (requires OHLC)
      8. Add macro features (optional — requires external series)
      9. Add cross-commodity features (optional — requires other commodity's RV)
     10. Add calendar features
     11. Drop rows with NaN (NaN rows are from rolling windows warming up)
     12. Save to gold layer

    Parameters
    ----------
    commodity_df : pd.DataFrame
        Must have DatetimeIndex and at minimum a 'close' column.
        For full features: also 'open', 'high', 'low'.
    dxy : pd.Series, optional
        DXY index series. If None, macro features are skipped.
    treasury_yield : pd.Series, optional
        10Y yield series. If None, macro features are skipped.
    cross_rv : pd.Series, optional
        Another commodity's RV series for cross-commodity features.
    cross_name : str
        Label for the cross-commodity feature column.
    commodity_name : str
        Used for saving the output CSV file.

    Returns
    -------
    pd.DataFrame
        Full feature store with targets and all features.
        Rows with NaN dropped (initial warm-up period).
    """
    logger.info(f"Building feature store for {commodity_name}")

    df = commodity_df.copy()

    # Ensure DatetimeIndex is sorted (always sort time series data!)
    df = df.sort_index()

    # --- Step 1: Log returns ---
    df = add_log_returns(df)

    # --- Step 2: Forward RV targets ---
    df = add_target_variables(df)

    # --- Step 3: Lagged RV features ---
    df = add_lagged_rv_features(df)

    # --- Step 4: Volatility of volatility ---
    df = add_volatility_of_volatility(df)

    # --- Step 5: Price momentum ---
    df = add_momentum_features(df)

    # --- Step 6: Garman-Klass (only if OHLC available) ---
    if all(c in df.columns for c in ["open", "high", "low"]):
        df = add_garman_klass(df)
    else:
        logger.warning(f"{commodity_name}: OHLC columns missing, skipping Garman-Klass")

    # --- Step 7: ATR ---
    if all(c in df.columns for c in ["high", "low"]):
        df = add_atr(df)
    else:
        logger.warning(f"{commodity_name}: OHLC columns missing, skipping ATR")

    # --- Step 8: Macro features ---
    if dxy is not None and treasury_yield is not None:
        df = add_macro_features(df, dxy, treasury_yield)
    else:
        logger.warning(f"{commodity_name}: Macro series not provided, skipping macro features")

    # --- Step 9: Cross-commodity ---
    if cross_rv is not None:
        df = add_cross_commodity_features(df, cross_rv, cross_name)

    # --- Step 10: Calendar features ---
    df = add_calendar_features(df)

    # --- Step 11: Drop NaN rows ---
    # Rolling windows of up to 21 days + shift(1) = first ~22 rows will have NaN.
    # Target variables create NaN at the END (last H rows have no future data yet).
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    logger.info(f"{commodity_name}: Dropped {dropped} NaN rows ({initial_rows} → {len(df)})")

    # --- Step 12: Save to gold layer ---
    output_path = FEATURE_DIR / f"{commodity_name}_features.csv"
    df.to_csv(output_path)
    logger.info(f"Saved feature store to {output_path}")

    return df


# ---------------------------------------------------------------------------
# 10. VALIDATION CHECKS
# ---------------------------------------------------------------------------

def validate_feature_store(df: pd.DataFrame, commodity_name: str = "") -> bool:
    """
    Sanity checks to run after feature engineering.

    These catch the most common mistakes before they silently corrupt your model.
    Add this to your test suite (Week 8).
    """
    passed = True

    # 1. No remaining NaN in feature columns (targets at tail are OK — check features)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    nan_count = df[feature_cols].isna().sum().sum()
    if nan_count > 0:
        logger.error(f"{commodity_name}: {nan_count} NaN values in feature columns!")
        passed = False

    # 2. Realized volatility must be positive
    rv_cols = [c for c in df.columns if "rv" in c]
    for col in rv_cols:
        if (df[col] < 0).any():
            logger.error(f"{commodity_name}: Negative values found in {col}!")
            passed = False

    # 3. Target columns must look forward — they should NOT be correlated ~1.0
    # with same-day features (that would indicate lookahead bias)
    for h in FORECAST_HORIZONS:
        target_col = f"target_rv_{h}d"
        if target_col in df.columns and f"rv_{h}d" in df.columns:
            corr = df[target_col].corr(df[f"rv_{h}d"])
            if corr > 0.98:
                logger.warning(
                    f"{commodity_name}: Suspiciously high correlation ({corr:.3f}) "
                    f"between {target_col} and rv_{h}d — check for lookahead bias!"
                )
                # Not a hard fail but a red flag

    # 4. Index must be DatetimeIndex and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f"{commodity_name}: Index is not DatetimeIndex!")
        passed = False
    elif not df.index.is_monotonic_increasing:
        logger.error(f"{commodity_name}: Index is not sorted!")
        passed = False

    # 5. Minimum row count — need enough data for Walk-Forward Validation in Week 5
    if len(df) < 500:
        logger.warning(
            f"{commodity_name}: Only {len(df)} rows after feature engineering. "
            "Consider fetching more historical data."
        )

    if passed:
        logger.info(f"{commodity_name}: All validation checks passed ✓")
    return passed


# ---------------------------------------------------------------------------
# EXAMPLE USAGE (run this file directly to test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    # ---- Load your silver-layer data ----
    # Adjust the paths and column names to match your ingest.py output.
    # Expected format: DatetimeIndex, columns = [open, high, low, close, volume]

    silver_path = Path("data/silver")

    commodities = {
        "wti":    silver_path / "wti_crude_daily.csv",
        "natgas": silver_path / "natgas_daily.csv",
        "wheat":  silver_path / "wheat_daily.csv",
        "corn":   silver_path / "corn_daily.csv",
    }

    macro_path = silver_path / "macro_daily.csv"

    # Load macro data (DXY and treasury yield)
    macro_df = None
    dxy_series = None
    treasury_series = None

    if macro_path.exists():
        macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        # Adjust column names to match your FRED ingest output
        if "dxy" in macro_df.columns:
            dxy_series = macro_df["dxy"]
        if "treasury_10y" in macro_df.columns:
            treasury_series = macro_df["treasury_10y"]

    # Build feature store for each commodity
    feature_stores = {}
    for name, path in commodities.items():
        if not path.exists():
            logger.warning(f"Data file not found: {path} — skipping {name}")
            continue

        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        raw.columns = raw.columns.str.lower()  # Normalize column names

        # For cross-commodity: WTI RV is a feature for natgas (and vice versa)
        cross_rv = None
        cross_name = "none"
        if name == "natgas" and "wti" in feature_stores:
            # Use WTI's rv_5d column from already-built WTI feature store
            cross_rv = feature_stores["wti"]["rv_5d"]
            cross_name = "wti"
        elif name == "wti" and "natgas" in feature_stores:
            cross_rv = feature_stores["natgas"]["rv_5d"]
            cross_name = "natgas"

        fs = build_feature_store(
            commodity_df=raw,
            dxy=dxy_series,
            treasury_yield=treasury_series,
            cross_rv=cross_rv,
            cross_name=cross_name,
            commodity_name=name
        )

        validate_feature_store(fs, commodity_name=name)
        feature_stores[name] = fs

        print(f"\n{'='*60}")
        print(f"Feature store: {name.upper()}")
        print(f"Shape: {fs.shape}")
        print(f"Date range: {fs.index[0].date()} → {fs.index[-1].date()}")
        print(f"\nFeature columns:")
        feature_cols = [c for c in fs.columns if not c.startswith("target_")]
        for col in feature_cols:
            print(f"  {col:<35} mean={fs[col].mean():>8.4f}  std={fs[col].std():>8.4f}")
        print(f"\nTarget columns:")
        target_cols = [c for c in fs.columns if c.startswith("target_")]
        for col in target_cols:
            print(f"  {col:<35} mean={fs[col].mean():>8.4f}  std={fs[col].std():>8.4f}")