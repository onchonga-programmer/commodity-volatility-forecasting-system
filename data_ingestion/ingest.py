# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import os
import random
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf      # used for CORN only — WEAT uses Stooq (see below)
from dotenv import load_dotenv
from fredapi import Fred
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

# API Keys — loaded from .env, never hardcoded
EIA_API_KEY  = os.getenv("EIA_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Date range
DEFAULT_START = "2015-01-01"
DEFAULT_END   = date.today().strftime("%Y-%m-%d")

# EIA series IDs
EIA_SERIES = {
    "wti":    "PET.RWTC.D",
    "natgas": "NG.RNGWHHD.D",
}

# ── Agricultural data source config ──────────────────────────────────────────
#
# WEAT → Stooq (stooq.com, no API key required)
# -----------------------------------------------
# WEAT executed a 1-for-5 reverse split on Nov 25 2025, which changed its
# CUSIP number. This permanently breaks yfinance for WEAT — it cannot stitch
# pre-split and post-split history across the CUSIP boundary, returning an
# empty DataFrame regardless of yfinance version. ZW=F (the CBOT futures
# fallback) is also broken in yfinance for the same underlying reason.
# Stooq serves the same WEAT ETF OHLCV data reliably, for free, with no
# authentication. Confirmed working as of May 2026.
#
# CORN → yfinance (yahoo finance)
# --------------------------------
# CORN has no CUSIP issue and downloads cleanly from yfinance after upgrading
# to the latest version (confirmed working May 2026). Stays on yfinance.
#
# WHY back_adjust MATTERS FOR CORN:
#   back_adjust=True retroactively rescales all historical prices relative to
#   today's price, making log returns across any past split dates economically
#   correct. Without it, a historical split creates a fake one-day return that
#   the volatility model would learn as an extreme event.
#
STOOQ_CONFIG: dict[str, str] = {
    # Stooq uses lowercase ticker + ".us" suffix for US-listed securities
    "weat": "weat.us",
}

YFINANCE_CONFIG: dict[str, str] = {
    # Standard Yahoo Finance tickers — verified working after yfinance upgrade
    "corn": "CORN",
}

# FRED series IDs
FRED_SERIES = {
    "dxy":       "DTWEXBGS",
    "yield_10y": "DGS10",
}

# Forward fill limit for calendar gaps (days)
MAX_FILL_DAYS = 3

# Retry configuration
MAX_RETRIES = 3

# Base delay for non-rate-limit retries (seconds).
# Actual wait = (2 ** attempt) + jitter — exponential backoff.
BASE_RETRY_DELAY = 2

# Multiplier applied on top of exponential backoff when a 429 is detected.
# 429 = "Too Many Requests" — Yahoo is throttling us. Wait much longer.
RATE_LIMIT_MULTIPLIER = 4

try:
    import great_expectations as gx
except Exception as exc:
    gx = None
    GX_IMPORT_ERROR = exc

# Storage paths
DATA_ROOT = Path("data")
BRONZE    = DATA_ROOT / "bronze"
SILVER    = DATA_ROOT / "silver"
GOLD      = DATA_ROOT / "gold"

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/ingest.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="10 MB",
)

# =============================================================================
# STORAGE UTILITIES
# =============================================================================

def init_storage() -> None:
    """
    Create the full medallion directory structure.
    Safe to call multiple times — uses exist_ok=True.
    """
    dirs = [
        BRONZE / "eia",
        BRONZE / "yfinance",
        BRONZE / "fred",
        SILVER,
        GOLD,
        Path("logs"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Medallion storage structure initialized")


def save_bronze(df: pd.DataFrame, source: str, name: str) -> Path:
    """
    Save raw API response to bronze layer.
    Bronze is immutable — this is the only place data is written as-is.
    """
    path = BRONZE / source / f"{name}_raw.parquet"
    df.to_parquet(path, index=True)
    logger.debug(f"Bronze saved → {path} | shape: {df.shape}")
    return path


def save_silver(df: pd.DataFrame, name: str) -> Path:
    """Save cleaned, calendar-aligned data to silver layer."""
    path = SILVER / f"{name}.parquet"
    df.to_parquet(path, index=True)
    logger.info(f"Silver saved → {path} | shape: {df.shape}")
    return path


def load_bronze(source: str, name: str) -> pd.DataFrame:
    path = BRONZE / source / f"{name}_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Bronze file not found: {path}")
    return pd.read_parquet(path)


def load_silver(name: str) -> pd.DataFrame:
    path = SILVER / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Silver file not found: {path}")
    return pd.read_parquet(path)

# =============================================================================
# CALENDAR UTILITIES
# =============================================================================

def build_master_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """
    Build NYSE trading day calendar — the master date spine.
    Every source is reindexed to this before any merging.

    Args:
        start: ISO date string e.g. '2015-01-01'
        end:   ISO date string e.g. '2024-12-31'

    Returns:
        DatetimeIndex of NYSE trading days (timezone-naive dates)
    """
    nyse     = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    raw_days = mcal.date_range(schedule, frequency="1D")

    # Normalize to timezone-naive date — strip time component
    calendar = pd.DatetimeIndex(
        [pd.Timestamp(d.date()) for d in raw_days]
    )
    logger.info(
        f"Master calendar: {calendar[0].date()} → {calendar[-1].date()} "
        f"({len(calendar)} trading days)"
    )
    return calendar


def align_to_calendar(
    series: pd.Series,
    master_calendar: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    """
    Reindex a price series to the master NYSE calendar.
    Forward-fills short gaps (holidays). Flags gaps exceeding MAX_FILL_DAYS.

    Args:
        series:          Raw price series with DatetimeIndex
        master_calendar: NYSE trading day index from build_master_calendar()
        name:            Series name for logging

    Returns:
        Series aligned to master_calendar with short gaps filled
    """
    # Normalize source index to timezone-naive
    series.index = pd.to_datetime(series.index).normalize()
    series.index = series.index.tz_localize(None)

    # Reindex — gaps become NaN
    aligned = series.reindex(master_calendar)

    # Count NaNs before fill
    n_before = aligned.isna().sum()

    # Forward fill short gaps only (holiday closures)
    aligned = aligned.ffill(limit=MAX_FILL_DAYS)

    # Count NaNs after fill
    n_after = aligned.isna().sum()

    if n_after > 0:
        missing = aligned[aligned.isna()].index.tolist()
        logger.warning(
            f"{name}: {n_after} unfilled NaNs after {MAX_FILL_DAYS}-day ffill "
            f"— first 5: {[str(d.date()) for d in missing[:5]]}"
        )
    else:
        logger.debug(f"{name}: aligned cleanly — {n_before} gaps filled via ffill")

    return aligned

# =============================================================================
# DATA PULLERS
# =============================================================================

def pull_eia(series_id: str, name: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily price series from EIA API v2.

    Args:
        series_id: EIA series ID e.g. 'PET.RWTC.D'
        name:      Human-readable name for logging
        start:     ISO start date
        end:       ISO end date

    Returns:
        DataFrame with DatetimeIndex and single 'price' column
        Saved to bronze/eia/{name}_raw.parquet
    """
    if not EIA_API_KEY:
        raise EnvironmentError(
            "EIA_API_KEY not found. Add it to your .env file.\n"
            "Register at: https://www.eia.gov/opendata/register.php"
        )

    url = (
        f"https://api.eia.gov/v2/seriesid/{series_id}"
        f"?api_key={EIA_API_KEY}"
        f"&frequency=daily"
        f"&start={start}"
        f"&end={end}"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=asc"
        f"&length=5000"
    )

    logger.info(f"Pulling EIA: {name} ({series_id})")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.exceptions.RequestException as e:
            wait = (BASE_RETRY_DELAY ** attempt) + random.uniform(0, 1)
            logger.warning(f"EIA attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {wait:.1f}s...")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"EIA pull failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(wait)

    # Parse response
    try:
        records = payload["response"]["data"]
    except KeyError:
        raise ValueError(
            f"Unexpected EIA response structure. "
            f"Keys found: {list(payload.keys())}"
        )

    if not records:
        raise ValueError(
            f"EIA returned 0 records for {series_id} "
            f"between {start} and {end}. Check series ID and date range."
        )

    df = pd.DataFrame(records)

    # EIA returns 'period' and 'value' columns
    df = df[["period", "value"]].copy()
    df.columns = ["date", "price"]
    df["date"]  = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.set_index("date").sort_index()

    # Drop rows where EIA explicitly returned null (market closures stored as null)
    n_nulls = df["price"].isna().sum()
    if n_nulls > 0:
        logger.debug(f"EIA {name}: dropping {n_nulls} null-value rows from raw data")
        df = df.dropna(subset=["price"])

    logger.info(f"EIA {name}: {len(df)} records | {df.index.min().date()} → {df.index.max().date()}")

    # Save to bronze — raw, untouched
    save_bronze(df, source="eia", name=name)

    return df


def pull_stooq(name: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily OHLCV data from Stooq (stooq.com).

    Stooq is used for WEAT because yfinance cannot retrieve it after WEAT's
    Nov 2025 reverse split changed its CUSIP number. Stooq serves the same
    WEAT ETF data reliably, requires no API key, and is queried via a plain
    CSV download URL.

    URL format: https://stooq.com/q/d/l/?s=<ticker>&d1=<YYYYMMDD>&d2=<YYYYMMDD>&i=d
      s   = ticker symbol (lowercase, with .us suffix for US securities)
      d1  = start date in YYYYMMDD format
      d2  = end date in YYYYMMDD format
      i=d = daily interval

    Args:
        name:  Logical commodity name — must be a key in STOOQ_CONFIG e.g. 'weat'
        start: ISO start date e.g. '2015-01-01'
        end:   ISO end date   e.g. '2026-05-30'

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
        Saved to bronze/yfinance/{name}_raw.parquet

    Raises:
        KeyError:     if name is not in STOOQ_CONFIG
        RuntimeError: if all retry attempts are exhausted
        ValueError:   if Stooq returns an empty or malformed response
    """
    ticker = STOOQ_CONFIG[name]
    logger.info(f"Pulling Stooq: {name} ({ticker})")

    # Stooq date format: YYYYMMDD (no dashes)
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={ticker}&d1={d1}&d2={d2}&i=d"

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # pandas read_csv pulls the CSV directly from the URL
            df = pd.read_csv(url, parse_dates=["Date"])

            if df.empty:
                raise ValueError(f"Stooq returned empty CSV for {ticker}")

            # Stooq returns titled columns: Date, Open, High, Low, Close, Volume
            df = df.rename(columns=str.lower)
            df = df.set_index("date").sort_index()

            required_cols = {"open", "high", "low", "close", "volume"}
            missing_cols  = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Stooq response for {ticker} missing columns: {missing_cols}. "
                    f"Got: {df.columns.tolist()}"
                )

            df = df[["open", "high", "low", "close", "volume"]].copy()

            # Normalize index to timezone-naive dates
            df.index = pd.to_datetime(df.index).normalize()
            df.index = df.index.tz_localize(None)
            df.index.name = "date"

            logger.info(
                f"Stooq {name} ({ticker}): {len(df)} records | "
                f"{df.index.min().date()} → {df.index.max().date()}"
            )

            # Store in the same bronze/yfinance/ path as the other agricultural
            # sources — downstream code (build_silver) doesn't care about the
            # original data provider, only the logical commodity name
            save_bronze(df, source="yfinance", name=name)
            return df

        except Exception as exc:
            last_error = exc
            wait = (BASE_RETRY_DELAY ** attempt) + random.uniform(0, 1)
            logger.warning(
                f"Stooq {ticker} attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                f"Waiting {wait:.1f}s before retry..."
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(
        f"Stooq pull failed for '{name}' ({ticker}) after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def pull_yfinance(name: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily OHLCV data from Yahoo Finance via yfinance.

    Used for CORN only. WEAT is handled by pull_stooq() due to a permanent
    yfinance failure caused by WEAT's Nov 2025 reverse split CUSIP change.

    Args:
        name:  Logical commodity name — must be a key in YFINANCE_CONFIG
               e.g. 'corn'
        start: ISO start date
        end:   ISO end date

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
        Saved to bronze/yfinance/{name}_raw.parquet

    Raises:
        KeyError:     if name is not in YFINANCE_CONFIG
        RuntimeError: if all retry attempts are exhausted
    """
    ticker = YFINANCE_CONFIG[name]
    logger.info(f"Pulling yfinance: {name} ({ticker})")

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Small jitter before each attempt — prevents hammering Yahoo's
            # servers when pulling multiple tickers in quick succession
            time.sleep(random.uniform(0.5, 2.0))

            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,   # adjusts prices for dividends and splits
                back_adjust=True,   # retroactively rescales ALL historical prices
                                    # so log returns across past split dates are
                                    # economically correct (no fake volatility spikes)
                progress=False,
                timeout=30,
            )

            if df.empty:
                raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

            # yfinance with auto_adjust returns MultiIndex columns:
            #   ('Close', 'CORN'), ('Open', 'CORN'), ...
            # Flatten to plain lowercase: 'close', 'open', ...
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]

            required_cols = {"open", "high", "low", "close", "volume"}
            missing_cols  = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"yfinance response for {ticker} missing columns: {missing_cols}. "
                    f"Got: {df.columns.tolist()}"
                )

            df = df[["open", "high", "low", "close", "volume"]].copy()

            # Normalize index to timezone-naive dates
            df.index = pd.to_datetime(df.index).normalize()
            df.index = df.index.tz_localize(None)
            df.index.name = "date"

            logger.info(
                f"yfinance {name} ({ticker}): {len(df)} records | "
                f"{df.index.min().date()} → {df.index.max().date()}"
            )

            save_bronze(df, source="yfinance", name=name)
            return df

        except Exception as exc:
            last_error = exc
            # Detect 429 rate-limit — wait much longer in that case
            msg = str(exc).lower()
            is_rate_limit = any(k in msg for k in ["429", "too many requests", "rate limit"])

            if is_rate_limit:
                wait = ((BASE_RETRY_DELAY ** attempt) + random.uniform(0, 1)) * RATE_LIMIT_MULTIPLIER
                logger.warning(
                    f"Rate limited (429) on {ticker} attempt {attempt}/{MAX_RETRIES}. "
                    f"Waiting {wait:.1f}s..."
                )
            else:
                wait = (BASE_RETRY_DELAY ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"yfinance {ticker} attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                    f"Waiting {wait:.1f}s..."
                )

            if attempt < MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(
        f"yfinance pull failed for '{name}' ({ticker}) after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def pull_fred(series_id: str, name: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily macro series from FRED API.

    Args:
        series_id: FRED series ID e.g. 'DTWEXBGS'
        name:      Series name for storage e.g. 'dxy'
        start:     ISO start date
        end:       ISO end date

    Returns:
        DataFrame with DatetimeIndex and single 'value' column
        Saved to bronze/fred/{name}_raw.parquet
    """
    if not FRED_API_KEY:
        raise EnvironmentError(
            "FRED_API_KEY not found. Add it to your .env file.\n"
            "Register at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    logger.info(f"Pulling FRED: {name} ({series_id})")

    fred = Fred(api_key=FRED_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            series = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            if series.empty:
                raise ValueError(f"FRED returned empty series for {series_id}")
            break
        except Exception as e:
            wait = (BASE_RETRY_DELAY ** attempt) + random.uniform(0, 1)
            logger.warning(f"FRED attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {wait:.1f}s...")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"FRED pull failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(wait)

    df = series.to_frame(name="value")
    df.index = pd.to_datetime(df.index).normalize()
    df.index = df.index.tz_localize(None)
    df.index.name = "date"

    # FRED stores missing days as NaN — drop them before bronze save
    n_nulls = df["value"].isna().sum()
    if n_nulls > 0:
        logger.debug(f"FRED {name}: dropping {n_nulls} null rows from raw data")
        df = df.dropna()

    logger.info(f"FRED {name}: {len(df)} records | {df.index.min().date()} → {df.index.max().date()}")

    save_bronze(df, source="fred", name=name)

    return df

# =============================================================================
# SILVER LAYER — ALIGNMENT AND MERGE
# =============================================================================

def build_silver(
    eia_data:        dict[str, pd.DataFrame],
    yfinance_data:   dict[str, pd.DataFrame],
    fred_data:       dict[str, pd.DataFrame],
    master_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align all bronze sources to the NYSE master calendar and merge
    into a single validated silver DataFrame.

    Column naming convention:
        EIA close prices:       wti, natgas
        yfinance OHLCV:         weat_open, weat_high, weat_low, weat_close, weat_volume
                                corn_open, corn_high, corn_low, corn_close, corn_volume
        FRED macro:             dxy, yield_10y

    Args:
        eia_data:        dict of {name: raw DataFrame} from pull_eia()
        yfinance_data:   dict of {name: raw DataFrame} from pull_yfinance()
        fred_data:       dict of {name: raw DataFrame} from pull_fred()
        master_calendar: NYSE trading day index

    Returns:
        Merged, aligned, validated DataFrame saved to silver/aligned.parquet
    """
    logger.info("Building silver layer — aligning all sources to NYSE calendar")

    aligned = pd.DataFrame(index=master_calendar)
    aligned.index.name = "date"

    # --- EIA: close price only ---
    for name, df in eia_data.items():
        aligned[name] = align_to_calendar(df["price"], master_calendar, name)

    # --- yfinance: full OHLCV ---
    for name, df in yfinance_data.items():
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                col_name = f"{name}_{col}"
                aligned[col_name] = align_to_calendar(df[col], master_calendar, col_name)

    # --- FRED: single value columns ---
    for name, df in fred_data.items():
        aligned[name] = align_to_calendar(df["value"], master_calendar, name)

    # Summary
    logger.info(
        f"Silver aligned DataFrame: {aligned.shape[0]} rows × {aligned.shape[1]} columns"
    )
    logger.info(f"Columns: {aligned.columns.tolist()}")

    return aligned

# =============================================================================
# GREAT EXPECTATIONS VALIDATION
# =============================================================================

def _manual_validate_silver(df: pd.DataFrame) -> None:
    """Fallback validation for environments where Great Expectations cannot import."""
    price_cols = ["wti", "natgas", "weat_close", "corn_close", "dxy", "yield_10y"]
    price_cols = [c for c in price_cols if c in df.columns]

    for col in price_cols:
        if df[col].isna().any():
            raise ValueError(f"Validation failed: {col} contains null values")

    for col in ["wti", "natgas", "weat_close", "corn_close"]:
        if col in df.columns and (df[col] <= 0).any():
            raise ValueError(f"Validation failed: {col} must be strictly positive")

    if "dxy" in df.columns and ((df["dxy"] < 50.0) | (df["dxy"] > 200.0)).any():
        raise ValueError("Validation failed: dxy is outside the expected 50.0-200.0 range")

    if "yield_10y" in df.columns and ((df["yield_10y"] < 0.0) | (df["yield_10y"] > 25.0)).any():
        raise ValueError("Validation failed: yield_10y is outside the expected 0.0-25.0 range")

    logger.info("Built-in silver validation PASSED")


def validate_silver(df: pd.DataFrame) -> bool:
    logger.info("Running Great Expectations validation on silver layer")

    # --- Pre-GX checks (structural, not column-level) ---

    # Check 1: No duplicate dates
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        raise ValueError(f"Duplicate dates in silver DataFrame: {n_dupes} duplicates found")

    # Check 2: Date continuity — no gaps > MAX_FILL_DAYS calendar days
    date_diffs = pd.Series(df.index).diff().dt.days.dropna()
    # NYSE calendar naturally removes weekends so max expected diff is ~3 (long weekends)
    large_gaps = date_diffs[date_diffs > 5]
    if len(large_gaps) > 0:
        gap_dates = [str(df.index[i].date()) for i in large_gaps.index[:5]]
        logger.warning(f"Date gaps > 5 days found near: {gap_dates}")
        # Warning only — some years have extended closures (e.g. 9/11 2001)

    # --- Column-level validation ---
    if gx is None:
        logger.warning(
            "Great Expectations import failed; using built-in validation instead: {}",
            GX_IMPORT_ERROR,
        )
        _manual_validate_silver(df)
        return True

    price_cols = ["wti", "natgas", "weat_close", "corn_close", "dxy", "yield_10y"]
    price_cols = [c for c in price_cols if c in df.columns]

    context = gx.get_context(mode="ephemeral")

    # Build expectation suite
    suite = context.suites.add(
        gx.ExpectationSuite(name="silver_validation_suite")
    )

    # Expect no nulls in price columns
    for col in price_cols:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # Expect all prices strictly positive
    for col in ["wti", "natgas", "weat_close", "corn_close"]:
        if col in df.columns:
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col,
                    min_value=0.01,   # strictly > 0
                    max_value=500.0,  # sanity upper bound (WTI never > $500/bbl)
                )
            )

    # DXY reasonable range (USD index, typically 70–130)
    if "dxy" in df.columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="dxy",
                min_value=50.0,
                max_value=200.0,
            )
        )

    # 10Y yield reasonable range (0 to 20%)
    if "yield_10y" in df.columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="yield_10y",
                min_value=0.0,
                max_value=25.0,
            )
        )

    # Run validation
    ds        = context.data_sources.add_pandas("silver_ds")
    da        = ds.add_dataframe_asset("silver_asset")
    batch_def = da.add_batch_definition_whole_dataframe("silver_batch")

    vd = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="silver_vd",
            data=batch_def,
            suite=suite,
        )
    )

    result = vd.run(batch_parameters={"dataframe": df.reset_index()})

    if result.success:
        logger.info("Great Expectations validation PASSED — all checks clean")
    else:
        failed    = [r for r in result.results if not r.success]
        error_msgs = [str(f.expectation_config) for f in failed[:5]]
        raise ValueError(
            f"Great Expectations validation FAILED — {len(failed)} checks failed:\n"
            + "\n".join(error_msgs)
        )

    return True

# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    """Print a clean summary of the silver DataFrame after ingestion."""
    print("\n" + "=" * 65)
    print("  INGESTION COMPLETE — SILVER LAYER SUMMARY")
    print("=" * 65)
    print(f"  Date range  : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Trading days: {len(df)}")
    print(f"  Columns     : {df.shape[1]}")
    print()
    print(f"  {'Column':<20} {'Non-null':>10} {'NaN':>8} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        non_null = df[col].notna().sum()
        n_nan    = df[col].isna().sum()
        col_min  = df[col].min()
        col_max  = df[col].max()
        print(f"  {col:<20} {non_null:>10} {n_nan:>8} {col_min:>10.3f} {col_max:>10.3f}")

    print("=" * 65 + "\n")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_ingestion(start: str, end: str, dry_run: bool = False) -> pd.DataFrame:
    """
    Full ingestion pipeline — pull, align, validate, store.

    Steps:
        1. Initialize medallion storage structure
        2. Pull all sources (EIA, yfinance, FRED) → bronze
        3. Align all sources to NYSE master calendar
        4. Merge into single silver DataFrame
        5. Validate with Great Expectations
        6. Save to silver layer

    Args:
        start:   ISO start date e.g. '2015-01-01'
        end:     ISO end date e.g. '2024-12-31'
        dry_run: If True, run pipeline but do not save any files

    Returns:
        Validated silver DataFrame
    """
    logger.info(f"Starting ingestion pipeline: {start} → {end}")
    if dry_run:
        logger.warning("DRY RUN MODE — no files will be written")

    # -------------------------------------------------------------------------
    # Step 1: Initialize storage
    # -------------------------------------------------------------------------
    if not dry_run:
        init_storage()

    # -------------------------------------------------------------------------
    # Step 2: Build master calendar
    # -------------------------------------------------------------------------
    master_calendar = build_master_calendar(start, end)

    # -------------------------------------------------------------------------
    # Step 3: Pull all data sources → bronze
    # -------------------------------------------------------------------------
    logger.info("--- Pulling EIA (energy prices) ---")
    eia_data: dict[str, pd.DataFrame] = {}
    for name, series_id in EIA_SERIES.items():
        try:
            eia_data[name] = pull_eia(series_id, name, start, end)
        except Exception as e:
            logger.error(f"Failed to pull EIA {name}: {e}")
            raise

    logger.info("--- Pulling agricultural prices (Stooq + yfinance) ---")
    yfinance_data: dict[str, pd.DataFrame] = {}

    # WEAT → Stooq (yfinance permanently broken due to Nov 2025 CUSIP change)
    for name in STOOQ_CONFIG:
        try:
            yfinance_data[name] = pull_stooq(name, start, end)
        except Exception as e:
            logger.error(f"Failed to pull Stooq {name}: {e}")
            raise

    # CORN → yfinance (working cleanly after upgrade)
    for name in YFINANCE_CONFIG:
        try:
            yfinance_data[name] = pull_yfinance(name, start, end)
        except Exception as e:
            logger.error(f"Failed to pull yfinance {name}: {e}")
            raise

    logger.info("--- Pulling FRED (macro features) ---")
    fred_data: dict[str, pd.DataFrame] = {}
    for name, series_id in FRED_SERIES.items():
        try:
            fred_data[name] = pull_fred(series_id, name, start, end)
        except Exception as e:
            logger.error(f"Failed to pull FRED {name}: {e}")
            raise

    # -------------------------------------------------------------------------
    # Step 4: Build silver — align and merge
    # -------------------------------------------------------------------------
    silver_df = build_silver(
        eia_data=eia_data,
        yfinance_data=yfinance_data,
        fred_data=fred_data,
        master_calendar=master_calendar,
    )

    # -------------------------------------------------------------------------
    # Step 5: Validate with Great Expectations
    # -------------------------------------------------------------------------
    validate_silver(silver_df)

    # -------------------------------------------------------------------------
    # Step 6: Save to silver layer
    # -------------------------------------------------------------------------
    if not dry_run:
        save_silver(silver_df, name="aligned")
        logger.info("Silver layer written → data/silver/aligned.parquet")
    else:
        logger.info("Dry run — silver DataFrame built and validated but not saved")

    # -------------------------------------------------------------------------
    # Step 7: Print summary
    # -------------------------------------------------------------------------
    print_summary(silver_df)

    return silver_df


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Commodity Volatility Forecasting — Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                                  # full pull 2015 to today
  python ingest.py --start 2018-01-01               # custom start date
  python ingest.py --start 2015-01-01 --end 2023-12-31
  python ingest.py --dry-run                        # validate without saving
        """,
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START,
        help=f"Start date in YYYY-MM-DD format (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END,
        help=f"End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without writing any files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate date arguments
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    if args.start >= args.end:
        logger.error("--start must be earlier than --end")
        sys.exit(1)

    # Run pipeline
    try:
        silver = run_ingestion(
            start=args.start,
            end=args.end,
            dry_run=args.dry_run,
        )
        logger.info("Ingestion pipeline completed successfully")
        sys.exit(0)
    except EnvironmentError as e:
        # Missing API keys — clear actionable message
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Ingestion pipeline failed: {e}")
        sys.exit(1)