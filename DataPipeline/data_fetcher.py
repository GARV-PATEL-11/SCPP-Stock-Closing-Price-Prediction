# ===========================================================
# File Name: data_fetcher.py
# Title:     Daily Stock Data Fetcher (IST Timezone)
# Author:    Garv Patel
# Project:   SCPP - Stock Closing Price Prediction
# ===========================================================
# Description:
# This module provides a utility function to fetch historical
# daily stock price data using the yfinance API, properly
# aligned to Indian Standard Time (IST). It includes the
# current day's data only if the time is after 3:30 PM IST.
# ===========================================================

# === Import Libraries ===
import yfinance as yf
import pytz
from datetime import datetime, date, timedelta
import pandas as pd

# === Data Fetching Function ===
def fetch_daily_data_ist(ticker: str) -> pd.DataFrame:
    """
    Fetches daily historical stock data from yfinance, aligned to IST.

    Parameters:
    ----------
    ticker : str
        Ticker symbol for the stock (e.g., 'RELIANCE.NS').

    Returns:
    -------
    pd.DataFrame
        DataFrame indexed by IST-localized dates with stock OHLCV data.

    Raises:
    ------
    TypeError
        If the internal start_date conversion fails due to invalid type.

    Notes:
    -----
    - The data is fetched from January 1, 2010 to today or tomorrow
      depending on the current IST time.
    - If current time is after 3:30 PM IST, today's data is included.
    - Timezones are managed to ensure correct alignment with IST.
    """
    # --- Date Settings ---
    start_date = "2010-01-01"
    IST = pytz.timezone("Asia/Kolkata")
    UTC = pytz.UTC

    # --- Convert start_date to date object ---
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()
    elif not isinstance(start_date, date):
        raise TypeError("start_date must be a string, datetime, or date object")

    # --- Current time in IST ---
    now_ist = datetime.now(IST)

    # --- Determine end date based on 3:30 PM cutoff ---
    cutoff_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    if now_ist > cutoff_time:
        end_date = now_ist.date() + timedelta(days=1)
    else:
        end_date = now_ist.date()

    # --- Localize to IST and convert to UTC for API ---
    start_dt_ist = IST.localize(datetime.combine(start_date, datetime.min.time()))
    end_dt_ist = IST.localize(datetime.combine(end_date, datetime.min.time()))
    start_dt_utc = start_dt_ist.astimezone(UTC)
    end_dt_utc = end_dt_ist.astimezone(UTC)

    # --- Fetch Data using yfinance ---
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(
        interval="1d",
        start=start_dt_utc,
        end=end_dt_utc,
        prepost=False
    )

    # --- Convert index to IST-localized dates ---
    df.index = df.index.tz_convert(IST).date

    return df
