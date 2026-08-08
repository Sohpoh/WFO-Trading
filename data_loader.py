"""Data loading for the Walk-Forward Trading App.

Two sources are supported:
  1. Local historical CSVs under Data/<SYMBOL>/<timeframe>/... (continuous futures,
     Databento-style export). This is the preferred source since it has full
     intraday history from the start of data through the present.
  2. yfinance, for arbitrary tickers or when local data isn't available. yfinance
     intraday history is capped by Yahoo (~60 days for sub-hourly, ~730 days for
     60m), so it is best used with daily bars for long lookbacks.
"""
import glob
import os

import pandas as pd
import streamlit as st
import yfinance as yf

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

LOCAL_TIMEFRAMES = ["1min", "5min", "15min", "1h", "4h", "1d"]

# approximate trading bars/year, used only to annualize Sharpe/CAGR
BARS_PER_YEAR = {
    "1min": 252 * 23 * 60,
    "5min": 252 * 23 * 12,
    "15min": 252 * 23 * 4,
    "1h": 252 * 23,
    "4h": 252 * 6,
    "1d": 252,
}

YFINANCE_INTERVAL_MAP = {"1d": "1d", "1h": "60m"}


def list_local_symbols() -> list[str]:
    if not os.path.isdir(DATA_ROOT):
        return []
    return sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith(".")
    )


def list_local_timeframes(symbol: str) -> list[str]:
    sym_dir = os.path.join(DATA_ROOT, symbol)
    if not os.path.isdir(sym_dir):
        return []
    found = [d for d in os.listdir(sym_dir) if os.path.isdir(os.path.join(sym_dir, d))]
    return [tf for tf in LOCAL_TIMEFRAMES if tf in found]


@st.cache_data(show_spinner="Loading local futures data...")
def load_local_data(symbol: str, timeframe: str) -> pd.DataFrame:
    pattern = os.path.join(DATA_ROOT, symbol, timeframe, f"{symbol}_*_{timeframe}", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No local data found for {symbol} @ {timeframe}")

    frames = [
        pd.read_csv(f, usecols=["ts_event", "open", "high", "low", "close", "volume"])
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.drop_duplicates(subset="ts_event").sort_values("ts_event")
    df = df.set_index("ts_event")
    df.index.name = "datetime"
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(show_spinner="Downloading data from Yahoo Finance...")
def load_yfinance_data(ticker: str, timeframe: str) -> pd.DataFrame:
    interval = YFINANCE_INTERVAL_MAP.get(timeframe, "1d")
    period = "730d" if interval != "1d" else "max"
    df = yf.download(
        ticker, interval=interval, period=period, auto_adjust=True, progress=False, multi_level_index=False
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance returned no data for '{ticker}' @ {interval}")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_data(source: str, symbol: str, timeframe: str) -> pd.DataFrame:
    if source == "local":
        return load_local_data(symbol, timeframe)
    return load_yfinance_data(symbol, timeframe)
