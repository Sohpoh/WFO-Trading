"""Donchian-channel breakout + VWAP filter, with ATR-based stop/target.

Rules:
  - LONG when price closes above the N-bar Donchian upper channel (breakout)
    while price is also above session VWAP (directional filter — free, no
    extra grid dimension, just a sign check).
  - SHORT when price closes below the N-bar Donchian lower channel while
    price is also below session VWAP.
  - FLAT otherwise, or once the stop or target is hit while in a trade.
  - Stop: k * ATR from entry (k is `atr_mult`, grid-searched — magnitude is
    instrument-dependent, e.g. ES ~10-20pt / NQ ~25-50pt per the wiki's
    stop-placement notes, so it's left to the grid rather than hardcoded).
  - Target: either the measured-move (Donchian channel height projected off
    the breakout level itself — the textbook measured-move anchor, e.g. for
    a long that's `B_up + (B_up - B_down)`, not off the entry/close, since a
    breakout bar can already be some distance past B_up) or 2k * ATR off
    entry — `target_mode` picks between them and is itself swept in the grid
    so walk-forward decides which travels better fold to fold, per the
    strategy brief.

This module only decides *when the strategy wants to enter and where its
stop/target sit*. Because those exits are path-dependent (a stop or target
can fire mid-trade, not just on the next opposing signal), plain
`apply_session_constraint()` can't express them — see
`session.apply_session_constraint_with_stops()`'s docstring for why that
walk (and the session-boundary force-flatten) has to live in session.py
rather than here. This file only ever computes raw signals/distances; it
has no session awareness of its own, consistent with every other strategy
in this codebase.

Caveat carried from session.py: stops/targets are *detected* off intrabar
High/Low but the exit still prices at that bar's Close (this app has no
finer-than-bar price series and `metrics.py` costs everything off Close), so
a stop does not cap the realized loss on the triggering bar — see
`apply_session_constraint_with_stops()` for details before describing
results as risk-capped.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# CME/Globex NQ/ES trade on an "18:00 ET -> next 17:00 ET" trading day, not a
# midnight-ET calendar day. Session VWAP resets there rather than at UTC or
# ET midnight so early-window sessions (e.g. "London", 02:00-05:00 ET) start
# with most of the prior evening's volume already accumulated instead of
# only 2-5 hours of it — this is a fixed instrument-calendar fact, not a use
# of session.py's *configurable* day-trade session, so it doesn't reintroduce
# session-awareness into this module.
VWAP_ANCHOR_OFFSET_HOURS = 18


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    # Wilder smoothing, same ewm(alpha=1/period) convention this codebase
    # already uses for RSI's average gain/loss.
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative typical-price VWAP, resetting each CME/Globex trading day."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    trading_day = (
        df.index.tz_convert("America/New_York") - pd.Timedelta(hours=VWAP_ANCHOR_OFFSET_HOURS)
    ).normalize()
    cum_pv = pv.groupby(trading_day).cumsum()
    cum_vol = df["Volume"].groupby(trading_day).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def compute_donchian(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """Upper/lower channel from the N bars *before* the current one (shifted
    so the breakout check isn't comparing a bar's close to a channel that
    already includes that same bar's own high/low)."""
    b_up = df["High"].shift(1).rolling(n, min_periods=n).max()
    b_down = df["Low"].shift(1).rolling(n, min_periods=n).min()
    return b_up, b_down


def generate_positions(
    df: pd.DataFrame,
    donchian_n: int,
    atr_mult: float,
    atr_period: int = 14,
    target_mode: str = "channel",
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    b_up, b_down = compute_donchian(df, donchian_n)
    channel_height = b_up - b_down
    atr = compute_atr(df, atr_period)
    vwap = compute_session_vwap(df)

    long_signal = ((close > b_up) & (close > vwap)).fillna(False)
    short_signal = ((close < b_down) & (close < vwap)).fillna(False)

    stop_distance = atr_mult * atr

    # target_price is an absolute level, already resolved per direction —
    # unlike the stop it can't be a single from-entry distance, since the
    # channel-mode measured-move anchors off the breakout level (B_up/B_down)
    # rather than off entry/close.
    if target_mode == "channel":
        long_target = b_up + channel_height
        short_target = b_down - channel_height
    elif target_mode == "atr":
        long_target = close + 2 * atr_mult * atr
        short_target = close - 2 * atr_mult * atr
    else:
        raise ValueError(f"Unknown target_mode: {target_mode!r} (expected 'channel' or 'atr')")

    target_price = pd.Series(np.nan, index=df.index)
    target_price[long_signal] = long_target[long_signal]
    target_price[short_signal] = short_target[short_signal]

    return apply_session_constraint_with_stops(
        close=close,
        high=df["High"],
        low=df["Low"],
        long_signal=long_signal,
        short_signal=short_signal,
        stop_distance=stop_distance,
        target_price=target_price,
        session=session,
    )


DEFAULT_PARAMS = {
    "donchian_n": 20,
    "atr_mult": 2.0,
    "atr_period": 14,
    "target_mode": "channel",
    "session": "New York",
}
