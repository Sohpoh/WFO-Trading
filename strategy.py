"""Opening-Range Breakout (ORB) on ES 15min, New York session.

New family (`opening_range_breakout`), pivoting off iteration 65's
`multiple-testing` rejection. The edge is the first hour's High/Low as an
institutional accumulation/distribution marker: a one-bar break of a
session-open-anchored range (with a buffer) is a higher-conviction intraday
trend signal than an unconditional rolling-channel breakout.

Rules (all computed on continuous, session-unaware bars; session gating and
stop/target bookkeeping are delegated to `apply_session_constraint_with_stops`):

  - Opening range: per-ET-day High/Low over bars whose New-York time falls in
    [09:30, 09:30 + orb_minutes) — computed via index.tz_convert("America/
    New_York") + ET-date grouping (bars are timestamped at their open). The
    range level is NaN until its window closes (so no entry inside the first
    `orb_minutes`) and is then frozen through the rest of that ET day. The
    09:30 anchor is the strategy's signal definition, not session gating — it
    gates nothing and flattens nothing; session.py alone decides which bars are
    tradable and force-flattens at 16:00.

  - Breakout trigger: a one-bar Close cross. LONG fires on the first bar where
    Close crosses above OR_high + buffer_frac*(OR_high - OR_low) (prior Close
    <= threshold < current Close); SHORT fires on the first bar where Close
    crosses below OR_low - buffer_frac*(OR_high - OR_low) (prior Close >=
    threshold > current Close). One-bar cross, not a persistent above/below
    state, so a stop/target exit does not re-arm on the next in-session bar.

  - Exit: path-dependent stop + target via `apply_session_constraint_with_stops`.
    Long: stop = entry - stop_atr_mult * ATR(14), target = entry +
    target_atr_mult * ATR(14); short mirrored. Both distances are read at the
    entry bar and the position force-flattens at session close (session.py).
    No flip exit.

Warm-up: ATR_STOP_PERIOD (14 bars) is the only bar-count lookback and is
threaded through build_grid() as a plain int so _max_lookback_bars() sizes the
test-window buffer to at least (14+5)*3 = 57 bars (the day-bars floor of ~96
bars at 15min dominates), comfortably covering the Wilder ATR(14) warm-up. The
opening-range window is within-day and is covered by the day-bars floor, so no
extra lookback is needed. orb_minutes is a *minutes-of-day* window (not a
bar-count lookback) and buffer_frac / stop_atr_mult / target_atr_mult are
fractions/multipliers, so all four are floats and must not inflate the warm-up
buffer.

Contract notes: `generate_positions` builds session-unaware raw signals and
delegates all session gating, stop/target bookkeeping, and end-of-session
flatten to `apply_session_constraint_with_stops`; it never reimplements the
session mask. No position sizing is specified, so the delegate's default 1.0
unit exposure applies and the position is long +1 / short -1 / flat 0 — repeat
signals while already positioned are no-ops and the single-position contract
holds. NaN/unwarmed indicators (ATR, or a not-yet-closed opening range) fail
closed (no entry). Costs (metrics.py, cost model v2) are out of scope for this
file.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Fixed ATR window, in BARS, for the stop/target distance. Genuine bar-count
# lookback, threaded through build_grid() as a plain int so _max_lookback_bars()
# sizes the warm-up buffer to cover the Wilder warm-up. Never grid-searched.
ATR_STOP_PERIOD = 14

# Opening-range anchor, fixed: the New York cash open, in Eastern wall-clock
# time. This is the strategy's signal definition (where the first-hour range
# starts), NOT session gating — it is deliberately hardcoded here because
# session.py still owns which bars are tradable and the 16:00 force-flatten.
# Never grid-searched.
OR_OPEN_HOUR = 9
OR_OPEN_MINUTE = 30


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's Average True Range over `period` bars, index-aligned to `close`.

    True range uses the prior bar's close as the reference so session/overnight
    gaps count. The Wilder average is a recursive EMA with alpha = 1/period
    (adjust=False) so it carries state rather than using a fixed trailing
    window. The first `period` bars are NaN (the recursion hasn't seen a full
    window yet), so a stop distance built from them fails closed and cannot
    open a position.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
    atr.iloc[:period] = np.nan
    return atr


def generate_positions(
    df: pd.DataFrame,
    orb_minutes: float,
    buffer_frac: float,
    stop_atr_mult: float,
    target_atr_mult: float,
    session: str | None = "New York",
    atr_stop_period: int = ATR_STOP_PERIOD,
) -> pd.Series:
    """Build the two-sided opening-range-breakout position series.

    Opening range = per-ET-day High/Low over bars whose New-York time is in
    [09:30, 09:30 + orb_minutes). The level is NaN until its window closes and
    then frozen through that ET day. LONG/SHORT fire on a one-bar Close cross of
    the buffered range high/low; exit is a bounded ATR(14)-scaled stop + target
    delegated to session.py, force-flattened at session close. Returns a signed
    unit-exposure position series (+1 long / -1 short / 0 flat, no sizing).
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # ET wall-clock time per bar (bars are timestamped at their open). Grouping
    # by the normalized ET date makes the opening range per-ET-calendar-day;
    # tz_convert keeps this DST-safe without hardcoding UTC offsets.
    et = df.index.tz_convert("America/New_York")
    et_date = et.normalize().to_numpy()
    et_min = (et.hour * 60 + et.minute).to_numpy()

    or_open_min = OR_OPEN_HOUR * 60 + OR_OPEN_MINUTE  # 570 = 09:30
    or_close_min = or_open_min + int(orb_minutes)

    in_window = (et_min >= or_open_min) & (et_min < or_close_min)
    range_known = et_min >= or_close_min

    # Per-ET-day opening-range High/Low over the in-window bars only. NaN until
    # the window closes (range_known), then frozen through the rest of that ET
    # day (transform broadcasts the single day value to every bar of the day).
    or_high = high.where(in_window).groupby(et_date).transform("max")
    or_low = low.where(in_window).groupby(et_date).transform("min")
    or_high = or_high.where(range_known)
    or_low = or_low.where(range_known)

    or_range = or_high - or_low
    long_threshold = or_high + float(buffer_frac) * or_range
    short_threshold = or_low - float(buffer_frac) * or_range

    # One-bar cross, not a persistent above/below state. The threshold is the
    # frozen OR level, so the first bar after the window closes can fire if the
    # prior Close sat on the other side. NaN thresholds (pre-window) fail
    # closed. Long and short are mutually exclusive: long_threshold >=
    # short_threshold always, so Close cannot exceed both.
    prior_close = close.shift(1)
    long_signal = (close > long_threshold) & (prior_close <= long_threshold)
    short_signal = (close < short_threshold) & (prior_close >= short_threshold)

    # Stop/target distances, both read at the entry bar. stop_distance is a
    # positive from-entry distance (session.py resolves below-entry for a long,
    # above-entry for a short); target_price is absolute and direction-resolved
    # (above entry on long bars, below entry on short bars).
    atr = atr_series(high, low, close, int(atr_stop_period))
    stop_distance = float(stop_atr_mult) * atr
    target_price = close + float(target_atr_mult) * atr
    target_price = target_price.where(~short_signal, close - float(target_atr_mult) * atr)

    # session.py alone decides which bars are tradable, walks the stop/target
    # bookkeeping bar-by-bar, opens +1/-1 on an entry bar, and force-flattens
    # on the session's last bar.
    return apply_session_constraint_with_stops(
        close,
        high,
        low,
        long_signal,
        short_signal,
        stop_distance,
        target_price,
        session,
    )


DEFAULT_PARAMS = {
    # orb_minutes (grid {30,60,90}) is a minutes-of-day window, NOT a bar-count
    # lookback, so it is a float and must not feed _max_lookback_bars().
    # buffer_frac (grid {0.0,0.1,0.25,0.5}) is a range-buffer fraction and
    # stop_atr_mult (grid {0.75,1.0,1.5,2.0}) / target_atr_mult (grid
    # {1.0,2.0,3.0,4.0}) are ATR multipliers — all floats, NOT lookbacks.
    # atr_stop_period (14) is a fixed plain-int lookback, threaded through
    # build_grid() so _max_lookback_bars() returns 14. `session` is a fixed
    # param. These are the concrete set the sanity checker runs
    # generate_positions() against.
    "orb_minutes": 60.0,
    "buffer_frac": 0.1,
    "stop_atr_mult": 2.0,
    "target_atr_mult": 2.0,
    "atr_stop_period": ATR_STOP_PERIOD,
    "session": "New York",
}
