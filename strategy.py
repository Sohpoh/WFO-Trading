"""Long-only swing-horizon pullback buy (NQ 15min, New York session).

The previous family (long-only overnight gap-down fill buy) is retired as
structural, not varied: its in-sample leg came back CAGR -9.97% at PF 0.671
over 221 IS / 332 OOS trades, which fires both no-edge triggers, and the
iteration before it was no-edge too. This is a clean pivot to a new family
rather than another knob on a dead one.

The mechanism
-------------
mean-reversion.md's short-term-overreaction / forced-selling edge, applied at
a *swing* horizon instead of an intraday one, and taken only with the
prevailing trend. Taking it only with the trend is that same page's pitfall #1
read as a constraint ("in a strong uptrend, shorting strength is a losing
strategy"): the dip is bought only while price is above a multi-week mean, and
the dip is never sold.

Three construction choices are carried over because they are the only findings
this log has replicated across unrelated families:
  1. Long-only. A ~2,000-trade direction split across four families put the
     short leg as the worse half every time. Under `metrics.py`'s per-leg cost
     model a non-qualifying day now pays zero legs rather than the two a short
     entry would cost.
  2. Slow horizons over fast. In the one clean robust run in this repo the two
     slowest formation values took 41 of 48 folds, so the grid here starts at
     ~10 UTC days of trend and never goes faster.
  3. No stop. Two earlier iterations added a hardcoded stop as their only
     change and both got worse, and hard-capping the payoff has been measured
     flipping per-trade economics from +0.4bps to -11.6bps. The exit here is a
     plain hold to the session flatten.

Closest prior relative is the trend_pullback_continuation family (no-edge) and
four things differ, the horizon being load-bearing: this is long-only rather
than symmetric; the trend filter is 10-20 UTC days rather than 96-384 bars
(which was *shorter* than that iteration's own pullback window, so it was
measuring a 5-hour dip against a "1-4 day trend"); the pullback is measured as
depth below a multi-day rolling high in trailing-daily-range units rather than
price-vs-its-own-rolling-mean (the exact quantity two earlier iterations found
information-free on this data in either polarity); and there is no stop.

Entry — a dense state, not a crossing
-------------------------------------
All quantities are strictly backward-looking and fully session-unaware.

    daily_range_t = rolling_max(High, RANGE_WINDOW).shift(1)
                  - rolling_min(Low,  RANGE_WINDOW).shift(1)
    vol_unit_t    = rolling_mean(daily_range, VOL_WINDOW)
    depth_t       = rolling_max(High, dip_lookback).shift(1) - Close_t

    long_signal_t = (Close_t > SMA(Close, trend_ma))
                    AND (depth_t >= dip_frac * vol_unit_t)

`RANGE_WINDOW` = 96 (one UTC day at 15min) and `VOL_WINDOW` = 480 (~5 days)
are module constants, deliberately NOT searched — the volatility unit carries
zero degrees of freedom, so `dip_frac` is the only width knob.

The `.shift(1)` on both rolling extrema is required, not cosmetic: without it
the rolling max includes the current bar's own High, so on a spike bar the
threshold moves with the price it is supposed to be measured against and the
condition becomes self-referential.

This is a dense STATE, not a crossing. A cross-only form would arm on exactly
one bar per pullback, and that bar is overwhelmingly likely to fall outside
the New York window (the pullback deepens whenever it deepens, including
overnight), producing almost no trades. As a state, the first in-session bar
of an armed stretch is the entry.

NaN comparisons evaluate False, so an unwarmed bar stands aside with no
separate validity mask to keep in sync — the same fail-closed discipline as
the previous iteration.

Exit — hold to the session flatten, no stop, no target
------------------------------------------------------
The raw `entries` series carries only 1.0 (arm long) and NaN (no opinion); it
never carries 0.0 and never carries -1.0. `session.apply_session_constraint()`
forward-fills that into a position, so the trade opens on the first in-session
armed bar and is force-flattened by `session.py` on the session's last bar.
Upside is uncapped; duration is bounded by the session. One round trip per
armed session.

This is deliberately NOT `apply_session_constraint_with_stops` — see point 3
above. Because `entries` is NaN (not False) on non-qualifying bars, a position
already open is *held* through bars where the signal switches off; the exit is
the session boundary alone.

Long-only means -1.0 can never be emitted, so `apply_session_constraint`'s
forward fill can only ever produce {0.0, 1.0} and a non-qualifying day costs
nothing at all.

Cost discipline, designed into the grid rather than bolted on
------------------------------------------------------------
At `dip_frac` = 0.5 (the loosest grid point) the qualifying pullback measures
~0.95% of price at the median on the local NQ 15min frame (the volatility unit
itself runs ~1.9% of price at the median), roughly 9x `metrics.py`'s ~10.2bps
round-trip toll — the spec estimated ~0.65% / ~6x, so the measured margin is
wider than pre-registered, not narrower. The trade is a single round trip held
across an RTH range of ~1.0-1.3%. NQ is chosen over ES because ES's smaller
ranges would starve the same threshold.

Measured density (full local NQ 15min frame, New York session), since the
trend filter and the depth filter partially fight each other and the spec
pre-registered a >=120-OOS-trade power floor: at trend_ma=960 / dip_lookback=96
/ dip_frac=0.5 the strategy arms on 357 distinct in-session days and takes 354
entries (one round trip per armed session, exactly as designed); at
trend_ma=1920 / dip_lookback=288 / dip_frac=0.5, 459 armed days. The tight
corner is genuinely thin: trend_ma=960 / dip_lookback=96 / dip_frac=1.5 arms on
only 11 days across four years.

That thinness is NOT neutralized by the engine, and the honest version of this
is worth stating: `wfo_engine._score_params()` returns -inf only for combos
with FEWER THAN 2 position changes in the train window, and one complete round
trip is exactly 2 changes — so a combo that took a single lucky train trade
still gets scored and can still win the fold. Measured on a 2024-only 12w/3w
smoke run: `dip_frac` = 1.5 combos won 8 of 13 folds and 6 of 13 folds took
zero OOS trades. Whether the resulting trade count clears the pre-registered
>=120-OOS-trade power floor over the full frame is the evaluator's call.

`metrics.py`'s per-leg toll (0.001% fee + 0.05% slippage) is unchanged and out
of scope.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
---------------------------------------------------------------------------
  - `trend_ma` and `dip_lookback` are genuine bar counts and are `int` in
    `build_grid()` **on purpose**, so `_max_lookback_bars()` picks them up and
    sizes the pre-test-window warm-up buffer from them. `dip_frac` is a
    fraction of the volatility unit and is cast `float` so it cannot inflate
    that buffer.
  - At the intended grid the binding value is `trend_ma`: the buffer is
    `max((trend_ma + 5) * 3, bars_per_day + 5)`, i.e. 2,895 bars at
    trend_ma = 960 and 5,775 at trend_ma = 1920.
  - The fold-skip guard in `run_walk_forward()` is
    `len(train_df) < max_lookback + 10` = 1,930 at the top of the grid. A
    12-week train window at 15min is ~7,700 bars, so no fold is skipped. THIS
    STRATEGY IS INTENDED AT `--timeframe 15min`. At 1h a 12-week train window
    is only ~1,930 bars — right on the guard — so folds will be silently
    skipped there; at 4h/1d every fold is skipped.
  - Hand-check for the part `_max_lookback_bars()` cannot see: `vol_unit`
    needs RANGE_WINDOW + VOL_WINDOW = 96 + 480 = 576 bars of history before it
    is non-NaN, and 96/480 are module constants that never appear in the grid.
    The smallest buffer this grid can produce is 2,895 bars, which clears 576
    with room to spare at every grid corner. If it ever did not, the failure
    mode is a NaN `vol_unit` -> comparison False -> no entry on the first bars
    of a test window: a few missed trades, never lookahead.

This module decides only *when* the strategy wants to be long. All day-trade
gating and the end-of-session flatten are delegated to `session.py`; see its
docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Bars in one UTC day at the intended 15min timeframe. The trailing high-low
# range over this window is the raw "one day of movement" unit. A module
# constant on purpose — the volatility scale keeps zero searched degrees of
# freedom, so `dip_frac` alone controls how deep a pullback has to be.
RANGE_WINDOW = 96

# Bars over which those daily ranges are averaged into the volatility unit
# (~5 UTC days at 15min). Also a module constant, same reason.
VOL_WINDOW = 480


def volatility_unit(high: pd.Series, low: pd.Series) -> pd.Series:
    """Average trailing one-day high-low range, in price units.

    Strictly backward looking: both extrema are `.shift(1)`ed before the
    average, so the value attached to bar t is fully determined by bars
    strictly before t. The first RANGE_WINDOW + VOL_WINDOW bars are NaN and
    fail closed downstream.
    """
    daily_range = (
        high.rolling(RANGE_WINDOW).max().shift(1) - low.rolling(RANGE_WINDOW).min().shift(1)
    )
    return daily_range.rolling(VOL_WINDOW).mean()


def generate_positions(
    df: pd.DataFrame,
    trend_ma: int,
    dip_lookback: int,
    dip_frac: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Volatility unit: what "a normal day of movement" is worth in points right
    # now, so the depth threshold is scale-free across 2022-2025 regimes.
    vol_unit = volatility_unit(high, low)

    # Uptrend gate: price above its multi-week mean. This is what turns the
    # dip-buy from a naked mean-reversion bet into a with-trend continuation
    # entry, and it is why there is no short leg (shorting strength inside an
    # uptrend is mean-reversion.md's pitfall #1).
    trend_ok = close > close.rolling(int(trend_ma)).mean()

    # Pullback depth: how far the current Close sits below the highest High of
    # the last `dip_lookback` bars. The `.shift(1)` excludes the current bar's
    # own High — without it a spike bar would set its own reference and the
    # comparison would be self-referential.
    depth = high.rolling(int(dip_lookback)).max().shift(1) - close

    # Dense arming state. NaN on either side of a comparison is False, so an
    # unwarmed bar simply stands aside — no separate validity mask to keep in
    # sync with this expression.
    with np.errstate(invalid="ignore"):
        long_signal = trend_ok & (depth >= float(dip_frac) * vol_unit)

    # Raw, session-unaware entries: 1.0 where the strategy wants to be long,
    # NaN everywhere else. NaN (not False/0.0) is load-bearing — it is what
    # lets session.py's forward fill HOLD an open position through bars where
    # the arming condition has switched off, so the only exit is the session
    # flatten. There is no -1.0 branch: this family is long-only, so a
    # non-qualifying day pays zero cost legs.
    entries = pd.Series(np.nan, index=df.index)
    entries[long_signal] = 1.0

    # session.py alone decides which bars are tradable and force-flattens on
    # the session's last bar.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    # All three are real points the intended grid searches (trend_ma
    # 960/1440/1920, dip_lookback 96/192/288, dip_frac 0.5/0.75/1.0/1.5), so
    # the post-edit sanity check exercises a genuine combo. trend_ma and
    # dip_lookback are ints because they ARE bar counts and are meant to size
    # wfo_engine's warm-up buffer; dip_frac is a float fraction of the
    # volatility unit and must not.
    "trend_ma": 960,
    "dip_lookback": 96,
    "dip_frac": 0.5,
    "session": "New York",
}
