"""Failed-breakout reversal — an N-bar channel extreme is broken, the break
fails to hold, and the reversion back through the channel is traded.

`mean-reversion.md`'s microstructure mechanism read literally: "forced
buying/selling (e.g. stop-losses triggering) overshoots, then reverts". A
break of a well-established N-bar extreme that price then closes back inside
is the observable signature of that overshoot — a liquidity grab that failed.
The reversion target is the same page's documented Bollinger exit ("expecting
reversion to middle. Exit at middle or opposite band"), quoted here as a
fraction of the channel's own width.

What makes this a different measurement from the fades already in this log:
iterations 1, 12 and 21 all measured "extreme" as a *magnitude* — a z-score of
Close against its own rolling mean — and all three failed with ample power on
15min New York (461/444/757 OOS trades), iteration 21 decisively so (PF ~0.68
on both IS and OOS with *both* polarities live, i.e. the rolling-mean z-score
carries no information on this data in either direction). This measures a
*structural level plus a path event* instead: a new N-bar extreme, then a
close back inside it. Structural-level fades in this log so far are iteration
13 (London 5min, same-bar wick rejection, in a session iterations 15/17 later
established as the binding constraint) and iteration 16 (errored before it
ran), so no well-powered structural-level fade exists on New York.

Rules (all computed on the full continuous frame with no session awareness of
their own; every rolling window uses strict `min_periods` so an unwarmed leg
is NaN and the signal fails *closed* — no entry — rather than silently
degrading to an always-true no-op):

  - Channel, from prior bars only (both legs `.shift(1)`, so nothing about
    bar t's own High/Low enters bar t's channel):
        upper = High.rolling(range_lookback).max().shift(1)
        lower = Low.rolling(range_lookback).min().shift(1)
        width = upper - lower,  masked to NaN where width <= 0
    The width mask is a correctness guard, not an economic filter: it is what
    guarantees every entry bar has a strictly positive `stop_distance` and a
    `target_price` strictly on the correct side of the Close — both
    preconditions of `apply_session_constraint_with_stops()`.

  - Break events at bar t:  break_up = High_t > upper_t
                            break_dn = Low_t  < lower_t
    NaN during warm-up compares False on both, so the gate fails closed.

  - Raw signals — the break must be strictly in the *past* (never the current
    bar), and the current bar must have closed back inside the channel:
        failed_up = break_up.rolling(fail_window).max().shift(1) > 0
        short = failed_up & (Close_t < upper_t)
        long  = failed_dn & (Close_t > lower_t)      # mirror
    The `.rolling(fail_window).shift(1)` pair means "an upside break occurred
    on at least one of bars t-1 ... t-fail_window".

  - Tiebreak, stated here rather than left to `session.py`'s branch order: if
    a failed upside break and a failed downside break are both live on the
    same bar, *neither* fires. `long_signal` and `short_signal` are therefore
    mutually exclusive by construction and the single-position contract holds
    without depending on which branch of
    `apply_session_constraint_with_stops()` happens to be evaluated first.

  - No lookahead: every input is a backward-looking window ending at bar t,
    every value is knowable at that bar's Close, and the entry is priced at
    that same Close.

  Observed property worth stating plainly, because the prose "closed back
  inside the channel" is looser than it sounds: `upper_t` is the rolling max
  over the `range_lookback` bars ending at t-1, so if the break happened at
  t-1 then `upper_t` already *absorbed* that break bar's own High. Condition
  (b) at fail_window=1 is thus roughly `Close_t < High_{t-1}` rather than
  "price reclaimed the pre-break level". This is the entry rule as specified
  and is implemented literally; it is not a strict pre-break-level reclaim,
  and the trade count should be read with that in mind.

Exit is a path-dependent stop/target — deliberately bounded rather than the
run-to-session-flatten shape of iterations 6/10/11, all three of which
`loop-review.md` showed collapsing under gate v2's leave-top-5-out hard gate.
The design goal is a distribution of moderate winners, not a handful of
outliers:

  - `long_signal`/`short_signal`/`stop_distance`/`target_price` go to
    `session.apply_session_constraint_with_stops()` (NOT the plain flip
    variant), which owns the bar-by-bar walk, the session gating and the
    forced flatten on the session's last bar.
  - stop_distance = stop_width_frac * width, held fixed at 0.5 and NOT
    grid-searched. Quoting the stop in channel-width units is the same
    formulation that coexisted with this repo's only positive OOS runs. It is
    deliberately *not* a tight "risk to the failed extreme" stop:
    `session.py` detects a stop off the bar's High/Low but flattens at that
    bar's Close, so a tight stop over-realizes the triggering bar's whole
    adverse move — the pathology behind the 1.7x / 3.2x / 1.82x
    avg-loss:avg-win ratios in iterations 13/14/15.
  - target_price is quoted as a *distance from the entry Close*, so it is
    always strictly on the correct side by construction and no entry is
    silently dropped by that function's `target > close` / `target < close`
    checks:
        short target = Close - target_frac * width
        long  target = Close + target_frac * width
    `target_frac = 0.5` lands a reclaim near the band roughly at the channel
    middle (mean-reversion.md's "exit at middle"), 1.0 approximates the
    opposite band, 0.25 harvests a partial reversion.
  - Fill-price caveat is `session.py`'s: a stop/target hit is detected off the
    bar's High/Low but flattened at that bar's Close, so the stop caps *when*
    you exit, not the realized loss on the triggering bar.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here
and out of scope for this file. The smallest searched target (0.25 x a 24-bar
channel width on 15min NQ, typically ~0.18-0.25%) is only ~2x that toll — the
thinnest corner of the grid, and worth checking in fold selection.

Warm-up and param types:

  - `range_lookback` and `fail_window` are both genuine bar-count lookbacks,
    so both are passed as plain `int` from `build_grid()` and correctly feed
    `wfo_engine._max_lookback_bars()`'s pre-test-window buffer.
    `target_frac` and `stop_width_frac` are fractions of a channel width, not
    bar counts, and are passed as `float` so they cannot inflate that buffer.
  - Every window here is a grid param, so nothing is hidden from the engine:
    with the intended grid the largest int is 192, giving
        buffer_bars = max((192 + 5) * 3, day_bars + 5) = 591
    against an actual need of range_lookback + fail_window + 2 ~= 200. That
    clears comfortably, and unlike the previous iteration there is no floor
    constraint on how small the grid's largest lookback may be shrunk.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`session.apply_session_constraint_with_stops()`; see its docstring for that
contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops


def donchian_channel(
    high: pd.Series, low: pd.Series, range_lookback: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Prior-bars-only N-bar channel: (upper, lower, width).

    Both legs are `.shift(1)`-ed after the rolling extreme, so the channel at
    bar t is built purely from bars t-range_lookback ... t-1 and bar t's own
    High/Low can break it without having helped define it.

    `width` is masked to NaN wherever it is not strictly positive (a
    degenerate flat stretch). That is what makes the stop distance strictly
    positive and the target strictly on the correct side of the Close at every
    bar that can produce a signal.
    """
    upper = high.rolling(range_lookback, min_periods=range_lookback).max().shift(1)
    lower = low.rolling(range_lookback, min_periods=range_lookback).min().shift(1)
    width = upper - lower
    return upper, lower, width.where(width > 0)


def _failed_recently(break_event: pd.Series, fail_window: int) -> pd.Series:
    """True at bar t if `break_event` was True on any of bars t-1 ... t-fail_window.

    The `.shift(1)` after the rolling max is what keeps the break strictly in
    the past — a break on bar t itself never qualifies bar t as a failure.
    Strict `min_periods` leaves the first bars NaN, which `> 0` renders False,
    so the gate fails closed during warm-up.
    """
    recent = break_event.astype(float).rolling(fail_window, min_periods=fail_window).max().shift(1)
    return (recent > 0).fillna(False)


def generate_positions(
    df: pd.DataFrame,
    range_lookback: int,
    fail_window: int,
    target_frac: float,
    stop_width_frac: float = 0.5,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    upper, lower, width = donchian_channel(high, low, range_lookback)

    # Break events. NaN channel levels during warm-up compare False on both
    # sides, so no break is ever recorded on an unwarmed bar.
    break_up = (high > upper).fillna(False)
    break_dn = (low < lower).fillna(False)

    failed_up = _failed_recently(break_up, fail_window)
    failed_dn = _failed_recently(break_dn, fail_window)

    # Close back inside the channel. NaN levels fail closed here too, and the
    # width mask propagates: no valid width means no stop/target and hence no
    # entry regardless.
    reclaimed_down = (close < upper).fillna(False)
    reclaimed_up = (close > lower).fillna(False)

    raw_short = failed_up & reclaimed_down
    raw_long = failed_dn & reclaimed_up

    # Explicit tiebreak: if both sides are live on the same bar, take neither.
    # This makes long/short mutually exclusive by construction rather than by
    # apply_session_constraint_with_stops()'s branch order.
    both = raw_long & raw_short
    long_signal = raw_long & ~both
    short_signal = raw_short & ~both

    stop_distance = stop_width_frac * width

    # Direction-resolved absolute target level, NaN off signal bars (which is
    # how apply_session_constraint_with_stops() reads "no valid entry here").
    # Quoted as a distance from the entry Close, so it is always strictly on
    # the correct side wherever `width` is valid.
    target_price = pd.Series(np.nan, index=df.index)
    target_price[long_signal] = (close + target_frac * width)[long_signal]
    target_price[short_signal] = (close - target_frac * width)[short_signal]

    return apply_session_constraint_with_stops(
        close=close,
        high=high,
        low=low,
        long_signal=long_signal,
        short_signal=short_signal,
        stop_distance=stop_distance,
        target_price=target_price,
        session=session,
    )


DEFAULT_PARAMS = {
    "range_lookback": 48,
    "fail_window": 2,
    "target_frac": 0.5,
    "stop_width_frac": 0.5,
    "session": "New York",
}
