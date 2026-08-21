"""Quiet-session band rejection fade — Donchian wick rejection, channel-fraction target.

A mean-reversion *fade*, deliberately confined to the quiet part of the NQ day.
The market side is the opposite of this repo's accepted breakout iterations:
where those bought a close *through* the channel, this sells the poke that
pokes through and is immediately rejected back inside it.

The regime justification is the session gate itself, not an added filter.
volatility.md's regime table assigns low volatility to mean reversion
("Trend-following struggles"; "Low volatility? Use tight-stop mean
reversion"), and the 02:00-05:00 ET London window is structurally the quiet,
range-bound part of the NQ session day. mean-reversion.md's pitfall #1 — "Mean
reversion only works in choppy markets. In a strong uptrend, shorting strength
is a losing strategy" — is therefore answered by *when* this trades rather than
by a trend filter bolted on top. The window is not dead tape either:
nq-futures.md names "Asian market closes (2-3 AM ET affecting tech)" as a
specific NQ overnight volatility source.

Rules (all computed on continuous bars, with no session awareness of their own):

  - Donchian channel, shifted one bar so the current bar is excluded from its
    own level (without the shift, `High > upper` would compare against a
    maximum that already contains this bar's High — lookahead):
        upper = High.rolling(range_lookback).max().shift(1)
        lower = Low.rolling(range_lookback).min().shift(1)
        width = upper - lower

  - Same-bar wick rejection, not a two-bar reclaim:
        SHORT where High > upper AND Close < upper
        LONG  where Low  < lower AND Close > lower
    i.e. the bar traded outside the channel and closed back inside it. The
    two-bar "closed outside, then closed back inside" formulation is
    deliberately rejected: because `upper` is a *shifted* rolling max, a bar
    that closes above upper feeds its own High into the next bar's upper, so a
    two-bar reclaim would fire mechanically after essentially every breakout.
    The one-bar version requires the rejection to have happened while the level
    was still the level.

  - Strict `min_periods` (pandas' default = window) on both rolling extremes,
    so an unwarmed bar is NaN and every comparison against it is False. The
    signal fails *closed* (no trade) rather than open.

  - Mutual exclusivity is enforced explicitly rather than left to the
    delegate's long-first `if/elif` tie-break: on any bar where both sides
    somehow fire (a bar that pokes above a 12h+ channel's high *and* below its
    low and closes inside — effectively impossible), both are suppressed. The
    stop/target walk documents mutual exclusivity as an assumption, so it is
    made true here instead of relied upon.

Exit — path-dependent stop and a channel-fraction target, plus the session
flatten, all owned by `apply_session_constraint_with_stops()`.

  - target_price is absolute and already direction-resolved, anchored off the
    channel edge that was rejected (not off the entry price):
        SHORT -> upper - target_frac * width
        LONG  -> lower + target_frac * width
    `target_frac` is the one grid-searched exit param. It is the genuinely
    open question here: iteration 10's reading of iteration 6's oos_trades.csv
    found that essentially every trade exits at the session flatten, and London
    leaves only 2-3 hours after an entry, so *how far back into the channel is
    reachable before the flatten* is the load-bearing unknown.

  - stop_distance = STOP_WIDTH_FRAC * width, with STOP_WIDTH_FRAC = 0.25 a
    module constant that is NOT grid-searched (zero added degrees of freedom,
    and no grid/CLI plumbing change). Quoting it in the same channel-width unit
    the target uses makes it volatility-scaled by construction rather than a
    fitted point value that would need re-fitting as NQ's price level moves.
    Iterations 10 and 11 were both accepted with a hardcoded channel-width
    stop, so that dimension is treated as already settled in this repo and the
    single searched exit slot is spent on `target_frac` instead.

    Precisely where that stop sits, since the delegate anchors it at the entry
    *Close* and not at the rejection extreme: for a short it is
    `close + 0.25 * width`, while the rejection extreme is that bar's High
    (which is above `upper`, hence above `close`). So the stop sits beyond the
    wick that triggered the entry only when the poke itself was smaller than
    `0.25 * width` — typically true for one 5min bar against a 12-48h channel,
    but true empirically, not by construction. An unusually large poke can be
    stopped out by a move that never exceeds its own entry bar's High.

Two silent-refusal behaviours of the delegate, stated rather than assumed:

  - It rejects any entry whose target is not strictly on the correct side of
    that bar's Close. For a short that means it needs
    `upper - close < target_frac * width`; for a long,
    `close - lower < target_frac * width`. So an unusually wide rejection bar
    that already closed past its own target is skipped — a don't-chase filter,
    which means not every rejection is taken, and the smaller `target_frac`
    grid points are structurally the more selective ones.
  - A stop hit is *detected* intrabar on High/Low against the stored level but
    the exit is priced at that bar's Close. The stop therefore caps *when* you
    exit, not the realized loss on the triggering bar; results must not be
    described as guaranteeing a max loss of stop_distance.
  - Degenerate bars fail closed for free: an unwarmed or zero `width` makes
    stop_distance NaN/0 and the target NaN or exactly equal to the channel
    edge, and the delegate refuses to open without a strictly positive stop
    distance and a strictly correctly-sided target. No redundant masking here.
  - Consequence for reversals: the stop/target walk only opens when flat, and
    never flips long->short directly. A stopped-out trade followed by an
    opposite-side entry is two trades and two round-trip cost legs.

Iteration 6's ATR-ratio volatility-regime gate is deliberately omitted so this
family's first run is clean and attributable to the rejection entry alone. Its
*low*-volatility mirror is a reserved zero-param lever for a later in-family
variation.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here
and out of scope for this file. Stated up front rather than left for the
evaluator to discover: the reward leg here is asymmetric against the stop. The
target sits `target_frac * width` inside the channel edge, but entry is at the
rejection bar's Close, which is *already* inside that edge — so the distance
from entry to target is `target_frac * width - (edge - close)`, anywhere from
near zero up to `target_frac * width`, against a fixed `0.25 * width` stop. At
`target_frac = 0.15` the reward leg is always narrower than the stop, and some
fills will tag the target on the very next bar for a near-zero gross move while
still paying the full round trip. That is faithful to the design — no minimum
target distance is imposed — but it means the per-trade edge at the small
`target_frac` grid points has very little room above costs, and a high win rate
there should not be read as a strong edge.

Warm-up: `range_lookback` bars for the rolling extremes plus the one-bar shift,
i.e. `range_lookback + 1`. That's a genuine bar-count lookback, so it is passed
as a plain `int` from `build_grid()` and correctly sizes the engine's
pre-test-window buffer; `target_frac` is a fraction and is passed as a `float`
so it can never inflate that buffer. `STOP_WIDTH_FRAC` never enters the grid at
all, and correctly so — it introduces no new lookback, it reuses the channel
width already computed. Unlike the vol-gated iterations there is no hidden
module-constant lookback chain to count by hand here. With the intended grid
topping out at range_lookback = 576, the engine buffers
max((576 + 5) * 3, day_bars + 5) = 1743 bars, comfortably above the 577 needed,
so both extremes are fully warm at each test window's first bar. Caveat for
whoever changes the grid next: the binding floor is (max_lookback + 5) * 3 >=
max_lookback + 1, which always holds — raise the buffer rather than loosening
`min_periods` if that ever changes.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint_with_stops()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Stop distance, quoted as a fraction of the rejected channel's own width.
# Hardcoded, NOT grid-searched: 0.25 is the "a poke that then runs a quarter of
# the channel past its own rejection extreme was never a rejection" boundary,
# chosen for its economic reading rather than fitted, so it costs zero degrees
# of freedom and leaves the whole search budget for `target_frac`.
STOP_WIDTH_FRAC = 0.25


def donchian_channel(df: pd.DataFrame, range_lookback: int) -> tuple[pd.Series, pd.Series]:
    """Prior-`range_lookback`-bar high/low, shifted one bar.

    The shift excludes the current bar from its own channel level — without
    it, `High > upper` would be comparing this bar's High against a maximum
    that already contains it (lookahead, and the comparison could never be
    strictly true).

    Strict `min_periods` (pandas' default = window) so an unwarmed bar is NaN
    in both levels; every downstream comparison against NaN is False, so the
    signal fails closed.
    """
    upper = df["High"].rolling(range_lookback, min_periods=range_lookback).max().shift(1)
    lower = df["Low"].rolling(range_lookback, min_periods=range_lookback).min().shift(1)
    return upper, lower


def generate_positions(
    df: pd.DataFrame,
    range_lookback: int,
    target_frac: float,
    session: str | None = "London",
) -> pd.Series:
    close = df["Close"]
    upper, lower = donchian_channel(df, range_lookback)
    width = upper - lower

    # Same-bar wick rejection: traded outside the channel, closed back inside.
    short_raw = ((df["High"] > upper) & (close < upper)).fillna(False)
    long_raw = ((df["Low"] < lower) & (close > lower)).fillna(False)

    # Explicit mutual exclusivity — see the module docstring. Effectively never
    # binds against a 12h+ channel, but the stop/target walk assumes the two
    # sides are exclusive, so that is made true here rather than falling
    # through to its long-first if/elif tie-break.
    both = long_raw & short_raw
    long_signal = long_raw & ~both
    short_signal = short_raw & ~both

    # Stop as a fraction of the rejected channel's width — symmetric, resolved
    # to a side by the delegate. NaN/0 width (warm-up, or a degenerate flat
    # channel) propagates here and makes the delegate refuse the entry.
    stop_distance = STOP_WIDTH_FRAC * width

    # Absolute, direction-resolved target: a fixed fraction of the channel back
    # inside the edge that was rejected (anchored off the channel, not off the
    # entry price — which is exactly why the delegate takes an absolute level
    # for the target and a distance for the stop). Only read on actual entry
    # bars, so the value on non-signal bars is irrelevant; it is still sided
    # off `long_signal` for readability.
    target_price = pd.Series(
        np.where(long_signal, lower + target_frac * width, upper - target_frac * width),
        index=df.index,
    )

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
    "range_lookback": 288,
    "target_frac": 0.25,
    "session": "London",
}
