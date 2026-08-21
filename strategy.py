"""With-trend pullback reclaim — trend-gated z-dip re-entry, sigma stop, no target.

The market side is unchanged from the last accepted iterations: this is a
*continuation* strategy, long in uptrends and short in downtrends. What changes
is the entry primitive. Instead of "buy the new extreme" (a channel breakout),
this buys the *dip inside an already-established trend* and only at the moment
the dip is being reclaimed. Buying a pullback rather than a breakout buys a
materially better entry price, which in turn lets the stop sit much closer than
a channel-width stop could.

Read the exit framing before reading this as a fade: the pullback is only the
*entry timing*; the trend is the *P&L source*. There is therefore no
mean-reversion profit target — winners run to the forced session flatten,
exactly as in the accepted breakout iterations.

The discriminating difference versus this repo's earlier, no-edge
bollinger-band-fade iteration is regime gating, which is pitfall #1 in
mean-reversion.md: "Mean reversion only works in choppy markets. In a strong
uptrend, shorting strength is a losing strategy." That earlier iteration ran
the z-dip *ungated*, so it faded strength as readily as it bought weakness.
Here the z-dip is only ever taken *in the direction of* the prevailing trend
and never against it.

Rules (all computed with no session awareness of their own):

  - Trend filter, a single slow SMA:
        sma_trend = SMA(Close, trend_ma)
        uptrend   = Close >  sma_trend
        downtrend = Close <  sma_trend
    These are exact complements on the same MA (the only overlap case, Close
    == sma_trend exactly, makes both False), so long and short can never fire
    on the same bar — the mutual-exclusivity property the stop/target walk
    relies on.

  - Dip measure, a residual z-score against a fast SMA:
        resid = Close - SMA(Close, Z_LOOKBACK)
        sigma = rolling std(resid, Z_LOOKBACK)
        z     = resid / sigma
    `Z_LOOKBACK` is a hardcoded module constant (20), deliberately NOT
    grid-searched — see the constant's own note for why that ratio to
    `trend_ma` matters.

  - Entries are the *reclaim crossing*, not the level:
        LONG  where uptrend   AND z.shift(1) <= -entry_z AND z > -entry_z
        SHORT where downtrend AND z.shift(1) >= +entry_z AND z < +entry_z
    Using the crossing rather than "z is currently beyond the level" is what
    stops one deep dip re-triggering bar after bar for as long as it stays
    extended — the same fix an earlier iteration in this repo introduced for
    exactly this failure mode. It also means the entry is timed to the moment
    the pullback stops working against the trend, not to the moment it starts.

  - Strict `min_periods` on all three rolling windows (the trend SMA, the fast
    SMA, and the residual std), so an unwarmed bar is NaN. Every comparison
    against NaN is False in pandas, so an unwarmed bar produces no signal —
    the filter fails *closed* (suppress the trade) rather than open.

Exit — a residual-sigma stop, no target.

  - stop_distance = STOP_SIGMA_MULT * sigma, with STOP_SIGMA_MULT = 1.5 a
    module constant that is *not* grid-searched (zero added degrees of
    freedom, and no grid/CLI plumbing change). The economic reading: a reclaim
    that then extends 1.5 further residual-sigma against the entry was not a
    pullback, it was the trend breaking, and it is cut.
  - Quoting the stop in the *same sigma unit the entry already uses* makes it
    volatility- and price-level-adaptive by construction, rather than a fitted
    point value that would have to be re-fit as NQ's price level moves.
  - No profit target. `apply_session_constraint_with_stops()` requires a
    non-NaN, correctly-sided target on every entry bar, so the target is set
    to a deliberately unreachable `UNREACHABLE_TARGET_MULT` stop-distances
    from the entry Close. One session cannot travel 150 residual sigma, so the
    profit side is governed only by the session flatten.
  - Degenerate bars fail closed for free, and both guards agree: NaN or zero
    `sigma` makes `stop_distance` NaN/0, and the delegate refuses to open
    without a strictly positive stop distance; the same NaN/0 also makes the
    target NaN or exactly equal to Close, which fails the delegate's strict
    sidedness test. No redundant masking is added here for that.
  - Fill-price honesty (see the delegate's own docstring): a stop hit is
    *detected* intrabar on High/Low against the stored level, but the exit is
    priced at that bar's Close. So the stop caps *when* you exit, not the
    realized loss on the triggering bar — results must not be described as
    guaranteeing a maximum loss of stop_distance.
  - Consequence for reversals, stated explicitly: the stop/target walk never
    flips directly from long to short — it only opens when flat. A stop-out
    followed by an opposite-side entry is two trades and two round-trip cost
    legs, not one flip.

An ATR-ratio volatility-regime gate (used by the accepted breakout iterations)
is deliberately NOT included here, so this family's first run is clean and
attributable to the trend-gated pullback entry alone. It is a reserved
zero-param lever for a later in-family variation.

Warm-up arithmetic. Two independent chains:
  - the trend SMA needs `trend_ma` bars. That's a genuine bar-count lookback,
    so it is passed as a plain `int` from `build_grid()` and correctly sizes
    the engine's pre-test-window buffer.
  - the z chain needs SMA(Z_LOOKBACK) -> resid -> rolling std(Z_LOOKBACK)
    -> one bar of `.shift(1)`, i.e. 20 + 19 + 1 = 40 bars. This chain is
    invisible to `wfo_engine._max_lookback_bars()` (module constants are not
    grid params), so it is counted by hand here.
`entry_z` is a z-score threshold, not a bar count, and is passed as a `float`
so it can never inflate that buffer. With the intended grid topping out at
trend_ma = 384, the engine buffers max((384 + 5) * 3, day_bars + 5) = 1167
bars, comfortably above both 384 and 40, so every indicator is fully warm at
each test window's first bar. Caveat for whoever changes the grid next: the
binding floor on the longest searched `trend_ma` is (max_lookback + 5) * 3 >=
max_lookback, which always holds, so the trend MA is safe at any grid size;
raise the buffer rather than loosening `min_periods` if that ever changes.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here
and out of scope for this file. Sizing sanity, measured on NQ 15min (checked on
both a 2023 and a 2025 slice, which agree closely): `STOP_SIGMA_MULT * sigma`
runs a median ~0.19% of price, with a 10th-90th percentile band of roughly
0.07%-0.56%. So the typical stop is only ~1.9x the round-trip toll, and in the
quietest decile of bars the stop is *narrower than the toll itself* — a much
tighter stop than the breakout iterations' channel-width one. Two consequences
worth stating up front rather than discovering in the results: the per-trade
edge has very little room above costs, and low-volatility bars are structurally
the worst trades here (small sigma tightens the stop faster than it shrinks the
cost, which is fixed in percentage terms). A volatility floor is one obvious
zero-param lever for a later in-family variation, but is deliberately not
included in this first run.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint_with_stops()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Lookback, in bars, for the fast SMA and the residual standard deviation that
# together define the dip z-score. Hardcoded on purpose, NOT grid-searched:
# fixing it keeps the search at 2 dimensions (matching every accepted run in
# this log, and strategy-development.md's "complexity usually signals weak
# edge"), leaves one param slot free for the next in-family variation, and
# holds trend_ma / Z_LOOKBACK >= 4.8 at every intended grid point so the two
# windows measure genuinely different horizons — a multi-day trend versus a
# few-hours pullback — rather than two noisy views of the same one.
Z_LOOKBACK = 20

# Stop distance, quoted in the same residual-sigma unit the entry threshold
# uses. Hardcoded, NOT grid-searched: 1.5 is the "this was not a pullback, the
# trend broke" boundary, chosen for its economic reading rather than fitted,
# so it costs zero degrees of freedom.
STOP_SIGMA_MULT = 1.5

# Multiple of the stop distance used as the (deliberately unreachable) profit
# target. The stop/target walk requires a non-NaN, correctly-sided target on
# every entry bar; 100 stop-widths (= 150 residual sigma) inside one session is
# not attainable, so this reproduces "no target — winners run to the session
# flatten".
UNREACHABLE_TARGET_MULT = 100.0


def dip_zscore(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Residual z-score of Close against its own `Z_LOOKBACK`-bar SMA.

    Returns `(z, sigma)`: sigma is handed back as well because the exit stop is
    quoted in the very same unit, so it must not be recomputed from a different
    window.

    Strict `min_periods` (pandas' default = window) on both rolling windows, so
    an unwarmed bar is NaN in `sigma` and therefore NaN in `z`. Comparisons
    against NaN are False, so an unwarmed bar can produce no entry — it fails
    closed. A degenerate flat window (sigma == 0) yields inf/NaN in `z` and a
    zero stop distance, which the delegate refuses.
    """
    sma_fast = close.rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).mean()
    resid = close - sma_fast
    sigma = resid.rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).std()
    return resid / sigma, sigma


def generate_positions(
    df: pd.DataFrame,
    trend_ma: int,
    entry_z: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]

    # Trend gate — exact complements on one MA, so the two sides below are
    # mutually exclusive by construction.
    sma_trend = close.rolling(trend_ma, min_periods=trend_ma).mean()
    uptrend = close > sma_trend
    downtrend = close < sma_trend

    z, sigma = dip_zscore(close)
    z_prev = z.shift(1)

    # The *reclaim crossing*, not the level: the dip must have been beyond the
    # threshold on the prior bar and be back inside it now. A dip that stays
    # extended therefore fires exactly once, on the bar it is reclaimed.
    long_signal = (uptrend & (z_prev <= -entry_z) & (z > -entry_z)).fillna(False)
    short_signal = (downtrend & (z_prev >= entry_z) & (z < entry_z)).fillna(False)

    # Stop in the same residual-sigma unit the entry threshold uses. NaN/0
    # sigma (warm-up, or a degenerate flat window) propagates here and makes
    # the delegate refuse the entry.
    stop_distance = STOP_SIGMA_MULT * sigma

    # No real target — an unreachable level on the correct side of the entry
    # Close, so the profit side is governed only by the session flatten. Only
    # read on actual entry bars, so the value on non-signal bars is irrelevant
    # (but is still sided off `long_signal` for readability).
    reach = UNREACHABLE_TARGET_MULT * stop_distance
    target_price = pd.Series(
        np.where(long_signal, close + reach, close - reach),
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
    "trend_ma": 192,
    "entry_z": 1.0,
    "session": "New York",
}
