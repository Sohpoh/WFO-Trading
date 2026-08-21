"""Relative-volume surge continuation — ET-slot-normalized volume trigger,
trailing-return direction, plain flip exit.

Seventeen iterations in this repo have mined price and price-volatility only
(Donchian channels, ATR, drift t-stats, range expansion) — every one of those
statistics is mechanically correlated with the very move being traded. The
`Volume` column has never been used once. This iteration spends that unused
column: `es-futures.md`'s ORB recipe states the confirmation clause plainly —
"If price breaks above the range high on high volume, go long" — and this lifts
the volume-confirmation half as the *trigger*, dropping the session-anchored
opening-range half.

Session: New York. Four London primitives (iterations 13/14/15/17) all failed
at the gross level and iteration 17 met its own pre-registered hurdle for
retiring the London steer entirely, so this returns to the session where every
accepted run in the log lives (6, 10, 11), on the continuation side — the only
side that has ever produced positive IS and OOS here.

Rules (all computed on continuous, session-unaware bars):

  - Relative volume, normalized *within* the Eastern-Time time-of-day slot:

        rvol = Volume / median(prior RVOL_OBS same-ET-slot Volumes)

    Bars are bucketed on their Eastern-Time minute-of-day; inside each bucket
    the baseline is `rolling(RVOL_OBS, min_periods=RVOL_OBS).shift(1)` median,
    so the current bar is excluded from its own yardstick and unwarmed buckets
    stay NaN (fail closed).

    ET bucketing rather than UTC is load-bearing, not cosmetic: New York's UTC
    offset moves twice a year, so a UTC-slot baseline would compare open-slot
    volume against pre-open volume for ~10 sessions after each DST switch,
    inflating `rvol` and firing on nearly every bar in eight contaminated
    windows. Bucketing on ET keeps 09:30 compared against 09:30 year-round.

    This is time-of-day *normalization inside an indicator*. It gates nothing
    and flattens nothing — `apply_session_constraint()` still owns all session
    logic, per CLAUDE.md's Day-Trade Session Requirement.

    `.where(baseline > 0)` masks a zero or NaN baseline: without it a zero
    denominator gives +inf, and +inf clears every threshold in the grid, so a
    dead slot would fire at every setting.

  - Direction, from a pre-existing trailing return (orthogonal to the trigger):

        d = Close - Close.shift(thrust_lookback)

    Rejected iteration 14 took direction from close-location-in-bar and won
    28.3% of trades; the surge bar's own shape is not used here at all.

  - Entries:
        LONG  where rvol >= rvol_mult and d > 0
        SHORT where rvol >= rvol_mult and d < 0
        NaN   elsewhere
    `d == 0` and any unwarmed/NaN bar fail closed (NaN compares False on both
    sides). The two sides are mutually exclusive by construction.

Exit — plain flip, no stop and no target. The raw entries series is handed
straight to `apply_session_constraint()` (deliberately NOT the stops variant).
Position is the forward-fill of the gated entries, so a trade ends only on an
opposite-direction surge trigger or on the forced New York flatten the delegate
owns; consecutive same-direction triggers are no-ops. This keeps the first run
of the family attributable to the entry primitive alone and preserves the
let-winners-run-to-flatten shape of accepted iteration 6. A failed-breakout-
style stop is the reserved follow-up lever, exactly as iteration 6 -> 10 spent
it.

Iteration 6's volatility-regime gate is deliberately omitted: iterations 8 and
9 both tested "trade fewer, more selective" and both came back overfit-gap with
trade counts collapsing 183 -> 100 -> 74. A volume surge is already a de-facto
activity filter; stacking a second one is the known failure path.

Warm-up and param types:

  - `rvol_baseline_bars` (fixed at 960, one grid value, never searched) is
    expressed as a *bar count* rather than an observation count precisely so
    `wfo_engine._max_lookback_bars()` can see it: the observation count is
    derived as `rvol_baseline_bars // BARS_PER_DAY = 960 // 96 = 10`. It is
    passed as a plain `int`, giving `buffer_bars = max((960 + 5) * 3, ...) =
    2895` bars, roughly 30 same-slot observations against the 11 needed — warm
    at every test window's first bar. Making it a module constant instead would
    render it invisible to that derivation and leave the baseline NaN (signal
    closed) for most of every test window; making it a float would starve the
    buffer down to ~111 bars, i.e. about one same-slot observation, with the
    same effect. `tools/check_strategy.py` warns about ints > 500 in the grid —
    here that warning is expected and must not be "fixed".

  - `thrust_lookback` is a genuine bar-count lookback and is a plain `int`.
  - `rvol_mult` is a dimensionless ratio threshold and is a `float`, so it can
    never inflate the warm-up buffer.

  - `BARS_PER_DAY = 96` is a hardcoded module constant and is never a
    `build_grid()` param — that exact mistake was iteration 4's error. It is
    also deliberately not derived from the passed `df`: on a 1h run that would
    make the observation count 41 and leave most of every window cold, whereas
    the constant keeps it at 10 for any timeframe.

Cost caveat to carry into evaluation (NOT changed here — `metrics.py` is out of
scope for this file): the model charges a flat 0.05% slippage per leg. A volume
*surge* bar is, if anything, a bar where liquidity is unusually deep, so the
constant-slippage assumption is more defensible on this trigger than on the
quiet-session iterations — but it is still a constant, and no per-bar
adjustment is made.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint()`; see its docstring for that contract.
"""
import pandas as pd

from session import apply_session_constraint

# Bars in one 24h day at the intended 15min timeframe. Hardcoded module
# constant, NEVER a build_grid() param and never derived from the data — it is
# only the divisor that turns the `rvol_baseline_bars` bar count into a
# same-slot observation count.
BARS_PER_DAY = 96


def relative_volume(df: pd.DataFrame, baseline_bars: int) -> pd.Series:
    """Volume divided by its own recent median in the same Eastern-Time slot.

    `baseline_bars` is a bar count (so the engine's warm-up derivation can see
    it); the number of prior same-slot observations used is
    `baseline_bars // BARS_PER_DAY`, floored at 1.

    Bucketing is on Eastern-Time minute-of-day — an int64 grouper, and DST-safe
    because `tz_convert` resolves the offset per timestamp rather than assuming
    a fixed one. Within each bucket the median is rolled over strictly prior
    observations (`.shift(1)`), with strict `min_periods` so unwarmed buckets
    stay NaN and every comparison against them is False.
    """
    n_obs = max(1, int(baseline_bars) // BARS_PER_DAY)

    volume = df["Volume"].astype(float)
    et = df.index.tz_convert("America/New_York")
    slot = et.hour * 60 + et.minute

    baseline = volume.groupby(slot).transform(
        lambda s: s.rolling(n_obs, min_periods=n_obs).median().shift(1)
    )

    # A zero baseline would give +inf, which clears every threshold in the
    # grid; NaN (warm-up) is masked by the same expression.
    return (volume / baseline).where(baseline > 0)


def generate_positions(
    df: pd.DataFrame,
    rvol_mult: float,
    thrust_lookback: int,
    rvol_baseline_bars: int = 960,
    session: str | None = "New York",
) -> pd.Series:
    rvol = relative_volume(df, rvol_baseline_bars)

    close = df["Close"]
    thrust = close - close.shift(thrust_lookback)

    surge = (rvol >= rvol_mult).fillna(False)
    long_signal = (surge & (thrust > 0)).fillna(False)
    short_signal = (surge & (thrust < 0)).fillna(False)

    entries = pd.Series(index=df.index, dtype=float)  # all-NaN = "no signal"
    entries[long_signal] = 1.0
    entries[short_signal] = -1.0

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "rvol_mult": 4.0,
    "thrust_lookback": 8,
    "rvol_baseline_bars": 960,
    "session": "New York",
}
