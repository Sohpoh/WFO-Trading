"""Bollinger mid-band fade, with a sigma-scaled stop and the mid-band as target.

Rules (mean-reversion.md, "Simple Mean Reversion Strategies": *enter a short
when price touches the upper band, expecting reversion to the middle; exit at
the middle or the opposite band; reverse for longs at the lower band*):

  - mean = Close.rolling(band_lookback).mean(),
    sigma = Close.rolling(band_lookback).std(),
    z = (Close - mean) / sigma. Non-positive/degenerate sigma is dropped to
    NaN so it can never produce an infinite z or a zero-width stop.
  - Entry is a CROSSING of the band, not a level: LONG when z crosses *down*
    through -entry_z (z <= -entry_z with the prior bar above it, i.e. the bar
    price pushes through the lower band), SHORT when z crosses *up* through
    +entry_z (upper band). This matters because
    `session.apply_session_constraint_with_stops()` skips same-bar re-entry
    after a stop (`continue`) but re-evaluates on the very next bar — a
    persistent level condition (`z <= -entry_z`) would re-open the same
    losing fade bar after bar all the way down a trend, which is exactly the
    high-turnover-fade failure the prior strategies were built to avoid. A
    crossing gives exactly one entry per excursion. Zero-param structural
    guard.
  - No flip on an opposite signal: the walk only opens when flat, so an
    opposing band touch while in a trade is a no-op.
  - Exits are path-dependent and owned entirely by session.py's walk:
    a stop `stop_sigma_mult * sigma` from the entry Close, a target at the
    mid-band (`mean`) as of the entry bar, and the forced flatten on the
    session's last bar.

Why the target needs no direction-resolution: a long can only fire when
z <= -entry_z < 0, so mean - Close = -z * sigma >= entry_z * sigma > 0 and
the mid-band necessarily sits *above* that bar's Close; symmetrically a short
only fires with the mid-band below it. So passing `mean` straight through as
`target_price` satisfies the walk's side-validity check on both sides without
a `.where()` split (unlike the prior ATR strategy, whose symmetric
`close +/- k*ATR` target genuinely needed one).

Both the entry threshold and the stop are quoted in the *same* rolling-sigma
units — deliberately not ATR — so the stop:target ratio does not silently
rescale as `band_lookback` changes and the grid cannot win a fold on a pure
scaling artifact.

Design intent: a low-turnover fade with a move-to-cost ratio well above 1. A
2.0-3.0 sigma entry puts the mid-band target roughly 0.3-0.5% away on ES
15min against `metrics.py`'s ~0.102% round trip — a 3-5x ratio, versus
transaction-costs.md's warning that a 5bp edge dies on 4bp of costs.

sigma/mean are deliberately NOT shifted by one bar: the numerator already
uses the current Close and the entry fills at that same Close, so including
the current bar in the rolling window is not look-ahead. Don't "fix" it to
`.shift(1)`.

Like this codebase's prior strategies, the bands are computed on the full
continuous-Globex df rather than session-scoped, so an early-session signal's
lookback can reach back into the prior evening's bars. Intentional.

This module only decides *when the strategy wants to enter and where its
stop/target sit*; it has no session awareness of its own. See
`session.apply_session_constraint_with_stops()`'s docstring for why the
path-dependent walk and the session-boundary flatten live there.

Caveat carried from session.py: stops/targets are *detected* off intrabar
High/Low but the exit still prices at that bar's Close (this app has no
finer-than-bar price series and `metrics.py` costs everything off Close), so
a stop does not cap the realized loss on the triggering bar — see
`apply_session_constraint_with_stops()` for details before describing
results as risk-capped.
"""
import pandas as pd

from session import apply_session_constraint_with_stops


def compute_bands(df: pd.DataFrame, band_lookback: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(z-score, mid-band, sigma) for the given rolling Bollinger lookback.

    Includes the current bar (see module docstring: not look-ahead, since
    entries fill at that bar's Close). sigma is masked to NaN where it is
    non-positive or still warming up, which propagates into both `z` (no
    signal) and the stop distance (`stop_ok` False in the walk) from one
    place.
    """
    close = df["Close"]
    mean = close.rolling(band_lookback, min_periods=band_lookback).mean()
    sigma = close.rolling(band_lookback, min_periods=band_lookback).std()
    sigma = sigma.where(sigma > 0)
    z = (close - mean) / sigma
    return z, mean, sigma


def generate_positions(
    df: pd.DataFrame,
    band_lookback: int,
    entry_z: float,
    stop_sigma_mult: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    z, mean, sigma = compute_bands(df, band_lookback)
    prior = z.shift(1)

    # Crossing, not level — one entry per excursion (see module docstring).
    # Long fades the lower band, short fades the upper band.
    long_signal = ((z <= -entry_z) & (prior > -entry_z)).fillna(False)
    short_signal = ((z >= entry_z) & (prior < entry_z)).fillna(False)

    stop_distance = stop_sigma_mult * sigma

    # The mid-band is the target for both sides and is direction-correct by
    # construction (mean > Close on every long signal, mean < Close on every
    # short), so it needs no per-side resolution here.
    target_price = mean

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
    "band_lookback": 30,
    "entry_z": 2.5,
    "stop_sigma_mult": 1.5,
    "session": "New York",
}
