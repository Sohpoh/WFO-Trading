"""ES 1h Bollinger-Band Reversion to Mean — a *mean-reversion* reading of
`Trading Vault wiki/mean-reversion.md`: fade a Close that has touched or
penetrated a trailing Bollinger band back to the trailing mean, with a
volatility-scaled hard stop and the trailing mean itself as the profit target.

This is the opposite side of the vault's regime table from the momentum
families that immediately preceded it. It trades ES (the vault's smoother,
lower-vol, more mean-reverting instrument) at 1h — a symbol/timeframe slice
never run as a fade — and the bounded "exit at the middle" target gives the
"frequent small wins" shape the evaluator's leave-top-5-out check rewards.

The mechanism
-------------
All quantities are session-unaware and strictly causal: the trailing mean and
standard deviation windows end at the current bar (inclusive), so there is no
lookahead. `metrics.bar_returns_with_costs` prices a position off
`position.shift(1)`, so a position set at bar t earns the Close_t -> Close_{t+1}
return the signal never sees.

  - On each bar compute the trailing mean μ and population std σ of Close over
    `band_lookback` bars (causal — the window ends at the current bar). σ is
    population (ddof=0), the canonical Bollinger-band convention.

  - Upper band = μ + `entry_z`·σ; lower band = μ − `entry_z`·σ.

  - A short (−1) is emitted where Close >= upper band (touch/penetration); a
    long (+1) where Close <= lower band (touch). Symmetric both directions.
    During warm-up μ/σ are NaN, so both comparisons fail closed and no entry
    is emitted on the first `band_lookback − 1` bars.

  - There is no opposite-signal flip: the band itself decides direction when
    flat, and once a position is open it exits only via its stop, its target,
    or the forced end-of-session flatten (the stop/target walk in session.py
    only ever opens from flat).

Exit — path-dependent stop/target, both quoted in trailing σ
------------------------------------------------------------
`apply_session_constraint_with_stops` (NOT the plain flip variant) — a
path-dependent stop/target pair, per mean-reversion.md's stop-placement rule
and "exit at the middle" target.

  - Target = the trailing mean μ. Direction-resolved by the walk: a long
    entered at the lower band has μ sitting *above* entry (μ = lower + z·σ),
    a short entered at the upper band has μ *below* entry (μ = upper − z·σ),
    so the one μ series serves both sides. Only μ's value at the actual entry
    bar is read (session.py stores it as a fixed `target_state`), so the
    target is the mean *at entry*, not a moving target.

  - Stop = entry ± `stop_sigma_mult`·σ, volatility-scaled (per volatility.md's
    stop-placement rule and mean-reversion.md pitfall #4 stop-loss discipline).
    A σ of 0 (perfectly flat window) makes the stop distance 0, which fails
    the walk's `stop_ok` gate and closes that bar to entries.

  - The New York session force-flattens at 16:00 ET, so every trade is closed
    within the session (day-trade only, no overnight).

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
--------------------------------------------------------------------------
  - `band_lookback` IS a genuine bar-count lookback (the rolling mean/σ
    window), so it is a plain `int` **on purpose**: it *should* feed
    `_max_lookback_bars()`, whose max over the grid ({20, 40, 60, 80} → 80)
    sizes the pre-test-window warm-up buffer to
    `max((80 + 5) * 3, day_bars + 5)` (~255 bars at 1h), comfortably covering
    the 79-bar warm-up before the first valid band.
  - `entry_z` and `stop_sigma_mult` are `float` **on purpose** — they are
    σ-multiples (dimensionless thresholds / stop scaling), not bar-count
    lookbacks, so they must NOT feed `_max_lookback_bars()`. Both are cast to
    float in `build_grid()` so a grid point written as `2` can never arrive as
    an int and silently inflate the buffer.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.005% slippage) is
unchanged and out of scope. This strategy is symmetric (both sides traded), so
a long↔short reversal costs two legs (close + open); no cost logic lives here.

This module decides only *when* the strategy wants to fade a band touch and how
far it lets a loser run before the σ-scaled stop exits. All day-trade gating
and the end-of-session flatten are delegated to `session.py`; see its docstring
for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops


def generate_positions(
    df: pd.DataFrame,
    band_lookback: int,
    entry_z: float,
    stop_sigma_mult: float,
    session: str | None = "New York",
) -> pd.Series:
    """Build the symmetric Bollinger-band fade position series from OHLCV bars.

    `band_lookback` is the trailing-window bar count for the mean μ and
    population std σ; `entry_z` is the σ-multiple that defines the upper/lower
    band (short at Close >= μ + entry_z·σ, long at Close <= μ − entry_z·σ);
    `stop_sigma_mult` scales the hard stop's distance from entry off σ. The
    profit target is the trailing mean μ (direction-resolved by session.py),
    and everything is routed through `apply_session_constraint_with_stops` so
    the New York session gates entries and force-flattens at 16:00 ET. Returns
    a {-1, 0, 1} position series.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Trailing (causal) mean and population std of Close. The first
    # `band_lookback - 1` bars are NaN, so bands built from them fail closed.
    band = close.rolling(band_lookback)
    mu = band.mean()
    sigma = band.std(ddof=0)

    # Direction-resolved band levels. entry_z and stop_sigma_mult are floats
    # (σ-multiples, NOT lookbacks); band_lookback is the genuine bar-count
    # lookback that feeds wfo_engine's warm-up buffer.
    upper = mu + float(entry_z) * sigma
    lower = mu - float(entry_z) * sigma

    # Band touch/penetration. NaN comparisons are False, so an unwarmed bar
    # (NaN μ/σ) emits no entry. Inclusive (>= / <=) so an exact touch counts.
    long_signal = close <= lower
    short_signal = close >= upper

    # Hard stop distance = stop_sigma_mult * σ, measured from the entry Close.
    # NaN during warm-up (and 0 in a flat window) fails the walk's stop_ok gate,
    # so no position can open there.
    stop_distance = float(stop_sigma_mult) * sigma

    # Profit target = the trailing mean μ. The walk reads only μ's value at the
    # actual entry bar and resolves direction from it (long: μ > entry at the
    # lower band; short: μ < entry at the upper band), so one series serves
    # both sides.
    target_price = mu

    # session.py alone decides which bars are tradable, walks the stop/target
    # bookkeeping bar-by-bar, and force-flattens on the session's last bar.
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
    # band_lookback is a plain int because it IS a genuine bar-count lookback
    # and must feed wfo_engine's warm-up buffer. entry_z and stop_sigma_mult
    # are floats because they are σ-multiples (not lookbacks) and must NOT feed
    # the buffer; all three are grid-searched ({20,40,60,80}, {2.0,2.5,3.0},
    # {1.0,1.5,2.0,2.5}). `session` is a fixed param. These are also the
    # concrete set the sanity checker runs generate_positions() against.
    "band_lookback": 20,
    "entry_z": 2.0,
    "stop_sigma_mult": 2.0,
    "session": "New York",
}
