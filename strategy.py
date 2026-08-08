"""ATR-normalized momentum thrust, with ATR-scaled stop and target.

Rules (see momentum-formulas.md eq. 269's risk-adjusted momentum
R_risk_adj = R_mean / sigma, applied time-series style to one instrument):

  - thrust = (Close - Close.shift(n)) / (ATR(n) * sqrt(n)), where
    n = `mom_lookback`. The n-bar return is divided by how big this market
    normally moves over one bar, and the sqrt(n) rescales that to a
    sigma-comparable z-score so `thrust_mult` means the same selectivity at
    every lookback — the grid can't win a fold on a pure scaling artifact.
  - Entry is a CROSSING, not a level: LONG when thrust crosses up through
    +thrust_mult, SHORT when it crosses down through -thrust_mult. This
    matters because `session.apply_session_constraint_with_stops()` skips
    same-bar re-entry after a stop (`continue`) but re-evaluates on the very
    next bar — a persistent level condition (`thrust >= mult`) would re-open
    the same failing move bar after bar, while a crossing gives exactly one
    entry per excursion. Zero-param structural guard, same spirit as the
    prior strategy's hardcoded trend guard.
  - No flip on an opposite signal: the walk only opens when flat, so an
    opposing thrust while in a trade is a no-op.
  - Exits are path-dependent and owned entirely by session.py's walk:
    a stop `stop_atr_mult * ATR(n)` from the entry Close, a target
    `target_atr_mult * ATR(n)` beyond the entry Close (direction-resolved
    here, since the walk validates that a long's target sits above and a
    short's below that bar's Close), and the forced flatten on the session's
    last bar.

Design intent: few, selective entries with a move-to-cost ratio well above
1 — on NQ 1h, ATR is roughly 0.35% of price, so a 2.5-ATR target is ~0.9%
gross against `metrics.py`'s ~0.10% round trip. This is deliberately the
opposite trade profile from a high-turnover fade, which dies of costs.

ATR is a simple rolling mean of the standard true range
TR = max(H-L, |H-C_prev|, |L-C_prev|) over the same n as the momentum
lookback (matched horizons — deliberately not Wilder smoothing, whose
effective lookback is ~2n-1 and would break both the matched horizon and
the sqrt(n) z-score scaling). It is deliberately NOT shifted by one bar:
the numerator already uses the current Close and the entry fills at that
same Close, so including the current bar's true range is not look-ahead.
Don't "fix" it to `.shift(1)`.

Like this codebase's prior strategies, ATR and thrust are computed on the
full continuous-Globex df rather than session-scoped, so an early-session
signal's lookback can reach back into the prior evening's bars. Intentional.

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
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple rolling mean of the standard true range over `period` bars.

    Includes the current bar (see module docstring: not look-ahead, since
    entries fill at that bar's Close). Non-positive/degenerate values are
    dropped to NaN so they can never produce an infinite thrust.
    """
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr.where(atr > 0)


def compute_thrust(df: pd.DataFrame, mom_lookback: int) -> tuple[pd.Series, pd.Series]:
    """(thrust z-score, ATR) for the given lookback."""
    atr = compute_atr(df, mom_lookback)
    momentum = df["Close"] - df["Close"].shift(mom_lookback)
    thrust = momentum / (atr * np.sqrt(mom_lookback))
    return thrust, atr


def generate_positions(
    df: pd.DataFrame,
    mom_lookback: int,
    thrust_mult: float,
    stop_atr_mult: float,
    target_atr_mult: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    thrust, atr = compute_thrust(df, mom_lookback)
    prior = thrust.shift(1)

    # Crossing, not level — one entry per excursion (see module docstring).
    long_signal = ((thrust >= thrust_mult) & (prior < thrust_mult)).fillna(False)
    short_signal = ((thrust <= -thrust_mult) & (prior > -thrust_mult)).fillna(False)

    stop_distance = stop_atr_mult * atr

    # Absolute, direction-resolved target: strictly beyond the entry Close on
    # the correct side, so the walk's target-side validity check passes for
    # both longs and shorts (a single unconditional `close + k*atr` series
    # would silently reject every short).
    long_target = close + target_atr_mult * atr
    short_target = close - target_atr_mult * atr
    target_price = long_target.where(long_signal, short_target.where(short_signal))

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
    "mom_lookback": 8,
    "thrust_mult": 1.75,
    "stop_atr_mult": 1.5,
    "target_atr_mult": 2.5,
    "session": "New York",
}
