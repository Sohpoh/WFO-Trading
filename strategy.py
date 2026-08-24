"""Swing-horizon time-series momentum, executed intraday.

`momentum-strategies.md`'s canonical construction read literally: rank by the
cumulative return over a *formation period*, skipping the most recent period
(Rcum = P(S)/P(S+T) - 1), and take the sign of that cumulative return as the
directional state. The one thing this repo has never measured is the
*horizon*: every directional lookback across 22 iterations tops out at 384
bars (~4 days), and iterations 7/20 measured drift over 24-192 bars (6h-2
days). The formation windows here are 480-1440 bars on 15min NQ, i.e. roughly
1-3 weeks — the horizon where the time-series momentum evidence is actually
strong, rather than the intraday horizon where it is weakest.

Second design driver is turnover economics. `metrics.py` charges ~10.2bps
round-trip off Close and is out of scope here, but it is the binding
constraint: the log's two gate-v2 runs both decomposed to roughly -11bps
net/trade on ~0bps gross across ~750-950 trades. So this is deliberately a
*state* signal, not an event trigger — the multi-week sign changes rarely, so
the strategy takes at most one round trip per session (open on the session's
first live bar, held to `session.py`'s forced flatten), rather than re-arming
after every stop.

Rules (all computed on the full continuous frame with no session awareness of
their own; every leg is a strict backward `.shift()` so unwarmed bars are NaN
and the signal fails *closed* — NaN compares False on both `> 0` and `< 0`, so
no entry can fire on an unwarmed bar):

  - anchor = Close.shift(skip_period)
    The formation window deliberately *ends* `skip_period` bars in the past.
    `skip_period = 0` is retained in the grid as a control: iteration 20
    already tested the skip mechanism at intraday horizon (hardcoded L/8) and
    found no edge there, so the skip must earn its place at this horizon.

  - formation_ret = anchor / Close.shift(skip_period + trend_lookback) - 1
    The Rcum of the formation window itself.

  - confirm_ret  = anchor / Close.shift(skip_period + max(1, trend_lookback // 4)) - 1
    A quarter-horizon confirmation over the *same* anchor. This is
    momentum-strategies.md pitfall 1 read literally — "when a trend breaks
    suddenly, momentum strategies take large losses; this is the main driver
    of drawdowns" — and it costs zero degrees of freedom: the quarter horizon
    is derived internally from `trend_lookback` and is deliberately NOT a
    grid param or a DEFAULT_PARAMS key.

  - Raw entries:  1.0 where formation_ret > 0 AND confirm_ret > 0
                 -1.0 where formation_ret < 0 AND confirm_ret < 0
                  NaN otherwise (the two horizons disagree, or warm-up).
    NaN means "do not open", not "exit": `apply_session_constraint()`'s ffill
    holds any position already open. Because that function zeroes the last bar
    of every session run and ffills that 0 through the off-session gap, no
    position ever survives the session close and every session starts flat.

  - No lookahead: every input is a backward shift of Close, every value is
    knowable at bar t's Close, and the entry is priced at that same Close.

Exit is a plain flip — `session.apply_session_constraint(entries, session)`.
There is no stop and no target, and deliberately no call to
`apply_session_constraint_with_stops()`: that path requires a valid stop AND a
valid target on every entry bar and would add nothing the forced session
flatten isn't already doing. Within a session the position is held, flipped
only if the multi-week state reverses sign intra-session (rare at this
horizon), and force-flattened by `session.py` on the session's last bar.

Warm-up and param types:

  - `trend_lookback` and `skip_period` are both genuine bar counts, so both
    are passed as plain `int` from `build_grid()` and correctly feed
    `wfo_engine._max_lookback_bars()`'s pre-test-window buffer. Nothing else
    is a bar count, so nothing else is an int.
  - The deepest reach back is `skip_period + trend_lookback` = 96 + 1440 =
    1536 bars with the intended grid, against a buffer of
    max((1440 + 5) * 3, day_bars + 5) = 4335 bars. That clears comfortably —
    the buffer is sized off the largest single int, and here the *sum* of the
    two ints is what matters, which is why the 3x multiplier is load-bearing.

Cost note: `metrics.py`'s ~0.102% round-trip is unchanged and out of scope.
At one round trip per session an entry has a full New York session to cover
that toll, which is the whole economic point of the state-based shape.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`session.apply_session_constraint()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd


from session import apply_session_constraint


def cumulative_return(close: pd.Series, skip_period: int, lookback: int) -> pd.Series:
    """Rcum over the window ending `skip_period` bars back, `lookback` bars long.

    `anchor = Close.shift(skip_period)` is the end of the formation window;
    `Close.shift(skip_period + lookback)` is its start. Both legs are strict
    backward shifts, so the result at bar t uses only bars <= t and the
    leading `skip_period + lookback` bars are NaN (fails closed).
    """
    anchor = close.shift(skip_period)
    base = close.shift(skip_period + lookback)
    return anchor / base - 1.0


def generate_positions(
    df: pd.DataFrame,
    trend_lookback: int,
    skip_period: int,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]

    # Quarter-horizon confirmation window, derived from trend_lookback rather
    # than searched — zero extra degrees of freedom. max(1, ...) keeps it a
    # strictly positive window even if a tiny trend_lookback is ever tried.
    confirm_lookback = max(1, int(trend_lookback) // 4)

    formation_ret = cumulative_return(close, skip_period, trend_lookback)
    confirm_ret = cumulative_return(close, skip_period, confirm_lookback)

    # Both horizons must agree in sign. NaN (warm-up) compares False on both
    # sides, so an unwarmed bar produces NaN entries = "do not open".
    long_signal = (formation_ret > 0) & (confirm_ret > 0)
    short_signal = (formation_ret < 0) & (confirm_ret < 0)

    entries = pd.Series(np.nan, index=df.index)
    entries[long_signal] = 1.0
    entries[short_signal] = -1.0

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "trend_lookback": 480,
    "skip_period": 24,
    "session": "New York",
}
