"""Long-only Hurst-persistence drift state (NQ 15min, New York session).

The previous family (long-only swing-horizon pullback buy) is retired as
structural, not varied: its in-sample leg came back IS Sharpe 0.02 / PF 1.01
on 65 IS trades, its own reasoning calling that leg "indistinguishable from no
edge". This is a clean pivot to a genuinely new family — no prior iteration in
this log computed Hurst at all (variance ratio appeared only as a single-lag
gate in iteration 9 and a polarity key in iteration 21, never as the
multi-tau variance-growth estimator used here).

The mechanism
-------------
statistical-mean-reversion-tests.md's persistence boundary read as the
*precondition for momentum*: H > 0.5 is that page's canonical definition of a
trending series (variance grows superlinearly — "H > 0.5: trending"), and
momentum only works when positive autocorrelation actually exists. So this
strategy refuses any directional bet unless a rolling multi-scale Hurst
estimate says continuation should be present — and it only ever bets long,
since iteration 33 measured the short leg as the worse half across ~2,000
trades in four unrelated families.

Three construction choices are carried over because they are the only findings
this log has replicated across unrelated families:
  1. Long-only. A ~2,000-trade direction split across four families put the
     short leg as the worse half every time. Under `metrics.py`'s per-leg cost
     model a non-qualifying day now pays zero legs rather than the two a short
     entry would cost.
  2. The dense self-normalizing state + decay-flip exit, no stop, no target.
     That construction produced the repo's only clean leave-top-5-out pass
     (iteration 36); stops and bounded targets have each been measured killing
     continuation gross edge in iterations 27/34/37, so neither is used here.
  3. A statistical-magnitude regime gate as a continuation conditioner.
     Iteration 6's vol-magnitude regime gate measurably improved a
     continuation (OOS Sharpe 0.32 -> 0.86), and H is the direct statistical
     cousin of that conditioning — the state here replaces the vol-magnitude
     test with the persistence test itself.

Pre-registered falsification (from the researcher, not this module): if the
H-gate is uninformative the state degenerates to a slow time-series-momentum
state (iteration 23, no-edge IS PF 0.946) and the family dies at the IS gate;
power floor >= 200 OOS trades (dense state, ~350-600 expected); h_threshold is
gridded rather than hardcoded because the naive overlapping variance-growth
estimator carries a known scale bias.

Entry — a dense state, not a crossing
-------------------------------------
All quantities are strictly backward-looking and fully session-unaware.

    r_s              = log(Close_s / Close_{s-1})
    tau-return_s     = sum of `tau` consecutive r's  (tau in {1,2,4,8,16,32})
    var_tau(s)       = variance of the trailing `hurst_window` tau-returns
    H(s)             = slope of log(var_tau) on log(tau) over the six taus, /2
    hurst_t          = H(t-1)          (strictly backward: bar t sees bars < t)
    drift_t          = Close_{t-1} / Close_{t-1-drift_lookback} - 1

    state_t          = (hurst_t > h_threshold) AND (drift_t > 0)

Hurst is estimated by the page's variance-growth method: for a series whose
tau-period variance grows as tau^(2H), log(Var) is linear in log(tau) with
slope 2H, so H = slope/2. A random walk has slope 1 -> H = 0.5; trending
(superlinear variance growth) -> H > 0.5; mean-reverting -> H < 0.5. The
tau-returns are *overlapping* (the standard variance-ratio construction), which
is what makes the estimator's finite-sample scale slightly biased — the exact
reason h_threshold is a searched parameter rather than a hardcoded 0.5.

Strict min_periods: `rolling(hurst_window).var()` is NaN until the full window
is present, so an unwarmed bar yields a NaN H. NaN (and inf) fail closed on
both sides of the state conjunction: the comparisons evaluate False (never a
phantom entry) AND the bar is marked "unknown" (never a forced flatten). There
is no entry threshold on price magnitude — the state itself is the
self-normalizing condition.

Exit — flip-to-flat, no stop, no target
---------------------------------------
The raw `entries` series carries exactly three values:
  - 1.0 while the state is ON (long),
  - 0.0 while the state is known-OFF (H <= h_threshold or drift <= 0) — the
    flip: `session.apply_session_constraint()` forward-fills this 0, so the
    position flattens on the first off bar and stays flat until a fresh
    on-state re-arms,
  - NaN while the state is unknown (unwarmed / non-finite) — forward-fill
    HOLDS an open position through these bars.
`session.py` force-flattens on the session's last bar, so there is at most one
round trip per on-session (a mid-session off->on re-arm is a fresh, intended
second trip). This is deliberately `apply_session_constraint()`, NOT the
`with_stops` variant: no stop, no target, upside uncapped, duration bounded by
the session.

This differs from the retired pullback family's entry shape in one load-bearing
way: there, raw entries were NaN (hold) on non-qualifying bars, so an open
position was HELD to the session flatten regardless of signal; here raw entries
are 0.0 on known-off bars, so the position FLIPS to flat the moment the
persistence/drift condition dies, and re-arms only on a fresh on-state.

Long-only means -1.0 can never be emitted, so the forward fill can only ever
produce {0.0, 1.0} and a non-qualifying day costs nothing at all.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
---------------------------------------------------------------------------
  - `hurst_window` and `drift_lookback` are genuine bar counts and are `int`
    in `build_grid()` **on purpose**, so `_max_lookback_bars()` picks them up
    and sizes the pre-test-window warm-up buffer from them. `h_threshold` is a
    Hurst gate, not a lookback, and is cast `float` so it cannot inflate that
    buffer.
  - At the intended grid the binding value is `hurst_window`: the buffer is
    `max((hurst_window + 5) * 3, bars_per_day + 5)`, i.e. 3,471 bars at
    hurst_window = 1152.
  - The fold-skip guard in `run_walk_forward()` is
    `len(train_df) < max_lookback + 10` = 1,162 at the top of the grid. A
    12-week train window at 15min is ~7,700 bars, so no fold is skipped — and
    even at 1h (~2,016 bars) folds still run, only with a heavier warm-up
    buffer. THIS STRATEGY IS INTENDED AT `--timeframe 15min`. At 4h/1d every
    fold is skipped.
  - Hand-check for the part `_max_lookback_bars()` cannot see: the regression's
    slowest column (tau = 32) is non-NaN only after hurst_window + 31 bars,
    and the `.shift(1)` adds one more, so `hurst` needs hurst_window + 32 bars
    of history before it is non-NaN; `drift` needs drift_lookback + 1 bars.
    The smallest buffer this grid can produce is (384 + 5) * 3 = 1,167 bars,
    which clears 384 + 32 = 416 (and drift_lookback + 1 = 385) at every grid
    corner. If it ever did not, the failure mode is a NaN H/drift -> state
    unknown -> no entry on the first bars of a test window: a few missed
    trades, never lookahead.

This module decides only *when* the strategy wants to be long. All day-trade
gating and the end-of-session flatten are delegated to `session.py`; see its
docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Tau lags (bars) for the variance-growth Hurst regression — the fixed
# multi-scale ladder from statistical-mean-reversion-tests.md. A module
# constant on purpose: the estimator's scale structure keeps zero searched
# degrees of freedom, so `h_threshold` alone controls the persistence gate.
HURST_TAUS = (1, 2, 4, 8, 16, 32)


def hurst_exponent(close: pd.Series, window: int) -> pd.Series:
    """Multi-scale variance-growth Hurst estimate, strictly backward-looking.

    For each bar s, regress log(Var(tau-bar log returns)) over the trailing
    `window` observations on log(tau) for tau in HURST_TAUS; H = slope / 2.
    The tau-returns are overlapping (tau consecutive one-bar log returns,
    equivalently log(Close_s / Close_{s-tau})). The returned series is shifted
    one bar, so the value attached to bar t uses only data strictly before t.

    NaN discipline: the rolling variance uses the strict full window
    (min_periods = window), so the slowest column (tau = 32) is NaN for the
    first window + 31 bars (the one-bar shift of log_ret adds one), and a NaN
    in any of the six regression points makes the row-wise slope NaN — an
    unwarmed bar therefore fails closed downstream instead of producing a
    spuriously finite H.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(close / close.shift(1))

    # Fixed design: slope = dot(log(var_tau), (log(tau) - mean) / Sxx), since
    # the tau ladder is the same on every bar. Sxx is ddof-free and constant.
    x = np.log(np.asarray(HURST_TAUS, dtype=float))
    x_mean = x.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    weights = (x - x_mean) / sxx

    log_vars = {}
    for tau in HURST_TAUS:
        tau_ret = log_ret.rolling(tau).sum()
        var = tau_ret.rolling(window, min_periods=window).var()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_vars[tau] = np.log(var)

    # Row-wise slope of log(Var) on log(tau) = 2H; a NaN in any column
    # propagates through the dot product, so H is NaN before full warm-up.
    slope = pd.DataFrame(log_vars, index=close.index).dot(weights)
    return (slope / 2.0).shift(1)


def generate_positions(
    df: pd.DataFrame,
    hurst_window: int,
    drift_lookback: int,
    h_threshold: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)

    # Multi-scale variance-growth Hurst of the trailing `hurst_window` bars,
    # shifted so bar t sees only data strictly before t. NaN before
    # hurst_window + 32 bars -> fails closed downstream.
    hurst = hurst_exponent(close, int(hurst_window))

    # Backward drift over the `drift_lookback` bars ending at t-1:
    # Close_{t-1} / Close_{t-1-drift_lookback} - 1. NaN for the first
    # drift_lookback + 1 bars.
    drift = close.shift(1) / close.shift(1 + int(drift_lookback)) - 1.0

    # Dense state with fail-closed NaN semantics. `known` marks bars where
    # both sides are finite, i.e. where the state is a genuine boolean; on an
    # unwarmed (NaN) or non-finite bar both comparisons evaluate False, so it
    # can never arm a phantom entry, and `known` is False so it is never
    # treated as a deliberate "off" (which would force a flatten).
    known = np.isfinite(hurst) & np.isfinite(drift)
    state_on = (hurst > float(h_threshold)) & (drift > 0.0)

    # Raw, session-unaware entries: 1.0 while the state is on, 0.0 while it is
    # known-off (the flip-to-flat exit), NaN while unknown (hold). There is no
    # -1.0 branch: this family is long-only, so a non-qualifying day pays zero
    # cost legs. apply_session_constraint() forward-fills this, so a long opens
    # on the first in-session on-bar, flattens on the first off-bar (0.0
    # ffills through the off stretch), holds through NaN bars, and is
    # force-flattened by session.py on the session's last bar.
    entries = pd.Series(np.nan, index=df.index)
    entries[state_on & known] = 1.0
    entries[known & ~state_on] = 0.0

    # session.py alone decides which bars are tradable and force-flattens on
    # the session's last bar.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    # All three are real points the intended grid searches (hurst_window
    # 384/768/1152, drift_lookback 96/192/384, h_threshold 0.5/0.55/0.6), so
    # the post-edit sanity check exercises a genuine combo. hurst_window and
    # drift_lookback are ints because they ARE bar counts and are meant to
    # size wfo_engine's warm-up buffer; h_threshold is a float Hurst gate and
    # must not.
    "hurst_window": 384,
    "drift_lookback": 96,
    "h_threshold": 0.5,
    "session": "New York",
}
