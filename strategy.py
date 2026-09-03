"""Two-sided Hurst-persistence drift momentum (NQ 15min, New York session).

This is a deliberate variation of iteration 44 (this family's first run), not a
new family. Iteration 44 gated the entry long-only — `H > threshold AND
drift > 0` — which its own reasoning diagnosed as the reason its in-sample leg
was only 3 trades: `H > 0.5` ("variance grows superlinearly") flags trending in
EITHER direction, but `drift > 0` only ever armed longs, silently throwing away
the entire 2022 persistent bear — the most momentum-rich, most-persistent regime
in the sample. This variation restores the short branch so the persistence gate
governs both directions it was built to detect: trade sign(drift) whenever
H > threshold.

The mechanism
-------------
statistical-mean-reversion-tests.md's persistence boundary read as the
*precondition for momentum*: H > 0.5 is that page's canonical definition of a
trending series (variance grows superlinearly — "H > 0.5: trending"), and
momentum only works when positive autocorrelation actually exists. So this
strategy refuses any directional bet unless a rolling multi-scale Hurst
estimate says continuation should be present — and, unlike iteration 44, it
bets BOTH directions: the sign of the backward drift picks long vs short, and
the H-gate is the conditioning that restricts shorts to genuinely-persistent
downtrends rather than firing them on every dip.

This re-answers iteration 33's "the short leg is the worse half" finding head-on
rather than ignoring it: that was measured on *ungated* symmetric strategies
across ~2,000 trades in four unrelated families, where shorts fired on
ungated/mean-reverting dips. The H-gate is exactly the missing conditioning, so
the short leg is restored and governed by the persistence test itself instead of
being deleted.

Three construction choices are carried over because they are the only findings
this log has replicated across unrelated families:
  1. The dense self-normalizing state + decay-flip exit, no stop, no target.
     That construction produced the repo's only clean leave-top-5-out pass
     (iteration 36); stops and bounded targets have each been measured killing
     continuation gross edge in iterations 27/34/37, so neither is used here.
  2. A statistical-magnitude regime gate as a continuation conditioner.
     Iteration 6's vol-magnitude regime gate measurably improved a
     continuation (OOS Sharpe 0.32 -> 0.86), and H is the direct statistical
     cousin of that conditioning — the state here replaces the vol-magnitude
     test with the persistence test itself.
  3. Grids byte-identical to iteration 44 so the restored short leg is the
     only change; h_threshold is gridded rather than hardcoded because the
     naive overlapping variance-growth estimator carries a known scale bias.

Pre-registered falsification (from the researcher, not this module): if the
H-gate is uninformative the state degenerates to a slow time-series-momentum
state (the two-sided cousin of iteration 23's no-edge long-only TSMOM) and the
family dies at the IS gate; power floor >= 200 OOS trades (dense state,
~350-600 expected).

Entry — a dense state, not a crossing
-------------------------------------
All quantities are strictly backward-looking and fully session-unaware.

    r_s              = log(Close_s / Close_{s-1})
    tau-return_s     = sum of `tau` consecutive r's  (tau in {1,2,4,8,16,32})
    var_tau(s)       = variance of the trailing `hurst_window` tau-returns
    H(s)             = slope of log(var_tau) on log(tau) over the six taus, /2
    hurst_t          = H(t-1)          (strictly backward: bar t sees bars < t)
    drift_t          = Close_{t-1} / Close_{t-1-drift_lookback} - 1

    raw entries_t    = sign(drift_t)   when hurst_t > h_threshold
                     = 0.0             when both finite and hurst_t <= h_threshold
                     = NaN             when unwarmed / non-finite

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
is no entry threshold on price magnitude — the persistence gate plus drift sign
is the self-normalizing condition.

Exit — flip-to-flat and drift-cross reversal, no stop, no target
----------------------------------------------------------------
The raw `entries` series carries exactly three values:
  - sign(drift) (+1.0 long / -1.0 short) while the persistence state is on
    (H > h_threshold),
  - 0.0 while the state is known-OFF (H <= h_threshold) — the flip:
    `session.apply_session_constraint()` forward-fills this 0, so the position
    flattens on the first off bar and stays flat until a fresh on-state
    re-arms,
  - NaN while the state is unknown (unwarmed / non-finite) — forward-fill
    HOLDS an open position through these bars.

While H stays above h_threshold, a drift-sign crossover emits the opposite
sign, so forward-fill reverses the position long<->short — a 2-leg
(close+open) reversal under `metrics.py`'s per-leg cost model. `session.py`
force-flattens on the session's last bar, so no position carries past the close;
a mid-session off->on re-arm is a fresh, intended second trip. This is
deliberately `apply_session_constraint()`, NOT the `with_stops` variant: no
stop, no target, upside uncapped, duration bounded by the session.

This differs from the retired pullback family's entry shape in one load-bearing
way: there, raw entries were NaN (hold) on non-qualifying bars, so an open
position was HELD to the session flatten regardless of signal; here raw entries
are 0.0 on known-off bars, so the position FLIPS to flat the moment the
persistence condition dies, and re-arms only on a fresh on-state.

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

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to `session.py`;
see its docstring for that contract.
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
    # treated as a deliberate "off" (which would force a flatten). `above`
    # marks bars where the persistence gate is satisfied in EITHER direction;
    # the sign of drift picks long vs short.
    hurst_v = hurst.to_numpy()
    drift_v = drift.to_numpy()
    known = np.isfinite(hurst_v) & np.isfinite(drift_v)
    above = hurst_v > float(h_threshold)

    # Raw, session-unaware entries: sign(drift) (+1.0 long / -1.0 short) while
    # the state is on, 0.0 while it is known-off (H <= h_threshold, the
    # flip-to-flat exit), NaN while unknown (hold). apply_session_constraint()
    # forward-fills this, so a position opens on the first in-session on-bar,
    # REVERSES long<->short when drift crosses zero while H stays above the
    # threshold (a 2-leg close+open flip), flattens on the first off-bar (0.0
    # ffills through the off stretch), holds through NaN bars, and is
    # force-flattened by session.py on the session's last bar.
    entries_v = np.full(len(df), np.nan)
    entries_v[known & above] = np.sign(drift_v[known & above])
    entries_v[known & ~above] = 0.0
    entries = pd.Series(entries_v, index=df.index)

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
