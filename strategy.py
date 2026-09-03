"""Long-only trend-gated Hurst drift momentum (NQ 15min, New York session).

A deliberate variation of iteration 45 (this family's two-sided run), not a
new family. Iteration 45's own trade data contained the two facts this
variation acts on: the short leg was dead even under the persistence gate
(OOS mean ~0bp at 42.6% win rate, and negative ex-top-5 in BOTH in-sample
and out-of-sample — a pure tail with no body), while the long leg's OOS was
regime-directional (+5bp in the 2024 bull, -18bp in the 2022 bear). This
variation (a) retires the short branch entirely — drift < 0 emits 0.0, never
-1.0 — and (b) adds a zero-param higher-timeframe uptrend gate
(Close > SMA(960)) so longs fire only in bull regimes, turning the 2022-bear
long bleed into flat no-trade periods.

The mechanism
-------------
Same persistence-first construction as 45: H > 0.5 is the canonical
"trending" boundary (variance grows superlinearly), and momentum only works
when positive autocorrelation actually exists. A long is armed only while all
three gates hold simultaneously:
  1. multi-scale variance-growth Hurst estimate H_t > h_threshold,
  2. backward drift Close_{t-1}/Close_{t-1-drift_lookback} - 1 > 0,
  3. Close_t > SMA(Close, TREND_MA) with TREND_MA = 960 bars hardcoded
     (zero-param, NOT grid-searched).
There is no short branch: a negative (or zero) drift is an off-bar, not a
short. The trend gate is the response to momentum-strategies.md pitfall #1
(trend reversal drives momentum drawdowns): it keeps the strategy flat
through bear regimes instead of betting long into them, and it re-answers
iteration 33's "the short leg is the worse half" finding by deleting the
short rather than trying to rescue it with a gate (45 already showed the gate
didn't rescue it).

Entry — a dense long-only state
-------------------------------
All quantities are session-unaware. The Hurst estimate and the backward
drift are strictly backward-looking (bar t sees bars < t); the trend gate
uses the current bar's Close_t against its own 960-bar SMA — the standard
close-vs-SMA convention. This is not lookahead: `metrics.bar_returns_with_costs`
prices a position off `position.shift(1)`, so a position set at bar t (and
entered at Close_t) earns the Close_t -> Close_{t+1} return, which the signal
never sees.

    r_s              = log(Close_s / Close_{s-1})
    tau-return_s     = sum of `tau` consecutive r's  (tau in {1,2,4,8,16,32})
    var_tau(s)       = variance of the trailing `hurst_window` tau-returns
    H(s)             = slope of log(var_tau) on log(tau), /2
    hurst_t          = H(t-1)          (strictly backward: bar t sees bars < t)
    drift_t          = Close_{t-1} / Close_{t-1-drift_lookback} - 1
    trend_t          = Close_t - SMA(Close, TREND_MA)

    raw entries_t    = +1.0  when hurst_t > h_threshold AND drift_t > 0
                              AND trend_t > 0
                     =  0.0  when all three are finite and ANY gate is off
                     =  NaN  when any gate is unwarmed / non-finite (fail closed)

NaN discipline: the Hurst rolling variance, the drift shifts, and the SMA all
produce NaN until their own warm-up is complete, so an unwarmed bar fails
closed (never a phantom entry) and is marked "unknown" (never a forced
flatten). `known` is the conjunction finite(hurst) & finite(drift) &
finite(trend); only bars where all three are finite carry a genuine boolean
state.

Exit — flip-to-flat, no stop, no target
---------------------------------------
Plain `apply_session_constraint` (NOT the `with_stops` variant). Raw entries
carry +1.0 while the state is on, 0.0 while it is known-off (any of the three
gates false), NaN while unknown. `apply_session_constraint` forward-fills, so
the position flattens on the first off-bar and holds through NaN bars. Because
there is no -1 branch, a gate that turns off can only close/re-open a long —
never reverse long<->short. session.py force-flattens on the session's last
bar, so no position carries past the close. No stop, no target, upside
uncapped, duration bounded by the session.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
---------------------------------------------------------------------------
  - `hurst_window` and `drift_lookback` are genuine bar counts and are `int`
    in `build_grid()` **on purpose**, so `_max_lookback_bars()` picks them up
    and sizes the pre-test-window warm-up buffer from them. `h_threshold` is a
    Hurst gate, not a lookback, and is cast `float` so it cannot inflate that
    buffer.
  - `TREND_MA = 960` is a MODULE CONSTANT (zero-param, hardcoded), exactly
    like `HURST_TAUS`. It never appears in `build_grid()` or `cli.py`'s flags,
    so `_max_lookback_bars()` cannot see it — but that is safe: the intended
    grid's binding int is `hurst_window = 1152`, and the buffer
    `max((1152 + 5) * 3, bars_per_day + 5)` = 3,471 bars clears the SMA's
    960-bar warm-up (and Hurst's 1152 + 32 = 1,184) at every grid corner.
  - At the intended grid the fold-skip guard in `run_walk_forward()` is
    `len(train_df) < max_lookback + 10` = 1,162 bars. A 12-week train window
    at 15min is ~7,700 bars, so no fold is skipped — THIS STRATEGY IS INTENDED
    AT `--timeframe 15min`. At 1h a 12-week train window is ~2,016 bars
    (above the guard, so folds still run, but with a heavy 3,471-bar buffer);
    at 4h/1d every fold is skipped.
  - Hand-check for the legs `_max_lookback_bars()` cannot see (the tau
    ladder, the shift, and the hardcoded trend SMA): the regression's slowest
    column (tau = 32) is non-NaN only after hurst_window + 31 bars and the
    `.shift(1)` adds one, so `hurst` needs hurst_window + 32 bars of history
    before it is non-NaN; `drift` needs drift_lookback + 1 bars; the trend
    gate's SMA needs TREND_MA = 960 bars. The intended buffer (3,471 bars)
    clears all three at every grid corner (max needs: 1,184 / 385 / 960).
    Even the smallest buffer this grid could produce — (384 + 5) * 3 = 1,167
    bars — clears every one of those. If it ever did not, the failure mode is
    a NaN gate -> state unknown -> no entry on the first bars of a test
    window: missed trades, never lookahead.

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

# Higher-timeframe uptrend gate period (bars). The researcher's `trend_ma`,
# deliberately a module constant rather than a param: zero-param, hardcoded,
# NOT grid-searched. A long fires only while Close is above its 960-bar SMA,
# so the strategy sits flat through bear regimes instead of betting long into
# them. Kept below the grid's binding `hurst_window = 1152`, so the warm-up
# buffer already sized from the Hurst window clears this SMA's 960-bar
# warm-up at every grid corner (see the module docstring's hand-check).
TREND_MA = 960


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

    # Zero-param higher-timeframe uptrend gate: Close_t minus its TREND_MA-bar
    # SMA. Positive means Close above SMA. NaN until the SMA's strict
    # min_periods warm-up completes (subtraction propagates NaN, unlike a
    # comparison which would collapse it to False).
    sma = close.rolling(int(TREND_MA), min_periods=int(TREND_MA)).mean()
    trend = close - sma

    hurst_v = hurst.to_numpy()
    drift_v = drift.to_numpy()
    trend_v = trend.to_numpy()

    # Dense long-only state with fail-closed NaN semantics. `known` marks bars
    # where all three gates are finite, i.e. where the state is a genuine
    # boolean; on an unwarmed (NaN) or non-finite bar the comparisons evaluate
    # False, so it can never arm a phantom entry, and `known` is False so it
    # is never treated as a deliberate "off" (which would force a flatten).
    # `long_ok` marks bars where all three gates hold. There is NO short
    # branch: drift <= 0 is simply an off-bar (0.0), never -1.0.
    known = np.isfinite(hurst_v) & np.isfinite(drift_v) & np.isfinite(trend_v)
    long_ok = (hurst_v > float(h_threshold)) & (drift_v > 0.0) & (trend_v > 0.0)

    # Raw, session-unaware entries: +1.0 while the state is on (all three
    # gates), 0.0 while it is known-off (any gate false — the flip-to-flat
    # exit), NaN while unknown (hold). apply_session_constraint() forward-fills
    # this, so a position opens on the first in-session on-bar, flattens on
    # the first off-bar (0.0 ffills through the off stretch), holds through
    # NaN bars, and is force-flattened by session.py on the session's last
    # bar. Because there is no -1 branch, a gate turning off can only close a
    # long, never reverse long<->short.
    entries_v = np.full(len(df), np.nan)
    entries_v[known & long_ok] = 1.0
    entries_v[known & ~long_ok] = 0.0
    entries = pd.Series(entries_v, index=df.index)

    # session.py alone decides which bars are tradable and force-flattens on
    # the session's last bar.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    # hurst_window and drift_lookback are ints because they ARE bar counts and
    # are meant to size wfo_engine's warm-up buffer; h_threshold is a float
    # Hurst gate and must not. All three are real points the intended grid
    # searches (384/768/1152, 96/192/384, 0.5/0.55/0.6), so the post-edit
    # sanity check exercises a genuine combo. TREND_MA (the 960-bar uptrend
    # gate) is a module constant, NOT a param, so it deliberately has no entry
    # here; `session` is threaded as a fixed param.
    "hurst_window": 384,
    "drift_lookback": 96,
    "h_threshold": 0.5,
    "session": "New York",
}
