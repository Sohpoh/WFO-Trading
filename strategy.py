"""Long-only dual-horizon moving-average trend state (NQ 15min, New York).

The previous family (long-only percentile-rank momentum) is retired, not
varied: its mandatory holdout run came back a no-edge verdict on data it was
never fit to, with the in-sample leg at CAGR -3.48% / PF 0.877. What carries
forward from it is the *construction*, not the primitive.

What is kept
------------
Long-only. It is the only shape in this log that has ever produced a clean
leave-top-5-out result with upside uncapped, and splitting ~2,000 OOS trades
by direction across four unrelated families said the same thing every time:
the short leg was the worse half. So momentum-strategies.md's Long-Only
portfolio construction ("Buy only the winners... w_i >= 0") stays, and under
`metrics.py`'s per-leg cost model a down-state pays zero legs rather than the
two a reversal would cost.

What is discarded
-----------------
The percentile-rank formation-return entry that failed the holdout. Every
part of the signal below is new.

The signal — a two-horizon moving-average relationship
------------------------------------------------------
Across 40 prior iterations, direction has come from channel breaks, z-scores,
gap sign, drift t-stats, close location, run length, VWAP deviation, CMF, ATR
displacement, regression residuals and formation-return rank. A relationship
between *two* moving averages has never once supplied direction (two earlier
iterations used a single MA as a confirmation gate against price — never two
MAs against each other). regime-changes.md names exactly this primitive
("Strategy A: Simple moving average momentum (MA10 > MA50 = BUY)") and defines
a trending market as one where "price moves consistently in one direction...
momentum strategies work". The same page presents it as regime-fragile; that
is stated here rather than hidden.

Computed on the full continuous frame, session-unaware, strictly backward
looking (both rolling windows end on the current completed bar and use only
prior closes):

    SMA_fast = Close.rolling(fast_ma, min_periods=fast_ma).mean()
    SMA_slow = Close.rolling(slow_ma, min_periods=slow_ma).mean()

    trend state at bar t  ==  SMA_fast_t > SMA_slow_t     (strict)

`min_periods` equal to the window is what NaNs out every unwarmed bar; NaN on
either side of `>` compares False, so an unwarmed bar is flat and there is no
separate validity mask to keep in sync.

Entry / exit
------------
  - `entries = 1.0` on every bar where the state is True.
  - `entries = 0.0` on every bar where it is False — a plain flip to flat.
    -1.0 is never emitted; there is no short leg.
  - No stop, no profit target, no path dependence, so the non-stops delegate
    `apply_session_constraint(entries, session)` is used unmodified. It gates
    entries to the session, forward-fills the instruction series, and
    force-flattens on the session's last bar.

The series is **dense** (a real 1.0 or 0.0 on every bar, no NaN hold band),
which is a deliberate departure from the sparse/hysteresis shape of the last
several iterations: the in-session position is then exactly the trend state,
with no possibility of holding a stale long through a NaN. The comparison is
cast with `.astype(float)` before it is handed to the delegate — a bool Series
would be upcast to `object` by the delegate's `.where(in_session)` and break
the engine's `position.diff()` scoring.

A still-trending winner runs uncapped to the forced session flatten. That is
intentional: the two iterations that passed leave-top-5-out both had uncapped
upside, while iteration 27 showed a hard R-cap inverting a continuation
family's per-trade economics from +0.4 bps to -11.6 bps.

Cost discipline, designed into the grid rather than bolted on as a filter
------------------------------------------------------------------------
Iteration 38 measured this session/instrument's binding constraint as cost
drag (gross 8.6 bps/trade against the fixed 10.2 bps toll). So `fast_ma`
floors at 192 bars — far longer than the 26-bar NY session at 15min — which
means the state cannot flip *inside* a session. Every fast/slow pair in the
intended grid is a multi-session state with 2x-9x horizon separation. NQ is
chosen over ES because ES state-signal shapes are 0-for-2 with no-edge here,
and NQ's larger relative excursion makes the fixed-percentage toll a smaller
fraction of a session's move.

No vol gate, no trend filter and no stop is added, so this first run of the
family stays clean and attributable.

`metrics.py`'s per-leg toll (0.001% fee + 0.05% slippage) is unchanged and out
of scope.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
---------------------------------------------------------------------------
  - `fast_ma` and `slow_ma` are both plain `int` **on purpose** — both are
    genuine bar counts, and both are exactly what should size the pre-test
    warm-up buffer. There are no non-lookback integer params in this
    strategy, so nothing here can inflate that buffer incorrectly.
  - `slow_ma` is the larger, so it alone sizes the buffer: at the grid max of
    1728, `buffer_bars = max((1728 + 5) * 3, day_bars + 5)` = 5199 bars,
    which more than covers the true requirement of `slow_ma` = 1728 bars.
  - COVERAGE HAZARD: raising `slow_ma` also raises `run_walk_forward()`'s
    fold-skip guard to `max_lookback + 10` = 1738 bars of *train* window. A
    12-week train window at 15min is ~5,500 bars, so the guard is comfortable
    at the intended timeframe — but at 1h (~1,380 bars) it is NOT met and
    every fold would be silently skipped. See `wfo_engine.build_grid()`.

This module decides only *when* the strategy wants to be long. All day-trade
gating and the end-of-session flatten are delegated to `session.py`; see its
docstring for that contract.
"""
import pandas as pd

from session import apply_session_constraint


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average over the trailing `window` completed bars.

    Strictly warm (`min_periods == window`), so the first `window - 1` bars
    are NaN and fail closed downstream.
    """
    w = int(window)
    return close.rolling(w, min_periods=w).mean()


def generate_positions(
    df: pd.DataFrame,
    fast_ma: int,
    slow_ma: int,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)

    sma_fast = sma(close, fast_ma)
    sma_slow = sma(close, slow_ma)

    # Trend state: fast above slow (strict). NaN on either side — an unwarmed
    # fast or slow window — compares False, so such a bar is flat with no
    # separate validity mask. `.astype(float)` is load-bearing: the delegate's
    # `.where(in_session)` would upcast a bool Series to `object`, which then
    # breaks `wfo_engine._score_params`'s `position.diff()`.
    #
    # Dense by design — 1.0 (long) or 0.0 (flat) on every bar, never NaN — so
    # the forward fill inside the delegate has nothing to hold and the
    # in-session position is exactly the trend state. There is no short leg;
    # -1.0 is never emitted, and a downtrend simply pays no cost legs.
    entries = (sma_fast > sma_slow).astype(float)

    # session.py alone decides which bars are tradable, forward-fills the
    # instruction series into a held position, and force-flattens on the
    # session's last bar.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    # Both are values the intended grid actually searches (fast 192/288/384,
    # slow 768/1152/1728) so the post-edit sanity check exercises a real combo.
    "fast_ma": 192,
    "slow_ma": 768,
    "session": "New York",
}
