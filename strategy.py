"""Run-exhaustion reversal — fade a run of consecutive same-direction bars once.

Every prior fade in this project measured "extreme" as a *level*: price
against its own rolling mean, band, or regression line. This one measures a
*path event* instead. A sign sequence — N consecutive lower closes — says
nothing about where price sits relative to any reference; it says only that
selling has been uninterrupted for N bars. Pairing that sequence with a
magnitude floor quoted in ATRs turns it into "a flush that has been both
persistent and material", which is the shape mean-reversion.md describes when
it talks about forced selling overshooting before it reverts.

The trade is the fade: a down-run is bought, an up-run is sold. It fires
*once* per run, on the first bar the run qualifies, not on every bar of a
continuing run.

Rules (all computed on the full continuous frame with no session awareness of
their own; every window is strictly backward with strict `min_periods`, so
unwarmed bars are NaN and the signal fails *closed*):

  - ATR: Wilder true range, simple-mean averaged over `ATR_PERIOD`, shifted
    one bar so the entry bar cannot size its own stop. The *same* shifted
    series scales the run-magnitude floor and the stop/target — one volatility
    yardstick throughout, measured strictly before the entry bar. Bars whose
    ATR is NaN or non-positive are masked out, so they can neither qualify a
    run nor pass the delegate's `stop_distance > 0` guard.

  - Down-run at bar t, with `run_len` = N:

        Close[t-i] < Close[t-i-1]   for i = 0 .. N-1        (N lower closes)
        (Close[t-N] - Close[t]) / ATR[t] >= run_move_atr     (magnitude floor)

    The streak arm is a rolling count of strictly-negative close-to-close
    diffs over exactly N bars ending at t, which spans Close[t-N]..Close[t] —
    the same span the magnitude arm measures, so the two arms are indexed
    consistently. The up-run is the mirror (N higher closes, and
    (Close[t] - Close[t-N]) / ATR[t] >= run_move_atr).

  - Raw entries, a *crossing* of the run condition (each fires on exactly one
    bar):
        long  where down_run is True at t and False at t-1
        short where up_run   is True at t and False at t-1
    Firing on the crossing rather than the level is what makes this one entry
    per flush: while the run keeps extending, the condition stays True, so no
    new crossing occurs and a stopped-out trade cannot immediately re-arm.
    (It does not eliminate that exposure entirely — a single counter-direction
    close resets the condition to False, so a stair-step decline can re-fire.)

    The two sides are mutually exclusive by construction: bar t's close-to-
    close diff cannot be simultaneously negative and positive, so t cannot be
    the terminus of both a lower-close run and a higher-close run.

    `.fillna(False)` on both signals is load-bearing — the delegate reads them
    as raw numpy values and `bool(np.nan)` is True, so a NaN left in the array
    would fire an entry on an unwarmed bar.

  - Direction is the fade, and this is the one place a sign slip would be
    silent rather than loud: down-run -> long, up-run -> short.

  - Stop: `stop_atr_mult * ATR` from the entry Close, symmetric — the delegate
    resolves it below the entry for longs and above for shorts. Held fixed at
    1.5 so this iteration is a clean test of the run-exhaustion signal itself.

  - Target: entry Close +/- `target_atr_mult * ATR`, supplied as an absolute,
    already-direction-resolved level (the shape the delegate requires). The
    grid spans 1.0 / 1.5 / 2.5 ATRs. Both legs are bounded, so P&L cannot be
    carried by a single open-ended tail.

Exit is path-dependent, so this delegates to
`session.apply_session_constraint_with_stops()` (both legs supplied on every
entry bar, as that function requires). `session.py`'s forced flatten on the
session's last bar remains the backstop — no position survives the session.

There is no flip. The delegate only opens when it is flat (by design — it owns
the session bookkeeping a flip would have to respect), and faking a flip here
would require session awareness inside `strategy.py`, which the contract
forbids. So an opposite-side crossing does not reverse an open trade; a trade
ends at its stop, its target, or the session flatten.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):
  - `run_len` is a genuine bar count, passed as plain `int`, so it correctly
    sizes the pre-test-window warm-up buffer.
  - `run_move_atr`, `target_atr_mult` and `stop_atr_mult` are unitless ATR
    multipliers, not bar counts, so all three are passed as `float` and are
    correctly ignored by that buffer sizing.
  - `ATR_PERIOD` is a module constant rather than a param, so it never enters
    a grid combo and cannot touch the buffer either way.

  Warm-up headroom is tighter here than in previous iterations and worth
  stating explicitly: the largest grid int is `run_len` (max 4), so
  `buffer_bars = max((4+5)*3, bars_per_day+5)` is ~27-29 bars at NQ 1h,
  versus the ~300 a 96-bar lookback used to produce. The signal's actual
  reach is ATR_PERIOD (14) + 1 bar for the TR's own `Close.shift(1)` + 1 bar
  for the ATR shift + `run_len` + 1 bar for the t-1 crossing arm ~= 21 bars at
  run_len=4, comfortably inside that buffer — and it stays inside even if a
  caller narrows the grid to `--run-len 2`. No artificial floor is needed;
  were the buffer ever to fall short it would fail closed (a few no-trade bars
  at the head of a test window), not open.

Cost note: `metrics.py`'s ~0.102% round-trip (0.001% fee + 0.05% slippage per
leg, 2 legs) is unchanged and out of scope. Running at 1h rather than a finer
timeframe is what keeps that fixed toll a modest fraction of a ~1-ATR NQ
target, which is the arithmetic this idea rests on.

This module decides only *when* the strategy wants to be long or short and at
what levels it wants out. All day-trade gating and the end-of-session flatten
are delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd


from session import apply_session_constraint_with_stops

# ATR lookback in bars, hardcoded rather than parameterized: it is the
# volatility yardstick both the run-magnitude floor and the stop are quoted
# in, not a lever this iteration is testing. Plain int is fine here — it never
# enters a grid combo, so it can't affect the warm-up buffer sizing either way.
ATR_PERIOD = 14


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder true range, one row per bar, unsmoothed and unshifted.

    `skipna=False` on the row-wise max keeps the first bar's TR NaN (its
    `Close.shift(1)` is NaN) rather than silently falling back to High-Low.
    """
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)


def average_true_range(df: pd.DataFrame, atr_period: int = ATR_PERIOD) -> pd.Series:
    """Wilder true range, simple-mean averaged over `atr_period`, shifted one bar.

    The shift excludes the current bar from its own volatility baseline, so the
    stop is sized off strictly prior information — the entry bar's own range
    can't widen or narrow the stop it is about to be given.

    Strict `min_periods` keeps every unwarmed bar NaN, so the delegate's
    `stop_distance > 0` guard refuses entries during warm-up.
    """
    tr = true_range(df)
    return tr.rolling(atr_period, min_periods=atr_period).mean().shift(1)


def _run_condition(
    close: pd.Series, atr: pd.Series, run_len: int, run_move_atr: float, down: bool
) -> pd.Series:
    """True at bars terminating a qualifying `run_len`-bar directional run.

    Two arms over the *same* span, Close[t-run_len] .. Close[t]:

      - persistence: every one of the `run_len` close-to-close diffs ending at
        t is strictly in the run's direction. Counted as a rolling sum of a
        bool over exactly `run_len` bars with strict `min_periods`, so the
        head of the series is NaN (-> False after the comparison) rather than
        being scored off a partial window.
      - magnitude: the run's total displacement, divided by the one-bar-
        shifted ATR at t, clears `run_move_atr`. `.where(atr > 0)` masks bars
        with a NaN or non-positive ATR to NaN; `NaN >= x` is False, so those
        bars fail closed instead of dividing by zero.

    Returns a plain bool Series (never NaN), safe to shift and negate.
    """
    n = int(run_len)
    diff = close.diff()
    step = (diff < 0) if down else (diff > 0)
    streak = step.rolling(n, min_periods=n).sum() == n

    displacement = (close.shift(n) - close) if down else (close - close.shift(n))
    magnitude = (displacement / atr).where(atr > 0) >= float(run_move_atr)

    return (streak & magnitude).fillna(False)


def generate_positions(
    df: pd.DataFrame,
    run_len: int,
    run_move_atr: float,
    target_atr_mult: float,
    stop_atr_mult: float = 1.5,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # One shifted ATR series, used both to normalize the run's displacement
    # and to size the stop/target — so "1 ATR of flush" and "1 ATR of stop"
    # mean the same thing on any given bar.
    atr = average_true_range(df, ATR_PERIOD)

    down_run = _run_condition(close, atr, run_len, run_move_atr, down=True)
    up_run = _run_condition(close, atr, run_len, run_move_atr, down=False)

    # Crossings, not levels: fire on the first bar the run qualifies and not
    # again while it keeps qualifying. `fill_value=False` keeps the shifted
    # series bool (no NaN at position 0) so the negation is well-defined, and
    # `.fillna(False)` guards the delegate's raw-numpy read regardless.
    #
    # The fade: a down-run is BOUGHT, an up-run is SOLD. Mutually exclusive by
    # construction — bar t's diff can't be both negative and positive.
    long_signal = (down_run & ~down_run.shift(1, fill_value=False)).fillna(False)
    short_signal = (up_run & ~up_run.shift(1, fill_value=False)).fillna(False)

    # Protective stop, in ATR units from the entry Close. Symmetric — the
    # delegate resolves the side. NaN/zero ATR makes it refuse the entry.
    stop_distance = float(stop_atr_mult) * atr

    # Target as an absolute, already direction-resolved level, which is the
    # shape the delegate expects. Only read on actual entry bars.
    tgt = float(target_atr_mult)
    target_price = pd.Series(
        np.where(long_signal, close + tgt * atr, close - tgt * atr),
        index=df.index,
    )

    return apply_session_constraint_with_stops(
        close=close,
        high=high,
        low=low,
        long_signal=long_signal,
        short_signal=short_signal,
        stop_distance=stop_distance,
        target_price=target_price,
        session=session,
    )


DEFAULT_PARAMS = {
    "run_len": 3,
    "run_move_atr": 1.25,
    "target_atr_mult": 1.5,
    "stop_atr_mult": 1.5,
    "session": "New York",
}
