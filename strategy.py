"""VWAP-deviation continuation cross — rolling volume-weighted level as the direction signal.

Every prior iteration derived direction from price alone (Donchian bands,
stochastic location, moving-average slope). This one weights the reference
level by *volume*: a rolling VWAP over the trailing `vwap_lookback` bars, so
the level sits where the bulk of contracts actually changed hands rather than
at the midpoint of an unweighted window. NQ 1h volume is far from uniform
across the day (the 09:30 and 15:00-16:00 ET bars dwarf midday), so a
volume-weighted anchor is a genuinely different series from a simple SMA, not
a cosmetic reskin of one.

The signal is a *continuation* cross, not a mean-reversion fade: price
stretching away from the volume-weighted level by more than `entry_dev` ATRs
is read as participation confirming the move, so the trade is taken in the
direction of the stretch. Using an event (a crossing) rather than a level
means one entry per excursion — a stopped-out trade does not immediately
re-arm while the deviation stays wide, which is the frequency problem
iteration 25's level signal had.

Rules (all computed on the full continuous frame with no session awareness of
their own; every window is strictly backward with strict `min_periods`, so
unwarmed bars are NaN and the signal fails *closed*):

  - Typical price and rolling VWAP over the trailing `vwap_lookback` bars,
    both windows ending at (and including) bar t, so both are knowable at t's
    Close and nothing after t is touched:

        TP_t   = (High_t + Low_t + Close_t) / 3
        VWAP_t = sum(TP * Volume)_t / sum(Volume)_t

    A window whose summed volume is 0 is masked to NaN rather than dividing
    by zero.

  - ATR: Wilder true range, simple-mean averaged over `ATR_PERIOD`, shifted
    one bar so the entry bar cannot size its own stop. The *same* shifted
    series scales the deviation and the stop/target — one volatility yardstick
    for both, measured strictly before the entry bar.

  - Deviation in ATR units:

        dev_t = (Close_t - VWAP_t) / ATR_t

    masked to NaN where ATR is not strictly positive: an ATR of 0 would make
    dev infinite, and while the delegate's `stop_distance > 0` guard would
    refuse that bar's entry, the infinity would still poison the *next* bar's
    `dev.shift(1)` crossing comparison asymmetrically.

  - Raw entries, a *crossing* (each fires on exactly one bar):
        +1.0 where dev crosses up through +entry_dev
             (dev_{t-1} <= entry_dev AND dev_t > entry_dev)
        -1.0 where dev crosses down through -entry_dev
             (dev_{t-1} >= -entry_dev AND dev_t < -entry_dev)
    `.fillna(False)` on the results is load-bearing — the delegate reads these
    as raw numpy values and `bool(np.nan)` is True, so a NaN left in the array
    would fire an entry on an unwarmed bar. The two sides are mutually
    exclusive by construction for every `entry_dev > 0` in the grid (dev can't
    be simultaneously above +entry_dev and below -entry_dev).

  - Volatility-regime gate (NEW this iteration, and the *only* change from the
    previous run of this family — hardcoded, deliberately not grid-searched):
    from the same one-bar-shifted true-range series the ATR is built on,

        atr_fast = mean(TR_shifted) over ATR_FAST_BARS (24)
        atr_slow = mean(TR_shifted) over ATR_SLOW_BARS (96)

    and both sides are AND-ed with `atr_fast >= atr_slow`. Realized volatility
    clusters over multi-day stretches, so a slow ratio of two realized-vol
    means is a persistent regime switch rather than a bar-by-bar flicker. A
    continuation signal has nothing to continue in a compressed tape; this
    stands the strategy down in exactly those stretches instead of letting the
    optimizer keep re-fitting a trend rule to a range. At 1h, 24/96 bars is
    24h vs. 96h of wall clock. Both windows use strict `min_periods`, so both
    sides are NaN until warm — and `NaN >= NaN` is already False, so the gate
    fails *closed* by construction; `.fillna(False)` on the AND result is kept
    as the documented guard against a NaN reaching the delegate's raw-numpy
    read (where `bool(np.nan)` would be True).

  - Stop: `stop_atr_mult * ATR` from the entry Close, symmetric — the delegate
    resolves it below the entry for longs and above for shorts. Held fixed at
    2.0 so this iteration is a clean test of the VWAP-deviation signal itself.

  - Target: entry Close +/- `target_atr_mult * ATR`, supplied as an absolute,
    already-direction-resolved level (the shape the delegate requires). The
    grid spans 1.5 / 2.5 / 4.0 ATRs — from tighter than the stop to twice it —
    so the optimizer can pick the reward:risk that the data supports rather
    than having one baked in.

Exit is path-dependent, so this delegates to
`session.apply_session_constraint_with_stops()` (both legs supplied on every
entry bar, as that function requires). `session.py`'s forced flatten on the
session's last bar remains the backstop — no position survives the session.

Consistent with the idea's stated exit rule, there is no flip. The delegate
only opens when it is flat (by design — it owns the session bookkeeping a flip
would have to respect), and faking a flip here would require session awareness
inside `strategy.py`, which the contract forbids. So an opposite-side crossing
does not reverse an open trade; a trade ends at its stop, its target, or the
session flatten, and the opposite side can only be taken from flat on a later
in-session bar.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):
  - `vwap_lookback` is a genuine bar count, passed as plain `int`, so it
    correctly sizes the pre-test-window warm-up buffer (max 96 -> (96+5)*3 =
    303 bars, ample for the 96-bar VWAP and the 14-bar ATR alike).
  - `entry_dev`, `target_atr_mult` and `stop_atr_mult` are unitless ATR
    multipliers, not bar counts, so all three are passed as `float` and are
    correctly ignored by that buffer sizing.
  - `ATR_PERIOD`, `ATR_FAST_BARS` and `ATR_SLOW_BARS` are module constants
    rather than params, so none of them ever enters a grid combo — they are
    plain ints, but `_max_lookback_bars()` only scans grid combos, so they
    cannot touch the buffer either way. The gate's own reach (96 + 1 shift +
    ~1 = ~98 bars) sits comfortably inside the 303-bar buffer the unchanged
    96-bar `vwap_lookback` maximum already produces. Caveat: a caller who
    narrows the lookback grid to only small values (e.g. `--vwap-lookback 12`)
    shrinks the buffer below the gate's reach and gets a gate-dead prefix at
    the head of each test window — which fails closed (no trades), not open.

Cost note: `metrics.py`'s ~0.102% round-trip (0.001% fee + 0.05% slippage per
leg, 2 legs) is unchanged and out of scope. Staying at 1h keeps that fixed
toll amortized over a session-sized excursion.

This module decides only *when* the strategy wants to be long or short and at
what levels it wants out. All day-trade gating and the end-of-session flatten
are delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd


from session import apply_session_constraint_with_stops

# ATR lookback in bars, hardcoded rather than parameterized: it is the
# volatility yardstick both the deviation and the stop are quoted in, not a
# lever this iteration is testing. Plain int is fine here — it never enters a
# grid combo, so it can't affect the warm-up buffer sizing either way.
ATR_PERIOD = 14

# Volatility-regime gate windows, in bars. Both are hardcoded rather than
# parameterized on purpose: this iteration's whole point is to add the regime
# filter *without* spending a grid axis on it, so the entry primitive and both
# searched grids stay identical to the previous run and the gate is the only
# variable. At 1h these are 24h (fast) and 96h (slow) of wall clock — slow
# enough that the ratio flips on a multi-day cadence, not bar to bar. Plain
# ints are safe here: they never enter a grid combo, so the name-agnostic
# max-int scan in `wfo_engine._max_lookback_bars()` never sees them.
ATR_FAST_BARS = 24
ATR_SLOW_BARS = 96


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder true range, one row per bar, unsmoothed and unshifted.

    Factored out so the ATR and the volatility-regime gate below are provably
    reading the *same* underlying range series at different smoothing lengths,
    rather than two independently-written definitions that could drift apart.

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


def volatility_regime_ok(df: pd.DataFrame) -> pd.Series:
    """True where short-horizon realized range is at least its slow baseline.

    Two simple means of the *one-bar-shifted* true range — so, like the ATR,
    the gate reads strictly prior bars and the entry bar cannot vote on its own
    regime. Strict `min_periods` on both means every unwarmed bar is NaN on
    both sides, and `NaN >= NaN` evaluates to False, so the comparison already
    yields a clean bool Series that fails closed with no masking needed.
    """
    tr_prior = true_range(df).shift(1)
    atr_fast = tr_prior.rolling(ATR_FAST_BARS, min_periods=ATR_FAST_BARS).mean()
    atr_slow = tr_prior.rolling(ATR_SLOW_BARS, min_periods=ATR_SLOW_BARS).mean()
    return atr_fast >= atr_slow


def rolling_vwap(df: pd.DataFrame, vwap_lookback: int) -> pd.Series:
    """Volume-weighted average typical price over the trailing `vwap_lookback` bars.

    The window ends at (and includes) the current bar, so the value is fully
    knowable at that bar's Close. NaN until the window is full, and NaN when
    the window's total volume is 0 — both cases propagate into `dev` and make
    every downstream crossing comparison False, so the signal fails closed.
    """
    n = int(vwap_lookback)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"]
    pv_sum = (typical * volume).rolling(n, min_periods=n).sum()
    v_sum = volume.rolling(n, min_periods=n).sum()
    return (pv_sum / v_sum).where(v_sum > 0)


def generate_positions(
    df: pd.DataFrame,
    vwap_lookback: int,
    entry_dev: float,
    target_atr_mult: float,
    stop_atr_mult: float = 2.0,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # One shifted ATR series, used both to scale the deviation and to size the
    # stop/target — so "1 ATR of stretch" and "1 ATR of stop" mean the same
    # thing on any given bar. `.where(atr > 0)` keeps dev finite (see docstring).
    atr = average_true_range(df, ATR_PERIOD)
    vwap = rolling_vwap(df, vwap_lookback)
    dev = ((close - vwap) / atr).where(atr > 0)

    thr = float(entry_dev)
    prev_dev = dev.shift(1)

    # Volatility-regime gate: hardcoded, applied identically to both sides, so
    # it can only ever remove entries the ungated version would have taken —
    # never add or move one.
    regime_ok = volatility_regime_ok(df)

    # Crossings, not levels: each fires on exactly one bar. Mutually exclusive
    # for any thr > 0. `fillna(False)` guards the delegate's raw-numpy read.
    long_signal = ((prev_dev <= thr) & (dev > thr) & regime_ok).fillna(False)
    short_signal = ((prev_dev >= -thr) & (dev < -thr) & regime_ok).fillna(False)

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
    "vwap_lookback": 24,
    "entry_dev": 0.5,
    "target_atr_mult": 2.5,
    "stop_atr_mult": 2.0,
    "session": "New York",
}
