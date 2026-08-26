"""Volatility-level polarity switch — one displacement primitive, sign set by regime.

Across this project's history the ATR ratio has only ever been used as an
on/off *gate* on a momentum strategy: trade when volatility is expanding,
stand aside when it isn't. The one polarity switch tried before keyed off a
variance ratio — a measure of *persistence*, not of volatility *level*. This
iteration reads volatility.md's Strategy Selection line literally instead —
"High volatility? Use wide-stop momentum strategies. Low volatility? Use
tight-stop mean-reversion" — and lets the measured vol regime set the *sign*
of a single entry primitive, plus the width of its stop.

There is exactly one signal primitive: an ATR-normalized displacement over
`move_lookback` bars. In the HIGH-vol regime it is traded *with* (momentum);
in the LOW-vol regime it is traded *against* (reversion). Nothing else about
the entry differs between the two legs, so the run is attributable to the
polarity switch alone.

The reversion leg is deliberately NOT a price-vs-rolling-mean / band /
regression z-score. Two earlier iterations concluded that exact quantity is
information-free on this data in either polarity, so the fade here is of a
*displacement over a fixed horizon*, which is a different measurement.

Rules (all computed on the full continuous frame with no session awareness of
their own; every window is strictly backward with strict `min_periods`, so
unwarmed bars are NaN and the signal fails *closed*):

  - ATR: Wilder true range, simple-mean averaged over `ATR_PERIOD`, shifted
    one bar so the entry bar cannot size its own stop. The same shifted
    series normalizes the displacement and scales the stop — one volatility
    yardstick throughout, measured strictly before the entry bar.

  - Regime (zero-param, hardcoded module constants, so the grid searches only
    the signal itself):

        vol_fast[t] = mean TR over VOL_FAST_N bars, shifted 1
        vol_slow[t] = mean TR over VOL_SLOW_N bars, shifted 1
        HIGH-vol  <=>  vol_fast >= vol_slow
        LOW-vol   <=>  vol_fast <  vol_slow

    A hard split with no deadband — the same construction earlier ATR-ratio
    gates used, so this iteration differs from them in polarity, not in how
    the regime is measured. Both comparisons return False when either leg is
    NaN, so an unwarmed bar is in *neither* regime and takes no trade.

  - Displacement:

        d[t] = (Close[t] - Close[t - move_lookback]) / ATR[t]

    masked to NaN where ATR is NaN or non-positive (so those bars can neither
    arm a signal nor pass the delegate's `stop_distance > 0` guard).

  - Arming and the crossing:

        armed[t] = |d[t]| >= move_atr
        raw entry only where armed[t] and not armed[t-1]

    Firing on the crossing rather than the level means a continuing move does
    not re-arm bar after bar, and a stopped-out trade cannot immediately
    re-enter on the same displacement.

  - Direction — the one place a sign slip would be silent rather than loud:

        HIGH-vol:  d > 0 -> LONG,   d < 0 -> SHORT   (trade *with* the move)
        LOW-vol:   d > 0 -> SHORT,  d < 0 -> LONG    (trade *against* it)

    Long and short are mutually exclusive by construction: `d` cannot be both
    positive and negative, and the two regimes cannot both be True.

    `.fillna(False)` on both signals is load-bearing — the delegate reads them
    as raw numpy values and `bool(np.nan)` is True, so a NaN left in the array
    would fire an entry on an unwarmed bar.

  - Stop, regime-scaled at the entry bar (volatility.md's Stop-Loss Placement
    section — wide stops for the momentum leg, tight for the reversion leg):

        HIGH-vol entry:  stop_atr_high * ATR
        LOW-vol  entry:  stop_atr_low  * ATR

    Bars in neither regime get a NaN stop distance, which is a second,
    independent fail-closed path: the delegate refuses any entry there.

  - Target: entry Close +/- `RR_MULT * stop_distance`, supplied as an
    absolute, already-direction-resolved level (the shape the delegate
    requires). Because the target is quoted off the *stop*, its ATR distance
    inherits the regime scaling too — at the default 1.5 / 1.0 stops that is
    2.25 ATR in HIGH-vol and 1.5 ATR in LOW-vol, both a constant 1.5R.

    Sizing is deliberately tighter than a 2.0/3.0-ATR shape: an earlier
    iteration showed a ~4-ATR target essentially never binds inside a ~6.5-bar
    1h New York session, so it degenerated into a run-to-session-flatten whose
    P&L was carried by a handful of tails. Both legs here are bounded, so P&L
    cannot be carried by an open-ended run.

Semantics worth stating explicitly, because they *reduce* trade count by
design and should not be read as bugs:

  - `armed` is computed on |d| alone, with no regime term. So if the regime
    flips while `armed` stays True, no new entry fires; and a crossing bar
    whose regime is still cold (NaN) consumes that crossing with no trade,
    which won't re-arm until |d| drops back below `move_atr` and crosses up
    again. This is what keeps "one entry per displacement event" true
    regardless of what the regime is doing.

  - There is no flip. The delegate only opens when it is flat (by design — it
    owns the session bookkeeping a flip would have to respect), and faking a
    flip here would require session awareness inside `strategy.py`, which the
    contract forbids. A trade ends at its stop, its target, or the session
    flatten.

Exit is path-dependent, so this delegates to
`session.apply_session_constraint_with_stops()` (both legs supplied on every
entry bar, as that function requires). `session.py`'s forced flatten on the
session's last bar remains the backstop — no position survives the session.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):
  - `move_lookback` is a genuine bar count, passed as plain `int`, so it
    correctly sizes the pre-test-window warm-up buffer.
  - `move_atr`, `stop_atr_high` and `stop_atr_low` are unitless ATR
    multipliers, not bar counts, so all three are passed as `float` and are
    correctly ignored by that buffer sizing.
  - `ATR_PERIOD`, `VOL_FAST_N`, `VOL_SLOW_N` and `RR_MULT` are module
    constants rather than params, so none of them enters a grid combo and
    none can touch the buffer either way.

  WARM-UP HAZARD — read before narrowing the grid. Because `VOL_SLOW_N` is a
  constant it never reaches `_max_lookback_bars()`, so the buffer is sized off
  `move_lookback` alone. With the intended grid top of 24,
  `buffer_bars = max((24 + 5) * 3, bars_per_day + 5)` = 87 bars at NQ 1h,
  against a regime reach of ~75 bars (72-bar window + 1 bar for the TR's own
  `Close.shift(1)` + 1 bar for the shift). That is ~12 bars of headroom. Run
  with, say, `--move-lookback 3` and the buffer collapses to ~24 bars, leaving
  the regime series NaN for roughly the first 50 bars of every test window —
  which fails *closed* (no trades there, not wrong trades), but silently guts
  the strategy. Do not narrow the grid below a top of 24 without also
  shortening `VOL_SLOW_N`.

Cost note: `metrics.py`'s ~0.102% round-trip (0.001% fee + 0.05% slippage per
leg, 2 legs) is unchanged and out of scope. At NQ 1h (ATR ~0.3%) the chosen
legs imply breakeven hit rates of roughly 49% (HIGH-vol, 2.25 ATR target vs
1.5 ATR stop) and 54% (LOW-vol, 1.5 ATR vs 1.0 ATR) against that toll.

This module decides only *when* the strategy wants to be long or short and at
what levels it wants out. All day-trade gating and the end-of-session flatten
are delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd


from session import apply_session_constraint_with_stops

# ATR lookback in bars — the volatility yardstick the displacement is
# normalized by and the stop is quoted in, not a lever this iteration tests.
ATR_PERIOD = 14

# Regime measure, both in bars. Fast vs. slow mean true range: vol_fast >=
# vol_slow means realized volatility is at/above its own slower baseline
# (HIGH), below means it is compressed (LOW). Hardcoded so the polarity switch
# is the only thing the grid can tune — and so a fold cannot pick a regime
# definition that happens to flatter the sample.
VOL_FAST_N = 24
VOL_SLOW_N = 72

# Reward:risk. The target is placed RR_MULT * stop_distance from entry, so it
# inherits whichever regime-scaled stop the entry was given and every trade
# carries the same R multiple regardless of regime. Hardcoded, same status as
# ATR_PERIOD.
RR_MULT = 1.5

# All four are plain ints/floats used only inside this module — none enters a
# grid combo, so none can reach `wfo_engine._max_lookback_bars()`.


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


def _vol_regime(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(high_vol, low_vol) bool Series — the measured volatility *level* regime.

    Fast and slow simple means of true range, each shifted one bar so the
    entry bar's own range cannot classify the bar it is about to trade.
    Strict `min_periods` leaves both NaN until warm; `NaN >= x` and `NaN < x`
    are both False, so an unwarmed bar lands in *neither* regime and is
    therefore untradeable — the regime series fails closed with no explicit
    validity mask needed.
    """
    tr = true_range(df)
    vol_fast = tr.rolling(VOL_FAST_N, min_periods=VOL_FAST_N).mean().shift(1)
    vol_slow = tr.rolling(VOL_SLOW_N, min_periods=VOL_SLOW_N).mean().shift(1)
    return (vol_fast >= vol_slow), (vol_fast < vol_slow)


def generate_positions(
    df: pd.DataFrame,
    move_lookback: int,
    move_atr: float,
    stop_atr_high: float = 1.5,
    stop_atr_low: float = 1.0,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # One shifted ATR series, used both to normalize the displacement and to
    # size the stop/target — so "1 ATR of move" and "1 ATR of stop" mean the
    # same thing on any given bar.
    atr = average_true_range(df, ATR_PERIOD)

    high_vol, low_vol = _vol_regime(df)

    # ATR-normalized displacement over a fixed horizon. `.where(atr > 0)`
    # masks bars with a NaN or non-positive ATR to NaN, so they fail closed
    # instead of dividing by zero.
    n = int(move_lookback)
    displacement = ((close - close.shift(n)) / atr).where(atr > 0)

    # Arming is on magnitude alone (no regime term) — see the module docstring
    # for why. `NaN >= x` is False, so unwarmed bars are simply not armed.
    armed = (displacement.abs() >= float(move_atr)).fillna(False)

    # Crossing, not level: fire on the first bar the displacement qualifies and
    # not again while it keeps qualifying. `fill_value=False` keeps the shifted
    # series bool (no NaN at position 0) so the negation is well-defined.
    crossing = armed & ~armed.shift(1, fill_value=False)

    up = (displacement > 0).fillna(False)
    down = (displacement < 0).fillna(False)

    # THE POLARITY SWITCH. HIGH-vol trades *with* the displacement, LOW-vol
    # trades *against* it. Mutually exclusive: `up`/`down` cannot both be
    # True, and `high_vol`/`low_vol` cannot both be True.
    long_signal = (crossing & ((high_vol & up) | (low_vol & down))).fillna(False)
    short_signal = (crossing & ((high_vol & down) | (low_vol & up))).fillna(False)

    # Regime-scaled protective stop, in ATR units from the entry Close.
    # Symmetric — the delegate resolves the side. Bars in neither regime keep
    # a NaN multiplier, so `stop_distance` is NaN there and the delegate
    # refuses the entry (a second fail-closed path, independent of the
    # signals themselves).
    stop_mult = pd.Series(np.nan, index=df.index)
    stop_mult[high_vol] = float(stop_atr_high)
    stop_mult[low_vol] = float(stop_atr_low)
    stop_distance = stop_mult * atr

    # Target as an absolute, already direction-resolved level, which is the
    # shape the delegate expects. Quoted off the *stop distance*, so every
    # trade is the same RR_MULT R regardless of regime. Only read on actual
    # entry bars.
    target_price = pd.Series(
        np.where(
            long_signal,
            close + RR_MULT * stop_distance,
            close - RR_MULT * stop_distance,
        ),
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
    "move_lookback": 6,
    "move_atr": 1.75,
    "stop_atr_high": 1.5,
    "stop_atr_low": 1.0,
    "session": "New York",
}
