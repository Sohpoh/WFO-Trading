"""Long-only volatility-squeeze Donchian breakout with a bounded ATR target.

New family (`volatility_squeeze_breakout`), following the retired KNN
return-prediction family. Entry reads the trading-vault volatility.md
clustering cycle ("Calm period: Low volatility ... -> Shock event: Spike in
volatility") as a tradeable setup: a multi-day realized-vol contraction
precedes expansion, so a breakout from a tight range *while squeezed* is a
higher-win-rate event than an unconditional channel breakout. Long-only (the
continuation side with demonstrated in-sample edge) and a bounded target
harvests the expansion into "frequent small wins" rather than an uncapped
tail.

Rules (all computed on continuous, session-unaware bars; session gating and
stop/target bookkeeping are delegated to `apply_session_constraint_with_stops`):

  - Squeeze state (zero-param, both ATR windows hardcoded): ATR_fast =
    ATR(96 bars ~ 1 day), ATR_slow = ATR(960 bars ~ 10 days), both Wilder
    averages. The setup is armed on any bar where
    ATR_fast / ATR_slow <= squeeze_ratio (a realized-vol contraction).

  - Breakout trigger: Donchian upper channel = the highest High of the
    trailing `range_lookback` PRIOR bars (no lookahead). A long fires on the
    single bar where Close crosses ABOVE that channel (previous Close <=
    previous channel, current Close > current channel) while the squeeze is
    armed. This is a one-bar crossing, NOT a persistent above-state, so a
    stop/target exit does not re-arm on the next in-session bar (the exact
    churn bug that sank the prior iteration's stop variant). No short branch.

  - Exit: path-dependent stop + target via `apply_session_constraint_with_stops`.
    Stop = entry - stop_atr_mult * ATR(14) (stop_atr_mult fixed 2.0);
    target = entry + target_atr_mult * ATR(14). Both distances are read at the
    entry bar and the position force-flattens at session close (session.py).

Warm-up: ATR_SLOW_PERIOD (960 bars) is the largest bar-count lookback and is
threaded through build_grid() as a plain int so _max_lookback_bars() sizes the
test-window buffer off it (960 -> (960+5)*3 = 2895 bars ~ 30 days at 15min),
comfortably covering the ATR(960) Wilder warm-up, the Donchian window
(range_lookback grid {8,12,24,48}), and ATR(14). range_lookback is a genuine
bar-count lookback and is a plain int on purpose. squeeze_ratio and
target_atr_mult are thresholds/multipliers, NOT lookbacks, so they are floats
and must not inflate the warm-up buffer. stop_atr_mult (fixed 2.0) is also a
float multiplier.

Contract notes: `generate_positions` builds session-unaware raw signals and
delegates all session gating, stop/target bookkeeping, and end-of-session
flatten to `apply_session_constraint_with_stops`; it never reimplements the
session mask. The delegate returns one scalar {-1,0,1} per bar, so repeat
signals while already long are no-ops and the single-position contract holds.
NaN/unwarmed indicators fail closed (no entry). Costs (metrics.py, cost model
v2) are out of scope for this file.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Fixed ATR windows, in BARS, for the squeeze state. Both are genuine bar-count
# lookbacks (Wilder smoothing spans) and are threaded through build_grid() as
# plain ints on purpose: ATR_SLOW_PERIOD (960) is the largest int in the grid,
# so _max_lookback_bars() returns 960 and sizes the warm-up buffer to ~30 days
# at 15min. Never grid-searched.
ATR_FAST_PERIOD = 96    # ~1 day at 15min (96 bars/day)
ATR_SLOW_PERIOD = 960   # ~10 days at 15min

# Fixed ATR window, in BARS, for the stop/target distance. Genuine bar-count
# lookback (subsumed by ATR_SLOW_PERIOD when sizing the warm-up buffer).
ATR_STOP_PERIOD = 14

# Fixed stop multiplier (float, NOT a lookback). Never grid-searched.
STOP_ATR_MULT = 2.0


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's Average True Range over `period` bars, index-aligned to `close`.

    True range uses the prior bar's close as the reference so session/overnight
    gaps count. The Wilder average is a recursive EMA with alpha = 1/period
    (adjust=False) so it carries state rather than using a fixed trailing
    window. The first `period` bars are NaN (the recursion hasn't seen a full
    window yet), so a squeeze ratio or stop distance built from them fails
    closed and cannot open a position.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
    atr.iloc[:period] = np.nan
    return atr


def generate_positions(
    df: pd.DataFrame,
    range_lookback: int,
    squeeze_ratio: float,
    target_atr_mult: float,
    session: str | None = "New York",
    stop_atr_mult: float = STOP_ATR_MULT,
    atr_fast_period: int = ATR_FAST_PERIOD,
    atr_slow_period: int = ATR_SLOW_PERIOD,
    atr_stop_period: int = ATR_STOP_PERIOD,
) -> pd.Series:
    """Build the long-only volatility-squeeze Donchian-breakout position series.

    Squeeze (armed) state = ATR(atr_fast_period)/ATR(atr_slow_period) <=
    squeeze_ratio. Breakout = a one-bar Close cross above the Donchian upper
    channel (highest High of the trailing `range_lookback` prior bars). Long
    fires on the crossing bar while squeezed; no short. Exit is a bounded
    ATR-scaled stop + target delegated to session.py, with the position
    force-flattened at session close. Returns a {-1,0,1} position series.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    atr_fast = atr_series(high, low, close, int(atr_fast_period))
    atr_slow = atr_series(high, low, close, int(atr_slow_period))

    # Squeeze/armed state. NaN (unwarmed ATR) compares False, so unwarmed bars
    # fail closed; a degenerate (zero/inf) ratio also fails closed.
    squeeze = (atr_fast / atr_slow) <= float(squeeze_ratio)

    # Donchian upper channel from the trailing `range_lookback` PRIOR bars
    # (high.shift(1) excludes the current bar -> no lookahead). A one-bar Close
    # cross ABOVE the channel, not a persistent above-state.
    upper = high.shift(1).rolling(int(range_lookback), min_periods=int(range_lookback)).max()
    cross_above = (close > upper) & (close.shift(1) <= upper.shift(1))

    long_signal = squeeze & cross_above
    short_signal = pd.Series(False, index=df.index, dtype=bool)

    # Stop/target distances, both read at the entry bar. stop_distance is a
    # positive from-entry distance (session.py resolves the below-entry
    # direction for a long); target_price is absolute and already above entry
    # because it anchors on Close + target_atr_mult * ATR(14).
    atr_stop = atr_series(high, low, close, int(atr_stop_period))
    stop_distance = float(stop_atr_mult) * atr_stop
    target_price = close + float(target_atr_mult) * atr_stop

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
    # range_lookback is a genuine bar-count lookback (the Donchian trailing
    # window; grid {8,12,24,48}) and is a plain int on purpose so it feeds
    # the warm-up buffer. squeeze_ratio (grid {0.8,0.9,1.0,1.1}) is a vol-ratio
    # threshold and target_atr_mult (grid {1.0,1.5,2.0,3.0}) is an ATR
    # multiplier — both floats, NOT lookbacks. stop_atr_mult (fixed 2.0) is a
    # float multiplier. atr_fast_period (96), atr_slow_period (960) and
    # atr_stop_period (14) are fixed plain-int lookbacks — never grid-searched
    # but threaded through build_grid() so _max_lookback_bars() returns 960.
    # `session` is a fixed param. These are the concrete set the sanity checker
    # runs generate_positions() against.
    "range_lookback": 24,
    "squeeze_ratio": 1.0,
    "target_atr_mult": 2.0,
    "stop_atr_mult": STOP_ATR_MULT,
    "atr_fast_period": ATR_FAST_PERIOD,
    "atr_slow_period": ATR_SLOW_PERIOD,
    "atr_stop_period": ATR_STOP_PERIOD,
    "session": "New York",
}
