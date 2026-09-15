"""ES 1h Bollinger-Band Reversion to Mean, variance-ratio-regime-gated — a
*mean-reversion* reading of `Trading Vault wiki/mean-reversion.md`, with a
zero-param variance-ratio regime gate that licenses the fade only while the
series is actually mean-reverting.

This is iteration 54's band fade (variation-of: 54) with one mechanism change
on the *conditioning variable* — not a param tweak. Iteration 54 faded every
Bollinger-band touch unconditionally and got `no-edge` (IS PF 0.87, worst trade
−1.34% vs +0.345% mean winner — the classic fade-in-a-trend signature).
mean-reversion.md's pitfall #1 is explicit that mean reversion "only works in
choppy markets" and "in a strong uptrend, shorting strength is a losing
strategy", yet 54 never checked regime, so it was shorting strength and buying
weakness throughout trending stretches. This iteration adds the gate named by
statistical-mean-reversion-tests.md ("If VR(n) << 1 … strongly mean-reverting
… VR(n) >> 1, trending"): emit a fade only when the rolling variance ratio is
< 1.0 (mean-reverting) and go flat otherwise. VR only licenses-or-blocks the
existing ES-1h band fade; it never flips to a momentum side.

The mechanism
-------------
All quantities are session-unaware and strictly causal: the trailing mean,
standard-deviation, and variance-ratio windows all end at the current bar
(inclusive), so there is no lookahead. `metrics.bar_returns_with_costs` prices a
position off `position.shift(1)`, so a position set at bar t earns the
Close_t -> Close_{t+1} return the signal never sees.

  - On each bar compute the trailing mean μ and population std σ of Close over
    `band_lookback` bars (causal — the window ends at the current bar). σ is
    population (ddof=0), the canonical Bollinger-band convention (unchanged
    from iter 54).

  - Upper band = μ + `entry_z`·σ; lower band = μ − `entry_z`·σ.

  - Zero-param variance-ratio regime gate (fixed module constants, NOT
    grid-searched): on each bar compute the rolling variance ratio
        VR = Var(Close.pct_change(vr_horizon)) over a vr_lookback-bar window
            ÷ (vr_horizon × Var(Close.pct_change(1)) over the same
               vr_lookback-bar window)
    — statistical-mean-reversion-tests.md's variance-ratio test. Both
    variances use the same vr_lookback window so the ddof choice cancels in
    the ratio (ddof=0 chosen here to match the band's population-σ convention,
    but it is immaterial).

  - A short (−1) is emitted where Close >= upper band AND VR < vr_threshold; a
    long (+1) where Close <= lower band AND VR < vr_threshold. When
    VR >= vr_threshold (random walk / trending) both signals are suppressed.
    During warm-up μ/σ/VR are NaN, so every comparison fails closed and no
    entry is emitted on the first max(band_lookback, vr_lookback+vr_horizon)
    − 1 bars.

  - There is no opposite-signal flip: the band decides direction when flat and
    VR < vr_threshold, and once a position is open it exits only via its stop,
    its target, or the forced end-of-session flatten (the stop/target walk in
    session.py only ever opens from flat).

Exit — path-dependent stop/target, both quoted in trailing σ
------------------------------------------------------------
`apply_session_constraint_with_stops` (NOT the plain flip variant) — a
path-dependent stop/target pair, per mean-reversion.md's stop-placement rule
and "exit at the middle" target. Byte-identical to iteration 54.

  - Target = the trailing mean μ. Direction-resolved by the walk: a long
    entered at the lower band has μ sitting *above* entry (μ = lower + z·σ),
    a short entered at the upper band has μ *below* entry (μ = upper − z·σ),
    so the one μ series serves both sides. Only μ's value at the actual entry
    bar is read (session.py stores it as a fixed `target_state`), so the
    target is the mean *at entry*, not a moving target.

  - Stop = entry ± `stop_sigma_mult`·σ, volatility-scaled (per volatility.md's
    stop-placement rule and mean-reversion.md pitfall #4 stop-loss discipline).
    A σ of 0 (perfectly flat window) makes the stop distance 0, which fails
    the walk's `stop_ok` gate and closes that bar to entries.

  - The New York session force-flattens at 16:00 ET, so every trade is closed
    within the session (day-trade only, no overnight).

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
-------------------------------------------------------------------------
  - `band_lookback` IS a genuine bar-count lookback (the rolling mean/σ
    window), so it is a plain `int` **on purpose**: it *should* feed
    `_max_lookback_bars()`, whose max over the grid sizes the pre-test-window
    warm-up buffer.
  - `vr_lookback` (96) and `vr_horizon` (12) are ALSO genuine bar-count
    lookbacks — the rolling-variance window and the pct_change lag in the
    variance-ratio numerator — so they are plain `int`s **on purpose** and are
    threaded as fixed (never grid-searched) values through `build_grid()`. The
    variance-ratio's total warm-up need is vr_lookback + vr_horizon = 108
    bars; `_max_lookback_bars()` takes the max over the grid (96 here, once
    vr_lookback is threaded), and the buffer formula's 3× headroom
    (max((96 + 5) * 3, day_bars + 5) = 303 bars at 1h) comfortably covers the
    108-bar total even if `band_lookback`'s grid later shrinks.
  - `vr_threshold` (1.0) is a dimensionless gate threshold, NOT a lookback, so
    it is a `float` **on purpose** and correctly ignored by
    `_max_lookback_bars()`.
  - `entry_z` and `stop_sigma_mult` are `float` **on purpose** — they are
    σ-multiples (dimensionless thresholds / stop scaling), not bar-count
    lookbacks, so they must NOT feed `_max_lookback_bars()`. Both are cast to
    float in `build_grid()` so a grid point written as `2` can never arrive as
    an int and silently inflate the buffer.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.005% slippage) is
unchanged and out of scope. This strategy is symmetric (both sides traded), so
a long↔short reversal costs two legs (close + open); no cost logic lives here.

This module decides only *when* the strategy wants to fade a band touch (and
now *whether the regime is mean-reverting*) and how far it lets a loser run
before the σ-scaled stop exits. All day-trade gating and the end-of-session
flatten are delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Fixed (never grid-searched) variance-ratio regime-gate constants. These are
# "zero-param" from the grid's point of view: they are not exposed as CLI
# flags and not swept, but they are genuine bar-count lookbacks (vr_horizon,
# vr_lookback) that must feed wfo_engine's warm-up buffer, so build_grid()
# threads them as fixed ints — see the warm-up note in the module docstring.
VR_HORIZON = 12       # bars: the pct_change lag in the variance-ratio numerator
VR_LOOKBACK = 96      # bars: the rolling-variance window for both VR terms
VR_THRESHOLD = 1.0    # gate: fade only while VR < this (mean-reverting); VR >= this -> flat


def generate_positions(
    df: pd.DataFrame,
    band_lookback: int,
    entry_z: float,
    stop_sigma_mult: float,
    session: str | None = "New York",
    vr_horizon: int = VR_HORIZON,
    vr_lookback: int = VR_LOOKBACK,
    vr_threshold: float = VR_THRESHOLD,
) -> pd.Series:
    """Build the variance-ratio-gated Bollinger-band fade position series.

    `band_lookback` is the trailing-window bar count for the mean μ and
    population std σ; `entry_z` is the σ-multiple that defines the upper/lower
    band (short at Close >= μ + entry_z·σ, long at Close <= μ − entry_z·σ);
    `stop_sigma_mult` scales the hard stop's distance from entry off σ. The
    profit target is the trailing mean μ (direction-resolved by session.py).
    `vr_horizon`/`vr_lookback`/`vr_threshold` are the fixed (never
    grid-searched) variance-ratio regime gate: a fade is licensed only while
    VR = Var(Close.pct_change(vr_horizon)) ÷ (vr_horizon ×
    Var(Close.pct_change(1))) over the vr_lookback window is below
    vr_threshold, and suppressed when the series is trending / a random walk.
    Everything is routed through `apply_session_constraint_with_stops` so the
    New York session gates entries and force-flattens at 16:00 ET. Returns a
    {-1, 0, 1} position series.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Trailing (causal) mean and population std of Close. The first
    # `band_lookback - 1` bars are NaN, so bands built from them fail closed.
    band = close.rolling(band_lookback)
    mu = band.mean()
    sigma = band.std(ddof=0)

    # Direction-resolved band levels. entry_z and stop_sigma_mult are floats
    # (σ-multiples, NOT lookbacks); band_lookback is the genuine bar-count
    # lookback that feeds wfo_engine's warm-up buffer.
    upper = mu + float(entry_z) * sigma
    lower = mu - float(entry_z) * sigma

    # Raw band touch/penetration (unchanged from iter 54). NaN comparisons are
    # False, so an unwarmed bar (NaN μ/σ) emits no entry. Inclusive (>= / <=)
    # so an exact touch counts.
    band_long = close <= lower
    band_short = close >= upper

    # Zero-param variance-ratio regime gate. Both rolling variances use the
    # same vr_lookback window, so the ddof normalization cancels in the ratio
    # (ddof=0 kept for consistency with the band's population σ). VR is NaN
    # until vr_lookback + vr_horizon bars have elapsed (the pct_change lag
    # needs vr_horizon bars and then the rolling var needs vr_lookback more),
    # and NaN < threshold is False, so unwarmed bars fail closed.
    ret_1 = close.pct_change(1, fill_method=None)
    ret_h = close.pct_change(vr_horizon, fill_method=None)
    var_1 = ret_1.rolling(vr_lookback).var(ddof=0)
    var_h = ret_h.rolling(vr_lookback).var(ddof=0)
    vr = var_h / (float(vr_horizon) * var_1)
    mean_reverting = vr < float(vr_threshold)

    # The regime gate licenses-or-blocks the fade; it never flips to a
    # momentum side. VR >= threshold (random walk / trending) suppresses both.
    long_signal = band_long & mean_reverting
    short_signal = band_short & mean_reverting

    # Hard stop distance = stop_sigma_mult * σ, measured from the entry Close.
    # NaN during warm-up (and 0 in a flat window) fails the walk's stop_ok gate,
    # so no position can open there.
    stop_distance = float(stop_sigma_mult) * sigma

    # Profit target = the trailing mean μ. The walk reads only μ's value at the
    # actual entry bar and resolves direction from it (long: μ > entry at the
    # lower band; short: μ < entry at the upper band), so one series serves
    # both sides.
    target_price = mu

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
    # band_lookback is a plain int because it IS a genuine bar-count lookback
    # and must feed wfo_engine's warm-up buffer. entry_z and stop_sigma_mult
    # are floats because they are σ-multiples (not lookbacks) and must NOT feed
    # the buffer; all three are grid-searched ({20,40,60,80}, {2.0,2.5,3.0},
    # {1.0,1.5,2.0,2.5}). vr_horizon/vr_lookback are fixed plain-int lookbacks
    # (12 / 96) and vr_threshold is a fixed float gate (1.0) — none of the
    # three are ever grid-searched, but they are threaded through build_grid()
    # as fixed params so _max_lookback_bars() accounts for the 108-bar VR
    # warm-up. `session` is a fixed param. These are also the concrete set the
    # sanity checker runs generate_positions() against.
    "band_lookback": 20,
    "entry_z": 2.0,
    "stop_sigma_mult": 2.0,
    "session": "New York",
    "vr_horizon": VR_HORIZON,
    "vr_lookback": VR_LOOKBACK,
    "vr_threshold": VR_THRESHOLD,
}
