"""Prior-day close-location reversal, held for one session.

Every prior iteration in this project measured its signal on a rolling window
that *includes* the bar it trades. This one does the opposite: the entire
signal is a statistic of the **previous completed UTC calendar day**, frozen
before the session opens, and it is faded.

The thesis is transplanted, and is flagged as such rather than presented as a
base rate. Cross-sectional momentum work skips the most recent completed
period ("S = skip period, usually 1 month, to avoid microstructure mean
reversion") precisely because that period is asserted to *reverse*; the
short-term-overreaction / forced-buying-and-selling literature supplies the
mechanism. Both claims are about individual equities at a monthly horizon.
Index futures at a daily horizon are much closer to a random walk, so this is
an analogy being tested, not a documented effect being harvested.

What makes this a different *shape* rather than a 31st indicator is the
holding geometry, not the statistic:

  - The signal is fully determined ~13.5h before the New York open, so there
    is no intraday event to chase and no extremity condition to select on.
  - The bias is constant for the whole day, so no opposing signal can fire
    intraday. Every trade is therefore closed by `session.py`'s forced
    flatten on the session's last bar: exactly one round trip per qualifying
    session, ~6h of hold (09:45 -> 15:45 ET at 15min bars).
  - One round trip means `metrics.py`'s ~10.2bps toll is charged once against
    a full open-to-close excursion, rather than once per intraday event.

Rules (all computed on the full continuous frame with no session awareness of
their own; every window is strictly backward with strict `min_periods`, so
unwarmed bars are NaN and the signal fails *closed*):

  - Daily aggregation, grouped by **UTC calendar day** (`index.normalize()`):

        High_d  = max(High) over day d
        Low_d   = min(Low)  over day d
        Close_d = last Close of day d
        Range_d = High_d - Low_d
        CL_d    = (Close_d - Low_d) / Range_d      (NaN where Range_d <= 0)

  - Range baseline, a strictly-warm 14-day mean of Range over the completed
    days ending at d (`min_periods=RANGE_BASELINE_DAYS`, so the first 13 days
    of any slice are NaN and disqualify):

        Baseline_d = mean(Range over the 14 days ending at d)

    Note that the window is inclusive of d, exactly as specced, so after the
    freeze `Baseline_{d-1}` contains `Range_{d-1}` as 1/14 of itself. A wide
    day therefore partly inflates its own threshold, compressing the effective
    qualifier toward 1.0. This is not an off-by-one — it is the specced
    "14 completed days ending at d" — but it should be kept in mind when
    reading where `range_mult` pins.

  - Freeze. Every bar of day d reads the statistics of day d-1 via a single
    positional `.shift(1)` on the *daily* frame. This is a strict one-day
    lag with no lookahead: day d-1 has fully closed (00:00 UTC) more than
    13h before the 09:30 ET open that lives inside day d.

  - Qualification (the volatility filter):

        Range_{d-1} >= range_mult * Baseline_{d-1}

    NaN on either side makes the comparison False, so an unwarmed day is
    disqualified for free.

  - Bias, given qualification — this is the fade, and the one place a sign
    slip would be silent rather than loud:

        CL_{d-1} >= cl_threshold        -> SHORT  (closed near the high)
        CL_{d-1} <= 1 - cl_threshold    -> LONG   (closed near the low)
        otherwise                       -> NaN    (no bias, stand aside)

    Mutually exclusive by construction for any `cl_threshold > 0.5`, which
    the whole intended grid satisfies. That bias is written onto *every* bar
    of day d; `session.py` alone decides which of those bars are tradable, so
    the position opens on the session's first bar and no session logic enters
    this module.

Exit is a plain flip with no stop, no target and no path dependence, so this
delegates to `session.apply_session_constraint()`. Non-qualifying days emit
NaN throughout, and the ffilled 0.0 left by the previous session's forced
flatten keeps the strategy flat across them.

ANCHOR CAVEAT (named deliberately, not hidden). Grouping by UTC calendar day
is an *approximation* of the cash-session statistic the order-flow intuition
describes. Two consequences, both accepted rather than patched:

  1. The "prior close" is the last bar before 00:00 UTC, i.e. ~19:45 ET, in
     thin trade — not the 16:00 ET cash close. The prior High/Low likewise
     include overnight extremes. Fixing this would require session awareness
     inside `strategy.py`, which the contract forbids.

  2. CME's Sunday 18:00 ET open lands in UTC Sunday, so UTC Sunday is a real
     group holding only ~8 bars at 15min (~2 at 1h). Two knock-on effects,
     both real and both left in place because the spec says UTC calendar
     days: every Monday session reads that ~2h stub as its "prior day", whose
     tiny Range almost never clears the qualifier, so Mondays are
     systematically disqualified; and those stub ranges enter the 14-day
     Baseline and deflate it, so other days clear `range_mult` more easily
     than the nominal threshold suggests. In particular this means a
     `range_mult` pin at the grid's low end should be read as "the qualifier
     is inert", not as evidence about a genuine boundary. No min-bars-per-day
     filter is applied — that would be a rule the spec does not have.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):

  - `cl_threshold` and `range_mult` are unitless (a [0,1] location and a ratio
    of ranges), never bar counts, so both are passed as `float` and are
    correctly ignored by the warm-up buffer sizing.

  - `history_bars` is a plain `int` and is deliberately **not read by any
    computation in this module**. It exists solely to reach
    `_max_lookback_bars()`, which scans every int-valued entry of every combo
    dict, and `build_grid()` threads it into every combo. This is the only
    lever this codebase has for telling the engine "my indicator is
    day-anchored and needs ~N bars of history": the engine's own
    `_bars_per_day()` floor only guarantees *one* day, while the Baseline
    needs 14 completed days plus the 1-day freeze shift.

      DO NOT DELETE IT AS DEAD CODE. Removing the kwarg (or casting it to
      float) collapses `buffer_bars` to `max(5*3, bars_per_day + 5)` = ~97
      bars at NQ 15min, i.e. about one day of history, which leaves
      `Baseline` NaN for the first ~15 days of every test window. That fails
      *closed* (no trades there, not wrong trades) but silently guts the
      strategy.

      Sizing at NQ 15min: buffer_bars = max((1440 + 5) * 3, 92 + 5) = 4335
      bars ~= 47 trading days, comfortably covering the 15 days of reach.

      COVERAGE HAZARD IN THE OTHER DIRECTION. `run_walk_forward()` skips any
      fold whose train window holds fewer than `max_lookback + 10` = 1450
      bars. A 12-week train window is ~5492 bars at 15min (fine) but only
      ~1373 bars at 1h and ~372 at 4h — both below the guard, so on those
      timeframes *every* fold is skipped and the run prints "Folds: 0/N" with
      an empty OOS column beside a populated in-sample one. Lower
      `--history-bars` for coarser timeframes, and check the Baseline still
      warms: measured 12-week train windows are 5492 bars at 15min, 1373 at
      1h and 372 at 4h, so use ~400 at 1h (guard 410, buffer 1215 bars ~= 53
      days at 23 bars/day) and ~100 at 4h (guard 110, buffer 315 bars ~= 52
      days at 6 bars/day). 400 at 4h would *itself* trip the guard.

  - `RANGE_BASELINE_DAYS` is a module constant, not a param, so it never
    enters a grid combo and cannot touch the buffer either way.

Cost note: `metrics.py`'s ~0.102% round trip (0.001% fee + 0.05% slippage per
leg, 2 legs) is unchanged and out of scope. With one round trip per session
against a ~6h NQ excursion, that toll is a far smaller fraction of the bet
than it is for event-triggered intraday entries — which is the point of the
holding geometry, not an assumption baked into the cost model.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to `session.py`;
see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Length of the daily-Range baseline, in completed UTC days. Hardcoded rather
# than grid-searched so the search tunes the fade itself (where the close sits
# in the prior day's range, and how big that range had to be) and cannot pick
# a volatility yardstick that happens to flatter its own sample. Being a
# module constant it never enters a grid combo, so it cannot reach
# `wfo_engine._max_lookback_bars()` — see `history_bars` for how the warm-up
# buffer is actually sized.
RANGE_BASELINE_DAYS = 14


def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-UTC-day High/Low/Close/Range/CL/Baseline, indexed by UTC midnight.

    `index.normalize()` on the frame's tz-aware UTC index is what defines a
    "day" here — see the ANCHOR CAVEAT in the module docstring for what that
    does and does not correspond to.

    `CL` is NaN wherever `Range <= 0` (a zero-range day carries no location
    information and would divide by zero). `Baseline` uses a strict
    `min_periods`, so the first `RANGE_BASELINE_DAYS - 1` days of any slice
    are NaN and cannot qualify.

    Every column here is a statistic *of* day d, not yet lagged — the one-day
    freeze is applied by the caller.
    """
    day = df.index.normalize()
    grouped = df.groupby(day)

    high = grouped["High"].max()
    low = grouped["Low"].min()
    close = grouped["Close"].last()

    rng = high - low
    cl = ((close - low) / rng).where(rng > 0)
    baseline = rng.rolling(RANGE_BASELINE_DAYS, min_periods=RANGE_BASELINE_DAYS).mean()

    return pd.DataFrame({"Range": rng, "CL": cl, "Baseline": baseline})


def generate_positions(
    df: pd.DataFrame,
    cl_threshold: float,
    range_mult: float,
    history_bars: int = 1440,
    session: str | None = "New York",
) -> pd.Series:
    # `history_bars` is intentionally unread here — it is a warm-up-buffer
    # hint consumed by wfo_engine._max_lookback_bars(). See the module
    # docstring's "DO NOT DELETE IT AS DEAD CODE" note before touching it.
    del history_bars

    stats = daily_stats(df)

    # THE FREEZE. A single positional shift on the *daily* frame, so every
    # bar of day d reads only statistics of the previous day present in the
    # data. No lookahead: day d-1 closed at 00:00 UTC, more than 13h before
    # the 09:30 ET open that lives inside day d.
    prev = stats.shift(1)

    # Broadcast the frozen daily statistics back onto every bar of their day.
    # `reindex` on the bar-level day labels (rather than `.map`) keeps this a
    # plain float64 alignment with no Timestamp-key dtype surprises.
    day = df.index.normalize()
    prev_range = prev["Range"].reindex(day).to_numpy()
    prev_cl = prev["CL"].reindex(day).to_numpy()
    prev_baseline = prev["Baseline"].reindex(day).to_numpy()

    # Qualification: the frozen day had to be a genuinely wide day relative to
    # its own 14-day baseline. NaN on either side makes this False, so an
    # unwarmed day (or a Sunday stub whose Baseline is still cold) fails
    # closed with no explicit validity mask needed.
    with np.errstate(invalid="ignore"):
        qualified = prev_range >= float(range_mult) * prev_baseline

    # THE FADE. Closed near the prior day's high -> short it; closed near the
    # prior day's low -> long it. Mutually exclusive for any cl_threshold >
    # 0.5 (the whole intended grid). NaN CL compares False both ways.
    thr = float(cl_threshold)
    with np.errstate(invalid="ignore"):
        short_bias = qualified & (prev_cl >= thr)
        long_bias = qualified & (prev_cl <= 1.0 - thr)

    # Raw, session-unaware entries: the day's constant bias on *every* bar of
    # that day, NaN on days with no bias. Explicit float64 so the delegate's
    # ffill/fillna arithmetic stays numeric.
    entries = pd.Series(np.nan, index=df.index, dtype=float)
    entries[long_bias] = 1.0
    entries[short_bias] = -1.0

    # session.py alone decides which of those bars are tradable: it opens the
    # position on the session's first bar and force-flattens on the session's
    # last. Because the bias is constant within a day, no opposing signal can
    # fire intraday, so that flatten is the only exit — exactly one round trip
    # per qualifying session.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "cl_threshold": 0.75,
    "range_mult": 0.9,
    "history_bars": 1440,
    "session": "New York",
}
