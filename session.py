"""Day-trade session enforcement — deliberately independent of `strategy.py`.

This app trades day trades only, gated to one configurable session, no matter
what the strategy's entry/exit logic is. `strategy.py` is expected to be
rewritten often as strategies are iterated on; this module is not — it's the
one place that guarantees no position ever opens outside the chosen session
or survives past that session's close.

Contract for `strategy.py`: build a raw `entries` series (1.0 at long-entry
bars, -1.0 at short-entry bars, NaN everywhere else — session-unaware), then
call `apply_session_constraint(entries, session)` to get the final tradable
position series. That's the only integration point strategies need to know
about.
"""
import numpy as np
import pandas as pd

# Trading hours per session, all quoted in Eastern Time — the home clock for
# NQ/ES (CME/Globex) — rather than each region's own local exchange hours.
# "Asia" and "London" here mean the Eastern-Time windows when that region's
# volume typically shows up for a US-based trader, not Tokyo/London cash
# session hours. Compared via tz_convert to America/New_York (DST-safe — no
# hardcoded UTC offsets that drift with the clock change).
SESSION_CONFIG = {
    "New York": {"tz": "America/New_York", "start": "09:30", "end": "16:00"},
    "London": {"tz": "America/New_York", "start": "02:00", "end": "05:00"},
    "Asia": {"tz": "America/New_York", "start": "20:00", "end": "00:00"},
}


def session_mask(index: pd.DatetimeIndex, session: str) -> pd.Series:
    """True for bars whose local time falls inside the session's trading hours."""
    cfg = SESSION_CONFIG[session]
    local_time = index.tz_convert(cfg["tz"]).time
    start_t = pd.Timestamp(cfg["start"]).time()
    end_t = pd.Timestamp(cfg["end"]).time()
    if end_t == pd.Timestamp("00:00").time():
        # session runs to midnight (e.g. Asia 20:00-24:00) — no upper bound
        # needed since local_time can never reach/exceed 24:00
        return pd.Series(local_time >= start_t, index=index)
    return pd.Series((local_time >= start_t) & (local_time < end_t), index=index)


def apply_session_constraint(entries: pd.Series, session: str | None) -> pd.Series:
    """Turn raw (session-unaware) entry signals into a day-trade-only position series.

    entries: 1.0 / -1.0 at bars where the strategy wants to open a position in
    that direction, NaN elsewhere (no signal that bar).

    session=None means no constraint (used for 1d bars, where session
    filtering is meaningless since there's only one price per day).
    """
    if session is None:
        return entries.ffill().fillna(0.0)

    in_session = session_mask(entries.index, session)

    # drop any entry that fired outside the session
    gated = entries.where(in_session)

    # force-flatten on the last bar of every session run so no position ever
    # carries past the session's close (ffill then propagates that 0 through
    # the overnight/off-session gap into the next session)
    session_end = in_session & ~in_session.shift(-1).fillna(False).astype(bool)
    gated[session_end] = 0.0

    return gated.ffill().fillna(0.0)


def apply_session_constraint_with_stops(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    long_signal: pd.Series,
    short_signal: pd.Series,
    stop_distance: pd.Series,
    target_price: pd.Series,
    session: str | None,
) -> pd.Series:
    """Session-aware position walk for strategies whose exits are
    path-dependent (a stop-loss or profit-target level hit mid-trade), not
    just "flatten on the next opposing entry" or "flatten at session end".

    `apply_session_constraint()` above builds a position by forward-filling a
    sparse entries series — it has no way to express "exit early because
    price touched a level between entries". Doing the stop/target bookkeeping
    in `strategy.py` instead wouldn't work either: a position force-flattened
    at session close needs the *next* bar's entry evaluation to start from
    flat, and only this module knows where session boundaries fall. So this
    function owns both concerns in one bar-by-bar walk, keeping strategy.py
    session-unaware as required — it hands in raw signals/levels, this
    function is the only place that knows about session hours.

    Inputs (all raw/session-unaware, computed by the strategy from its own
    indicators):
      - close/high/low: price series.
      - long_signal / short_signal: bool Series, True where the strategy
        would open that side *if flat and in-session*. Mutually exclusive by
        construction in every strategy so far (e.g. can't close above the
        Donchian upper band and below the lower band on the same bar).
      - stop_distance: positive price-distance Series (e.g. k*ATR) measured
        from the entry price — symmetric, so the direction (above/below
        entry) is resolved here based on long vs. short.
      - target_price: absolute price level Series, already direction-
        resolved by the strategy (e.g. a long target anchored off the
        breakout level, not off entry — so this can't be expressed as a
        simple from-entry distance the way the stop can). Only the value at
        an actual entry bar is read; elsewhere it's ignored.
      NaN in either (or a target on the wrong side of that bar's close) means
      "no valid entry this bar" — e.g. indicator warm-up not complete yet.

    Fill-price caveat: a stop/target hit is *detected* using that bar's
    High/Low against the stored trigger level, but the position is flattened
    as of that bar's Close (this codebase has no intrabar price series to
    fill at the level itself, and `metrics.bar_returns_with_costs` prices
    every bar off Close). So a bar that gaps or spikes through the stop
    still books that whole bar's close-to-close return before exiting — the
    stop caps *when* you exit, not the realized loss on the triggering bar.
    Don't describe results as guaranteeing a max loss of stop_distance.

    session=None means no session constraint at all (used for 1d bars) —
    entries are gated only by being flat and having a valid signal, and
    there's no forced end-of-session flatten.
    """
    n = len(close)
    if session is None:
        in_session = np.ones(n, dtype=bool)
        session_end = np.zeros(n, dtype=bool)
    else:
        in_session_s = session_mask(close.index, session)
        in_session = in_session_s.to_numpy()
        session_end = (in_session_s & ~in_session_s.shift(-1).fillna(False).astype(bool)).to_numpy()

    close_v = close.to_numpy()
    high_v = high.to_numpy()
    low_v = low.to_numpy()
    long_v = long_signal.to_numpy()
    short_v = short_signal.to_numpy()
    stop_dist_v = stop_distance.to_numpy()
    target_price_v = target_price.to_numpy()

    position = np.zeros(n)
    pos = 0
    stop_price = target_state = np.nan

    for i in range(n):
        if pos != 0:
            if pos == 1:
                hit_stop = low_v[i] <= stop_price
                hit_target = high_v[i] >= target_state
            else:
                hit_stop = high_v[i] >= stop_price
                hit_target = low_v[i] <= target_state
            if hit_stop or hit_target:
                pos = 0
                stop_price = target_state = np.nan
                position[i] = 0
                # deliberately no same-bar re-entry after a stop/target exit
                # — keeps the exit visible to metrics.extract_trades (which
                # walks position *changes*) instead of masking it as one
                # continuous trade with zero cost legs charged
                continue

        if session_end[i]:
            pos = 0
            stop_price = target_state = np.nan
            position[i] = 0
            continue

        if pos == 0 and in_session[i]:
            stop_ok = stop_dist_v[i] > 0 and not np.isnan(stop_dist_v[i])
            tgt_ok = not np.isnan(target_price_v[i])
            if stop_ok and tgt_ok and long_v[i] and target_price_v[i] > close_v[i]:
                pos = 1
                entry = close_v[i]
                stop_price = entry - stop_dist_v[i]
                target_state = target_price_v[i]
            elif stop_ok and tgt_ok and short_v[i] and target_price_v[i] < close_v[i]:
                pos = -1
                entry = close_v[i]
                stop_price = entry + stop_dist_v[i]
                target_state = target_price_v[i]

        position[i] = pos

    return pd.Series(position, index=close.index)
