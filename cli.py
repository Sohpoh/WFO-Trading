"""Command-line walk-forward runner — the non-interactive twin of app.py.

Runs the exact same engine (wfo_engine / metrics / session / strategy) the
Streamlit app uses, so results here always match the UI for the same inputs.
Useful for batch runs, cron/CI, or piping results into other tools instead of
clicking through the sidebar every time.

Examples:
    # the strategy's intended config (NQ 15min, New York session). NQ over ES
    # on purpose: ES's smaller overnight moves would starve the 0.40% floor on
    # --min-gap-pct. Every searched param is a float, so wfo_engine's warm-up
    # buffer comes from the one-day floor (~97 bars at 15min) and its fold-skip
    # guard drops to 10 bars of train window — unlike previous iterations there
    # is no timeframe that silently skips every fold.
    python cli.py --symbol NQ --timeframe 15min --train-weeks 12 --test-weeks 3

    # override strategy/grid + walk-forward schedule
    python cli.py --symbol NQ --timeframe 15min --session "New York" \\
        --min-gap-pct 0.004,0.006,0.009,0.013 --stop-gap-frac 0.5,0.75,1.0 \\
        --target-frac 0.75,1.0 --train-weeks 12 --test-weeks 3

    # yfinance source, daily bars, no session filter
    python cli.py --source yfinance --symbol NQ=F --timeframe 1d

    # save fold table / trade logs / equity curves to CSV
    python cli.py --out-dir results/run1

    # list what local data is available, then exit
    python cli.py --list
"""
import argparse
import sys

import pandas as pd

import streamlit.logger
streamlit.logger.set_log_level("ERROR")  # silence st.cache_data's "no ScriptRunContext" noise outside `streamlit run`

from data_loader import BARS_PER_YEAR, list_local_symbols, list_local_timeframes, load_data
from metrics import equity_curve, summarize
from session import SESSION_CONFIG
from wfo_engine import build_grid, run_retail_insample, run_walk_forward


def parse_num_list(text: str, cast=float) -> list:
    vals = sorted({cast(x.strip()) for x in text.split(",") if x.strip()})
    if not vals:
        raise ValueError(f"could not parse number list from {text!r}")
    return vals


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Walk-forward test the strategy in strategy.py from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--list", action="store_true", help="list local symbols/timeframes and exit")

    data = p.add_argument_group("data")
    data.add_argument("--source", choices=["local", "yfinance"], default="local")
    data.add_argument("--symbol", default="NQ", help="local symbol (NQ/ES) or yfinance ticker (e.g. NQ=F)")
    data.add_argument("--timeframe", default="1h", help="1min/5min/15min/1h/4h/1d (local) or 1h/1d (yfinance)")
    data.add_argument("--date-from", default=None, help="YYYY-MM-DD, defaults to earliest available")
    data.add_argument("--date-to", default=None, help="YYYY-MM-DD, defaults to latest available")

    strat = p.add_argument_group(
        "strategy grid (long-only swing-horizon pullback buy: a bar arms a LONG when BOTH "
        "(a) Close is above its --trend-ma bar simple moving average, i.e. the multi-week "
        "uptrend is intact, and (b) Close sits at least --dip-frac trailing-daily-ranges "
        "BELOW the highest High of the previous --dip-lookback bars, i.e. the pullback is "
        "deep enough to be worth buying. The volatility unit is the 480-bar average of the "
        "trailing 96-bar high-low range (strategy.RANGE_WINDOW / strategy.VOL_WINDOW, "
        "module constants that are deliberately NOT searched, so depth keeps exactly one "
        "degree of freedom). Both rolling extrema are shifted one bar so the threshold is "
        "never self-referential on a spike bar. There is NO short leg — a non-qualifying "
        "day pays zero cost legs instead of the two a short would cost. The exit is a "
        "plain hold to the session flatten via session.apply_session_constraint: NO stop "
        "and NO target, upside uncapped, duration bounded by the session, one round trip "
        "per armed session. The arming condition is a dense STATE, not a crossing — a "
        "pullback can deepen overnight, outside the session — and because raw entries are "
        "NaN rather than False on non-qualifying bars, an open position is HELD through "
        "them until session.py force-flattens on the session's last bar. NOTE: --trend-ma "
        "and --dip-lookback are genuine bar counts and size wfo_engine's warm-up buffer, "
        "so this grid is timeframe-sensitive — see --trend-ma)"
    )
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--trend-ma", default="960,1440,1920",
                        help="comma-separated simple-moving-average lengths IN BARS for the "
                             "uptrend gate; Close must be above the SMA for a bar to arm. At "
                             "15min these are ~10/15/20 UTC days — deliberately slow, since the "
                             "one clean robust run in this repo had its two slowest formation "
                             "values take 41 of 48 folds. Parsed as ints ON PURPOSE: they are "
                             "real lookbacks and are meant to feed wfo_engine's warm-up buffer. "
                             "TIMEFRAME WARNING: at the 1920 top of this grid the buffer is "
                             "(1920+5)*3 = 5,775 bars and run_walk_forward()'s fold-skip guard "
                             "becomes len(train_df) < 1930. A 12-week train window is ~7,700 "
                             "bars at 15min (fine, and that is the intended timeframe) but only "
                             "~1,930 bars at 1h — right on the guard, so folds get SILENTLY "
                             "SKIPPED there — and far below it at 4h/1d, where every fold is "
                             "skipped. Always pass --timeframe 15min")
    strat.add_argument("--dip-lookback", default="96,192,288",
                        help="comma-separated lookbacks IN BARS for the rolling high that "
                             "pullback depth is measured down from (the high is shifted one bar "
                             "so the current bar's own High cannot set its own reference). At "
                             "15min these are ~1/2/3 UTC days. Parsed as ints ON PURPOSE — same "
                             "reason as --trend-ma. May be longer or shorter than --trend-ma; "
                             "both orderings are meaningful and no combination is degenerate")
    strat.add_argument("--dip-frac", default="0.5,0.75,1.0,1.5",
                        help="comma-separated pullback depths, in units of the trailing average "
                             "daily range (0.5 = half a normal day of movement below the rolling "
                             "high). Floats, NOT bar counts, so they must not feed the warm-up "
                             "buffer. The 0.5 floor is the cost discipline: measured on the local "
                             "NQ 15min frame that is a ~0.95%% pullback at the median against "
                             "metrics.py's ~10.2bps round-trip toll (~9x), on a single round trip "
                             "held across an RTH range of ~1.0-1.3%%. Lowering it below 0.5 gives "
                             "that margin away. Note the top of this range is thin AND is not "
                             "screened out by the engine: at 1.5 with --trend-ma 960 "
                             "--dip-lookback 96 the strategy arms on only 11 days in four years, "
                             "and wfo_engine's guard rejects a combo only below 2 position "
                             "changes (= one complete round trip), so a single lucky train trade "
                             "can still win a fold. On a 2024-only 12w/3w run, dip_frac 1.5 won "
                             "8 of 13 folds and 6 of 13 folds took zero OOS trades")

    wfo = p.add_argument_group("walk-forward schedule")
    wfo.add_argument("--train-weeks", type=int, default=12)
    wfo.add_argument("--test-weeks", type=int, default=3)

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", default=None, help="if set, write fold_table.csv, oos_trades.csv, "
                      "insample_trades.csv, equity_curves.csv into this directory")
    out.add_argument("--quiet", action="store_true", help="suppress per-fold progress lines")
    return p


def print_progress(completed, total, fold):
    print(
        f"  fold {completed}/{total} — train {fold.train_start.date()}..{fold.train_end.date()} "
        f"-> test {fold.test_start.date()}..{fold.test_end.date()}",
        file=sys.stderr,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for sym in list_local_symbols():
            tfs = ", ".join(list_local_timeframes(sym))
            print(f"{sym}: {tfs}")
        return 0

    try:
        raw_df = load_data(args.source, args.symbol, args.timeframe)
    except Exception as e:
        print(f"error loading data for {args.symbol}: {e}", file=sys.stderr)
        return 1

    date_from = pd.Timestamp(args.date_from, tz="UTC") if args.date_from else raw_df.index.min()
    date_to = (pd.Timestamp(args.date_to, tz="UTC") + pd.Timedelta(days=1)) if args.date_to else (
        raw_df.index.max() + pd.Timedelta(days=1)
    )
    df = raw_df[(raw_df.index >= date_from) & (raw_df.index < date_to)]
    if df.empty:
        print("error: no bars in the selected date range", file=sys.stderr)
        return 1

    session = None if (args.session == "none" or args.timeframe == "1d") else args.session
    if args.timeframe == "1d" and args.session != "none":
        print("note: 1d bars ignore session filtering (one price per day) — running with session=None",
              file=sys.stderr)

    try:
        # trend_ma / dip_lookback are cast to int (genuine bar-count lookbacks
        # that are meant to size wfo_engine's warm-up buffer); dip_frac is a
        # float fraction of the volatility unit and must never feed it.
        trend_mas = parse_num_list(args.trend_ma, int)
        dip_lookbacks = parse_num_list(args.dip_lookback, int)
        dip_fracs = parse_num_list(args.dip_frac, float)
        grid = build_grid(trend_mas, dip_lookbacks, dip_fracs, session)
    except ValueError as e:
        print(f"error parsing strategy params: {e}", file=sys.stderr)
        return 1

    if not grid:
        print("error: parameter grid is empty", file=sys.stderr)
        return 1

    ann_factor = BARS_PER_YEAR.get(args.timeframe, 252)

    print(
        f"{args.symbol} · {args.timeframe} · {len(df):,} bars ({df.index.min().date()} -> {df.index.max().date()}) "
        f"· session={session or 'none'} · grid={len(grid)} combos · train={args.train_weeks}w / test={args.test_weeks}w",
        file=sys.stderr,
    )

    callback = None if args.quiet else print_progress
    try:
        oos_returns, oos_trades, folds = run_walk_forward(
            df, args.train_weeks, args.test_weeks, grid, ann_factor, progress_callback=callback
        )
        insample_returns, insample_trades, insample_params = run_retail_insample(df, grid, ann_factor)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    insample_stats = summarize(insample_returns, insample_trades, ann_factor)
    oos_stats = summarize(oos_returns, oos_trades, ann_factor)
    comparison = pd.DataFrame({"Retail In-Sample": insample_stats, "Walk-Forward OOS": oos_stats})

    print()
    print(f"Folds: {len([f for f in folds if f.best_params])}/{len(folds)} produced a valid parameter set")
    print()
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        print(comparison.to_string())
    print()
    print(f"Retail in-sample params (single curve-fit set): {insample_params}")

    chosen = [f.best_params for f in folds if f.best_params]
    if chosen:
        # Only the three grid-searched params are reported here; `session` is
        # fixed across every combo, so its "distinct values" would always be 1
        # and carry no stability information. (strategy.RANGE_WINDOW and
        # strategy.VOL_WINDOW are module constants, not params, so they never
        # appear in best_params.)
        trend_ma_vals = sorted({p["trend_ma"] for p in chosen})
        dip_lookback_vals = sorted({p["dip_lookback"] for p in chosen})
        dip_frac_vals = sorted({p["dip_frac"] for p in chosen})
        print(
            f"Fold param stability: {len(trend_ma_vals)} distinct trend ma "
            f"{trend_ma_vals}, {len(dip_lookback_vals)} distinct dip lookback "
            f"{dip_lookback_vals}, {len(dip_frac_vals)} distinct dip frac "
            f"{dip_frac_vals}"
        )

    # Per-fold OOS consistency: computed directly from each fold's own stitched
    # test-window trades (see wfo_engine.Fold), not the OOS/Retail-IS ratio —
    # that ratio divides by a single whole-period curve fit and degenerates
    # whenever that fit's CAGR is small or negative. "Active" folds are those
    # that actually took a trade; folds with a valid param set but zero trades
    # are excluded from the denominator rather than counted as a loss.
    active_folds = [f for f in folds if f.best_params and f.n_oos_trades > 0]
    if active_folds:
        fold_rets = pd.Series([f.oos_return for f in active_folds])
        fold_sharpes = pd.Series([f.oos_sharpe for f in active_folds])
        n_profitable = int((fold_rets > 0).sum())
        idle_note = (
            f" ({len(folds) - len(active_folds)} folds took no trades)"
            if len(active_folds) < len(folds) else ""
        )
        print(
            f"Fold OOS consistency: {n_profitable}/{len(active_folds)} active folds profitable "
            f"({n_profitable / len(active_folds) * 100:.1f}%), median fold OOS return "
            f"{fold_rets.median() * 100:.2f}%, median fold OOS Sharpe {fold_sharpes.median():.2f}"
            f"{idle_note}"
        )

    if args.out_dir:
        import os
        os.makedirs(args.out_dir, exist_ok=True)

        fold_rows = [
            {
                "fold": f.index + 1,
                "train_start": f.train_start, "train_end": f.train_end,
                "test_start": f.test_start, "test_end": f.test_end,
                "trend_ma": f.best_params.get("trend_ma"),
                "dip_lookback": f.best_params.get("dip_lookback"),
                "dip_frac": f.best_params.get("dip_frac"),
                "train_sharpe": f.train_sharpe,
                "test_bars": f.n_test_bars,
                "oos_trades": f.n_oos_trades,
                "oos_return": f.oos_return,
                "oos_sharpe": f.oos_sharpe,
            }
            for f in folds if f.best_params
        ]
        pd.DataFrame(fold_rows).to_csv(os.path.join(args.out_dir, "fold_table.csv"), index=False)
        oos_trades.to_csv(os.path.join(args.out_dir, "oos_trades.csv"), index=False)
        insample_trades.to_csv(os.path.join(args.out_dir, "insample_trades.csv"), index=False)

        eq = pd.DataFrame(
            {
                "insample_equity": equity_curve(insample_returns, start=100.0),
                "oos_equity": equity_curve(oos_returns, start=100.0),
            }
        )
        eq.to_csv(os.path.join(args.out_dir, "equity_curves.csv"))
        comparison.to_csv(os.path.join(args.out_dir, "comparison.csv"))

        print(f"\nWrote fold_table.csv, oos_trades.csv, insample_trades.csv, equity_curves.csv, "
              f"comparison.csv -> {args.out_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
