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
import math
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
        "strategy grid (ES 1h Bollinger-band reversion to mean: on each bar compute the "
        "trailing mean μ and population std σ of Close over --band-lookback bars, then "
        "fade a band touch back to the middle — SHORT where Close >= μ + --entry-z·σ "
        "(upper-band touch/penetration), LONG where Close <= μ − --entry-z·σ (lower-band "
        "touch). Symmetric both directions; no opposite-signal flip. Exit is a "
        "path-dependent stop/target via apply_session_constraint_with_stops: target = the "
        "trailing mean μ (direction-resolved by session.py), stop = entry ± "
        "--stop-sigma-mult·σ, both quoted in trailing σ so they re-fit each fold to the "
        "prevailing volatility regime. The New York session force-flattens at 16:00 ET, "
        "so every trade closes within the session (day-trade only)"
    )
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--band-lookback", default="20,40,60,80",
                        help="comma-separated trailing-window bar counts for the Bollinger "
                             "mean μ and population std σ (the window ends at the current "
                             "bar, so it is causal). A genuine bar-count lookback, passed "
                             "as ints so it correctly feeds wfo_engine's warm-up buffer")
    strat.add_argument("--entry-z", default="2.0,2.5,3.0",
                        help="comma-separated σ-multiples defining the entry bands. A bar "
                             "goes short where Close >= μ + entry_z·σ and long where "
                             "Close <= μ − entry_z·σ. A dimensionless threshold (NOT a "
                             "lookback), passed as floats and correctly ignored by the "
                             "warm-up sizing")
    strat.add_argument("--stop-sigma-mult", default="1.0,1.5,2.0,2.5",
                        help="comma-separated σ-multiples on the hard stop's distance from "
                             "entry (entry ± mult·σ, detected on the intrabar High/Low, "
                             "flattened at that bar's Close). Searched so each fold "
                             "re-fits the stop width to the prevailing volatility regime. "
                             "A σ-multiple (NOT a lookback), passed as floats and correctly "
                             "ignored by the warm-up sizing")

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
        # band_lookback IS a genuine bar-count lookback, so it is parsed as int
        # and correctly feeds wfo_engine's warm-up buffer. entry_z and
        # stop_sigma_mult are σ-multiples (not lookbacks), so they are parsed
        # as float and correctly ignored by _max_lookback_bars().
        band_lookbacks = parse_num_list(args.band_lookback, int)
        entry_zs = parse_num_list(args.entry_z, float)
        stop_sigma_mults = parse_num_list(args.stop_sigma_mult, float)
        grid = build_grid(band_lookbacks, entry_zs, stop_sigma_mults, session)
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
        # and carry no stability information.
        band_lookback_vals = sorted({p["band_lookback"] for p in chosen})
        entry_z_vals = sorted({p["entry_z"] for p in chosen})
        stop_sigma_mult_vals = sorted({p["stop_sigma_mult"] for p in chosen})
        print(
            f"Fold param stability: {len(band_lookback_vals)} distinct band lookback "
            f"{band_lookback_vals}, {len(entry_z_vals)} distinct entry z "
            f"{entry_z_vals}, {len(stop_sigma_mult_vals)} distinct stop sigma mult "
            f"{stop_sigma_mult_vals}"
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
        pct = n_profitable / len(active_folds)
        se = math.sqrt(pct * (1.0 - pct) / len(active_folds))
        band = "clear pass" if pct >= 0.60 else ("near-miss" if pct >= 0.60 - se else "clear miss")
        idle_note = (
            f" ({len(folds) - len(active_folds)} folds took no trades)"
            if len(active_folds) < len(folds) else ""
        )
        print(
            f"Fold OOS consistency: {n_profitable}/{len(active_folds)} active folds profitable "
            f"({pct * 100:.1f}%, SE {se * 100:.1f}% -> {band}), median fold OOS return "
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
                "band_lookback": f.best_params.get("band_lookback"),
                "entry_z": f.best_params.get("entry_z"),
                "stop_sigma_mult": f.best_params.get("stop_sigma_mult"),
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
