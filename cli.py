"""Command-line walk-forward runner — the non-interactive twin of app.py.

Runs the exact same engine (wfo_engine / metrics / session / strategy) the
Streamlit app uses, so results here always match the UI for the same inputs.
Useful for batch runs, cron/CI, or piping results into other tools instead of
clicking through the sidebar every time.

Examples:
    # quick run with defaults (NQ, 1h, New York session)
    python cli.py

    # override strategy/grid + walk-forward schedule
    python cli.py --symbol ES --timeframe 15min --session "New York" \\
        --band-lookback 20,30 --entry-z 2.0,2.5 --stop-sigma-mult 1.0,1.5 \\
        --train-weeks 24 --test-weeks 8

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

    strat = p.add_argument_group("strategy grid (Bollinger mid-band fade)")
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--band-lookback", default="20,30,40,60",
                        help="comma-separated Bollinger lookback bars (windows both the mid-band and its sigma)")
    strat.add_argument("--entry-z", default="2.0,2.5,3.0",
                        help="comma-separated sigma thresholds the z-score must cross to fade the band")
    strat.add_argument("--stop-sigma-mult", default="1.0,1.5,2.0,2.5",
                        help="comma-separated stop-distance sigma multipliers (target is the mid-band)")

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
        lookback_vals = sorted({p["band_lookback"] for p in chosen})
        entry_z_vals = sorted({p["entry_z"] for p in chosen})
        stop_vals = sorted({p["stop_sigma_mult"] for p in chosen})
        print(
            f"Fold param stability: {len(lookback_vals)} distinct band lookback {lookback_vals}, "
            f"{len(entry_z_vals)} distinct entry z {entry_z_vals}, "
            f"{len(stop_vals)} distinct stop sigma mult {stop_vals}"
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
