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
        "strategy grid (long-only trend-gated Hurst drift momentum: a bar is LONG only "
        "while ALL THREE gates hold — (1) the multi-scale variance-growth Hurst estimate "
        "of the trailing --hurst-window bars is strictly above --h-threshold (H > 0.5 is "
        "the canonical persistence boundary: variance grows superlinearly, i.e. "
        "trending), (2) the backward --drift-lookback-bar drift "
        "Close[t-1]/Close[t-1-drift_lookback]-1 is positive, and (3) Close_t is above "
        "its 960-bar SMA — a hardcoded zero-param higher-timeframe uptrend gate (NOT "
        "grid-searched). There is NO short branch: a negative or zero drift is an "
        "off-bar, never -1. Hurst is estimated per bar by regressing "
        "log(Var(tau-bar log returns)) on log(tau) for tau in {1,2,4,8,16,32}, "
        "H = slope/2, over OVERLAPPING tau-returns, shifted one bar so bar t sees only "
        "data strictly before t; the naive overlapping estimator carries a known scale "
        "bias, which is exactly why --h-threshold is gridded rather than hardcoded. "
        "NO entry threshold on price magnitude, NO stop and NO target. The exit is a "
        "FLIP-TO-FLAT — raw entries carry +1.0 while all three gates are on and 0.0 "
        "while any is known-off (H <= h_threshold, drift <= 0, or Close <= SMA(960)), "
        "so session.apply_session_constraint flattens on the first off bar and stays "
        "flat until a fresh on-state re-arms; NaN (unwarmed) bars hold. Because there "
        "is no -1 branch, a gate turning off can only close/re-open a long, never "
        "reverse long<->short. session.py force-flattens on the session's last bar. "
        "NOTE: --hurst-window and --drift-lookback are genuine bar counts and size "
        "wfo_engine's warm-up buffer — see --hurst-window)"
    )
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--hurst-window", default="384,768,1152",
                        help="comma-separated trailing-bar windows IN BARS over which the "
                             "multi-scale variance-growth Hurst estimate is computed "
                             "(384/768/1152 ~ 4/8/12 UTC days at 15min). Parsed as ints ON "
                             "PURPOSE: a genuine lookback meant to feed wfo_engine's "
                             "warm-up buffer. TIMEFRAME WARNING: at the 1152 top of this "
                             "grid the buffer is (1152+5)*3 = 3,471 bars and "
                             "run_walk_forward()'s fold-skip guard becomes len(train_df) < "
                             "1162. A 12-week train window is ~7,700 bars at 15min (fine, "
                             "and that is the intended timeframe) and ~2,016 bars at 1h "
                             "(above the guard, so folds still run, but with the heavy "
                             "buffer); at 4h/1d every fold is skipped. "
                             "Always pass --timeframe 15min")
    strat.add_argument("--drift-lookback", default="96,192,384",
                        help="comma-separated lookbacks IN BARS for the backward drift "
                             "Close[t-1]/Close[t-1-lookback]-1, which must be POSITIVE "
                             "to arm a long (drift <= 0 is an off-bar, never a short — "
                             "this iteration is long-only) (96/192/384 ~ 1/2/4 UTC days "
                             "at 15min). Parsed as ints ON PURPOSE — same reason as "
                             "--hurst-window. May be longer or shorter than --hurst-window; "
                             "both orderings are meaningful and no combination is degenerate")
    strat.add_argument("--h-threshold", default="0.5,0.55,0.6",
                        help="comma-separated Hurst gates: the state is on only while the "
                             "estimated H is strictly above this value. 0.5 is the "
                             "canonical persistence boundary (H > 0.5 = trending, variance "
                             "grows superlinearly; H < 0.5 = mean-reverting); 0.55/0.6 are "
                             "safety margins against the naive overlapping estimator's "
                             "known scale bias. Floats, NOT bar counts, so they must not "
                             "feed the warm-up buffer")

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
        # hurst_window / drift_lookback are cast to int (genuine bar-count
        # lookbacks that are meant to size wfo_engine's warm-up buffer);
        # h_threshold is a float Hurst gate and must never feed it.
        hurst_windows = parse_num_list(args.hurst_window, int)
        drift_lookbacks = parse_num_list(args.drift_lookback, int)
        h_thresholds = parse_num_list(args.h_threshold, float)
        grid = build_grid(hurst_windows, drift_lookbacks, h_thresholds, session)
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
        # and carry no stability information. (strategy.HURST_TAUS is a module
        # constant, not a param, so it never appears in best_params.)
        hurst_window_vals = sorted({p["hurst_window"] for p in chosen})
        drift_lookback_vals = sorted({p["drift_lookback"] for p in chosen})
        h_threshold_vals = sorted({p["h_threshold"] for p in chosen})
        print(
            f"Fold param stability: {len(hurst_window_vals)} distinct hurst window "
            f"{hurst_window_vals}, {len(drift_lookback_vals)} distinct drift lookback "
            f"{drift_lookback_vals}, {len(h_threshold_vals)} distinct h threshold "
            f"{h_threshold_vals}"
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
                "hurst_window": f.best_params.get("hurst_window"),
                "drift_lookback": f.best_params.get("drift_lookback"),
                "h_threshold": f.best_params.get("h_threshold"),
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
