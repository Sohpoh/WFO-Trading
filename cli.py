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
        "strategy grid (long-only Kalman-filter trend momentum: a local-linear-trend "
        "Kalman filter over log(Close) recursively estimates a level and a per-bar "
        "slope — slope_{t+1} = slope_t + w (process noise sigma_w), y_t = level_t + v "
        "(observation noise sigma_v). Only the noise RATIO sigma_w/sigma_v is searched "
        "(--kf-noise-ratio); sigma_v itself is estimated from the filter's residual as "
        "an EMA of squared innovations, so slope/sigma_v is scale-free. LONG where "
        "slope/sigma_v > --min-slope; there is NO short leg, so a down-state pays no "
        "legs at all. Exit is a path-dependent ATR hard stop (entry − --stop-atr-mult × "
        "ATR(14)) with NO profit target, so winners run uncapped to session.py's forced "
        "flatten; the ATR window is fixed at 14 (a genuine bar-count lookback, not "
        "grid-searched, so there is no flag for it). The noise-estimation and warm-up "
        "machinery are module constants in strategy.py and are NOT searched, so there "
        "is no flag for them)"
    )
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--kf-noise-ratio", default="0.5,1.0,2.0",
                        help="comma-separated Kalman noise ratios sigma_w/sigma_v (process "
                             "noise over observation noise). Larger = the slope re-rates "
                             "faster and tracks recent turns more tightly (shorter effective "
                             "memory); smaller = smoother, slower-to-adapt slope. Passed as "
                             "floats and correctly ignored by the warm-up sizing")
    strat.add_argument("--min-slope", default="0.25,0.5,1.0,2.0",
                        help="comma-separated dimensionless per-bar slope thresholds. A bar "
                             "goes long when the Kalman slope, normalized by the estimated "
                             "observation noise sigma_v, is strictly above this. Passed as "
                             "floats and correctly ignored by the warm-up sizing")
    strat.add_argument("--stop-atr-mult", default="1.0,1.5,2.0,3.0",
                        help="comma-separated ATR multipliers on the hard stop's distance "
                             "from entry (entry − mult × ATR(14), detected on the intrabar "
                             "Low, flattened at that bar's Close). Searched so each fold "
                             "re-fits the stop width to the prevailing volatility regime. "
                             "Passed as floats and correctly ignored by the warm-up sizing")

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
        # kf_noise_ratio, min_slope, and stop_atr_mult are all floats: none is
        # a bar-count lookback, so none must feed wfo_engine's warm-up buffer.
        # atr_period IS a genuine bar-count lookback (fixed at 14, not
        # grid-searched), so it is threaded as a plain int and correctly feeds
        # _max_lookback_bars().
        kf_noise_ratios = parse_num_list(args.kf_noise_ratio, float)
        min_slopes = parse_num_list(args.min_slope, float)
        stop_atr_mults = parse_num_list(args.stop_atr_mult, float)
        atr_period = 14
        grid = build_grid(kf_noise_ratios, min_slopes, stop_atr_mults, atr_period, session)
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
        # Only the three grid-searched params are reported here; `session` and
        # `atr_period` are fixed across every combo, so their "distinct values"
        # would always be 1 and carry no stability information.
        # (strategy.SIGMA_EMA_SPAN etc. are module constants, not params, so
        # they never appear in best_params.)
        kf_noise_ratio_vals = sorted({p["kf_noise_ratio"] for p in chosen})
        min_slope_vals = sorted({p["min_slope"] for p in chosen})
        stop_atr_mult_vals = sorted({p["stop_atr_mult"] for p in chosen})
        print(
            f"Fold param stability: {len(kf_noise_ratio_vals)} distinct kf noise ratio "
            f"{kf_noise_ratio_vals}, {len(min_slope_vals)} distinct min slope "
            f"{min_slope_vals}, {len(stop_atr_mult_vals)} distinct stop atr mult "
            f"{stop_atr_mult_vals}"
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
                "kf_noise_ratio": f.best_params.get("kf_noise_ratio"),
                "min_slope": f.best_params.get("min_slope"),
                "stop_atr_mult": f.best_params.get("stop_atr_mult"),
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
