"""Command-line walk-forward runner — the non-interactive twin of app.py.

Runs the exact same engine (wfo_engine / metrics / session / strategy) the
Streamlit app uses, so results here always match the UI for the same inputs.
Useful for batch runs, cron/CI, or piping results into other tools instead of
clicking through the sidebar every time.

Examples:
    # the strategy's intended config (NQ 15min, New York session)
    python cli.py --symbol NQ --timeframe 15min --train-weeks 12 --test-weeks 3

    # override strategy/grid + walk-forward schedule
    python cli.py --symbol NQ --timeframe 15min --session "New York" \\
        --formation-lookback 24,48,96,192 --rank-pct 0.80,0.875,0.925 \\
        --rank-window 960 --train-weeks 12 --test-weeks 3

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
        "strategy grid (long-only percentile-rank momentum: each bar's formation return "
        "Close_t/Close_{t-formation_lookback} - 1 is ranked against its own trailing "
        "distribution — the rank_pct quantile of the same statistic over the previous "
        "rank_window bars, current bar excluded. Top-quantile bars go LONG; there is NO "
        "short leg, so a down-state simply pays no legs at all instead of paying two to "
        "reverse. Exit is a volatility-scaled HARD STOP, walked bar-by-bar by "
        "session.apply_session_constraint_with_stops(): the stop sits 0.5 x the trailing "
        "daily range (mean over the last 960 bars of the rolling 96-bar high-low span, "
        "shifted 1) below the entry close, frozen at entry — it is a hard stop, not a "
        "trailing one. The previous iteration's momentum-decay flip is gone. There is no "
        "profit target: target_price is an unreachable Close x 2.0 that exists only to "
        "satisfy the delegate's 'target must be non-NaN and above the entry close' guard, "
        "so winners still run uncapped to session.py's forced flatten. The stop fraction "
        "and both daily-range windows are module constants in strategy.py and are NOT "
        "searched, so there is no flag for them; because a stopped-out trade can re-enter "
        "later in the same session if the rank still clears --rank-pct, a traded session "
        "is not exactly one round trip. Per session.py's fill-price caveat the stop caps "
        "WHEN you exit, not the realized loss on the triggering bar)"
    )
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--formation-lookback", default="96,192,288,384",
                        help="comma-separated momentum formation periods, in bars — how far back "
                             "Close_t is compared to when measuring the return that gets ranked. "
                             "Genuine bar-count lookbacks, passed as ints, so they feed "
                             "wfo_engine's pre-test-window warm-up buffer (though --rank-window, "
                             "being larger, is what actually sizes it). The default grid is the "
                             "SLOW end only: across the previous iteration's 48 folds the two "
                             "slowest values took 41 of them (96 in 22, grid-max 192 in 19) while "
                             "24 and 48 took 5 and 2, so the dead fast end is retired and 288/384 "
                             "are opened above the old boundary. Do not push past ~384 with "
                             "--rank-window at 960: the trailing quantile then rests on only "
                             "~960/L (~2.5 at 384) independent non-overlapping observations and "
                             "the threshold itself gets jumpy")
    strat.add_argument("--rank-pct", default="0.80,0.875,0.925",
                        help="comma-separated quantile levels in (0,1) — a bar goes long when its "
                             "formation return is strictly above the rank_pct quantile of the "
                             "trailing distribution, i.e. 0.90 means 'top decile'. Unlike every "
                             "absolute threshold used in earlier iterations this re-scales itself "
                             "with the volatility regime, so it should not churn across folds. "
                             "Passed as floats and correctly ignored by the warm-up sizing")
    strat.add_argument("--rank-window", type=int, default=960,
                        help="FIXED, never grid-searched: how many trailing bars the rank "
                             "threshold is computed over (960 = ~10 trading days of 15min bars). "
                             "As the largest int in the grid this alone sizes the warm-up buffer "
                             "to (960+5)*3 = 2895 bars, covering the true requirement of "
                             "rank_window + max(formation_lookback) = 960 + 384 = 1344. The stop "
                             "has a SECOND warm-up leg the buffer sizing cannot see (strategy.py's "
                             "DAILY_RANGE_BARS/DAILY_RANGE_WINDOW are module constants, not grid "
                             "params): hand-checked, it first goes finite at bar 1056, so the "
                             "signal leg's 1344 still binds. Raising "
                             "--rank-window also "
                             "raises wfo_engine's fold-skip guard (max_lookback + 10 = 970 bars "
                             "of train window); that is comfortable at 15min (~5,700 bars per "
                             "12-week train window) and still clears at 1h (~1,400), but at 4h "
                             "(~350) EVERY fold is silently skipped — so re-check both numbers "
                             "before raising it or moving to a coarser timeframe")

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
        formation_lookbacks = parse_num_list(args.formation_lookback, int)
        rank_pcts = parse_num_list(args.rank_pct, float)
        grid = build_grid(formation_lookbacks, rank_pcts, args.rank_window, session)
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
        # Only the two grid-searched params are reported here; `session` and
        # `rank_window` are fixed across every combo, so their "distinct
        # values" would always be 1 and carry no stability information.
        formation_lookback_vals = sorted({p["formation_lookback"] for p in chosen})
        rank_pct_vals = sorted({p["rank_pct"] for p in chosen})
        print(
            f"Fold param stability: {len(formation_lookback_vals)} distinct formation lookback "
            f"{formation_lookback_vals}, {len(rank_pct_vals)} distinct rank pct "
            f"{rank_pct_vals}"
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
                "formation_lookback": f.best_params.get("formation_lookback"),
                "rank_pct": f.best_params.get("rank_pct"),
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
