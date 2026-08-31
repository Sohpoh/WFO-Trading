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
        "strategy grid (long-only overnight gap-down fill buy: the anchor is the Close of "
        "the final bar of the PREVIOUS completed UTC day, broadcast forward across the "
        "current UTC day — session-unaware and strictly backward-looking. A bar arms a "
        "LONG when its Close sits between --min-gap-pct and 3x --min-gap-pct BELOW that "
        "anchor; the 3x upper edge is strategy.BAND_CAP_MULT, a module constant that is "
        "deliberately NOT searched (beyond it the move is read as news, not a fadeable "
        "overshoot). There is NO short leg — a gap-up day pays zero cost legs instead of "
        "the two a short would cost. The exit is path-dependent and routed through "
        "session.apply_session_constraint_with_stops: a bounded profit target at "
        "--target-frac of the way back up to the anchor (1.0 = the literal full fill) "
        "and a hard stop --stop-gap-frac of the same gap distance below the entry Close, "
        "with session.py force-flattening any survivor on the session's last bar. The "
        "arming condition is a dense STATE, not a crossing — the gap forms overnight, "
        "outside the session — so a stopped-out trade can re-arm on a later in-session "
        "bar while price is still inside the band; the 3x cap is what bounds that to a "
        "few attempts rather than unlimited averaging-down)"
    )
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--min-gap-pct", default="0.004,0.006,0.009,0.013",
                        help="comma-separated gap-down thresholds as fractions of the anchor "
                             "(0.004 = 0.40%%), i.e. the LOWER edge of the entry band; the upper "
                             "edge is always 3x this and is not searchable. Passed as floats — "
                             "these are return thresholds, not bar counts, so they must not feed "
                             "wfo_engine's warm-up buffer. The 0.40%% floor is the cost "
                             "discipline: with --target-frac >= 0.75 the smallest implied target "
                             "is ~30bps against metrics.py's ~10.2bps round-trip toll (~2.9x, "
                             "~3.9x at target_frac 1.0). Lowering it below 0.004 gives that "
                             "margin away")
    strat.add_argument("--stop-gap-frac", default="0.5,0.75,1.0",
                        help="comma-separated hard-stop distances, as fractions of the gap "
                             "distance (anchor - entry Close), measured from the entry Close and "
                             "frozen there by session.py (hard, not trailing). Floats, not bar "
                             "counts. 1.0 puts the stop a full gap-width below entry")
    strat.add_argument("--target-frac", default="0.75,1.0",
                        help="comma-separated profit targets, as fractions of the way back up to "
                             "the anchor: 1.0 is the literal full gap fill, 0.75 a partial fill. "
                             "Floats, not bar counts. Always > 0 so the target is strictly above "
                             "the entry Close by construction, satisfying "
                             "apply_session_constraint_with_stops' target_price > close guard. "
                             "NOTE: unlike previous iterations, every searched param here is a "
                             "float, so wfo_engine's _max_lookback_bars() returns 0, the warm-up "
                             "buffer falls back to the one-day floor (bars_per_day + 5 = 97 at "
                             "NQ 15min, enough for the prior-day anchor) and the fold-skip guard "
                             "drops to 10 train bars — there is no timeframe here that silently "
                             "skips every fold")

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
        min_gap_pcts = parse_num_list(args.min_gap_pct, float)
        stop_gap_fracs = parse_num_list(args.stop_gap_frac, float)
        target_fracs = parse_num_list(args.target_frac, float)
        grid = build_grid(min_gap_pcts, stop_gap_fracs, target_fracs, session)
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
        # and carry no stability information. (strategy.BAND_CAP_MULT is a
        # module constant, not a param, so it never appears in best_params.)
        min_gap_vals = sorted({p["min_gap_pct"] for p in chosen})
        stop_frac_vals = sorted({p["stop_gap_frac"] for p in chosen})
        target_frac_vals = sorted({p["target_frac"] for p in chosen})
        print(
            f"Fold param stability: {len(min_gap_vals)} distinct min gap pct "
            f"{min_gap_vals}, {len(stop_frac_vals)} distinct stop gap frac "
            f"{stop_frac_vals}, {len(target_frac_vals)} distinct target frac "
            f"{target_frac_vals}"
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
                "min_gap_pct": f.best_params.get("min_gap_pct"),
                "stop_gap_frac": f.best_params.get("stop_gap_frac"),
                "target_frac": f.best_params.get("target_frac"),
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
