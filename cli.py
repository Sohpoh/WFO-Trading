"""Command-line walk-forward runner — the non-interactive twin of app.py.

Runs the exact same engine (wfo_engine / metrics / session / strategy) the
Streamlit app uses, so results here always match the UI for the same inputs.
Useful for batch runs, cron/CI, or piping results into other tools instead of
clicking through the sidebar every time.

Examples:
    # the strategy's intended config (NQ 15min, New York session): long-only
    # KNN next-session return prediction. `k` is threaded as a float (neighbor
    # count, not a lookback); feature_lookback/norm_lookback are int bar
    # counts, so wfo_engine's warm-up buffer is sized off norm_lookback (1440
    # bars -> ~45 days).
    python cli.py --symbol NQ --timeframe 15min --train-weeks 12 --test-weeks 3

    # override strategy/grid + walk-forward schedule
    python cli.py --symbol NQ --timeframe 15min --session "New York" \\
        --k 5,10,15 --feature-lookback 288,480,768 \\
        --pred-threshold 0.0,0.0005,0.001 --train-weeks 12 --test-weeks 3

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

import numpy as np
import pandas as pd

import streamlit.logger
streamlit.logger.set_log_level("ERROR")  # silence st.cache_data's "no ScriptRunContext" noise outside `streamlit run`

from data_loader import BARS_PER_YEAR, list_local_symbols, list_local_timeframes, load_data
from metrics import equity_curve, summarize
from session import SESSION_CONFIG
from tools import statistics as st
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
        "strategy grid (opening-range breakout, bounded ATR(14) stop/target)"
    )
    strat.add_argument("--session", default="New York", choices=list(SESSION_CONFIG.keys()) + ["none"],
                        help="day-trade session, or 'none' to disable session gating (forced for --timeframe 1d)")
    strat.add_argument("--orb-minutes", default="30,60,90",
                        help="comma-separated opening-range window lengths in MINUTES (30/60/90 = "
                             "first half-hour/hour/90min of the New York cash session). The opening "
                             "range is the per-ET-day High/Low over bars whose New-York time is in "
                             "[09:30, 09:30 + orb_minutes). A minutes-of-day window (float, not a "
                             "bar-count lookback), so it does NOT feed wfo_engine's warm-up buffer")
    strat.add_argument("--buffer-frac", default="0.0,0.1,0.25,0.5",
                        help="comma-separated range-buffer fractions: LONG threshold = OR_high + "
                             "buffer_frac * (OR_high - OR_low), SHORT threshold = OR_low - "
                             "buffer_frac * (OR_high - OR_low). A fraction (float, not a lookback), "
                             "so it does NOT feed wfo_engine's warm-up buffer")
    strat.add_argument("--stop-atr-mult", default="0.75,1.0,1.5,2.0",
                        help="comma-separated stop-loss ATR multipliers: stop = entry - "
                             "stop_atr_mult * ATR(14) for a long (entry + stop_atr_mult * ATR(14) "
                             "for a short). An ATR multiplier (float, not a lookback), so it does "
                             "NOT feed wfo_engine's warm-up buffer")
    strat.add_argument("--target-atr-mult", default="1.0,2.0,3.0,4.0",
                        help="comma-separated profit-target ATR multipliers: target = entry + "
                             "target_atr_mult * ATR(14) for a long (entry - target_atr_mult * ATR(14) "
                             "for a short). An ATR multiplier (float, not a lookback), so it does NOT "
                             "feed wfo_engine's warm-up buffer")

    wfo = p.add_argument_group("walk-forward schedule")
    wfo.add_argument("--train-weeks", type=int, default=12)
    wfo.add_argument("--test-weeks", type=int, default=3)
    wfo.add_argument("--purge-days", type=int, default=0,
                     help="pre-OOS purge: drop the last N training days nearest each test "
                          "window from grid search (prevents label warm-up leakage)")
    wfo.add_argument("--rank-stability-k", type=int, default=0,
                     help="if >1, re-backtest the top-K in-sample trials on each fold's OOS "
                          "window and report where argmax selection ranks out-of-sample")
    wfo.add_argument("--dsr-n-trials", type=int, default=None,
                     help="total independent backtests to deflate the Sharpe over for DSR "
                          "(defaults to this run's grid-combo count; set to the loop's total "
                          "backtest count for a loop-level DSR)")

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


def _statistical_validation(oos_returns, insample_stats, oos_stats, folds, ann_factor, dsr_n_trials):
    """Ported validation layer (DSR/PSR/Sharpe-CI/overfit/decay/rank-stability).

    All Sharpe inputs for DSR/PSR stay in per-period units (Bailey & López de
    Prado require the observed Sharpe and the observation count to share a
    periodicity); only the printed/CI values are annualized.
    """
    valid = [f for f in folds if f.best_params]
    oos = np.asarray(oos_returns, dtype=float)
    oos = oos[np.isfinite(oos)]

    pp_sharpe = st.per_period_sharpe(oos)
    n_obs = int(oos.size)
    skew = st.skewness(oos)
    kurt = st.raw_kurtosis(oos)

    # Trial population = every grid combo's train Sharpe across folds.  These
    # are stored annualized (what _score_params optimizes), so convert to
    # per-bar to match the per-bar observed_sr and n_obs (Bailey & López de
    # Prado require a single periodicity throughout).
    trial_sharpes = [s / math.sqrt(ann_factor) for f in valid for s in f.grid_train_sharpes]
    n_trials = int(dsr_n_trials) if dsr_n_trials else (len(trial_sharpes) or 1)

    oos_sharpe_ann = float(oos_stats["Sharpe Ratio"])
    is_sharpe_ann = float(insample_stats["Sharpe Ratio"])

    psr = st.probabilistic_sharpe_ratio(pp_sharpe, benchmark_sr=0.0, n_obs=n_obs, skewness=skew, kurtosis=kurt)
    dsr = st.deflated_sharpe(trial_sharpes, n_trials=n_trials, observed_sr=pp_sharpe, n_obs=n_obs, skewness=skew, kurtosis=kurt)
    ci = st.bootstrap_sharpe_ci(oos, ann_factor=ann_factor)
    oratio = st.overfit_ratio(is_sharpe_ann, oos_sharpe_ann)
    decay = st.sharpe_decay([f.oos_sharpe for f in valid])
    eff = st.efficiency(float(insample_stats["CAGR"]), float(oos_stats["CAGR"]))
    logits = [f.oos_rank_logit for f in valid if f.oos_rank_logit is not None]
    rank_fail = st.walk_forward_top_k_rank_failure_rate(logits) if logits else float("nan")

    return {
        "per_period_sharpe": pp_sharpe,
        "n_obs": n_obs,
        "n_trials": n_trials,
        "psr": psr,
        "dsr": dsr,
        "sharpe_ci_lo": ci[0] if ci else None,
        "sharpe_ci_hi": ci[1] if ci else None,
        "overfit_ratio": oratio,
        "efficiency": eff,
        "sharpe_decay": decay,
        "rank_failure_rate": rank_fail,
        "n_rank_folds": len(logits),
    }


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
        # orb_minutes is a minutes-of-day window (not a bar-count lookback) and
        # buffer_frac / stop_atr_mult / target_atr_mult are a fraction / ATR
        # multipliers — all floats, NOT lookbacks, so they stay out of the
        # warm-up buffer. The stop/target ATR window (14) is a fixed value
        # threaded inside build_grid() (not a CLI flag). The build_grid
        # arguments are orb_minutess / buffer_fracs / stop_atr_mults /
        # target_atr_mults — the grid-search naming convention is
        # "<param> + 's'".
        orb_minutes_list = parse_num_list(args.orb_minutes)
        buffer_fracs = parse_num_list(args.buffer_frac)
        stop_atr_mults = parse_num_list(args.stop_atr_mult)
        target_atr_mults = parse_num_list(args.target_atr_mult)
        grid = build_grid(orb_minutes_list, buffer_fracs, stop_atr_mults, target_atr_mults, session)
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
            df, args.train_weeks, args.test_weeks, grid, ann_factor, progress_callback=callback,
            purge_days=args.purge_days, rank_stability_k=args.rank_stability_k
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
        # Only the four grid-searched params are reported here; `session` and
        # the fixed ATR window (14) are constant across every combo, so their
        # "distinct values" would always be 1 and carry no stability
        # information.
        orb_minutes_vals = sorted({p["orb_minutes"] for p in chosen})
        buffer_frac_vals = sorted({p["buffer_frac"] for p in chosen})
        stop_atr_mult_vals = sorted({p["stop_atr_mult"] for p in chosen})
        target_atr_mult_vals = sorted({p["target_atr_mult"] for p in chosen})
        print(
            f"Fold param stability: {len(orb_minutes_vals)} distinct OR minutes "
            f"{orb_minutes_vals}, {len(buffer_frac_vals)} distinct buffer fracs "
            f"{buffer_frac_vals}, {len(stop_atr_mult_vals)} distinct stop ATR mults "
            f"{stop_atr_mult_vals}, {len(target_atr_mult_vals)} distinct target ATR mults "
            f"{target_atr_mult_vals}"
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

    # Statistical validation layer (ported from quantjourney-bt): DSR/PSR for
    # selection-bias, a bootstrap Sharpe CI for "is this even nonzero?", overfit
    # ratio + Sharpe decay for IS->OOS transfer, and rank-stability for argmax
    # selection quality. All load-bearing for a loop whose edges are single-digit
    # bps and whose confirmation holdouts are 12-81 trades.
    if len(oos_returns):
        val = _statistical_validation(oos_returns, insample_stats, oos_stats, folds, ann_factor, args.dsr_n_trials)
        print()
        print("Statistical validation:")
        print(f"  OOS per-period Sharpe {val['per_period_sharpe']:.4f} over {val['n_obs']} bars "
              f"(annualized {oos_stats['Sharpe Ratio']:.2f})")
        print(f"  DSR (deflated over {val['n_trials']} trials): {val['dsr']:.3f} "
              f"-> {st.interpret(val['dsr'], 'deflated_sharpe')[1]}")
        print(f"  PSR (P[true SR > 0]): {val['psr']:.3f}")
        if val["sharpe_ci_lo"] is not None:
            print(f"  Sharpe 5/95% bootstrap CI: [{val['sharpe_ci_lo']:.2f}, {val['sharpe_ci_hi']:.2f}]")
        else:
            print("  Sharpe bootstrap CI: not computable (< 20 OOS observations)")
        print(f"  Overfit ratio (IS/OOS Sharpe): {val['overfit_ratio']:.2f} "
              f"-> {st.interpret(val['overfit_ratio'], 'overfit_ratio', oos_sharpe=oos_stats['Sharpe Ratio'])[1]}")
        print(f"  Sharpe decay: {val['sharpe_decay']:+.4f}/fold "
              f"-> {st.interpret(val['sharpe_decay'], 'sharpe_decay', oos_sharpe=oos_stats['Sharpe Ratio'])[1]}")
        print(f"  Efficiency (OOS/IS CAGR): {val['efficiency']:.2f} "
              f"-> {st.interpret(val['efficiency'], 'efficiency')[1]}")
        if val["n_rank_folds"]:
            print(f"  Rank-stability failure rate: {val['rank_failure_rate']:.2f} "
                  f"({val['n_rank_folds']} folds) -> {st.interpret(val['rank_failure_rate'], 'rank_failure_rate')[1]}")

    if args.out_dir:
        import os
        os.makedirs(args.out_dir, exist_ok=True)

        fold_rows = [
            {
                "fold": f.index + 1,
                "train_start": f.train_start, "train_end": f.train_end,
                "test_start": f.test_start, "test_end": f.test_end,
                "purged_train_end": f.train_end_effective,
                "orb_minutes": f.best_params.get("orb_minutes"),
                "buffer_frac": f.best_params.get("buffer_frac"),
                "stop_atr_mult": f.best_params.get("stop_atr_mult"),
                "target_atr_mult": f.best_params.get("target_atr_mult"),
                "train_sharpe": f.train_sharpe,
                "n_grid_combos": len(f.grid_train_sharpes),
                "test_bars": f.n_test_bars,
                "oos_trades": f.n_oos_trades,
                "oos_return": f.oos_return,
                "oos_sharpe": f.oos_sharpe,
                "oos_rank_logit": f.oos_rank_logit,
            }
            for f in folds if f.best_params
        ]
        pd.DataFrame(fold_rows).to_csv(os.path.join(args.out_dir, "fold_table.csv"), index=False)
        oos_trades.to_csv(os.path.join(args.out_dir, "oos_trades.csv"), index=False)
        insample_trades.to_csv(os.path.join(args.out_dir, "insample_trades.csv"), index=False)

        # Full grid-trial train-Sharpe population (one row per fold x valid combo),
        # so a later DSR can be recomputed from saved artifacts without re-running.
        trial_rows = [
            {"fold": f.index + 1, "train_sharpe_annualized": s}
            for f in folds if f.best_params
            for s in f.grid_train_sharpes
        ]
        pd.DataFrame(trial_rows).to_csv(os.path.join(args.out_dir, "grid_train_sharpes.csv"), index=False)

        eq = pd.DataFrame(
            {
                "insample_equity": equity_curve(insample_returns, start=100.0),
                "oos_equity": equity_curve(oos_returns, start=100.0),
            }
        )
        eq.to_csv(os.path.join(args.out_dir, "equity_curves.csv"))
        comparison.to_csv(os.path.join(args.out_dir, "comparison.csv"))

        if len(oos_returns):
            val_rows = {k: v for k, v in val.items()}
            pd.DataFrame([val_rows]).to_csv(os.path.join(args.out_dir, "validation.csv"), index=False)

        print(f"\nWrote fold_table.csv, oos_trades.csv, insample_trades.csv, equity_curves.csv, "
              f"comparison.csv" + (", validation.csv" if len(oos_returns) else "") + f" -> {args.out_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
