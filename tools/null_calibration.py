"""Null-model calibration: measure gate-v2's false-accept rate under pure noise.

Feeds a NO-SKILL strategy (deterministic coin-flip entries, param-keyed) through
the *real* walk-forward pipeline — grid search (argmax train Sharpe) + fold
stitching + the gate-v2 verdict — so the selection effect is reproduced, not
assumed. If a meaningful fraction of trials clear the bar, the bar is reachable
by luck and the strict gate is the problem; if ~0% clear it, the ideas are the
problem and the gate is doing its job.

Usage:
    /opt/anaconda3/envs/WFO/bin/python tools/null_calibration.py --trials 20
    /opt/anaconda3/envs/WFO/bin/python tools/null_calibration.py --trials 1 --probe
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

WFO_TRADING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WFO_TRADING_DIR)

import streamlit.logger  # noqa: E402
streamlit.logger.set_log_level("ERROR")

from data_loader import BARS_PER_YEAR, load_data  # noqa: E402
import metrics  # noqa: E402
import session as session_mod  # noqa: E402
import wfo_engine  # noqa: E402
from tools.gate_v2 import verdict  # noqa: E402

# Per-trial seed is stashed in a mutable cell so the injected null strategy can
# read it without changing the generate_positions() signature.
_TRIAL_SEED = {"value": 0}


def make_null_strategy(p_long: float, p_short: float):
    """Return a generate_positions() that emits a deterministic coin-flip signal.

    The RNG is seeded by (trial, param combo), so:
      - different param combos => different noise realisations => the grid
        search genuinely *selects* the combo whose noise looked best in train;
      - different trials => different noise, for a proper distribution.
    """
    def null_generate_positions(df, hurst_window, drift_lookback, h_threshold, session=None):
        seed = (
            _TRIAL_SEED["value"] * 1000003
            + int(hurst_window) * 7919
            + int(drift_lookback) * 104729
            + int(round(float(h_threshold) * 1000)) * 104743
        ) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)
        n = len(df)
        u = rng.random(n)
        entries = pd.Series(np.nan, index=df.index)
        entries[u < p_long] = 1.0
        entries[u > (1.0 - p_short)] = -1.0
        return session_mod.apply_session_constraint(entries, session)

    return null_generate_positions


def run_one_trial(df, grid, ann_factor, p_long, p_short, train_weeks, test_weeks):
    wfo_engine.generate_positions = make_null_strategy(p_long, p_short)
    oos_returns, trades_df, folds = wfo_engine.run_walk_forward(df, train_weeks, test_weeks, grid, ann_factor)

    active = [f for f in folds if f.best_params and f.n_oos_trades > 0]
    fold_rets = np.array([f.oos_return for f in active])
    fold_sharpes = np.array([f.oos_sharpe for f in active])
    net = trades_df["net_return"].to_numpy() if len(trades_df) else np.array([])

    oos_stats = metrics.summarize(oos_returns, trades_df, ann_factor)
    v = verdict(
        fold_rets,
        fold_sharpes,
        net,
        oos_stats["Total Return"],
        oos_stats["Sharpe Ratio"],
        oos_stats["Profit Factor"],
        int(oos_stats["Trades"]),
    )
    return v, len(folds), len(active)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--p-long", type=float, default=0.012, help="per-bar long signal probability")
    p.add_argument("--p-short", type=float, default=0.012, help="per-bar short signal probability")
    p.add_argument("--symbol", default="NQ")
    p.add_argument("--timeframe", default="15min")
    p.add_argument("--train-weeks", type=int, default=12)
    p.add_argument("--test-weeks", type=int, default=3)
    p.add_argument("--date-to", default="2024-12-31", help="exploration window end (holdout excluded)")
    p.add_argument("--probe", action="store_true", help="run one trial and print timing/trade count only")
    args = p.parse_args()

    df = load_data("local", args.symbol, args.timeframe)
    df = df[df.index < pd.Timestamp(args.date_to, tz="UTC") + pd.Timedelta(days=1)]
    ann_factor = BARS_PER_YEAR[args.timeframe]

    grid = wfo_engine.build_grid([384, 768, 1152], [96, 192, 384], [0.5, 0.55, 0.6], "New York")

    if args.probe:
        _TRIAL_SEED["value"] = 0
        t0 = time.time()
        v, n_folds, n_active = run_one_trial(df, grid, ann_factor, args.p_long, args.p_short, args.train_weeks, args.test_weeks)
        print(f"probe: {n_folds} folds, {n_active} active, {v['n_trades']} OOS trades, {time.time() - t0:.1f}s")
        print(f"       verdict={v['status']} sharpe={v['oos_sharpe']:.3f} pf={v['oos_pf']:.3f} "
              f"pct_prof={v['fold_consistency']['pct_profitable']*100:.1f}% robust={v['robustness']['robust']}")
        return

    rows = []
    accepts = 0
    t0 = time.time()
    for trial in range(args.trials):
        _TRIAL_SEED["value"] = trial + 1
        v, n_folds, n_active = run_one_trial(df, grid, ann_factor, args.p_long, args.p_short, args.train_weeks, args.test_weeks)
        is_accept = v["status"] == "accepted"
        accepts += int(is_accept)
        rows.append({
            "trial": trial + 1,
            "status": v["status"],
            "n_trades": v["n_trades"],
            "oos_sharpe": round(v["oos_sharpe"], 3),
            "oos_pf": round(v["oos_pf"], 3),
            "pf_ex_top5": round(v["robustness"].get("profit_factor_ex_top5") or 0.0, 3),
            "robust": v["robustness"]["robust"],
            "pct_profitable": round(v["fold_consistency"]["pct_profitable"], 3),
            "active_folds": v["fold_consistency"]["active_folds"],
        })
        if (trial + 1) % 5 == 0:
            print(f"  ... {trial + 1}/{args.trials} trials done ({time.time() - t0:.0f}s)", file=sys.stderr)

    elapsed = time.time() - t0
    table = pd.DataFrame(rows)
    print("\n=== Null-model calibration (gate v2 false-accept rate under pure noise) ===")
    print(f"setup: {args.symbol} {args.timeframe} {args.symbol and 'New York'} | grid={len(grid)} combos | "
          f"{args.train_weeks}w/{args.test_weeks}w | date-to {args.date_to} | p={args.p_long}/{args.p_short}")
    print(f"trials: {args.trials} in {elapsed:.0f}s ({elapsed/args.trials:.1f}s/trial)")
    print(f"\nFALSE-ACCEPT RATE: {accepts}/{args.trials} = {accepts/args.trials*100:.1f}%\n")

    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:,.3f}"):
        print(table.to_string(index=False))

    print("\n--- distributions (across trials) ---")
    def stats(col):
        s = table[col]
        return f"min={s.min():.3f} median={s.median():.3f} max={s.max():.3f}"
    print(f"OOS Sharpe        : {stats('oos_sharpe')}")
    print(f"OOS profit factor : {stats('oos_pf')}")
    print(f"PF ex-top-5       : {stats('pf_ex_top5')}")
    print(f"% folds profitable: {stats('pct_profitable')}")
    print(f"OOS trades        : min={table['n_trades'].min()} median={table['n_trades'].median()} max={table['n_trades'].max()}")
    robust_counts = table["robust"].value_counts().to_dict()
    print(f"robustness verdicts: {robust_counts}")
    print(f"\n% trials with PF_ex_top5 >= 1.15 (robust:true): "
          f"{(table['pf_ex_top5'] >= 1.15).mean()*100:.1f}%")
    print(f"% trials with fold consistency >= 60%: {(table['pct_profitable'] >= 0.60).mean()*100:.1f}%")
    print(f"% trials with OOS Sharpe > 0.8: {(table['oos_sharpe'] > 0.8).mean()*100:.1f}%")
    print(f"% trials with OOS PF > 1.2: {(table['oos_pf'] > 1.2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
