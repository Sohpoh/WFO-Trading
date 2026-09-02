"""Compute the gate-v2 verdict for a cli.py results directory (post cost-v2 re-run).

Reads comparison.csv + oos_trades.csv + fold_table.csv and reports the exact
quantities gate v2 decides on (fold consistency, leave-top-5-out robustness,
absolute checklist), plus the verdict. Handles pre-gate-v2 fold tables that
lack the per-fold oos_* columns by reconstructing fold membership from
entry_time against test windows (the evaluator's own fallback).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

WFO_TRADING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WFO_TRADING_DIR)

from tools.gate_v2 import verdict  # noqa: E402


def report(results_dir: str, label: str = "") -> dict:
    comp = pd.read_csv(os.path.join(results_dir, "comparison.csv"), index_col=0)
    oos = comp["Walk-Forward OOS"]
    trades = pd.read_csv(os.path.join(results_dir, "oos_trades.csv"))
    trades["net_return"] = trades["net_return"].astype(float)
    ft = pd.read_csv(os.path.join(results_dir, "fold_table.csv"))

    reconstructed = False
    if "oos_return" in ft.columns:
        active = ft[ft["oos_trades"] > 0]
        fold_rets = active["oos_return"].to_numpy()
        fold_sharpes = active["oos_sharpe"].to_numpy()
        n_active = len(active)
    else:
        reconstructed = True
        ft["test_start"] = pd.to_datetime(ft["test_start"], utc=True)
        ft["test_end"] = pd.to_datetime(ft["test_end"], utc=True)
        entry = pd.to_datetime(trades["entry_time"], utc=True).to_numpy()
        fold_of_trade = np.zeros(len(trades), dtype=int)
        for _, row in ft.iterrows():
            m = (entry >= row["test_start"]) & (entry < row["test_end"])
            fold_of_trade[m] = int(row["fold"])
        trades = trades.assign(_fold=fold_of_trade)
        g = trades.groupby("_fold")["net_return"]
        fold_rets = g.sum().to_numpy()
        fold_sharpes = np.array([
            (x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else 0.0
            for _, x in g
        ])
        n_active = len(fold_rets)

    v = verdict(
        fold_rets,
        fold_sharpes,
        trades["net_return"].to_numpy(),
        float(oos["Total Return"]),
        float(oos["Sharpe Ratio"]),
        float(oos["Profit Factor"]),
        int(oos["Trades"]),
    )

    fc = v["fold_consistency"]
    rb = v["robustness"]
    print(f"=== {label or results_dir} ===")
    print(f"  status           : {v['status']}")
    print(f"  OOS Sharpe / PF  : {v['oos_sharpe']:.3f} / {v['oos_pf']:.3f}  (n={v['n_trades']})")
    print(f"  OOS total return : {float(oos['Total Return'])*100:.2f}%  (CAGR {float(oos['CAGR'])*100:.2f}%)")
    print(f"  fold consistency : {fc['pct_profitable']*100:.1f}% of {fc['active_folds']} active folds "
          f"(median fold ret {fc['median_fold_return']*100:.2f}%, median Sharpe {fc['median_fold_sharpe']:.2f})"
          + (" [reconstructed]" if reconstructed else ""))
    print(f"  robustness       : PF_full={rb['profit_factor_full']:.3f}  PF_ex5={rb['profit_factor_ex_top5']:.3f} "
          f"-> {rb['robust']}")
    if v["status"] == "rejected":
        conds = [v["cond1_robust_not_false"], v["cond2_fold_consistency"], v["cond3_absolute"]]
        print(f"  failed on        : " + ", ".join(
            n for n, ok in zip(["robustness", "fold consistency", "absolute"], conds) if not ok))
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=[])
    args = ap.parse_args()
    labels = args.labels or [""] * len(args.results_dirs)
    for d, l in zip(args.results_dirs, labels):
        report(d, l)
        print()
