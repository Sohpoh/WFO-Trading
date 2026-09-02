"""Cost-sensitivity audit: does a near-miss survive a more realistic cost model?

The engine charges COST_RATE = 0.051% per leg => 10.2bp round trip, and each
trade's saved `net_return` is `gross - 0.00102`. This script reconstructs gross
(`net + 0.00102`) and recomputes the gate-v2 load-bearing quantities at cost
multipliers {1x, 0.5x, 0.2x, 0.1x, 0x}:

    net'(m) = net + 0.00102 * (1 - m)

For each candidate we report PF, PF-ex-top-5 (the hard gate), win rate, total
return, and fold consistency — the exact quantities gate v2 decides on. Sharpe
is *not* recomputed (it needs bar-level returns; a re-run would be required for
that), so this script answers "does the edge survive the toll?" via PF / PF_ex5
/ fold consistency, which are the accept/reject drivers.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

WFO_TRADING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WFO_TRADING_DIR)

from tools.gate_v2 import _pf  # noqa: E402

COST_RATE = 0.00051          # per leg
ROUND_TRIP = 2 * COST_RATE   # 0.00102

# iteration -> (label, results dir)
CANDIDATES = {
    6: ("iter-006 rolling-range-breakout vol-gate (v1 accept, fragile)", "results/iter-006-rolling-range-breakout-vol-regime-gate"),
    10: ("iter-010 rolling-range-breakout failed-breakout-stop (v1 accept, robust:false)", "results/iter-010-rolling-range-breakout-failed-breakout-stop"),
    11: ("iter-011 ...narrow-buffer-grid (v1 accept, fragile borderline)", "results/iter-011-rolling-range-breakout-failed-breakout-stop-narrow-buffer"),
    36: ("iter-036 long-only momentum slow-formation (best robust:true)", "results/iter-036-long-only-momentum-slow-formation-grid"),
    38: ("iter-038 long-only momentum widened-entry (cost-drag: 8.6bp gross)", "results/iter-038-long-only-momentum-widened-entry-base"),
    40: ("iter-040 long-only momentum 12w/2w (v2 accept, unconfirmed)", "results/iter-040-long-only-momentum-slow-formation-grid-12w2w-schedule"),
}

MULTIPLIERS = [1.0, 0.5, 0.2, 0.1, 0.0]


def load_trades(results_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(WFO_TRADING_DIR, results_dir, "oos_trades.csv"))
    df["net_return"] = df["net_return"].astype(float)

    # gate-v1 runs predate the per-fold column; reconstruct it by bucketing
    # entry_time against fold_table.csv's test windows (the evaluator's own
    # fallback path) so fold consistency stays available for those rows.
    if "fold" not in df.columns:
        ft = pd.read_csv(os.path.join(WFO_TRADING_DIR, results_dir, "fold_table.csv"))
        ft["test_start"] = pd.to_datetime(ft["test_start"], utc=True)
        ft["test_end"] = pd.to_datetime(ft["test_end"], utc=True)
        entry = pd.to_datetime(df["entry_time"], utc=True).to_numpy()
        folds = np.zeros(len(df), dtype=int)
        for _, row in ft.iterrows():
            m = (entry >= row["test_start"]) & (entry < row["test_end"])
            folds[m] = int(row["fold"])
        df["fold"] = folds
    return df


def rescore(trades: pd.DataFrame, m: float) -> dict:
    net = trades["net_return"].to_numpy() + ROUND_TRIP * (1.0 - m)
    n = len(net)
    pf = _pf(net)
    win = float((net > 0).mean())

    remaining = np.sort(net)[::-1][5:] if n >= 15 else net
    pf_ex5 = _pf(remaining)

    # total return: plain sum (matches evaluator Step 3) and compounded
    total_sum = float(net.sum())
    total_comp = float(np.prod(1.0 + net) - 1.0) if n else 0.0

    # fold consistency: % active folds whose (cost-adjusted) trade sum > 0
    fold_sum = trades.assign(_net=net).groupby("fold")["_net"].sum()
    pct_prof = float((fold_sum > 0).mean()) if len(fold_sum) else 0.0

    return {
        "m": m,
        "n_trades": n,
        "pf": pf,
        "pf_ex_top5": pf_ex5,
        "win_rate": win,
        "total_return_sum": total_sum,
        "total_return_comp": total_comp,
        "pct_folds_profitable": pct_prof,
        "round_trip_bp": ROUND_TRIP * m * 10000,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", default="6,10,11,36,38,40", help="comma-separated iteration numbers")
    args = ap.parse_args()

    iters = [int(x) for x in args.iters.split(",") if x.strip()]

    print(f"COST_RATE per leg = {COST_RATE*100:.3f}% ; round trip = {ROUND_TRIP*10000:.2f}bp "
          f"({'~'}{ROUND_TRIP*100*20000*5/100:.0f} per contract round trip on NQ ~20k)\n")

    for it in iters:
        if it not in CANDIDATES:
            print(f"!! iteration {it} not in CANDIDATES — skipping")
            continue
        label, rdir = CANDIDATES[it]
        trades = load_trades(rdir)
        print("=" * 110)
        print(f"{label}")
        print(f"   {rdir}  ({len(trades)} OOS trades at m=1.0)")
        header = f"   {'mult':>5} {'rtrip(bp)':>9} {'PF':>7} {'PF_ex5':>7} {'win%':>6} {'totRet(sum)':>11} {'folds%':>7}"
        print(header)
        for m in MULTIPLIERS:
            r = rescore(trades, m)
            print(f"   {r['m']:>5.1f} {r['round_trip_bp']:>9.2f} {r['pf']:>7.3f} {r['pf_ex_top5']:>7.3f} "
                  f"{r['win_rate']*100:>6.1f} {r['total_return_sum']*100:>10.2f}% {r['pct_folds_profitable']*100:>6.1f}%")
        print()


if __name__ == "__main__":
    main()
