"""Re-validate a saved cli.py results directory without re-running the engine.

Reconstructs the OOS bar-return series from equity_curves.csv and computes the
ported statistical-validation layer on top of the artifacts already on disk:

  - PSR (P[true Sharpe > 0]) and DSR (Deflated Sharpe, selection-bias corrected)
  - a stationary block-bootstrap 5%/95% CI for the annualized Sharpe
  - overfit ratio (IS/OOS Sharpe) and Sharpe decay (slope of per-fold OOS Sharpe)
  - rank-stability failure rate (only for runs made with --rank-stability-k)
  - cost robustness: mean gross edge per trade vs the round-trip toll

DSR trial population is taken from grid_train_sharpes.csv when present (runs
made with the current engine), else --loop reconstructs a loop-level population
from iterations/log.jsonl (every strategy ever run's OOS Sharpe), else DSR is
reported as not computable.

Usage:
    python tools/validate.py results/iter-040-...            # within-run DSR
    python tools/validate.py results/iter-040-... --loop     # loop-level DSR (64 strategies)
    python tools/validate.py results/iter-040-... --n-trials 2000
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

WFO_TRADING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WFO_TRADING_DIR)

from metrics import COST_RATE  # noqa: E402  (per-leg cost, cost v2)
from tools import statistics as st  # noqa: E402

BARS_PER_YEAR = {"1min": 252 * 23 * 60, "5min": 252 * 23 * 12, "15min": 252 * 23 * 4,
                 "1h": 252 * 23, "4h": 252 * 6, "1d": 252}


def _infer_ann_factor(index: pd.DatetimeIndex) -> int:
    """Annualization factor from median bar spacing."""
    if len(index) < 2:
        return 252
    delta_min = float(np.median(np.diff(index.values).astype("timedelta64[m]").astype(float)))
    if delta_min <= 0:
        return 252
    if delta_min >= 1440:
        return 252
    return int(round(252 * 23 * 60 / delta_min))


def _reconstruct_oos_returns(equity_csv: str) -> tuple[np.ndarray, int]:
    eq = pd.read_csv(equity_csv, index_col=0, parse_dates=True)
    ann = _infer_ann_factor(eq.index)
    oos = eq["oos_equity"].dropna()
    rets = oos.pct_change().dropna().astype(float).to_numpy()
    return rets[np.isfinite(rets)], ann


def _load_loop_population() -> tuple[list[float], int]:
    """Every strategy's OOS Sharpe from iterations/log.jsonl, in per-period units."""
    path = os.path.join(WFO_TRADING_DIR, "iterations", "log.jsonl")
    pops = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            oos = e.get("oos")
            if not isinstance(oos, dict):
                continue
            sharpe = oos.get("sharpe")
            if sharpe is None or not math.isfinite(float(sharpe)):
                continue
            cmd = e.get("cli_command") or ""
            m = re.search(r"--timeframe\s+(\S+)", cmd)
            tf = m.group(1) if m else None
            ann = BARS_PER_YEAR.get(tf, 252) if tf else 252
            pops.append(float(sharpe) / math.sqrt(ann))
    return pops, len(pops)


def validate(results_dir: str, *, n_trials: int | None = None, loop: bool = False) -> dict:
    comp = pd.read_csv(os.path.join(results_dir, "comparison.csv"), index_col=0)
    oos_stats = comp["Walk-Forward OOS"]
    is_stats = comp["Retail In-Sample"]

    rets, ann = _reconstruct_oos_returns(os.path.join(results_dir, "equity_curves.csv"))
    pp_sharpe = st.per_period_sharpe(rets)
    n_obs = int(rets.size)
    skew = st.skewness(rets)
    kurt = st.raw_kurtosis(rets)

    ft = pd.read_csv(os.path.join(results_dir, "fold_table.csv"))

    # Trial population for DSR
    trial_sharpes: list[float] = []
    population_label = ""
    trials_csv = os.path.join(results_dir, "grid_train_sharpes.csv")
    if os.path.exists(trials_csv):
        tr = pd.read_csv(trials_csv)
        trial_sharpes = [float(s) / math.sqrt(ann) for s in tr["train_sharpe_annualized"]]
        population_label = f"within-run grid ({len(trial_sharpes)} combos)"
    elif loop:
        trial_sharpes, n_loop = _load_loop_population()
        population_label = f"loop ({n_loop} strategies from log.jsonl)"
    else:
        population_label = "unavailable (no grid_train_sharpes.csv; use --loop for loop-level, or re-run)"

    eff_n = int(n_trials) if n_trials else (len(trial_sharpes) or 1)
    dsr = None
    if trial_sharpes:
        dsr = st.deflated_sharpe(trial_sharpes, n_trials=eff_n, observed_sr=pp_sharpe, n_obs=n_obs,
                                 skewness=skew, kurtosis=kurt)

    psr = st.probabilistic_sharpe_ratio(pp_sharpe, benchmark_sr=0.0, n_obs=n_obs, skewness=skew, kurtosis=kurt)
    ci = st.bootstrap_sharpe_ci(rets, ann_factor=ann)

    oos_sharpe_ann = float(oos_stats["Sharpe Ratio"])
    is_sharpe_ann = float(is_stats["Sharpe Ratio"])
    oratio = st.overfit_ratio(is_sharpe_ann, oos_sharpe_ann)
    decay = st.sharpe_decay(ft["oos_sharpe"].astype(float)) if "oos_sharpe" in ft.columns else float("nan")
    eff_ratio = st.efficiency(float(is_stats["CAGR"]), float(oos_stats["CAGR"]))

    logits = ft["oos_rank_logit"].dropna().astype(float).tolist() if "oos_rank_logit" in ft.columns else []
    rank_fail = st.walk_forward_top_k_rank_failure_rate(logits) if logits else float("nan")

    # Cost robustness: mean gross edge per trade vs the round-trip toll
    trades = pd.read_csv(os.path.join(results_dir, "oos_trades.csv"))["net_return"].astype(float)
    round_trip = 2 * COST_RATE
    mean_gross_bp = (float(trades.mean()) + round_trip) * 10000.0

    return {
        "results_dir": results_dir,
        "ann_factor": ann,
        "per_period_sharpe": pp_sharpe,
        "n_obs": n_obs,
        "oos_sharpe_ann": oos_sharpe_ann,
        "psr": psr,
        "dsr": dsr,
        "dsr_n_trials": eff_n,
        "dsr_population": population_label,
        "sharpe_ci": ci,
        "overfit_ratio": oratio,
        "efficiency": eff_ratio,
        "sharpe_decay": decay,
        "rank_failure_rate": rank_fail,
        "n_rank_folds": len(logits),
        "mean_gross_bp": mean_gross_bp,
        "round_trip_bp": round_trip * 10000.0,
    }


def _report(v: dict) -> None:
    print(f"=== {v['results_dir']} ===")
    print(f"  OOS Sharpe {v['oos_sharpe_ann']:.2f} (per-period {v['per_period_sharpe']:.5f} over {v['n_obs']} bars)")
    if v["dsr"] is not None:
        print(f"  DSR  (over {v['dsr_n_trials']} trials, {v['dsr_population']}): {v['dsr']:.3f} "
              f"-> {st.interpret(v['dsr'], 'deflated_sharpe')[1]}")
    else:
        print(f"  DSR  : not computable — {v['dsr_population']}")
    print(f"  PSR  (P[true SR > 0]): {v['psr']:.3f}")
    if v["sharpe_ci"] is not None:
        lo, hi = v["sharpe_ci"]
        print(f"  Sharpe 5/95% bootstrap CI: [{lo:.2f}, {hi:.2f}]")
    else:
        print("  Sharpe 5/95% bootstrap CI: not computable (< 20 OOS observations)")
    print(f"  Overfit ratio (IS/OOS): {v['overfit_ratio']:.2f} "
          f"-> {st.interpret(v['overfit_ratio'], 'overfit_ratio', oos_sharpe=v['oos_sharpe_ann'])[1]}")
    print(f"  Sharpe decay: {v['sharpe_decay']:+.4f}/fold "
          f"-> {st.interpret(v['sharpe_decay'], 'sharpe_decay', oos_sharpe=v['oos_sharpe_ann'])[1]}")
    print(f"  Efficiency (OOS/IS CAGR): {v['efficiency']:.2f} -> {st.interpret(v['efficiency'], 'efficiency')[1]}")
    if v["n_rank_folds"]:
        print(f"  Rank-stability failure rate: {v['rank_failure_rate']:.2f} ({v['n_rank_folds']} folds) "
              f"-> {st.interpret(v['rank_failure_rate'], 'rank_failure_rate')[1]}")
    print(f"  Cost robustness: mean gross edge {v['mean_gross_bp']:+.2f}bp/trade vs {v['round_trip_bp']:.2f}bp "
          f"round-trip -> {'cost-robust' if v['mean_gross_bp'] > 20 else ('marginal' if v['mean_gross_bp'] >= 10 else 'cost-fragile')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--n-trials", type=int, default=None, help="override DSR trial count")
    ap.add_argument("--loop", action="store_true", help="use loop-level population from iterations/log.jsonl")
    args = ap.parse_args()
    _report(validate(args.results_dir, n_trials=args.n_trials, loop=args.loop))
