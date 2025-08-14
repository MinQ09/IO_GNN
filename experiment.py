# experiment.py  ──────────────────────────────────────────────────────────────
"""
Master driver for IO-GNN experiments.

Runs one or both tasks:
  * kind="Z"  – edge-flow prediction
  * kind="VA" – node value-added prediction

Grid:
  seeds × lambda_candidates × beta_candidates

Output layout
-------------
<out_dir>/<kind>/seed_<s>/         (created inside run_single)
<out_dir>/summary_<kind>.json      per-kind, cross-seed metrics
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import Config
from utils import set_seed
from run_single import run_single

# Metrics collected from `run_single`. Note: IOIS is Z-only; for VA it will be NaN.
Metrics: List[str] = ["RMSE", "MAE", "SMAPE", "R2", "RHO", "IOIS"]


def run_all(cfg: Config, kinds: List[str]) -> None:
    """Launch the full sweep and write summary JSON files."""
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # summary[kind][(lambda, beta)][metric] -> list[float across seeds]
    summary: Dict[str, Dict[Tuple[float, float], Dict[str, List[float]]]] = {
        k: {
            (lam, beta): {m: [] for m in Metrics}
            for lam in cfg.lambda_candidates
            for beta in cfg.beta_candidates
        }
        for k in kinds
    }

    # ────────────────────────────────────────────────────────────────── sweep
    for seed in cfg.seeds:
        set_seed(seed)

        for lam in cfg.lambda_candidates:
            for beta in cfg.beta_candidates:
                exp_cfg = copy.deepcopy(cfg)
                exp_cfg.lambda_max = lam
                exp_cfg.beta_init = beta

                for kind in kinds:
                    _, _, _, metrics = run_single(exp_cfg, seed=seed, kind=kind)
                    for m in Metrics:
                        # IOIS is absent for VA; append NaN in that case
                        summary[kind][(lam, beta)][m].append(metrics.get(m, float("nan")))

    # ───────────────────────────────────────────────────────────── write summaries
    for kind in kinds:
        print(f"\n=== {kind} summary across seeds ===")
        serial = {}
        for (lam, beta), vals in summary[kind].items():
            tag = f"lam_{lam:g}_beta_{beta:g}"
            stats = {}
            for m in Metrics:
                arr = np.asarray(vals[m], dtype=float)
                valid = arr[~np.isnan(arr)]
                if valid.size > 0:
                    mu = float(valid.mean())
                    sd = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
                    print(f"λ={lam:g}, β={beta:g}  {m:<6}: {mu:.4f} ± {sd:.4f}")
                else:
                    print(f"λ={lam:g}, β={beta:g}  {m:<6}: N/A")
                # store raw values (keep NaNs) for downstream processing
                stats[m] = [None if np.isnan(x) else float(x) for x in arr]
            serial[tag] = stats

        (out_root / f"summary_{kind}.json").write_text(json.dumps(serial, indent=2))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IO-GNN experiment runner")
    parser.add_argument("--kinds", nargs="+", default=["Z", "VA"],
                        help="Which models to train (choices include 'Z' and 'VA')")
    parser.add_argument("--out_dir", type=str, help="Override cfg.out_dir")
    args = parser.parse_args()

    cfg = Config()
    if args.out_dir:
        cfg.out_dir = Path(args.out_dir)

    run_all(cfg, kinds=args.kinds)