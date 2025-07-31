# experiment.py  ──────────────────────────────────────────────────────────────

"""experiment.py – master driver for IO‑GNN experiments
------------------------------------------------------
Runs one or both tasks:
    * kind="Z"  – edge‑flow prediction
    * kind="VA" – node value‑added prediction

Grid:
    seeds × lambda_candidates × beta_candidates

Output layout
-------------
<out_dir>/<kind>/seed_<s>/   (created inside run_single)
<out_dir>/summary_<kind>.json – per‑kind cross‑seed metrics
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import Config
from utils import set_seed
from run_single import run_single

# ----------------------------------------------------------------------
Metrics: List[str] = ["RMSE", "MAE", "SMAPE", "R2", "RHO", "CVR"]  

# ----------------------------------------------------------------------
def run_all(cfg: Config, kinds: List[str]) -> None:
    """Launch the full sweep and write summary JSON files."""
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # summary[kind][(λ,β)][metric] -> list[float]
    summary: Dict[str, Dict[tuple[float, float], Dict[str, List[float]]]] = {
        k: {
            (lam, beta): {m: [] for m in Metrics}
            for lam in cfg.lambda_candidates
            for beta in cfg.beta_candidates
        }
        for k in kinds
    }

    # sweep ────────────────────────────────────────────────────────────
    for seed in cfg.seeds:
        set_seed(seed)

        for lam in cfg.lambda_candidates:
            for beta in cfg.beta_candidates:
                exp_cfg = copy.deepcopy(cfg)
                exp_cfg.lambda_max = lam
                exp_cfg.beta_init  = beta

                for kind in kinds:
                    _, _, _, metrics = run_single(exp_cfg, seed=seed, kind=kind)
                    for m in Metrics:
                        # CVR은 Z 모델에만 있으므로 안전하게 처리
                        if m in metrics:
                            summary[kind][(lam, beta)][m].append(metrics[m])
                        else:
                            # VA 모델에서 CVR이 없는 경우 NaN 추가
                            summary[kind][(lam, beta)][m].append(float('nan'))

    # write summaries ─────────────────────────────────────────────────
    for kind in kinds:
        print(f"\n=== {kind} summary across seeds ===")
        serial = {}
        for (lam, beta), vals in summary[kind].items():
            tag = f"lam_{lam:g}_beta_{beta:g}"
            stats = {}
            for m in Metrics:
                arr = np.asarray(vals[m], dtype=float)
                # NaN 값들을 필터링 (VA 모델의 CVR 등)
                valid_arr = arr[~np.isnan(arr)]
                if len(valid_arr) > 0:
                    mu = float(valid_arr.mean())
                    sd = float(valid_arr.std(ddof=1)) if len(valid_arr) > 1 else 0.0
                    print(f"λ={lam:g}, β={beta:g}  {m:<6}: {mu:.4f} ± {sd:.4f}")
                else:
                    print(f"λ={lam:g}, β={beta:g}  {m:<6}: N/A")
                stats[m] = vals[m]  # raw list for JSON
            serial[tag] = stats

        (out_root / f"summary_{kind}.json").write_text(json.dumps(serial, indent=2))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IO‑GNN experiment runner")
    parser.add_argument("--kinds", nargs="+", default=["Z", "VA"],
                        help="Which models to train (Z VA)")
    parser.add_argument("--out_dir", type=str, help="Override cfg.out_dir")
    args = parser.parse_args()

    cfg = Config()

    if args.out_dir:
        cfg.out_dir = args.out_dir

    run_all(cfg, kinds=args.kinds)
