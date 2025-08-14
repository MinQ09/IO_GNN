# main.py  ──────────────────────────────────────────────────────────────
"""
Entry point for running IO-GNN experiments across seeds and (lambda, beta) grids.

This script:
  - Builds a base Config
  - Sweeps over seeds × lambda_candidates × beta_candidates
  - Trains/evaluates both kinds ("Z" edge flows and "VA" value-added)
  - Aggregates metrics across seeds and writes per-kind JSON summaries
"""

from pathlib import Path
import copy
import json
import numpy as np
import torch

from config import Config
from utils import set_seed
from run_single import run_single


def main() -> None:
    base_cfg = Config()
    out_root = Path(base_cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Which model kinds to run
    kinds = ["Z", "VA"]

    # Metrics we expect from `run_single`
    # - IOIS is Z-only; we will record NaN for VA
    metrics_all = ["RMSE", "MAE", "SMAPE", "R2", "RHO", "IOIS"]

    # Prepare summary accumulator:
    # summary[kind][(lam, beta)][metric] -> list[float] across seeds
    summary: dict[str, dict[tuple[float, float], dict[str, list[float]]]] = {
        kind: {
            (lam, beta): {m: [] for m in metrics_all}
            for lam in base_cfg.lambda_candidates
            for beta in base_cfg.beta_candidates
        }
        for kind in kinds
    }

    # ---------------------------- sweep ----------------------------
    for seed in base_cfg.seeds:
        set_seed(seed)

        for lam in base_cfg.lambda_candidates:
            for beta in base_cfg.beta_candidates:
                # Clone base config and inject grid-specific hyperparameters
                cfg = copy.deepcopy(base_cfg)
                cfg.lambda_max = lam
                cfg.beta_init = beta

                for kind in kinds:
                    # 1) Train + test
                    model, hist, _, test_metrics = run_single(cfg, seed, kind=kind)

                    # 2) Update summary (IOIS may be absent for VA)
                    for m in metrics_all:
                        val = test_metrics.get(m, float("nan"))
                        summary[kind][(lam, beta)][m].append(val)

                    # 3) Save per-run artifacts
                    run_dir = out_root / kind / f"seed_{seed}" / f"lam_{lam:g}_beta_{beta:g}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    torch.save(model.cpu().state_dict(), run_dir / "model.pth")
                    model.to(cfg.device)  # move back if needed later

                    with open(run_dir / "train_history.json", "w") as f:
                        json.dump(
                            {k: [float(v) for v in lst] for k, lst in hist.items()},
                            f,
                            indent=2,
                        )

    # ---------------------- print & write summaries ----------------------
    for kind in kinds:
        print(f"\n=== {kind} cross-seed summary (mean ± std) ===")
        serial = {}

        for (lam, beta), vals in summary[kind].items():
            tag = f"lam_{lam:g}_beta_{beta:g}"
            serial[tag] = {}

            print(f"λ {lam:g} | β {beta:g}")
            for m, arr in vals.items():
                arr_np = np.array(arr, dtype=float)
                # Compute mean/std ignoring NaNs (IOIS for VA will be NaN)
                valid = arr_np[~np.isnan(arr_np)]
                if valid.size > 0:
                    mu = float(valid.mean())
                    sd = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
                    print(f"  {m:<6}: {mu:.4f} ± {sd:.4f}")
                else:
                    print(f"  {m:<6}: N/A")
                # Store raw list per grid point
                serial[tag][m] = [None if np.isnan(x) else float(x) for x in arr_np]

        (out_root / f"summary_{kind}.json").write_text(json.dumps(serial, indent=2))


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()