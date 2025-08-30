# experiment.py  — robust sweep runner for IO-GNN
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import Config
from utils import set_seed
from run_single import run_single

# Metrics returned by run_single(...). IOIS applies to Z only (NaN for VA).
METRICS: tuple[str, ...] = ("RMSE", "MAE", "SMAPE", "R2", "RHO", "IOIS")


def _normalize_kinds(kinds: List[str]) -> List[str]:
    allowed = {"Z", "VA"}
    normed = []
    for k in kinds:
        k2 = k.strip().upper()
        if k2 not in allowed:
            raise ValueError(f"Unknown kind '{k}'. Allowed: {sorted(allowed)}")
        if k2 not in normed:
            normed.append(k2)
    return normed


def _init_summary(kinds: List[str], lambdas: List[float], betas: List[float]
                  ) -> Dict[str, Dict[Tuple[float, float], Dict[str, List[float]]]]:
    return {
        kind: {
            (lam, beta): {m: [] for m in METRICS}
            for lam in lambdas
            for beta in betas
        }
        for kind in kinds
    }


def _add_metrics(summary, kind: str, lam: float, beta: float, metrics: Dict[str, float]) -> None:
    bucket = summary[kind][(lam, beta)]
    for m in METRICS:
        bucket[m].append(float(metrics.get(m, float("nan"))))


def _finalize_and_write(out_root: Path, kinds: List[str], summary) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        serial = {}
        print(f"\n=== {kind} summary across seeds ===")
        # Sort keys for stable output
        for (lam, beta) in sorted(summary[kind].keys(), key=lambda x: (x[0], x[1])):
            vals = summary[kind][(lam, beta)]
            tag = f"lam_{lam:g}_beta_{beta:g}"
            stats: Dict[str, List[float]] = {}
            for m in METRICS:
                arr = np.asarray(vals[m], dtype=float)
                valid = arr[~np.isnan(arr)]
                if valid.size > 0:
                    mu = float(valid.mean())
                    sd = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
                    print(f"λ={lam:g}, β={beta:g}  {m:<6}: {mu:.4f} ± {sd:.4f}")
                else:
                    print(f"λ={lam:g}, β={beta:g}  {m:<6}: N/A")
                # Keep raw values (NaNs allowed) for downstream analysis
                stats[m] = [None if np.isnan(x) else float(x) for x in arr]
            serial[tag] = stats

        (out_root / f"summary_{kind}.json").write_text(json.dumps(serial, indent=2))

        # Optional: also emit a compact CSV with mean/std for quick plotting
        lines = ["lambda,beta," + ",".join([f"{m}_mean,{m}_std" for m in METRICS])]
        for (lam, beta) in sorted(summary[kind].keys(), key=lambda x: (x[0], x[1])):
            vals = summary[kind][(lam, beta)]
            row = [f"{lam}", f"{beta}"]
            for m in METRICS:
                arr = np.asarray(vals[m], dtype=float)
                valid = arr[~np.isnan(arr)]
                if valid.size > 0:
                    mu = float(valid.mean())
                    sd = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
                else:
                    mu, sd = np.nan, np.nan
                row.extend([f"{mu}", f"{sd}"])
            lines.append(",".join(row))
        (out_root / f"summary_{kind}.csv").write_text("\n".join(lines))


def run_all(cfg: Config, kinds: List[str]) -> None:
    """Run seeds × lambda_candidates × beta_candidates across kinds; write summaries."""
    kinds = _normalize_kinds(kinds)
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = _init_summary(kinds, cfg.lambda_candidates, cfg.beta_candidates)

    # Sweep
    t0 = time.time()
    failures: list[dict] = []

    for seed in cfg.seeds:
        set_seed(seed)  # also seed DataLoader generator/worker in your training code
        for lam in cfg.lambda_candidates:
            for beta in cfg.beta_candidates:
                # Create an immutable variant of cfg for this run
                exp_cfg = replace(cfg, lambda_max=lam, beta_init=beta)
                for kind in kinds:
                    try:
                        _, _, _, metrics = run_single(exp_cfg, seed=seed, kind=kind)
                        _add_metrics(summary, kind, lam, beta, metrics)
                    except Exception as e:
                        failures.append(
                            {"seed": seed, "lambda": lam, "beta": beta, "kind": kind, "error": repr(e)}
                        )
                        # Record NaNs for this failed cell to keep shapes aligned
                        _add_metrics(summary, kind, lam, beta, {m: float("nan") for m in METRICS})

    # Write summaries (JSON + CSV) and a small manifest
    _finalize_and_write(out_root, kinds, summary)

    manifest = {
        "kinds": kinds,
        "seeds": cfg.seeds,
        "lambda_candidates": cfg.lambda_candidates,
        "beta_candidates": cfg.beta_candidates,
        "out_dir": str(cfg.out_dir),
        "elapsed_sec": round(time.time() - t0, 3),
        "failures": failures,
    }
    (out_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    if failures:
        print(f"\n⚠ Completed with {len(failures)} failures. See experiment_manifest.json for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IO-GNN experiment runner")
    parser.add_argument(
        "--kinds", nargs="+", default=["Z", "VA"],
        help="Which models to train (choices: Z, VA)"
    )
    parser.add_argument("--out_dir", type=str, help="Override cfg.out_dir")
    args = parser.parse_args()

    cfg = Config()
    if args.out_dir:
        cfg.out_dir = Path(args.out_dir)

    run_all(cfg, kinds=args.kinds)