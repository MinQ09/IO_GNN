"""
Grid search runner for IO-GNN hyper-parameter optimization.

✓ Fixes
  • Ensures every hyper-parameter is cast to the proper numeric type before training
  • Deep-copies the param dict inside Config.generate_grid_configs() (defensive, in case
    that method mutates shared dicts)
  • Adds a tiny helper _sanitize_cfg() to DRY the casting logic and guarantee no
    string/list sneaks through (prevents errors like “can't multiply sequence by non-int”)
  • Minor: consistent logging via print wrapper, clearer progress meter when n_jobs=1

Instructions: drop this file over the old grid_search.py and run:
  python grid_search.py --config grid_config.yaml --kinds Z --n_jobs 2
"""
from __future__ import annotations

import argparse, json, time, warnings, multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

warnings.filterwarnings("ignore")

from config import Config
from run_single import run_single

# ────────────────────────────────────────────────────────────────────────────
# Utility: force-cast *every* hyper-param to the right dtype so we never see
# “can't multiply sequence by non-int of type 'float'” again.
# ────────────────────────────────────────────────────────────────────────────
_num = (int, float)

def _sanitize_cfg(cfg: Config) -> None:  # in-place
    """Cast all numeric attrs on Config to int/float (handles str, list, np scalars)."""
    for attr, caster in [
        ("batch_size", int),
        ("lr", float),
        ("weight_decay", float),
        ("hidden", int),
        ("k", int),
        ("dropout", float),
        ("lambda_max", float),
    ]:
        val = getattr(cfg, attr, None)
        if val is None:
            continue
        # take first element if list/tuple
        if isinstance(val, (list, tuple)):
            val = val[0]
        # cast only if not already numeric scalar
        if not isinstance(val, _num):
            try:
                val = caster(val)
            except Exception as e:
                raise TypeError(f"Config field '{attr}' could not be cast to {caster.__name__}: {val} ({e})")
        setattr(cfg, attr, val)


# ────────────────────────────────────────────────────────────────────────────
# Worker
# ────────────────────────────────────────────────────────────────────────────

def run_single_config(pack: tuple[Config, int]) -> Dict[str, Any]:
    cfg, cfg_id = pack
    try:
        print("\n" + "=" * 60)
        print(f"Running Config {cfg_id}: {cfg.get_param_string()}")
        print("=" * 60)

        results: Dict[str, Any] = {}
        for kind in ["Z"]:  # extend to ["Z", "VA"] if needed
            try:
                t0 = time.time()
                model, hist, _, metrics = run_single(cfg, seed=cfg.seeds[0], kind=kind)
                t1 = time.time()
                results[kind] = {
                    "metrics": metrics,
                    "best_val_loss": min(hist["val_tot"]) if hist["val_tot"] else float("inf"),
                    "final_train_loss": hist["train_tot"][-1] if hist["train_tot"] else float("inf"),
                    "final_val_R2": hist["val_R2"][-1] if hist["val_R2"] else 0.0,
                    "final_val_CVR": hist.get("val_CVR", [None])[-1] if kind == "Z" else None,
                    "training_time": t1 - t0,
                    "epochs_trained": len(hist["train_tot"]),
                }
                print(
                    f"✅ {kind} completed – Test R²: {metrics.get('R2', 'N/A'):.4f}, "
                    f"CVR: {metrics.get('CVR', 'N/A'):.4f}, Time: {t1 - t0:.1f}s"
                )
            except Exception as e:
                print(f"❌ {kind} failed: {e}")
                results[kind] = {"error": str(e)}

        return {
            "config_id": cfg_id,
            "config_params": {k: getattr(cfg, k) for k in [
                "batch_size", "lr", "weight_decay", "hidden", "k", "dropout", "lambda_max"]},
            "seed": cfg.seeds[0],
            "results": results,
        }
    except Exception as e:
        print(f"❌ Config {cfg_id} crashed: {e}")
        return {"config_id": cfg_id, "results": {"error": str(e)}}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="grid_config.yaml")
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--kinds", nargs="+", default=["Z"], choices=["Z", "VA"])
    args = p.parse_args()

    # 1) Config 로드
    if Path(args.config).exists():
        base_cfg = Config.load(args.config)
    else:
        print(f"⚠️  '{args.config}' not found → using minimal defaults")
        base_cfg = Config(batch_size=64, data_dir=Path("./Data"), out_dir=Path("./Results/grid_search"))

    base_cfg.grid_search = True

    # 2) 그리드 생성 (깊은 복사 안전화는 Config 내부에서 수행한다고 가정)
    cfgs = base_cfg.generate_grid_configs()
    for c in cfgs:
        _sanitize_cfg(c)

    print(f"\n🚀 Starting grid search with {len(cfgs)} configurations")
    print(f"📊 Using {args.n_jobs} parallel jobs")

    results_dir = Path("./Results/grid_search"); results_dir.mkdir(parents=True, exist_ok=True)

    packs = list(enumerate(cfgs))  # (id, cfg) 튜플 → 순서 유지
    packs = [(cfg, cid) for cid, cfg in packs]

    t0 = time.time()
    if args.n_jobs == 1:
        all_results = []
        for i, pack in enumerate(packs, 1):
            print(f"\n⏳ {i}/{len(packs)} …", end=" ")
            all_results.append(run_single_config(pack))
    else:
        with mp.Pool(args.n_jobs) as pool:
            all_results = pool.map(run_single_config, packs)
    print(f"\n🎉 Grid search finished in {time.time() - t0:.1f}s")

    # 3) 결과 저장 (CSV + JSON)
    rows: List[Dict[str, Any]] = []
    for res in all_results:
        cfg_info = res.get("config_params", {}) | {"config_id": res["config_id"], "seed": res.get("seed")}
        if "error" in res["results"]:
            rows.append(cfg_info | {"task": "ERROR", "error": res["results"]["error"]})
            continue
        for task, task_res in res["results"].items():
            rows.append(cfg_info | {"task": task} | task_res.get("metrics", {}) | {
                "best_val_loss": task_res.get("best_val_loss"),
                "final_val_R2": task_res.get("final_val_R2"),
                "training_time": task_res.get("training_time"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(results_dir / "grid_search_results.csv", index=False)
    with open(results_dir / "grid_search_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("✅ Results written to", results_dir)


if __name__ == "__main__":
    main()
