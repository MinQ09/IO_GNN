# experiment.py  ───────────────────────────────────────────
"""
Master driver for IO-GNN experiments.

▶ 두 모델 모두 수행:
    kind="Z"  : edge-flow prediction
    kind="VA" : node Value-Added prediction

▶ 조합:
    seeds × lambda_candidates × beta_candidates

산출물
------
<out_dir>/<kind>/seed_<s>/<lam_*>_beta_<*>/
    ├─ model.pth
    ├─ train_history.json
    ├─ csv_pred/   (pred/true/attn  CSV)
    └─ alpha.txt

<out_dir>/summary_<kind>.json
"""

# ───────── import 3rd-party ─────────
from pathlib import Path
import argparse, copy, json, numpy as np, torch

# ───────── local modules (동일 디렉터리) ─────────
from config       import Config
from utils        import set_seed            # set_seed 함수 utils.py에 위치
from run_single   import run_single          # kind 파라미터 버전
from helper   import dump_pred_matrices  # 만약 run_single 내부에서 안 쓰면

# ──────────────────────────────────────────
def run_all(cfg: Config, kinds: list[str]) -> None:
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    metrics = ["RMSE", "MAE", "SMAPE", "R2", "rho", "CVR"]

    # {kind: {(λ,β): {metric: [values]}}}
    summary: dict[str, dict[tuple[float, float], dict[str, list[float]]]] = {
        k: { (lam, beta): {m: [] for m in metrics}
             for lam in cfg.lambda_candidates
             for beta in cfg.beta_candidates }
        for k in kinds
    }

    for seed in cfg.seeds:
        set_seed(seed)

        for lam in cfg.lambda_candidates:
            for beta in cfg.beta_candidates:
                # ── 하이퍼 셋 복사 주입 ──
                exp_cfg = copy.deepcopy(cfg)
                exp_cfg.lambda_max = lam
                exp_cfg.beta_init  = beta

                for kind in kinds:
                    model, hist, _, test_metrics = run_single(
                        exp_cfg, seed, kind=kind
                    )

                    # summary update
                    for m in metrics:
                        summary[kind][(lam, beta)][m].append(test_metrics[m])

    # ───────── summary 출력 & 저장 ─────────
    for kind in kinds:
        print(f"\n=== {kind} cross-seed summary (mean ± std) ===")
        serial = {}
        for (lam, beta), vals in summary[kind].items():
            print(f"λ={lam:g}, β={beta:g}")
            for m in metrics:
                arr = np.array(vals[m], dtype=float)
                mu  = arr.mean()
                sd  = arr.std(ddof=1) if len(arr) > 1 else 0.0
                print(f"  {m:<6}: {mu:.4f} ± {sd:.4f}")
            serial[f"lam_{lam:g}_beta_{beta:g}"] = {
                k: list(map(float, v)) for k, v in vals.items()
            }

        (out_root / f"summary_{kind}.json").write_text(
            json.dumps(serial, indent=2)
        )


# ──────────────── CLI entrypoint ────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="IO-GNN Z / VA experiment runner")
    p.add_argument("--kinds", nargs="+", default=["Z", "VA"],
                   help="Which models to train (Z VA)")
    p.add_argument("--out_dir", type=str, help="Override cfg.out_dir")
    args = p.parse_args()

    cfg = Config()
    if args.out_dir:
        cfg.out_dir = args.out_dir

    run_all(cfg, kinds=args.kinds)
