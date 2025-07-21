# main.py  ──────────────────────────────────────────────────────────────
from pathlib import Path
import copy, json, numpy as np, torch
from config import Config
from utils  import set_seed
from run_single import run_single

# ───────────────────────────────────────────────────────────────────────
def main() -> None:
    base_cfg  = Config()
    out_root  = Path(base_cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    kinds       = ["Z", "VA"]                     # NEW
    metrics_all = ["RMSE", "MAE", "SMAPE", "R2", "rho", "CVR"]

    # summary[kind][(λ,β)][metric] = list over seeds
    summary: dict[str, dict[tuple[float,float], dict[str, list[float]]]] = {
        k: {
            (lam, beta): {m: [] for m in metrics_all}
            for lam in base_cfg.lambda_candidates
            for beta in base_cfg.beta_candidates
        }
        for k in kinds
    }

    # ───────── sweep ─────────
    for seed in base_cfg.seeds:
        set_seed(seed)

        for lam in base_cfg.lambda_candidates:
            for beta in base_cfg.beta_candidates:

                # cfg 복사 → 조합별 하이퍼 주입
                cfg = copy.deepcopy(base_cfg)
                cfg.lambda_max = lam
                cfg.beta_init  = beta

                for kind in kinds:
                    # 1) 학습 + 테스트
                    model, hist, _, test_metrics = run_single(cfg, seed, kind=kind)

                    # 2) summary 갱신
                    for m in metrics_all:
                        summary[kind][(lam, beta)][m].append(test_metrics[m])

                    # 3) 디렉터리
                    run_dir = (
                        out_root / kind / f"seed_{seed}" / f"lam_{lam:g}_beta_{beta:g}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)

                    # 4) artefacts 저장
                    torch.save(model.cpu().state_dict(), run_dir / "model.pth")
                    model.to(cfg.device)

                    with open(run_dir / "train_history.json", "w") as f:
                        json.dump({k: [float(v) for v in lst] for k,lst in hist.items()},
                                  f, indent=2)

    # ─────── summary 출력 & 저장 ───────
    for kind in kinds:
        print(f"\n=== {kind} cross-seed summary (mean ± std) ===")
        for (lam, beta), vals in summary[kind].items():
            print(f"λ {lam:g} | β {beta:g}")
            for m, arr in vals.items():
                arr = np.array(arr, dtype=float)
                mu, sd = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0
                print(f"  {m:<6}: {mu:.4f} ± {sd:.4f}")

        # JSON dump
        serial = {
            f"lam_{lam:g}_beta_{beta:g}": {k: list(map(float, v))
                                           for k, v in vals.items()}
            for (lam, beta), vals in summary[kind].items()
        }
        (out_root / f"summary_{kind}.json").write_text(json.dumps(serial, indent=2))


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()