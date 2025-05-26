import argparse, torch
from pathlib import Path

from config  import Config
from trainer import run

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--out_dir",  default="./runs")
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

cfg = Config(
    data_dir=args.data_dir,
    out_dir =args.out_dir,
)

Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

best_model, (best_lam, best_beta), best_val = run(cfg, seed=args.seed)

torch.save(best_model.state_dict(), Path(cfg.out_dir)/"best_model.pth")
(cfg.out_dir/ "best_cfg.yaml").write_text(f"lambda: {best_lam}\nbeta: {best_beta}\nval_smape: {best_val}\n")

print(f"✓ done  |  λ={best_lam:g}  β={best_beta:g}  Val-SMAPE={best_val:.4f}")
