# run_single.py ─────────────────────────────────────────────────────────
"""
Single-run trainer for IO-GNN (edge flow “Z” or node value-added “VA”).

Key points
----------
1.  The PINN term is computed **directly in standardized space** (no inverse
    transform required).                        ← critical
2.  A mini-batch–wise adaptive weight λₜ is used:
       λₜ = λ_max · MSE / (PINN + ε)
3.  Training log format
       train : loss | MSE | PINN | λ̄ | R²
       val   : tot  | RMSE | MAE | SMAPE | R² | (CVR)
4.  All metrics (RMSE, MAE, …) are evaluated **in the original scale**.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
import torch_geometric.data as pyg

from data_io import GraphWindowDataset, collate_window, inverse_transform_1d
from metrics  import (
    rmse, mae, smape, r2, cvr_tensor_standardized,
    mean_ignore_nan, safe_pearson,
)
from losses   import get_pinn_loss_function
from model    import IOGNN_Z, IOGNN_VA
from utils    import set_seed
from helper   import dump_pred_matrices, save_edge_attention

# ────────────────────────── helpers ──────────────────────────
def _slice_batch(
    cat: torch.Tensor,
    graphs: List[pyg.data.Data],
    edge_mode: bool,
):
    """Iterate over (slice, graph) pairs within a concatenated batch tensor."""
    off = 0
    for g in graphs:
        span = g.edge_attr.numel() if edge_mode else g.num_nodes
        yield cat[off : off + span], g
        off += span


EPS_LAM = 1e-12  # numerical stability for adaptive λₜ denominator
# ──────────────────────────── main ───────────────────────────
def run_single(
    cfg: Any,
    seed: int,
    *,
    kind: str = "Z",  # "Z" (edge) or "VA" (node)
) -> Tuple[torch.nn.Module, Dict[str, list], None, Dict[str, float]]:

    assert kind in {"Z", "VA"}
    set_seed(seed)
    edge_mode = kind == "Z"

    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) DATA ────────────────────────────────────────────────
    years       = list(range(1, 67))
    train_y, val_y, test_y = years[:-12], years[-12:-6], years[-6:]

    # fit scalers on training split only
    train_ds = GraphWindowDataset(train_y, cfg, scalers=None, fit_scalers=True)
    scalers  = train_ds.get_scalers()
    if cfg.save_scalers:
        pickle.dump(scalers, open(cfg.scalers_path, "wb"))

    def make_loader(Y, shuffle, bs):
        return DataLoader(
            GraphWindowDataset(Y, cfg, scalers, fit_scalers=False),
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_window,
            pin_memory=True,
        )

    train_ld = make_loader(train_y, True,  cfg.batch_size)
    val_ld   = make_loader(val_y,   False, cfg.batch_size)
    test_ld  = make_loader(test_y,  False, 1)

    # 2) MODEL & LOSS ───────────────────────────────────────
    model   = (IOGNN_Z if edge_mode else IOGNN_VA)(nfeat=3, cfg=cfg).to(cfg.device)
    pinn_fn = get_pinn_loss_function(kind)  # operates in standardized space
    optim   = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)

    scaler_Z  = scalers["edge_Z"]
    scaler_VA = scalers["node"]["value_added"]
    get_scaler = lambda: scaler_Z if edge_mode else scaler_VA

    # 3) LOGGING SETUP ──────────────────────────────────────
    keys = (
        "train_tot", "train_mse", "train_pinn", "train_R2",
        "val_tot", "val_mse", "val_pinn",
        "val_RMSE", "val_MAE", "val_SMAPE", "val_R2", "val_RHO", "val_CVR",
        "lambda_t",
    )
    hist: Dict[str, List[float]] = {k: [] for k in keys}
    best_metric, best_state, bad_epochs = float("inf"), None, 0

    # ═════════════════════ Train / Validate ═════════════════════
    for ep in trange(1, cfg.epochs + 1, desc=f"{kind}-seed{seed}"):
        # ───── TRAIN ─────
        model.train()
        tot = mse_sum = pinn_sum = r2_sum = lam_sum = 0.0

        for seqs, tgts in train_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, *_ = model(seqs, tgts)  # standardized output
            tgt_std      = torch.cat(
                [g.edge_attr if edge_mode else g.va for g in tgts]
            )

            # PINN uses standardized predictions – keep graph detached
            pinn = pinn_fn(pred_std, tgts, scalers)
            mse  = F.mse_loss(pred_std, tgt_std)

            # mini-batch adaptive λₜ
            lam_t = cfg.lambda_max * (mse.detach() /
                     (pinn.detach() + EPS_LAM)).clamp(max=1.0)

            loss = mse + lam_t * pinn
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            # aggregate logs
            tot += loss.item()
            mse_sum += mse.item()
            pinn_sum += pinn.item()
            lam_sum += lam_t.item()

            # batch-wise R² in original scale
            pred_orig = inverse_transform_1d(pred_std, get_scaler())
            tgt_orig  = inverse_transform_1d(tgt_std,  get_scaler())
            r2_sum += r2(pred_orig, tgt_orig)

        nb        = len(train_ld)
        lam_mean  = lam_sum / nb
        hist["train_tot"].append(tot / nb)
        hist["train_mse"].append(mse_sum / nb)
        hist["train_pinn"].append(pinn_sum / nb)
        hist["train_R2"].append(r2_sum / nb)
        hist["lambda_t"].append(lam_mean)

        tqdm.write(
            f"[EP {ep:03d}] train: "
            f"loss {tot/nb:.4f} | MSE {mse_sum/nb:.4f} | "
            f"PINN {pinn_sum/nb:.4f} | λ̄ {lam_mean:.3e} | R² {r2_sum/nb:.3f}"
        )

        # ───── VALIDATION ─────
        model.eval()
        v_tot = v_mse = v_pinn = 0.0
        acc = {k: [] for k in ("rmse", "mae", "smape", "r2", "rho", "cvr")}

        with torch.no_grad():
            for seqs, tgts in val_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [g.to(cfg.device) for g in tgts]

                pred_std, *_ = model(seqs, tgts)
                tgt_std      = torch.cat(
                    [g.edge_attr if edge_mode else g.va for g in tgts]
                )

                pinn  = pinn_fn(pred_std, tgts, scalers)
                mse   = F.mse_loss(pred_std, tgt_std)
                v_tot += (mse + lam_mean * pinn).item()
                v_mse += mse.item()
                v_pinn += pinn.item()

                pred_orig = inverse_transform_1d(pred_std, get_scaler())
                tgt_orig  = inverse_transform_1d(tgt_std,  get_scaler())

                acc["rmse"].append(rmse(pred_orig, tgt_orig))
                acc["mae" ].append(mae (pred_orig, tgt_orig))
                acc["smape"].append(smape(pred_orig, tgt_orig))
                acc["r2"   ].append(r2   (pred_orig, tgt_orig))
                acc["rho"  ].append(safe_pearson(pred_orig, tgt_orig))

                if edge_mode:
                    for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                        acc["cvr"].append(
                            cvr_tensor_standardized(p_slice, g, scalers)
                        )

        nh = len(val_ld)
        hist["val_tot" ].append(v_tot / nh)
        hist["val_mse" ].append(v_mse / nh)
        hist["val_pinn"].append(v_pinn / nh)
        hist["val_RMSE"].append(mean_ignore_nan(acc["rmse"]))
        hist["val_MAE" ].append(mean_ignore_nan(acc["mae" ]))
        hist["val_SMAPE"].append(mean_ignore_nan(acc["smape"]))
        hist["val_R2"  ].append(mean_ignore_nan(acc["r2"  ]))
        hist["val_RHO" ].append(mean_ignore_nan(acc["rho" ]))
        hist["val_CVR" ].append(
            mean_ignore_nan(acc["cvr"]) if edge_mode else np.nan
        )

        if ep % cfg.log_every == 0:
            cvr_str = f"  CVR {hist['val_CVR'][-1]:.3e}" if edge_mode else ""
            tqdm.write(
                f"[VAL {ep:03d}] tot {hist['val_tot'][-1]:.4f} | "
                f"RMSE {hist['val_RMSE'][-1]:.2f}  "
                f"MAE {hist['val_MAE'][-1]:.2f}  "
                f"SMAPE {hist['val_SMAPE'][-1]:.3f}  "
                f"R² {hist['val_R2'][-1]:.3f}{cvr_str}"
            )

        # Early-stopping check
        monitor = hist["val_SMAPE"][-1] if edge_mode else hist["val_MAE"][-1]
        if monitor < best_metric - 1e-8:
            best_metric, best_state, bad_epochs = monitor, \
                {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stop @ {ep} (best={best_metric:.4f})")
                break

    # restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # 4) TEST ──────────────────────────────────────────────
    model.eval()
    res = {k: [] for k in ("rmse", "mae", "smape", "r2", "rho", "cvr")}

    with torch.no_grad():
        for seqs, tgts in test_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, att_out, att_in = model(seqs, tgts)
            tgt_std = torch.cat(
                [g.edge_attr if edge_mode else g.va for g in tgts]
            )

            pred_orig = inverse_transform_1d(pred_std, get_scaler())
            tgt_orig  = inverse_transform_1d(tgt_std,  get_scaler())

            res["rmse"].append(rmse(pred_orig, tgt_orig))
            res["mae" ].append(mae (pred_orig, tgt_orig))
            res["smape"].append(smape(pred_orig, tgt_orig))
            res["r2"   ].append(r2   (pred_orig, tgt_orig))
            res["rho"  ].append(safe_pearson(pred_orig, tgt_orig))

            if edge_mode:
                for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                    res["cvr"].append(
                        cvr_tensor_standardized(p_slice, g, scalers)
                    )
                save_edge_attention(
                    att_out, att_in,
                    tgts[0].edge_index, tgts[0].num_nodes,
                    kind, save_dir,
                )

    metrics = {k.upper(): mean_ignore_nan(v) for k, v in res.items()}
    if not edge_mode:
        metrics.pop("CVR", None)

    # 5) SAVE ARTEFACTS ────────────────────────────────────
    scalers_path = save_dir / "scalers.pkl"
    if not scalers_path.exists():
        pickle.dump(scalers, open(scalers_path, "wb"))

    dump_pred_matrices(
        model,
        scalers_path,      # pass path, not dict
        years=test_y,
        save_dir=save_dir,
        cfg=cfg,
        kind=kind,
        save_x=False,
    )

    torch.save(model.cpu().state_dict(), save_dir / "model.pth")
    (save_dir / "alpha.txt").write_text(f"{model.cell.Ox.alpha.item():.6f}")
    (save_dir / "val_history.json").write_text(
        json.dumps({k: list(map(float, v)) for k, v in hist.items()}, indent=2)
    )

    print("\n[Test]")
    for k, v in metrics.items():
        print(f"{k:<5}: {v:.4f}")

    return model, hist, None, metrics