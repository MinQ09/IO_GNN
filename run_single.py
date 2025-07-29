# run_single.py ────────────────────────────────────────────────
"""
Single‑run trainer for the IO‑GNN models with selective scaling strategy.

Key changes for selective scaling:
1. **Input features only** are standardized (Import/Export/Final_Demand)  
2. **Z matrix and VA targets** remain in raw scale to preserve variance structure
3. **Raw-scale PINN losses** replace standardized versions
4. **Conditional inverse transform** handles identity scalers
5. **Adjusted learning rate and gradient clipping** for raw-scale stability

Supported tasks
---------------
* ``kind="Z"``  –   predict **edge flows** (inter‑industry transactions)
* ``kind="VA"`` –   predict **node value‑added**
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

from data_io import GraphWindowDataset, collate_window
from helper import (
    dump_pred_matrices,
    inverse_transform_predictions,
    inverse_transform_targets,
    save_edge_attention,
)
from losses import (
    pinn_loss_z_batch_raw,
    pinn_loss_va_batch_raw
)
from metrics import (
    mae,
    mean_ignore_nan,
    rmse,
    r2,
    safe_pearson,
    smape,
)
from model import IOGNN_VA, IOGNN_Z
from utils import set_seed

# ───────────────────────── helpers ──────────────────────────

def _slice_batch(cat: torch.Tensor, graphs: List[pyg.data.Data], edge: bool):
    """Yield *cat* slices belonging to individual *graphs*."""
    offs = 0
    for g in graphs:
        span = g.edge_attr.numel() if edge else g.num_nodes
        yield cat[offs : offs + span], g
        offs += span


def _adaptive_lambda(
    *, mse: torch.Tensor, pinn: torch.Tensor, cfg, global_step: int
) -> torch.Tensor:
    """Equation (7) in the paper – with warm‑up and numerical guard."""
    scale = (
        1.0
        if cfg.warmup == 0
        else min(1.0, float(global_step) / cfg.warmup)
    )
    return (
        scale
        * cfg.lambda_max
        * (mse.detach() / (pinn.detach() + 1e-12)).clamp(max=10.0)
    )


def is_identity_scaler(scaler):
    """Check if scaler is identity (mean=0, scale=1)."""
    return (hasattr(scaler, 'scale_') and 
            hasattr(scaler, 'mean_') and
            abs(scaler.scale_[0] - 1.0) < 1e-6 and 
            abs(scaler.mean_[0] - 0.0) < 1e-6)


def cvr_tensor_raw(pred_slice: torch.Tensor, graph: pyg.data.Data) -> float:
    """Calculate CVR (Coefficient of Variation Ratio) for raw-scale predictions."""
    if pred_slice.numel() == 0 or graph.edge_attr.numel() == 0:
        return float('nan')
    
    pred_var = pred_slice.var().item()
    true_var = graph.edge_attr.var().item()
    
    if true_var == 0:
        return float('nan')
    
    return pred_var / true_var


def cvr_tensor_standardized(pred_slice: torch.Tensor, graph: pyg.data.Data, scalers: Dict) -> float:
    """Calculate CVR for standardized predictions (fallback)."""
    # This is a placeholder - implement if needed for backward compatibility
    return cvr_tensor_raw(pred_slice, graph)


# ─────────────────────────── main ───────────────────────────

def run_single(
    cfg: Any, seed: int, *, kind: str = "Z"
) -> Tuple[torch.nn.Module, Dict[str, List[float]], None, Dict[str, float]]:
    """Train once with *cfg* and *seed*, return the fitted model & metrics."""

    if kind not in {"Z", "VA"}:
        raise ValueError("kind must be either 'Z' or 'VA'")

    set_seed(seed)
    edge_mode = kind == "Z"

    # ─── I/O dirs ──────────────────────────────────────────
    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / f"lam_{cfg.lambda_max:.4g}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # ─── DATASETS & SCALERS with selective scaling ───────────
    years = list(range(1, 21))  
    tr_y, vl_y, ts_y = years[:-4], years[-4:-2], years[-2:]

    # Training dataset with selective scaling
    tr_ds = GraphWindowDataset(
        tr_y, 
        cfg, 
        scalers=None, 
        fit_scalers=True,
        scale_targets=False  # Z·VA를 원본 스케일로 유지
    )
    scalers = tr_ds.get_scalers()
    
    if cfg.save_scalers and not (Path(cfg.scalers_path).exists()):
        with open(cfg.scalers_path, "wb") as f:
            pickle.dump(scalers, f)

    def _mk_loader(Y, shuffle: bool, bs: int):
        return DataLoader(
            GraphWindowDataset(
                Y, 
                cfg, 
                scalers=scalers, 
                fit_scalers=False,
                scale_targets=False  # 모든 로더에서 일관되게 적용
            ),
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_window,
            pin_memory=False,
        )

    tr_ld, vl_ld, ts_ld = (
        _mk_loader(tr_y, True, cfg.batch_size),
        _mk_loader(vl_y, False, cfg.batch_size),
        _mk_loader(ts_y, False, 1),
    )

    # ─── MODEL / OPTIMISER with adjusted hyperparameters ────────────────
    model_cls = IOGNN_Z if edge_mode else IOGNN_VA
    model = model_cls(nfeat=3, cfg=cfg).to(cfg.device)

    # Raw-scale PINN functions
    if kind == "Z":
        pinn_fn = pinn_loss_z_batch_raw
        target_scaler = scalers["edge_Z"]
    elif kind == "VA":
        pinn_fn = pinn_loss_va_batch_raw
        target_scaler = scalers["node"]["value_added"]

    # Adjusted optimizer for raw-scale learning
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr * 0.2,  # 학습률을 1/5로 낮춤
        weight_decay=cfg.weight_decay
    )

    # ─── LOGGERS ──────────────────────────────────────────
    keys = (
        "train_tot",
        "train_mse",
        "train_pinn",
        "train_R2",
        "val_tot",
        "val_mse",
        "val_pinn",
        "val_RMSE",
        "val_MAE",
        "val_SMAPE",
        "val_R2",
        "val_RHO",
        "val_CVR",
        "lambda_t",
    )
    hist: Dict[str, List[float]] = {k: [] for k in keys}

    best_metric, best_state, bad_epochs = float("inf"), None, 0
    global_step = 0

    # Check scaling configuration
    print(f"\n=== Scaling Configuration ===")
    print(f"Kind: {kind}")
    print(f"Target scaler - mean: {target_scaler.mean_[0]:.6f}, scale: {target_scaler.scale_[0]:.6f}")
    print(f"Is identity scaler: {is_identity_scaler(target_scaler)}")
    print("="*35)

    # ════════════════════ TRAIN / VAL LOOP ════════════════════
    for ep in trange(1, cfg.epochs + 1, desc=f"{kind}-seed{seed}"):
        # ─── TRAIN ──────────────────────────────────────
        model.train()
        tot = mse_acc = pinn_acc = r2_acc = lam_acc = 0.0

        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_raw, *_ = model(seqs, tgts)  # Raw scale output
            tgt_raw = torch.cat(
                [g.edge_attr if edge_mode else g.va for g in tgts]
            )

            # Raw-scale PINN loss (no scalers parameter)
            pinn = pinn_fn(pred_raw, tgts)
            mse = F.mse_loss(pred_raw, tgt_raw)
            lam_t = _adaptive_lambda(
                mse=mse, pinn=pinn, cfg=cfg, global_step=global_step
            )

            loss = mse + lam_t * pinn
            optim.zero_grad()
            loss.backward()
            
            # Enhanced gradient clipping for raw-scale stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            # Conditional inverse transform for metrics
            if is_identity_scaler(target_scaler):
                pred_o, tgt_o = pred_raw, tgt_raw
            else:
                pred_o = inverse_transform_predictions(pred_raw, scalers, kind)
                tgt_o = inverse_transform_targets(tgt_raw, scalers, kind)

            # ─── logging on‑the‑fly
            tot += loss.item()
            mse_acc += mse.item()
            pinn_acc += pinn.item()
            lam_acc += lam_t.item()
            r2_acc += r2(pred_o, tgt_o)
            global_step += 1

        nb = len(tr_ld)
        lam_avg = lam_acc / nb
        hist["train_tot"].append(tot / nb)
        hist["train_mse"].append(mse_acc / nb)
        hist["train_pinn"].append(pinn_acc / nb)
        hist["train_R2"].append(r2_acc / nb)
        hist["lambda_t"].append(lam_avg)

        # Debug logging for first epoch
        if ep == 1:
            pred_mean = pred_raw.mean().item()
            pred_std_val = pred_raw.std().item()
            tgt_mean = tgt_raw.mean().item()
            tgt_std_val = tgt_raw.std().item()
            
            print(f"\n[EP 1 DEBUG] Predictions - mean: {pred_mean:.4f}, std: {pred_std_val:.4f}")
            print(f"[EP 1 DEBUG] Targets - mean: {tgt_mean:.4f}, std: {tgt_std_val:.4f}")

        tqdm.write(
            f"[EP {ep:03d}] train: loss {tot/nb:.4f} | MSE {mse_acc/nb:.4f} | "
            f"PINN {pinn_acc/nb:.6f} | λ̄ {lam_avg:.3e} | R² {r2_acc/nb:.3f}"
        )

        # ─── VALIDATION ─────────────────────────────────
        model.eval()
        v_tot = v_mse = v_pinn = 0.0
        acc = {k: [] for k in ("rmse", "mae", "smape", "r2", "RHO", "cvr")}

        with torch.no_grad():
            for seqs, tgts in vl_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [g.to(cfg.device) for g in tgts]

                pred_raw, *_ = model(seqs, tgts)
                tgt_raw = torch.cat(
                    [g.edge_attr if edge_mode else g.va for g in tgts]
                )

                pinn = pinn_fn(pred_raw, tgts)
                mse = F.mse_loss(pred_raw, tgt_raw)

                v_tot += (mse + lam_avg * pinn).item()
                v_mse += mse.item()
                v_pinn += pinn.item()

                # Conditional inverse transform
                if is_identity_scaler(target_scaler):
                    pred_o, tgt_o = pred_raw, tgt_raw
                else:
                    pred_o = inverse_transform_predictions(pred_raw, scalers, kind)
                    tgt_o = inverse_transform_targets(tgt_raw, scalers, kind)

                acc["rmse"].append(rmse(pred_o, tgt_o))
                acc["mae"].append(mae(pred_o, tgt_o))
                acc["smape"].append(smape(pred_o, tgt_o))
                acc["r2"].append(r2(pred_o, tgt_o))
                acc["RHO"].append(safe_pearson(pred_o, tgt_o))

                # CVR calculation with raw-scale support
                if edge_mode:
                    for p_slice, g in _slice_batch(pred_raw.cpu(), tgts, True):
                        acc["cvr"].append(cvr_tensor_raw(p_slice, g))

        nh = len(vl_ld)
        hist["val_tot"].append(v_tot / nh)
        hist["val_mse"].append(v_mse / nh)
        hist["val_pinn"].append(v_pinn / nh)
        hist["val_RMSE"].append(mean_ignore_nan(acc["rmse"]))
        hist["val_MAE"].append(mean_ignore_nan(acc["mae"]))
        hist["val_SMAPE"].append(mean_ignore_nan(acc["smape"]))
        hist["val_R2"].append(mean_ignore_nan(acc["r2"]))
        hist["val_RHO"].append(mean_ignore_nan(acc["RHO"]))
        hist["val_CVR"].append(mean_ignore_nan(acc["cvr"]) if edge_mode else np.nan)

        if ep % cfg.log_every == 0:
            extra = f"  CVR {hist['val_CVR'][-1]:.3e}" if edge_mode else ""
            tqdm.write(
                f"[VAL {ep:03d}] tot {hist['val_tot'][-1]:.4f} | "
                f"RMSE {hist['val_RMSE'][-1]:.2f}  MAE {hist['val_MAE'][-1]:.2f}  "
                f"SMAPE {hist['val_SMAPE'][-1]:.3f}  R² {hist['val_R2'][-1]:.3f}{extra}"
            )

        # Early stopping
        monitor = hist["val_SMAPE"][-1] if edge_mode else hist["val_MAE"][-1]
        if monitor < best_metric - 1e-8:
            best_metric, best_state, bad_epochs = (
                monitor,
                {k: v.cpu() for k, v in model.state_dict().items()},
                0,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stop @ {ep} (best={best_metric:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ─── TEST ────────────────────────────────────────────
    model.eval()
    res = {k: [] for k in ("rmse", "mae", "smape", "r2", "RHO", "cvr")}

    with torch.no_grad():
        for seqs, tgts in ts_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_raw, att_out, att_in = model(seqs, tgts)
            tgt_raw = torch.cat(
                [g.edge_attr if edge_mode else g.va for g in tgts]
            )

            # Conditional inverse transform
            if is_identity_scaler(target_scaler):
                pred_o, tgt_o = pred_raw, tgt_raw
            else:
                pred_o = inverse_transform_predictions(pred_raw, scalers, kind)
                tgt_o = inverse_transform_targets(tgt_raw, scalers, kind)

            res["rmse"].append(rmse(pred_o, tgt_o))
            res["mae"].append(mae(pred_o, tgt_o))
            res["smape"].append(smape(pred_o, tgt_o))
            res["r2"].append(r2(pred_o, tgt_o))
            res["RHO"].append(safe_pearson(pred_o, tgt_o))

            if edge_mode:
                for p_slice, g in _slice_batch(pred_raw.cpu(), tgts, True):
                    res["cvr"].append(cvr_tensor_raw(p_slice, g))

                save_edge_attention(
                    att_out,
                    att_in,
                    tgts[0].edge_index,
                    tgts[0].num_nodes,
                    kind,
                    save_dir,
                )

    metrics = {k.upper(): mean_ignore_nan(v) for k, v in res.items()}
    if not edge_mode:
        metrics.pop("CVR", None)

    # ─── SAVE ARTEFACTS ──────────────────────────────────
    scalers_path = save_dir / "scalers.pkl"
    if not scalers_path.exists():
        pickle.dump(scalers, open(scalers_path, "wb"))

    dump_pred_matrices(
        model,
        scalers_path,
        years=ts_y,
        save_dir=save_dir,
        cfg=cfg,
        kind=kind,
        save_x=False,
    )

    (save_dir / f"metrics_lambda_{cfg.lambda_max:.4g}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    
    (save_dir / f"val_history_lambda_{cfg.lambda_max:.4g}.json").write_text(
        json.dumps({k: list(map(float, v)) for k, v in hist.items()}, indent=2)
    )

    torch.save(
        model.cpu().state_dict(),
        save_dir / f"model_lambda_{cfg.lambda_max:.4g}.pth"
    )

    (save_dir / "alpha.txt").write_text(f"{model.cell.Ox.alpha.item():.6f}")

    print("\n[Test Results]")
    for k, v in metrics.items():
        print(f"{k:<5}: {v:.4f}")

    print(f"\n[Scaling Summary]")
    print(f"Target scaling: {'Identity (Raw)' if is_identity_scaler(target_scaler) else 'StandardScaler'}")
    print(f"Learning rate: {cfg.lr * 0.2:.2e} (reduced from {cfg.lr:.2e})")
    print(f"Gradient clipping: 5.0 (enhanced)")

    return model, hist, None, metrics
