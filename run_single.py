# run_single.py ────────────────────────────────────────────────
"""
Single-run trainer for the IO-GNN models with selective scaling strategy,
with optional Rolling-Window cross-validation.

Key changes:
1. Input features standardized; Z/VA targets raw-scale.
2. Raw-scale PINN losses.
3. Conditional inverse transform for metrics.
4. Adjustable learning rate & gradient clipping.
5. Rolling Window Validation when cfg.rolling_val=True.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import BaseCrossValidator
from tqdm.auto import trange, tqdm
from copy import deepcopy

import torch_geometric.data as pyg

from data_io import GraphWindowDataset, collate_window
from helper import (
    dump_pred_matrices,
    inverse_transform_predictions,
    inverse_transform_targets,
    save_edge_attention,
)
from losses import pinn_loss_z_batch_raw, pinn_loss_va_batch_raw
from metrics import mae, mean_ignore_nan, rmse, r2, safe_pearson, smape
from model import IOGNN_Z, IOGNN_VA
from utils import set_seed

# ───────────────────────── helpers ──────────────────────────

def _slice_batch(cat: torch.Tensor, graphs: List[pyg.data.Data], edge: bool):
    offs = 0
    for g in graphs:
        span = g.edge_attr.numel() if edge else g.num_nodes
        yield cat[offs:offs+span], g
        offs += span

def _adaptive_lambda(
    *, mse: torch.Tensor, pinn: torch.Tensor,
    cfg, global_step: int
) -> torch.Tensor:
    warm = 1.0 if cfg.warmup == 0 else min(1.0, global_step / cfg.warmup)
    ratio = torch.sqrt(mse.detach() / (pinn.detach() + 1e-12))
    ratio = ratio.clamp(min=0.1, max=10.0)
    return warm * cfg.lambda_max * ratio

def is_identity_scaler(scaler):
    return (hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_')
            and abs(scaler.scale_[0] - 1.0) < 1e-6 and abs(scaler.mean_[0]) < 1e-6)

def cvr_tensor_raw(pred_slice: torch.Tensor, graph: pyg.data.Data) -> float:
    if pred_slice.numel() == 0 or graph.edge_attr.numel() == 0:
        return float('nan')
    pred_var, true_var = pred_slice.var().item(), graph.edge_attr.var().item()
    return pred_var / true_var if true_var != 0 else float('nan')

# ───────────────────── Rolling Window CV ─────────────────────

class RollingWindowSplit(BaseCrossValidator):
    """Time-series Rolling-Window CV over a list X (e.g., years)."""
    def __init__(self,
                 n_splits: int = 3,
                 train_size: Optional[int] = None,
                 test_size: int = 2,
                 gap: int = 0):
        self.n_splits, self.train_size, self.test_size, self.gap = \
            n_splits, train_size, test_size, gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        tr_len = self.train_size or int(n * 0.7)
        step = max(1, (n - tr_len - self.test_size) // (self.n_splits - 1)) if self.n_splits > 1 else 0
        for k in range(self.n_splits):
            start = k * step
            tr_end = start + tr_len
            ts_start = tr_end + self.gap
            ts_end = ts_start + self.test_size
            if ts_end > n:
                break
            yield np.arange(start, tr_end), np.arange(ts_start, ts_end)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def train_and_eval_fold(
    fold_id: int,
    model_cls,
    cfg,
    years: List[int],
    tr_idx: np.ndarray,
    vl_idx: np.ndarray,
    edge_mode: bool
) -> float:
    """Train on train_years, eval on val_years, return validation R²."""
    # Select years per fold
    train_years = [years[i] for i in tr_idx]
    val_years   = [years[i] for i in vl_idx]

    # Fit scalers on training subset
    tr_ds = GraphWindowDataset(train_years, cfg,
                               scalers=None, fit_scalers=True, scale_targets=False)
    scalers = deepcopy(tr_ds.get_scalers())
    vl_ds = GraphWindowDataset(val_years, cfg,
                               scalers=scalers, fit_scalers=False, scale_targets=False)

    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_window)
    vl_ld = DataLoader(vl_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_window)

    model = model_cls(nfeat=tr_ds.nfeat if hasattr(tr_ds, 'nfeat') else 3, cfg=cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pinn_fn = pinn_loss_z_batch_raw if edge_mode else pinn_loss_va_batch_raw

    # Short training per fold
    for _ in range(getattr(cfg, 'fold_epochs', 5)):
        model.train()
        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]
            pred, *_ = model(seqs, tgts)
            tgt = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
            loss = F.mse_loss(pred, tgt) + pinn_fn(pred, tgts)
            optim.zero_grad(); loss.backward(); optim.step()

    # Validation R²
    model.eval()
    scores = []
    with torch.no_grad():
        for seqs, tgts in vl_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]
            pred, *_ = model(seqs, tgts)
            tgt = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
            scores.append(r2(pred, tgt))
    fold_r2 = float(np.mean(scores)) if scores else 0.0
    tqdm.write(f"[Fold {fold_id+1}] R² = {fold_r2:.3f}")
    return fold_r2

# ────────────────────────── run_single ─────────────────────────

def run_single(cfg: Any, seed: int, *, kind: str = "Z"):
    set_seed(seed)
    edge_mode = (kind == "Z")
    model_cls = IOGNN_Z if edge_mode else IOGNN_VA

    # All years data
    years = list(range(1, 21))

    # Rolling-Window CV branch
    if getattr(cfg, 'rolling_val', False):
        splitter = RollingWindowSplit(
            n_splits=cfg.rolling_splits,
            train_size=cfg.rolling_train_size,
            test_size=cfg.rolling_test_size,
            gap=cfg.rolling_gap
        )
        fold_r2 = []
        for k, (tr_idx, vl_idx) in enumerate(splitter.split(years)):
            set_seed(seed + k)
            fold_r2.append(
                train_and_eval_fold(k, model_cls, cfg, years, tr_idx, vl_idx, edge_mode)
            )
        mean_r2, std_r2 = float(np.mean(fold_r2)), float(np.std(fold_r2))
        print(f"\n>> Rolling-CV R² = {mean_r2:.3f} ± {std_r2:.3f}")
        return None, None, None, {'ROLLING_R2': mean_r2, 'ROLLING_R2_STD': std_r2}

    # ── Standard train/val/test branch ──────────────────────────

    # I/O dirs
    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / f"lam_{cfg.lambda_max:.4g}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # train/val/test splits
    tr_y, vl_y, ts_y = years[:-4], years[-4:-2], years[-2:]
    tr_ds = GraphWindowDataset(tr_y, cfg, scalers=None, fit_scalers=True, scale_targets=False)
    scalers = deepcopy(tr_ds.get_scalers())
    vl_ds = GraphWindowDataset(vl_y, cfg, scalers=scalers, fit_scalers=False, scale_targets=False)
    ts_ds = GraphWindowDataset(ts_y, cfg, scalers=scalers, fit_scalers=False, scale_targets=False)

    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_window)
    vl_ld = DataLoader(vl_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_window)
    ts_ld = DataLoader(ts_ds, batch_size=1,             shuffle=False, collate_fn=collate_window)

    model = model_cls(nfeat=tr_ds.nfeat if hasattr(tr_ds, 'nfeat') else 3, cfg=cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr*0.2, weight_decay=cfg.weight_decay)
    pinn_fn = pinn_loss_z_batch_raw if edge_mode else pinn_loss_va_batch_raw
    target_scaler = scalers['edge_Z'] if edge_mode else scalers['node']['value_added']

    keys = ("train_tot","train_mse","train_pinn","train_R2",
            "val_tot","val_mse","val_pinn","val_RMSE","val_MAE",
            "val_SMAPE","val_R2","val_RHO","val_CVR","lambda_t")
    hist = {k: [] for k in keys}
    best_metric, best_state, bad_epochs = float('inf'), None, 0
    global_step = 0

    print(f"\n=== Scaling Configuration ===")
    print(f"Kind: {kind}")
    print(f"Target scaler - mean: {target_scaler.mean_[0]:.6f}, scale: {target_scaler.scale_[0]:.6f}")
    print(f"Is identity scaler: {is_identity_scaler(target_scaler)}")
    print("="*35)

    # TRAIN/VAL loop
    for ep in trange(1, cfg.epochs+1, desc=f"{kind}-seed{seed}"):
        # TRAIN
        model.train()
        tot = mse_acc = pinn_acc = r2_acc = lam_acc = 0.0
        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]
            pred_raw, *_ = model(seqs, tgts)
            tgt_raw = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
            pinn = pinn_fn(pred_raw, tgts)
            mse = F.mse_loss(pred_raw, tgt_raw)
            lam_t = _adaptive_lambda(mse=mse, pinn=pinn, cfg=cfg, global_step=global_step)
            loss = mse + lam_t * pinn
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            if is_identity_scaler(target_scaler):
                pred_o, tgt_o = pred_raw, tgt_raw
            else:
                pred_o = inverse_transform_predictions(pred_raw, scalers, kind)
                tgt_o = inverse_transform_targets(tgt_raw, scalers, kind)

            tot += loss.item(); mse_acc += mse.item(); pinn_acc += pinn.item()
            r2_acc += r2(pred_o, tgt_o); lam_acc += lam_t.item(); global_step += 1

        nb = len(tr_ld)
        lam_avg = lam_acc / nb
        hist['train_tot'].append(tot/nb)
        hist['train_mse'].append(mse_acc/nb)
        hist['train_pinn'].append(pinn_acc/nb)
        hist['train_R2'].append(r2_acc/nb)
        hist['lambda_t'].append(lam_avg)

        tqdm.write(f"[EP {ep:03d}] train: loss {tot/nb:.4f} | MSE {mse_acc/nb:.4f} | "
                f"PINN {pinn_acc/nb:.4f} | λ̄ {lam_avg:.3e} | R² {r2_acc/nb:.3f}")
            
        # VALIDATION
        model.eval()
        v_tot = v_mse = v_pinn = 0.0
        acc = {k: [] for k in ("rmse","mae","smape","r2","RHO","cvr")}
        with torch.no_grad():
            for seqs, tgts in vl_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [g.to(cfg.device) for g in tgts]
                pred_raw, *_ = model(seqs, tgts)
                tgt_raw = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
                pinn = pinn_fn(pred_raw, tgts)
                mse = F.mse_loss(pred_raw, tgt_raw)
                v_tot += (mse + lam_avg * pinn).item()
                v_mse += mse.item(); v_pinn += pinn.item()

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
                if edge_mode:
                    for ps, g in _slice_batch(pred_raw.cpu(), tgts, True):
                        acc["cvr"].append(cvr_tensor_raw(ps, g))

        hist["val_tot"].append(v_tot/len(vl_ld))
        hist["val_mse"].append(v_mse/len(vl_ld))
        hist["val_pinn"].append(v_pinn/len(vl_ld))
        hist["val_RMSE"].append(mean_ignore_nan(acc["rmse"]))
        hist["val_MAE"].append(mean_ignore_nan(acc["mae"]))
        hist["val_SMAPE"].append(mean_ignore_nan(acc["smape"]))
        hist["val_R2"].append(mean_ignore_nan(acc["r2"]))
        hist["val_RHO"].append(mean_ignore_nan(acc["RHO"]))
        hist["val_CVR"].append(mean_ignore_nan(acc["cvr"]) if edge_mode else np.nan)

        if ep % cfg.log_every == 0:
            msg = (
                f"[VAL {ep:03d}] "
                f"loss {v_tot/len(vl_ld):.4f} | "
                f"RMSE {hist['val_RMSE'][-1]:.3f} | "
                f"MAE {hist['val_MAE'][-1]:.3f} | "
                f"SMAPE {hist['val_SMAPE'][-1]:.3f} | "
                f"R² {hist['val_R2'][-1]:.3f}"
            )
            if edge_mode:
                msg += f" | CVR {hist['val_CVR'][-1]:.2e}"
            tqdm.write(msg)

        # Early stopping
        monitor = hist["val_SMAPE"][-1] if edge_mode else hist["val_MAE"][-1]
        if monitor < best_metric - 1e-8:
            best_metric = monitor
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stop @ {ep} (best={best_metric:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    # TEST
    model.eval()
    res = {k: [] for k in ("rmse","mae","smape","r2","RHO","cvr")}
    with torch.no_grad():
        for seqs, tgts in ts_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]
            pred_raw, att_o, att_i = model(seqs, tgts)
            tgt_raw = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
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
                for ps, g in _slice_batch(pred_raw.cpu(), tgts, True):
                    res["cvr"].append(cvr_tensor_raw(ps, g))
                save_edge_attention(att_o, att_i, tgts[0].edge_index, tgts[0].num_nodes, kind, save_dir)

    metrics = {k.upper(): float(mean_ignore_nan(v)) for k, v in res.items()}
    if not edge_mode:
        metrics.pop("CVR", None)

    # Save artifacts
    scalers_path = save_dir / "scalers.pkl"
    pickle.dump(scalers, open(scalers_path, "wb"))
    dump_pred_matrices(
        model, scalers_path,
        years=ts_y,
        save_dir=save_dir,
        cfg=cfg,
        kind=kind,
        save_x=False,
    )
    (save_dir / f"metrics_lambda_{cfg.lambda_max:.4g}.json").write_text(json.dumps(metrics, indent=2))
    (save_dir / f"val_history_lambda_{cfg.lambda_max:.4g}.json").write_text(json.dumps({k:list(map(float,v)) for k,v in hist.items()}, indent=2))
    torch.save(model.cpu().state_dict(), save_dir / f"model_lambda_{cfg.lambda_max:.4g}.pth")
    (save_dir / "alpha.txt").write_text(f"{model.cell.Ox.alpha.item():.6f}")

    print("\n[Test Results]")
    for k, v in metrics.items():
        print(f"{k:<5}: {v:.4f}")
    print(f"\n[Scaling Summary]")
    print(f"Target scaling: {'Identity' if is_identity_scaler(target_scaler) else 'StandardScaler'}")
    print(f"Learning rate: {cfg.lr*0.2:.2e} (reduced from {cfg.lr:.2e})")
    print(f"Gradient clipping: 5.0 (enhanced)")

    return model, hist, None, metrics
