# run_single.py ────────────────────────────────────────────────
"""
Single-run trainer for IO-GNN with:
  - Raw-scale targets (Z/VA)
  - Relative PINN losses
  - Optional rolling-window CV
  - Adaptive lambda with warm-up
  - Full-forward PINN (compute constraint on the entire graph set)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Tuple

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

from constants import EPS
from data_io import GraphWindowDataset, collate_window
from helper import (
    dump_pred_matrices,
    inverse_transform_predictions,
    inverse_transform_targets,
)
from losses import pinn_loss_z_batch_rel, pinn_loss_va_batch_raw
from metrics import mae, mean_ignore_nan, rmse, r2, safe_pearson, smape
from model import IOGNN_Z, IOGNN_VA
from utils import set_seed


# ------------------------------- Helpers --------------------------------

def _slice_batch(cat: torch.Tensor, graphs: List[pyg.data.Data], edge: bool):
    """Yield (slice, graph) pairs for concatenated predictions."""
    offs = 0
    for g in graphs:
        span = g.edge_attr.numel() if edge else g.num_nodes
        yield cat[offs:offs + span], g
        offs += span


def _adaptive_lambda(*, mse: torch.Tensor, pinn: torch.Tensor, cfg, global_step: int) -> torch.Tensor:
    """
    Adaptive lambda with warm-up and scale matching.
    λ_t = warmup_factor * λ_max * sqrt(MSE / (PINN + ε)), clipped to [0.1, 10] factor.
    """
    warm_steps = getattr(cfg, "warmup", 0)
    warm = 1.0 if warm_steps == 0 else min(1.0, global_step / max(1, warm_steps))
    ratio = torch.sqrt(mse.detach() / (pinn.detach() + 1e-12)).clamp(min=0.1, max=10.0)
    return warm * cfg.lambda_max * ratio


def is_identity_scaler(scaler) -> bool:
    """Detect if a StandardScaler is effectively identity."""
    return (hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_')
            and abs(float(scaler.scale_[0]) - 1.0) < 1e-6
            and abs(float(scaler.mean_[0])) < 1e-6)


def _full_forward_concat(
    model: torch.nn.Module,
    full_loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, List[pyg.data.Data]]:
    """
    Forward the model over ALL graphs in `full_loader` and concatenate predictions.
    Keeps autograd graph so gradients flow from PINN to model parameters.
    Returns:
      out_cat : 1D tensor concatenating predictions across graphs
      graphs  : list[Data] aligned with `out_cat` slicing
    """
    outs, graphs = [], []
    for seqs, tgts in full_loader:
        seqs = [[g.to(device) for g in s] for s in seqs]
        tgts = [g.to(device) for g in tgts]
        pred_cat, *_ = model(seqs, tgts)
        outs.append(pred_cat)
        graphs.extend(tgts)
    out_cat = torch.cat(outs, dim=0) if outs else torch.tensor([], device=device)
    return out_cat, graphs


# ------------------------- Rolling Window CV -----------------------------

class RollingWindowSplit(BaseCrossValidator):
    """Time-series rolling-window cross-validation over an ordered list X (e.g., years)."""
    def __init__(self, n_splits: int = 3, train_size: Optional[int] = None, test_size: int = 2, gap: int = 0):
        self.n_splits, self.train_size, self.test_size, self.gap = n_splits, train_size, test_size, gap

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
    """Train on train_years, eval on val_years, return validation R² for this fold."""
    train_years = [years[i] for i in tr_idx]
    val_years   = [years[i] for i in vl_idx]

    tr_ds = GraphWindowDataset(train_years, cfg, scalers=None, fit_scalers=True, scale_targets=False)
    scalers = deepcopy(tr_ds.get_scalers())
    vl_ds = GraphWindowDataset(val_years, cfg, scalers=scalers, fit_scalers=False, scale_targets=False)

    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_window)
    vl_ld = DataLoader(vl_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_window)

    model = model_cls(nfeat=getattr(tr_ds, 'nfeat', 3), cfg=cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pinn_fn = pinn_loss_z_batch_rel if edge_mode else pinn_loss_va_batch_raw

    for _ in range(getattr(cfg, 'fold_epochs', 5)):
        model.train()
        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]
            pred, *_ = model(seqs, tgts)
            tgt = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
            loss = F.mse_loss(pred, tgt) + pinn_fn(pred, tgts)
            optim.zero_grad(); loss.backward(); optim.step()

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


# ------------------------------- Main Run --------------------------------

def run_single(cfg: Any, seed: int, *, kind: str = "Z"):
    set_seed(seed)
    edge_mode = (kind == "Z")
    model_cls = IOGNN_Z if edge_mode else IOGNN_VA

    # Use a simple sequential list as "years" to demo splits; replace with real indices if needed
    years = list(range(1, 21))

    # Optional rolling-window CV branch
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
            fold_r2.append(train_and_eval_fold(k, model_cls, cfg, years, tr_idx, vl_idx, edge_mode))
        mean_r2, std_r2 = float(np.mean(fold_r2)), float(np.std(fold_r2))
        print(f"\n>> Rolling-CV R² = {mean_r2:.3f} ± {std_r2:.3f}")
        return None, None, None, {'ROLLING_R2': mean_r2, 'ROLLING_R2_STD': std_r2}

    # --------------------- Standard train/val/test ----------------------

    # Output directory
    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / f"lam_{cfg.lambda_max:.4g}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # Splits
    tr_y, vl_y, ts_y = years[:-4], years[-4:-2], years[-2:]
    tr_ds = GraphWindowDataset(tr_y, cfg, scalers=None, fit_scalers=True, scale_targets=False)
    scalers = deepcopy(tr_ds.get_scalers())
    vl_ds = GraphWindowDataset(vl_y, cfg, scalers=scalers, fit_scalers=False, scale_targets=False)
    ts_ds = GraphWindowDataset(ts_y, cfg, scalers=scalers, fit_scalers=False, scale_targets=False)

    # Loaders
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_window)
    vl_ld = DataLoader(vl_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_window)
    ts_ld = DataLoader(ts_ds, batch_size=1,             shuffle=False, collate_fn=collate_window)

    # Full-forward loader for PINN (train set, no shuffle)
    full_pin_loader = DataLoader(tr_ds, batch_size=1, shuffle=False, collate_fn=collate_window)

    # Model & optimizer
    model = model_cls(nfeat=getattr(tr_ds, 'nfeat', 3), cfg=cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.2, weight_decay=cfg.weight_decay)

    # PINN loss function
    pinn_fn = pinn_loss_z_batch_rel if edge_mode else pinn_loss_va_batch_raw

    target_scaler = scalers['edge_Z'] if edge_mode else scalers['node']['value_added']

    keys = ("train_tot","train_mse","train_pinn","train_R2",
            "val_tot","val_mse","val_pinn","val_RMSE","val_MAE",
            "val_SMAPE","val_R2","val_RHO","val_IOIS","lambda_t")
    hist = {k: [] for k in keys}
    best_metric, best_state, bad_epochs = float('inf'), None, 0
    global_step = 0

    print(f"\n=== Scaling Configuration ===")
    print(f"Kind: {kind}")
    print(f"Target scaler - mean: {target_scaler.mean_[0]:.6f}, scale: {target_scaler.scale_[0]:.6f}")
    print(f"Is identity scaler: {is_identity_scaler(target_scaler)}")
    print("="*35)

    # Full-forward recompute cadence for PINN
    PINN_FULL_EVERY = int(getattr(cfg, "pinn_full_every", 1))  # 1 = recompute every step
    full_pred_cache = None
    full_graphs_cache = None
    full_cache_step = -1

    # ------------------------------ Train/Val loop ------------------------------
    for ep in trange(1, cfg.epochs + 1, desc=f"{kind}-seed{seed}"):
        # ------------------------------ Train ------------------------------
        model.train()
        tot = mse_acc = pinn_acc = r2_acc = lam_acc = 0.0

        for b_idx, (seqs, tgts) in enumerate(tr_ld):
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            # 1) Batch prediction loss (L_pred)
            pred_raw, *_ = model(seqs, tgts)
            tgt_raw = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
            mse = F.mse_loss(pred_raw, tgt_raw)

            # 2) Full-forward PINN (L_bal) with current params
            need_refresh = (full_cache_step < 0) or ((b_idx % PINN_FULL_EVERY) == 0)
            if need_refresh:
                full_pred, full_graphs = _full_forward_concat(model, full_pin_loader, cfg.device)
                full_pred_cache, full_graphs_cache = full_pred, full_graphs
                full_cache_step = b_idx
            else:
                full_pred, full_graphs = full_pred_cache, full_graphs_cache

            pinn = pinn_fn(full_pred, full_graphs)

            # 3) Adaptive lambda and total loss
            lam_t = _adaptive_lambda(mse=mse, pinn=pinn, cfg=cfg, global_step=global_step)
            loss = mse + lam_t * pinn

            # 4) Optimize
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            # 5) Optional output-space metrics (identity fast-path)
            if is_identity_scaler(target_scaler):
                pred_o, tgt_o = pred_raw, tgt_raw
            else:
                pred_o = inverse_transform_predictions(pred_raw, scalers, kind)
                tgt_o = inverse_transform_targets(tgt_raw, scalers, kind)

            tot += float(loss.item())
            mse_acc += float(mse.item())
            pinn_acc += float(pinn.item())
            r2_acc += r2(pred_o, tgt_o)
            lam_acc += float(lam_t.item())
            global_step += 1

        nb = len(tr_ld)
        lam_avg = lam_acc / max(1, nb)
        hist['train_tot'].append(tot / max(1, nb))
        hist['train_mse'].append(mse_acc / max(1, nb))
        hist['train_pinn'].append(pinn_acc / max(1, nb))
        hist['train_R2'].append(r2_acc / max(1, nb))
        hist['lambda_t'].append(lam_avg)

        tqdm.write(f"[EP {ep:03d}] train: loss {tot/nb:.4f} | MSE {mse_acc/nb:.4f} | "
                   f"PINN {pinn_acc/nb:.4f} | λ̄ {lam_avg:.3e} | R² {r2_acc/nb:.3f}")

        # ------------------------------ Validation ------------------------------
        model.eval()
        v_tot = v_mse = v_pinn = 0.0
        acc = {k: [] for k in ("rmse", "mae", "smape", "r2", "RHO", "iois")}
        with torch.no_grad():
            for seqs, tgts in vl_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [g.to(cfg.device) for g in tgts]
                pred_raw, *_ = model(seqs, tgts)
                tgt_raw = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

                # Use same PINN definition; for validation we can reuse lam_avg for reporting
                pinn = pinn_fn(pred_raw, tgts)
                mse = F.mse_loss(pred_raw, tgt_raw)
                v_tot += float((mse + lam_avg * pinn).item())
                v_mse += float(mse.item())
                v_pinn += float(pinn.item())

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
                    iois_num = pred_raw.new_tensor(0.0)
                    iois_den = pred_raw.new_tensor(0.0)
                    for ps, g in _slice_batch(pred_raw, tgts, True):
                        s, t = g.edge_index
                        n = g.num_nodes
                        row_z = ps.new_zeros(n).index_add_(0, s, ps)
                        col_z = ps.new_zeros(n).index_add_(0, t, ps)
                        imp, exp, fd = g.x_raw.T
                        va = g.va_raw
                        row = row_z + fd + exp - imp
                        col = col_z + va
                        iois_num += (row - col).abs().sum()
                        iois_den += g.tot_raw.sum().clamp_min(EPS)
                    acc["iois"].append((iois_num / iois_den).item())

        hist["val_tot"].append(v_tot / len(vl_ld))
        hist["val_mse"].append(v_mse / len(vl_ld))
        hist["val_pinn"].append(v_pinn / len(vl_ld))
        hist["val_RMSE"].append(mean_ignore_nan(acc["rmse"]))
        hist["val_MAE"].append(mean_ignore_nan(acc["mae"]))
        hist["val_SMAPE"].append(mean_ignore_nan(acc["smape"]))
        hist["val_R2"].append(mean_ignore_nan(acc["r2"]))
        hist["val_RHO"].append(mean_ignore_nan(acc["RHO"]))
        hist["val_IOIS"].append(mean_ignore_nan(acc["iois"]) if edge_mode else np.nan)

        if ep % cfg.log_every == 0:
            msg = (f"[VAL {ep:03d}] loss {v_tot/len(vl_ld):.4f} | RMSE {hist['val_RMSE'][-1]:.3f} | "
                   f"MAE {hist['val_MAE'][-1]:.3f} | SMAPE {hist['val_SMAPE'][-1]:.3f} | "
                   f"R² {hist['val_R2'][-1]:.3f}")
            if edge_mode:
                msg += f" | IOIS {hist['val_IOIS'][-1]:.2e}"
            tqdm.write(msg)

        # Early stopping on total val objective
        monitor = hist["val_tot"][-1]
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

    # ------------------------------ Test ------------------------------
    model.eval()
    res = {k: [] for k in ("rmse", "mae", "smape", "r2", "RHO", "iois")}
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
                iois_num = pred_raw.new_tensor(0.0)
                iois_den = pred_raw.new_tensor(0.0)
                for ps, g in _slice_batch(pred_raw, tgts, True):
                    s, t = g.edge_index
                    n = g.num_nodes
                    row_z = ps.new_zeros(n).index_add_(0, s, ps)
                    col_z = ps.new_zeros(n).index_add_(0, t, ps)
                    imp, exp, fd = g.x_raw.T
                    va = g.va_raw
                    row = row_z + fd + exp - imp
                    col = col_z + va
                    iois_num += (row - col).abs().sum()
                    iois_den += g.tot_raw.sum().clamp_min(EPS)
                res["iois"].append((iois_num / iois_den).item())

    metrics = {k.upper(): float(mean_ignore_nan(v)) for k, v in res.items()}
    if not edge_mode:
        metrics.pop("IOIS", None)

    # ------------------------------ Save ------------------------------
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
    (save_dir / f"val_history_lambda_{cfg.lambda_max:.4g}.json").write_text(
        json.dumps({k: list(map(float, v)) for k, v in hist.items()}, indent=2)
    )
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