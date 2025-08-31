# run_single.py ────────────────────────────────────────────────────────────────
"""
Single-run trainer for IO-GNN without validation (standard path):
  - Train on first N-4 years, test on last 4 years (fixed split)
  - Raw-scale Z / (optionally standardized) VA targets
  - Relative PINN losses with adaptive lambda + warm-up
  - Full-forward PINN (constraint computed on the entire train set)

Optional branch:
  - Rolling-window CV (unchanged): computes fold-wise R^2 and returns summary,
    but does not save models or predictions.
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
    """Yield (slice, graph) pairs for a concatenated prediction tensor."""
    offs = 0
    for g in graphs:
        if edge:
            span = g.edge_attr.view(-1).numel()
        else:
            span = g.va.view(-1).numel() if hasattr(g, "va") else g.num_nodes
        yield cat[offs:offs + span], g
        offs += span


def _adaptive_lambda(*, mse: torch.Tensor, pinn: torch.Tensor, cfg, global_step: int) -> torch.Tensor:
    """
    Adaptive lambda with warm-up and basic scale matching.
      λ_t = warmup_factor * λ_max * sqrt(MSE / (PINN + ε))
    """
    eps = getattr(cfg, "eps", 1e-12)
    clip_min = getattr(cfg, "lambda_ratio_min", 0.3)
    clip_max = getattr(cfg, "lambda_ratio_max", 5.0)
    lam_cap  = getattr(cfg, "lambda_cap", None)

    warm_steps = int(getattr(cfg, "warmup", 0))
    warm = 1.0 if warm_steps <= 0 else min(1.0, float(global_step) / float(max(1, warm_steps)))

    ratio = torch.sqrt(mse.detach() / (pinn.detach() + eps)).clamp(min=clip_min, max=clip_max)
    lam_t = ratio * float(getattr(cfg, "lambda_max", 0.0)) * warm
    if lam_cap is not None:
        lam_t = torch.clamp(lam_t, max=float(lam_cap))
    return lam_t


def is_identity_scaler(scaler) -> bool:
    """Detect if a StandardScaler is effectively identity-like (all dims)."""
    if not (hasattr(scaler, "scale_") and hasattr(scaler, "mean_")):
        return False
    try:
        scale = np.asarray(scaler.scale_, dtype=float)
        mean  = np.asarray(scaler.mean_, dtype=float)
        return np.allclose(scale, 1.0, atol=1e-6) and np.allclose(mean, 0.0, atol=1e-6)
    except Exception:
        try:
            return (
                abs(float(scaler.scale_[0]) - 1.0) < 1e-6
                and abs(float(scaler.mean_[0])) < 1e-6
            )
        except Exception:
            return False


def _full_forward_concat(
    model: torch.nn.Module,
    full_loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, List[pyg.data.Data]]:
    """
    Forward the model over ALL graphs in `full_loader` and concatenate predictions.
    Keeps autograd graph so gradients can flow from the PINN loss to model parameters.
    """
    outs: List[torch.Tensor] = []
    graphs: List[pyg.data.Data] = []

    was_training = model.training
    model.eval()  # no dropout/bn updates; keep grads

    for seqs, tgts in full_loader:
        seqs = [[g.to(device) for g in s] for s in seqs]
        tgts = [g.to(device) for g in tgts]
        pred_cat, *_ = model(seqs, tgts)
        outs.append(pred_cat)
        graphs.extend(tgts)

    if was_training:
        model.train()

    if outs:
        out_cat = torch.cat(outs, dim=0)
    else:
        try:
            p = next(model.parameters())
            out_cat = torch.empty(0, dtype=p.dtype, device=p.device)
        except StopIteration:
            out_cat = torch.empty(0, device=device)

    return out_cat, graphs


def _inverse_standardize_torch_flat(x_flat: torch.Tensor,
                                    mean: torch.Tensor,
                                    scale: torch.Tensor,
                                    feature_dim: int = 1) -> torch.Tensor:
    """Differentiable inverse transform for flattened targets."""
    if feature_dim == 1:
        return x_flat * scale + mean
    total = x_flat.numel()
    assert total % feature_dim == 0, "x_flat length must be divisible by feature_dim"
    x = x_flat.reshape(-1, feature_dim)
    x = x * scale + mean
    return x.reshape(-1)


def _va_scaler_stats_as_tensors(scalers, device, dtype) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract VA scaler stats from sklearn StandardScaler into torch tensors."""
    va_scaler = scalers['node']['value_added']
    mean_np = getattr(va_scaler, "mean_", 0.0)
    scale_np = getattr(va_scaler, "scale_", 1.0)

    mean_t = torch.as_tensor(mean_np, device=device, dtype=dtype)
    scale_t = torch.as_tensor(scale_np, device=device, dtype=dtype)

    if mean_t.ndim == 0:
        mean_t = mean_t.view(1)
    if scale_t.ndim == 0:
        scale_t = scale_t.view(1)

    feature_dim = int(mean_t.numel())
    return mean_t, scale_t, feature_dim

# ------------------------- Rolling Window CV (unchanged) -----------------

class RollingWindowSplit(BaseCrossValidator):
    """Time-series rolling-window CV over an ordered list X (e.g., years)."""
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
    """Train on train_years, eval on val_years (used only in RW-CV branch)."""
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

    va_mean_t = va_scale_t = None
    va_feat_dim = 1
    if (not edge_mode) and bool(getattr(cfg, "scale_targets", False)):
        va_mean_t, va_scale_t, va_feat_dim = _va_scaler_stats_as_tensors(
            scalers, device=cfg.device, dtype=torch.float32
        )

    for _ in range(getattr(cfg, 'fold_epochs', 5)):
        model.train()
        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]
            pred, *_ = model(seqs, tgts)
            tgt = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

            if edge_mode:
                pinn_term = pinn_fn(pred, tgts)
            else:
                if bool(getattr(cfg, "scale_targets", False)):
                    pred_raw_va = _inverse_standardize_torch_flat(pred, va_mean_t, va_scale_t, feature_dim=va_feat_dim)
                else:
                    pred_raw_va = pred
                pinn_term = pinn_fn(pred_raw_va, tgts)

            loss = F.mse_loss(pred, tgt) + pinn_term
            optim.zero_grad(); loss.backward(); optim.step()

    model.eval()
    scores = []
    with torch.no_grad():
        for seqs, tgts in vl_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]
            pred, *_ = model(seqs, tgts)
            tgt = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

            if edge_mode or (not bool(getattr(cfg, "scale_targets", False))):
                pred_o, tgt_o = pred, tgt
            else:
                pred_o = _inverse_standardize_torch_flat(pred, va_mean_t, va_scale_t, feature_dim=va_feat_dim)
                tgt_o  = _inverse_standardize_torch_flat(tgt,  va_mean_t, va_scale_t, feature_dim=va_feat_dim)

            if edge_mode or (not bool(getattr(cfg, "scale_targets", False))):
                tgt_o = tgt
            scores.append(r2(pred_o, tgt_o))

    fold_r2 = float(np.mean(scores)) if scores else 0.0
    tqdm.write(f"[Fold {fold_id+1}] R² = {fold_r2:.3f}")
    return fold_r2

# ------------------------------- Main Run --------------------------------

def run_single(cfg: Any, seed: int, *, kind: str = "Z"):
    set_seed(seed)
    edge_mode = (kind == "Z")
    model_cls = IOGNN_Z if edge_mode else IOGNN_VA

    # Years
    years = getattr(cfg, "years", None)
    if not years:
        years = list(range(1, 21))

    # Optional rolling-window CV branch (unchanged)
    if getattr(cfg, 'rolling_val', False):
        splitter = RollingWindowSplit(
            n_splits=int(cfg.rolling_splits),
            train_size=cfg.rolling_train_size,
            test_size=int(cfg.rolling_test_size),
            gap=int(cfg.rolling_gap)
        )
        fold_r2: List[float] = []
        for k, (tr_idx, vl_idx) in enumerate(splitter.split(years)):
            set_seed(seed + k)
            r2_k = train_and_eval_fold(k, model_cls, cfg, years, tr_idx, vl_idx, edge_mode)
            fold_r2.append(r2_k)

        if len(fold_r2) == 0:
            raise ValueError(
                "RollingWindowSplit produced 0 folds. Check time series length vs "
                f"window={getattr(cfg, 'window', 1)}, train_size={cfg.rolling_train_size or 'auto(≈70%)'}, "
                f"test_size={cfg.rolling_test_size}, gap={cfg.rolling_gap}, n_splits={cfg.rolling_splits}."
            )

        mean_r2 = float(np.mean(fold_r2))
        std_r2 = float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0
        print(f"\n>> Rolling-CV R² over {len(fold_r2)} folds = {mean_r2:.3f} ± {std_r2:.3f}")
        return None, None, None, {'ROLLING_R2': mean_r2, 'ROLLING_R2_STD': std_r2}

    # --------------------- Standard train/test (no validation) ----------------------

    # Output directory
    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / f"lam_{cfg.lambda_max:.4g}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fixed splits: train = all but last 4, test = last 4
    tr_y, ts_y = years[:-4], years[-4:]

    # Datasets
    tr_ds = GraphWindowDataset(tr_y, cfg, scalers=None, fit_scalers=True, scale_targets=False)
    scalers = deepcopy(tr_ds.get_scalers())
    ts_ds = GraphWindowDataset(ts_y, cfg, scalers=scalers, fit_scalers=False, scale_targets=False)

    # Loaders
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_window)
    ts_ld = DataLoader(ts_ds, batch_size=1,             shuffle=False, collate_fn=collate_window)

    # Safety: ensure training has batches
    nb = len(tr_ld)
    if nb == 0:
        raise ValueError(
            f"No training batches. Check window={getattr(cfg,'window',1)} and train years={len(tr_y)}."
        )

    # Full-forward loader for PINN (train set, no shuffle)
    full_pin_loader = DataLoader(tr_ds, batch_size=1, shuffle=False, collate_fn=collate_window)

    # Model & optimizer
    model = model_cls(nfeat=getattr(tr_ds, 'nfeat', 3), cfg=cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.2, weight_decay=cfg.weight_decay)

    # PINN loss function
    pinn_fn = pinn_loss_z_batch_rel if edge_mode else pinn_loss_va_batch_raw

    target_scaler = scalers['edge_Z'] if edge_mode else scalers['node']['value_added']

    # History (train only)
    keys = ("train_tot","train_mse","train_pinn","train_R2","lambda_t")
    hist = {k: [] for k in keys}

    global_step = 0

    print(f"\n=== Scaling Configuration ===")
    print(f"Kind: {kind}")
    print(f"Target scaler - mean: {target_scaler.mean_[0]:.6f}, scale: {target_scaler.scale_[0]:.6f}")
    print(f"Is identity scaler: {is_identity_scaler(target_scaler)}")
    print("="*35)

    # Full-forward recompute cadence for PINN
    PINN_FULL_EVERY = int(getattr(cfg, "pinn_full_every", 1))  # 1 = every batch
    full_pred_cache = None
    full_graphs_cache = None
    full_cache_step = -1

    # ------------------------------ Train loop (no validation) ------------------------------
    for ep in trange(1, cfg.epochs + 1, desc=f"{kind}-seed{seed}"):
        model.train()
        tot = mse_acc = pinn_acc = r2_acc = lam_acc = 0.0

        for b_idx, (seqs, tgts) in enumerate(tr_ld):
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            # 1) Prediction loss
            pred_raw, *_ = model(seqs, tgts)
            tgt_raw = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])
            mse = F.mse_loss(pred_raw, tgt_raw)

            # 2) Full-forward PINN
            need_refresh = (full_cache_step < 0) or ((b_idx % PINN_FULL_EVERY) == 0)
            if need_refresh:
                full_pred, full_graphs = _full_forward_concat(model, full_pin_loader, cfg.device)
                if not edge_mode:  # VA → back to raw scale for PINN
                    mean_t, scale_t, feat_dim = _va_scaler_stats_as_tensors(scalers, full_pred.device, full_pred.dtype)
                    full_pred = _inverse_standardize_torch_flat(full_pred, mean_t, scale_t, feature_dim=feat_dim)
                full_pred_cache, full_graphs_cache = full_pred, full_graphs
                full_cache_step = b_idx
            else:
                full_pred, full_graphs = full_pred_cache, full_graphs_cache

            pinn = pinn_fn(full_pred, full_graphs)

            # 3) Adaptive lambda
            lam_t = _adaptive_lambda(mse=mse, pinn=pinn, cfg=cfg, global_step=global_step)
            loss = mse + lam_t * pinn

            # 4) Optimize
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            # 5) Output-space metrics
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

        lam_avg = lam_acc / max(1, nb)
        hist['train_tot'].append(tot / max(1, nb))
        hist['train_mse'].append(mse_acc / max(1, nb))
        hist['train_pinn'].append(pinn_acc / max(1, nb))
        hist['train_R2'].append(r2_acc / max(1, nb))
        hist['lambda_t'].append(lam_avg)

        tqdm.write(f"[EP {ep:03d}] train: loss {tot/nb:.4f} | MSE {mse_acc/nb:.4f} | "
                   f"PINN {pinn_acc/nb:.4f} | λ̄ {lam_avg:.3e} | R² {r2_acc/nb:.3f}")

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
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)
    dump_pred_matrices(
        model, scalers_path,
        years=ts_y,
        save_dir=save_dir,
        cfg=cfg,
        kind=kind,
        save_x=False,
    )
    (save_dir / f"metrics_lambda_{cfg.lambda_max:.4g}.json").write_text(json.dumps(metrics, indent=2))
    (save_dir / f"train_history_lambda_{cfg.lambda_max:.4g}.json").write_text(  # ← 파일명 변경
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