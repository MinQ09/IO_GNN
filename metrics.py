"""metrics.py – evaluation & utility metrics for IO‑GNN.
Keys expected in `scalers_dict` (aligned with data_io):
    "node"   – {"node_features", "value_added", "total"}
    "edge_Z" – StandardScaler for Z edges
    "edge_A" – identity scaler for Af (unused here)
"""
from __future__ import annotations

from typing import List
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from constants import EPS

# ───────────────── base metrics ───────────────── #
@torch.inference_mode()
def rmse(pred: Tensor, true: Tensor) -> float:
    return torch.sqrt(((pred - true).float().pow(2)).mean()).item()

@torch.inference_mode()
def mae(pred: Tensor, true: Tensor) -> float:
    return (pred.float() - true.float()).abs().mean().item()

@torch.inference_mode()
def r2(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.float(), true.float()
    ss_tot = ((true - true.mean()) ** 2).sum()
    if ss_tot < EPS:
        return float("nan")
    ss_res = ((pred - true) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()

@torch.inference_mode()
def smape(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.float(), true.float()
    denom = (pred.abs() + true.abs()).clamp(min=EPS)
    return (2 * (pred - true).abs() / denom).mean().item()

# ───────────── safe Pearson helper ───────────── #
@torch.inference_mode()
def safe_pearson(x: Tensor, y: Tensor) -> float:
    x_np, y_np = x.detach().cpu().numpy().ravel(), y.detach().cpu().numpy().ravel()
    mask = np.isfinite(x_np) & np.isfinite(y_np)
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(np.corrcoef(x_np[mask], y_np[mask])[0, 1])
    except Exception:
        return float("nan")

# ───────────── CVR in original scale ─────────── #
@torch.inference_mode()
def cvr_tensor_standardized(pred_std: Tensor, g: Data, scalers: dict) -> float:
    # edge inverse transform (Z only)
    edge_scaler: StandardScaler = scalers["edge_Z"]
    pred_orig = torch.from_numpy(edge_scaler.inverse_transform(
        pred_std.cpu().numpy().reshape(-1, 1)
    ).squeeze()).to(pred_std.device)

    # node inverse transform
    n_scalers = scalers["node"]
    x_orig = torch.from_numpy(n_scalers["node_features"].inverse_transform(g.x.cpu().numpy())).to(pred_std.device)
    va_orig = torch.from_numpy(n_scalers["value_added"].inverse_transform(g.va.cpu().numpy().reshape(-1, 1)).squeeze()).to(pred_std.device)
    tot_orig= torch.from_numpy(n_scalers["total"].inverse_transform(g.tot.cpu().numpy().reshape(-1, 1)).squeeze()).to(pred_std.device)

    imp, exp, fd = x_orig.T
    src, trg = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=pred_std.device).index_add_(0, src, pred_orig) + fd + exp
    col = torch.zeros(n, device=pred_std.device).index_add_(0, trg, pred_orig) + va_orig + imp

    mismatch = (row - col).abs().sum()
    total_output = tot_orig.sum().clamp(min=EPS)
    return (mismatch / total_output).item()

# nano‑mean helper remains unchanged

def mean_ignore_nan(vals: List[float]) -> float:
    return float(np.nanmean(vals)) if vals else float("nan")