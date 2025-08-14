"""metrics.py – evaluation & utility metrics for IO-GNN.

Expected keys in `scalers_dict` (aligned with data_io), when needed:
    "node"   – {"node_features", "value_added", "total"}
    "edge_Z" – StandardScaler for Z edges
    "edge_A" – identity scaler for Af (unused here)

This module provides:
  - Base regression metrics (RMSE/MAE/R2/SMAPE, Pearson)
  - IOIS metrics on RAW scale:
      * iois_z_raw:     absolute imbalance over total (classic)
      * iois_z_rel_raw: node-wise relative imbalance (matches PINN loss)
      * iois_z_rel_batch: average of graph-wise iois_z_rel_raw
"""

from __future__ import annotations

from typing import List
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from constants import EPS


# ------------------------------ Base metrics ------------------------------ #

@torch.inference_mode()
def rmse(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.float(), true.float()
    return torch.sqrt(((pred - true) ** 2).mean()).item()


@torch.inference_mode()
def mae(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.float(), true.float()
    return (pred - true).abs().mean().item()


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
    return (2.0 * (pred - true).abs() / denom).mean().item()


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


def mean_ignore_nan(vals: List[float]) -> float:
    return float(np.nanmean(vals)) if vals else float("nan")


# ------------------------------ IOIS metrics (RAW) ------------------------------ #

@torch.inference_mode()
def iois_z_raw(pred_raw: Tensor, g: Data) -> float:
    """
    Input-Output Imbalance Score (absolute), RAW scale:
      IOIS_abs = (Σ_i |row_i - col_i|) / (Σ_i TOT_i)

    With exogenous terms:
      row_i = Σ_j Z_ij(pred) + FD_i + EXP_i - IMP_i
      col_i = Σ_j Z_ji(pred) + VA_i
    """
    src, trg = g.edge_index
    n = g.num_nodes

    imp, exp, fd = g.x_raw.T  # [N]
    va = g.va_raw             # [N]
    tot = g.tot_raw           # [N]

    row = torch.zeros(n, device=pred_raw.device).index_add_(0, src, pred_raw) + fd + exp - imp
    col = torch.zeros(n, device=pred_raw.device).index_add_(0, trg, pred_raw) + va

    mismatch = (row - col).abs().sum()
    total_output = tot.sum().clamp(min=EPS)
    return (mismatch / total_output).item()


@torch.inference_mode()
def iois_z_rel_raw(pred_raw: Tensor, g: Data) -> float:
    """
    Node-wise relative IOIS on RAW scale (matches training PINN definition):
      IOIS_rel = mean_i  |row_i - col_i| / (TOT_i + eps)

    This balances sectors by size and better reflects constraint satisfaction
    when using relative PINN losses.
    """
    src, trg = g.edge_index
    n = g.num_nodes

    imp, exp, fd = g.x_raw.T  # [N]
    va = g.va_raw             # [N]
    tot = g.tot_raw           # [N]

    row = torch.zeros(n, device=pred_raw.device).index_add_(0, src, pred_raw) + fd + exp - imp
    col = torch.zeros(n, device=pred_raw.device).index_add_(0, trg, pred_raw) + va

    rel = (row - col).abs() / (tot.abs() + EPS)
    return rel.mean().item()


@torch.inference_mode()
def iois_z_rel_batch(pred_cat: Tensor, graphs: List[Data]) -> float:
    """
    Batch version of relative IOIS:
      average over graphs of each graph's node-wise mean relative imbalance.
    """
    acc = 0.0
    off = 0
    num_graphs = 0
    for g in graphs:
        E = g.edge_index.size(1)
        pred = pred_cat[off:off + E]
        off += E
        acc += iois_z_rel_raw(pred, g)
        num_graphs += 1
    return acc / max(1, num_graphs)