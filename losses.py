# losses.py  ─────────────────────────────────────────────────────────────
"""
Physics-informed loss helpers.

* pinn_loss_z_batch  : use when training the edge-level Z model
* pinn_loss_va_batch : use when training the node-level VA model
"""

from __future__ import annotations
from typing import List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data


# ──────────────────── 1) Z-model PINN ────────────────────
def _pinn_single_z(
    z_raw: Tensor, g: Data, scale: float,
    *, eps: float = 1e-8
) -> Tensor:
    """
    PINN residuals for one graph using *predicted* Ẑ_raw.

    Row, column, net balances are all enforced.
    """
    src, trg = g.edge_index
    n = g.num_nodes

    # Row / Column sums of predicted Z
    row = torch.zeros(n, device=z_raw.device).index_add_(0, src, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, trg, z_raw)

    # True node attributes (rescaled)
    x   = g.x * scale          # [Imports, Exports, Final_Demand]
    va  = g.va * scale
    tot = g.tot * scale

    imp, exp, fd = x[:, 0], x[:, 1], x[:, 2]

    row_res = (row + fd + exp - tot) / (tot + eps)
    col_res = (col + va + imp - tot) / (tot + eps)
    net_res = (row - col + fd + exp - va - imp) / (tot + eps)

    return (row_res.square().mean() +
            col_res.square().mean() +
            net_res.square().mean()) / 3


def pinn_loss_z_batch(
    z_raw_concat: Tensor,          # concatenated Ẑ_raw (edges)
    tgt_batch:     List[Data],
    scale:         float,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """
    Average PINN loss over a mini-batch –– **edge model** version.
    """
    e_offs = 0
    losses: list[Tensor] = []

    for g in tgt_batch:
        e = g.edge_attr.numel()
        z_slice = z_raw_concat[e_offs : e_offs + e]
        losses.append(_pinn_single_z(z_slice, g, scale, eps=eps))
        e_offs += e

    return torch.stack(losses).mean()


# ──────────────────── 2) VA-model PINN ────────────────────
def _pinn_single_va(
    va_pred: Tensor, g: Data, scale: float,
    *, eps: float = 1e-8
) -> Tensor:
    """
    Column-only balance using *predicted* Value Added VÂ_raw.

    Row residual is ignored (unchanged), because VA only affects columns.
    """
    src, trg = g.edge_index
    n = g.num_nodes

    # True Z (log1p stored) → raw  (row/col sums **fixed** here)
    z_true_raw = torch.expm1(g.edge_attr)      # assume original scale_Z = 1
    row_true = torch.zeros(n, device=va_pred.device).index_add_(0, src, z_true_raw)
    col_true = torch.zeros(n, device=va_pred.device).index_add_(0, trg, z_true_raw)

    # Other true attributes
    x   = g.x * scale
    imp = x[:, 0]
    tot = g.tot * scale

    # Replace Value Added with prediction
    col_pred = col_true + va_pred + imp
    col_res  = (col_pred - tot) / (tot + eps)

    # Only column residual enforced
    return col_res.square().mean()


def pinn_loss_va_batch(
    va_pred_concat: Tensor,        # concatenated VÂ_raw (nodes)
    tgt_batch:      List[Data],
    scale:          float,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """
    Average PINN loss over a mini-batch –– **VA model** version.
    """
    n_offs = 0
    losses: list[Tensor] = []

    for g in tgt_batch:
        n = g.num_nodes
        va_slice = va_pred_concat[n_offs : n_offs + n]
        losses.append(_pinn_single_va(va_slice, g, scale, eps=eps))
        n_offs += n

    return torch.stack(losses).mean()