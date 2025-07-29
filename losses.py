import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Any, Dict, Callable

EPS = 1e-12                 

# --- util ----------------------------------------------------------
def _rel_err(pred: Tensor, true: Tensor, eps: float = EPS) -> Tensor:
    """|pred-true| / (|true|+eps)  – 크기 차이 완화용."""
    return (pred - true).abs() / (true.abs() + eps)

# --- Z PINN (raw-scale) -----------------------------------------------
def pinn_single_z_raw(
    z_raw: Tensor,
    g: Data,
    *,
    w_row: float = 1.0,
    w_col: float = 1.0,
) -> Tensor:
    src, trg = g.edge_index
    n = g.num_nodes


    row = torch.zeros(n, device=z_raw.device).index_add_(0, src, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, trg, z_raw)

    SCALE = 1e6
    imp_raw, exp_raw, fd_raw = g.x_raw.T / SCALE

    row_res = _rel_err(row + fd_raw + exp_raw - imp_raw - g.tot, g.tot)   # Σrow = TOT
    col_res = _rel_err(col + g.va - g.tot,             g.tot)   # Σcol = TOT

    return w_row * row_res.mean() + w_col * col_res.mean()

# --- VA PINN (raw-scale) ----------------------------------------------
def pinn_single_va_raw(
    va_raw: Tensor,
    g: Data,
    *,
    w_col: float = 1.0,
) -> Tensor:
    src, trg = g.edge_index
    n = g.num_nodes

    col_z = torch.zeros(n, device=va_raw.device).index_add_(0, trg, g.edge_attr)
    col_pred = col_z + va_raw
    col_res  = _rel_err(col_pred, g.tot)     # Σ(Z + VA) = TOT

    return w_col * col_res.mean()

# ───────── batch wrappers ────────────────────
def pinn_loss_z_batch_raw(z_cat, batch):
    off, losses = 0, []
    for g in batch:
        e = g.edge_attr.numel()
        losses.append(pinn_single_z_raw(z_cat[off:off+e], g))
        off += e
    return torch.stack(losses).mean()

def pinn_loss_va_batch_raw(va_cat, batch):
    off, losses = 0, []
    for g in batch:
        n = g.num_nodes
        losses.append(pinn_single_va_raw(va_cat[off:off+n], g))
        off += n
    return torch.stack(losses).mean()

def get_pinn_loss_function(kind: str) -> Callable:
    if kind == "Z":
        return pinn_loss_z_batch_raw
    if kind == "VA":
        return pinn_loss_va_batch_raw
    raise ValueError(kind)

