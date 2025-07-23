# losses.py  –  PINN in *standardised* space
from __future__ import annotations
from typing import List, Dict, Any, Callable
import torch
from torch import Tensor
from torch_geometric.data import Data
from constants import EPS  # 1e-8

# ───────── Z single graph ──────────────────────────────────────────
def _pinn_single_z_std(z_std: Tensor, g: Data, _scalers: Dict[str, Any]) -> Tensor:
    """row / col / net residuals in *standardised* space."""
    src, trg = g.edge_index
    n        = g.num_nodes

    row = torch.zeros(n, device=z_std.device).index_add_(0, src, z_std)
    col = torch.zeros(n, device=z_std.device).index_add_(0, trg, z_std)

    imp, exp, fd = g.x.T            # already std-space
    va_std       = g.va             # std
    tot_std      = g.tot            # std

    row_res = (row + fd + exp - tot_std) / (tot_std.abs() + EPS)
    col_res = (col + va_std + imp - tot_std) / (tot_std.abs() + EPS)
    net_res = (row - col + fd + exp - va_std - imp) / (tot_std.abs() + EPS)

    return (row_res.square().mean() +
            col_res.square().mean() +
            net_res.square().mean()) / 3.0

# ───────── VA single graph ─────────────────────────────────────────
def _pinn_single_va_std(va_std: Tensor, g: Data, _scalers: Dict[str, Any]) -> Tensor:
    src, trg = g.edge_index
    n        = g.num_nodes

    row_true = torch.zeros(n, device=va_std.device).index_add_(0, src, g.edge_attr)
    col_true = torch.zeros(n, device=va_std.device).index_add_(0, trg, g.edge_attr)

    imp = g.x[:, 0]           # std-space
    tot = g.tot               # std

    col_pred = col_true + va_std + imp
    col_res  = (col_pred - tot) / (tot.abs() + EPS)
    return col_res.square().mean()

# ───────── batch wrappers & factory (변경 없음) ─────────────────────
def pinn_loss_z_batch_standardized(z_cat, batch, scalers):  # scalers 인자 유지
    off = 0; losses = []
    for g in batch:
        e = g.edge_attr.numel()
        losses.append(_pinn_single_z_std(z_cat[off:off+e], g, scalers))
        off += e
    return torch.stack(losses).mean()

def pinn_loss_va_batch_standardized(va_cat, batch, scalers):
    off = 0; losses = []
    for g in batch:
        n = g.num_nodes
        losses.append(_pinn_single_va_std(va_cat[off:off+n], g, scalers))
        off += n
    return torch.stack(losses).mean()

def get_pinn_loss_function(kind: str, *, use_standardized: bool = True) -> Callable:
    if kind == "Z":  return pinn_loss_z_batch_standardized
    if kind == "VA": return pinn_loss_va_batch_standardized
    raise ValueError(kind)