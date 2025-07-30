import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Any, Dict, Callable

EPS = 1e-12

def _rel_err(residual: Tensor, scale: Tensor, eps: float = EPS) -> Tensor:
    """|residual| / (|scale|+eps) - scale은 g.tot 같이 항상 양수인 기준."""
    return residual.abs() / (scale.abs() + eps)

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

    # ✅ 올바른 잔차 정의 (= 0이어야 하는 값)
    row_imb = row + fd_raw + exp_raw - imp_raw - g.tot
    col_imb = col + g.va - g.tot
    net = row + fd_raw + exp_raw - imp_raw - col - g.va

    row_res = _rel_err(row_imb, g.tot)
    col_res = _rel_err(col_imb, g.tot)
    net_res = _rel_err(net, g.tot)

    # ✅ 올바른 합산 방식
    return net_res.abs().mean()

def pinn_single_va_raw(
    va_raw: Tensor,
    g: Data,
    *,
    w_col: float = 1.0,
) -> Tensor:
    src, trg = g.edge_index
    n = g.num_nodes

    col_z = torch.zeros(n, device=va_raw.device).index_add_(0, trg, g.edge_attr)
    col_imb = col_z + va_raw - g.tot  # ✅ 올바른 잔차

    col_res = _rel_err(col_imb, g.tot)
    return w_col * col_res.mean()

# 배치 래퍼는 그대로
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

