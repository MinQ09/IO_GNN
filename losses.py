import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Any, Dict, Callable

EPS = 1e-12                   # 수치 안전용

# ───────── Z PINN 3-in-1 ──────────────────────────────────
def pinn_single_z_std(
    z_std: Tensor,
    g: Data,
    scalers: Dict[str, Any],
    *,
    w_row: float = 1.0,
    w_col: float = 1.0,
    w_net: float = 1.0,
) -> Tensor:
    """
    • row_res : ∑_j Z_ij + FD_i + EXP_i − TOT_i
    • col_res : ∑_i Z_ij + VA_j + IMP_j − TOT_j
    • net_res : row_res − col_res   (중복 상쇄)
      → 세 잔차 모두 σ_tot 로 나눠 표준편차 1 스케일에 맞춤
    """

    src, trg = g.edge_index
    n = g.num_nodes

    # ── 집계(표준화 공간) ────────────────────────────────
    row = torch.zeros(n, device=z_std.device).index_add_(0, src, z_std)
    col = torch.zeros(n, device=z_std.device).index_add_(0, trg, z_std)

    imp, exp, fd = g.x.T            # std-space
    va_std       = g.va
    tot_std      = g.tot

    # ── σ_tot (노드별 표준편차) 로 스케일 평준화 ──────────
    σ_tot = torch.tensor(
        scalers["node"]["total"].scale_,
        device=tot_std.device,
        dtype=tot_std.dtype,
    ) + EPS                          # 안전 분모

    row_res = (row + fd + exp  - tot_std)        
    col_res = (col + va_std + imp - tot_std)      
    net_res = row_res - col_res                             

    loss = (
        w_row * row_res.abs().mean()
        + w_col * col_res.abs().mean()
    ) / 2
    return loss


# ───────── VA PINN (열 합계) ────────────────────────────
def pinn_single_va_std(
    va_std: Tensor,
    g: Data,
    scalers: Dict[str, Any],
    *,
    w_col: float = 1.0,
) -> Tensor:
    """
    • col_res : ∑_i Z_ij + VA_j + IMP_j − TOT_j  (표준화 후 절댓값 평균)
    """
    src, trg = g.edge_index
    n = g.num_nodes

    col_true = torch.zeros(n, device=va_std.device).index_add_(0, trg, g.edge_attr)
    imp      = g.x[:, 0]           # std-space
    tot_std  = g.tot

    σ_tot = torch.tensor(
        scalers["node"]["total"].scale_,
        device=tot_std.device,
        dtype=tot_std.dtype,
    ) + EPS

    col_pred = col_true + va_std + imp
    col_res  = (col_pred - tot_std)               # ★

    return w_col * col_res.abs().mean()


# ───────── batch wrappers (변경 없음) ────────────────────
def pinn_loss_z_batch_standardized(z_cat, batch, scalers):
    off, losses = 0, []
    for g in batch:
        e = g.edge_attr.numel()
        losses.append(pinn_single_z_std(z_cat[off:off+e], g, scalers))
        off += e
    return torch.stack(losses).mean()

def pinn_loss_va_batch_standardized(va_cat, batch, scalers):
    off, losses = 0, []
    for g in batch:
        n = g.num_nodes
        losses.append(pinn_single_va_std(va_cat[off:off+n], g, scalers))
        off += n
    return torch.stack(losses).mean()

def get_pinn_loss_function(kind: str) -> Callable:
    if kind == "Z":
        return pinn_loss_z_batch_standardized
    if kind == "VA":
        return pinn_loss_va_batch_standardized
    raise ValueError(kind)
