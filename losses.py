import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Any, Dict, Callable

EPS = 1e-12                     # 수치 안전용

# ───────── Z PINN 3-in-1  ─────────────────────────────────────────
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
    • row_res  : ∑_j Z_ij + FD_i + EXP_i − TOT_i = 0
    • col_res  : ∑_i Z_ij + VA_j + IMP_j − TOT_j = 0
    • net_res  : (row_res − col_res) = 0   ← 중복 상쇄용
      (세 식 모두 *표준화 공간*에서 계산)

    가중치 w_* 로 항별 비중을 조정할 수 있습니다.
    """
    src, trg = g.edge_index
    n = g.num_nodes

    # ── 집계 ───────────────────────────────────────────
    row = torch.zeros(n, device=z_std.device).index_add_(0, src, z_std)
    col = torch.zeros(n, device=z_std.device).index_add_(0, trg, z_std)

    imp, exp, fd = g.x.T        # (std-space)
    va_std       = g.va         # (std)
    tot_std      = g.tot        # (std)

    # ── σ_tot 로 나눠 스케일 평준화 ─────────────────────
    σ_tot = torch.tensor(
        scalers["node"]["total"].scale_, device=tot_std.device, dtype=tot_std.dtype
    ) + EPS                      # safe_den

    row_res = (row + fd + exp - tot_std) 
    col_res = (col + va_std + imp - tot_std) 
    net_res = row_res - col_res

    return net_res.square().mean()

# ───────── VA PINN 1-eq  (열 합계) ────────────────────────────────
def pinn_single_va_std(
    va_std: Tensor,
    g: Data,
    scalers: Dict[str, Any],
    *,
    w_col: float = 1.0,
) -> Tensor:
    """
    • col_res  : ∑_i Z_ij + VA_j + IMP_j − TOT_j = 0
      (실제 Z·IMP 는 그래프에서, VA_j 예측값은 va_std)
    """
    src, trg = g.edge_index
    n = g.num_nodes

    row_true = torch.zeros(n, device=va_std.device).index_add_(0, src, g.edge_attr)
    col_true = torch.zeros(n, device=va_std.device).index_add_(0, trg, g.edge_attr)

    imp = g.x[:, 0]           # std-space (imports)
    tot_std = g.tot

    σ_tot = torch.tensor(
        scalers["node"]["total"].scale_, device=tot_std.device, dtype=tot_std.dtype
    ) + EPS

    col_pred = col_true + va_std + imp
    col_res  = (col_pred - tot_std)

    return col_res.abs().mean()

# ───────── batch wrappers & factory (변경 없음) ─────────────────────
def pinn_loss_z_batch_standardized(z_cat, batch, scalers):  # scalers 인자 유지
    off = 0; losses = []
    for g in batch:
        e = g.edge_attr.numel()
        losses.append(pinn_single_z_std(z_cat[off:off+e], g, scalers))
        off += e
    return torch.stack(losses).mean()

def pinn_loss_va_batch_standardized(va_cat, batch, scalers):
    off = 0; losses = []
    for g in batch:
        n = g.num_nodes
        losses.append(pinn_single_va_std(va_cat[off:off+n], g, scalers))
        off += n
    return torch.stack(losses).mean()

def get_pinn_loss_function(kind: str, *, use_standardized: bool = True) -> Callable:
    if kind == "Z":  return pinn_loss_z_batch_standardized
    if kind == "VA": return pinn_loss_va_batch_standardized
    raise ValueError(kind)