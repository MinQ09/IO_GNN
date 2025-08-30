import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Callable, List
from constants import EPS

# Toggle this to True if you want one-time Z debug printouts
DEBUG_Z_PRINT: bool = False
_DEBUG_PRINTED_Z: int = 0
_DEBUG_MAX_Z: int = 1

# -------------------- helpers --------------------

def _stable_den(tot_raw: Tensor, *, q: float = 0.05, gamma: float = 0.5, floor: float = 1e-6) -> Tensor: 
    """
    Build a robust denominator per node:
      den = max(|TOT|, quantile_q(|TOT|)) ** gamma
    """
    den = tot_raw.abs()
    p = torch.quantile(den.detach(), q).clamp_min(floor)  # detach: no grad needed
    den = torch.maximum(den, p).pow(gamma)
    return den

def _rel_err(residual: Tensor, den: Tensor) -> Tensor:
    """Relative error with robust denominator."""
    return residual.abs() / (den + EPS)

# -------------------- single-graph PINNs -----------------

def pinn_single_z_raw(
    z_raw: Tensor,
    g: Data,
    *,
    w_row: float = 1.0,
    w_col: float = 0.0,
    q: float = 0.05,
    gamma: float = 0.5,
) -> Tensor:
    """
    Single-graph PINN term for Z using robust relative residuals.
    - IOIS-like: |row - col| / den
    - Optional net-balance: |(row + FD + EXP - IMP) - (col + VA)| / den
    """
    s, t = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=z_raw.device).index_add_(0, s, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, t, z_raw)

    den = _stable_den(g.tot_raw, q=q, gamma=gamma)

    # IOIS-style residual (relative per node)
    resid_io_rel = _rel_err(row - col, den)

    # Optional net residual (also relative)
    imp_raw, exp_raw, fd_raw = g.x_raw.T
    row_imb = row + fd_raw + exp_raw - imp_raw
    col_imb = col + g.va_raw
    resid_net_rel = _rel_err(row_imb - col_imb, den)

    if DEBUG_Z_PRINT and _should_print_z_debug():
        _print_z_debug(z_raw, g, tag="pinn_single_z_raw")

    return w_row * resid_io_rel.mean() + w_col * resid_net_rel.mean()

def pinn_single_va_raw(
    va_raw: Tensor,
    g: Data,
    *,
    w_col: float = 1.0,
    q: float = 0.05,
    gamma: float = 0.5,
) -> Tensor:
    """
    Single-graph PINN term for VA (column balance), robust relative residual:
      |(col_sum(Z) + VA - TOT)| / den
    """
    _, t = g.edge_index
    n = g.num_nodes

    col_z = torch.zeros(n, device=va_raw.device).index_add_(0, t, g.edge_attr)
    col_imb = col_z + va_raw - g.tot_raw

    den = _stable_den(g.tot_raw, q=q, gamma=gamma)
    col_res = _rel_err(col_imb, den).mean()
    return w_col * col_res  # ← 중복 mean 제거

# -------------------- batch PINNs --------------------

def pinn_loss_z_batch_rel(
    z_cat: Tensor,
    batch: List[Data],
    *,
    include_exogenous: bool = True,
    q: float = 0.05,
    gamma: float = 0.5,
) -> Tensor:
    """
    Batch PINN loss for Z with robust node-wise relative residuals.
    """
    total = z_cat.new_tensor(0.0)
    num_graphs = 0
    off = 0

    for g in batch:
        E = g.edge_index.size(1)
        z = z_cat[off:off + E]
        off += E

        s, t = g.edge_index
        n = g.num_nodes
        row_z = z.new_zeros(n).index_add_(0, s, z)
        col_z = z.new_zeros(n).index_add_(0, t, z)

        if include_exogenous:
            imp, exp, fd = g.x_raw.T
            va = g.va_raw
            row = row_z + fd + exp - imp
            col = col_z + va
        else:
            row, col = row_z, col_z

        den = _stable_den(g.tot_raw, q=q, gamma=gamma)
        rel = _rel_err(row - col, den).mean()

        total = total + rel
        num_graphs += 1

    return total / max(1, num_graphs)

def pinn_loss_va_batch_raw(
    va_cat: Tensor,
    batch: List[Data],
    *,
    q: float = 0.05,
    gamma: float = 0.5,
) -> Tensor:
    """
    Batch PINN loss for VA with robust column-balance residuals per graph.
    """
    off, losses = 0, []
    for g in batch:
        n = g.num_nodes
        va_slice = va_cat[off:off + n]
        off += n
        losses.append(pinn_single_va_raw(va_slice, g, q=q, gamma=gamma))
    return torch.stack(losses).mean()

def get_pinn_loss_function(kind: str) -> Callable:
    if kind == "Z":
        return pinn_loss_z_batch_rel
    if kind == "VA":
        return pinn_loss_va_batch_raw
    raise ValueError(f"Unknown kind: {kind}")

# ---------------------- Optional debug utilities ----------------------

def _should_print_z_debug() -> bool:
    global _DEBUG_PRINTED_Z
    if _DEBUG_PRINTED_Z < _DEBUG_MAX_Z:
        _DEBUG_PRINTED_Z += 1
        return True
    return False

def _print_z_debug(z_raw: Tensor, g: Data, tag: str) -> None:
    s, t = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=z_raw.device).index_add_(0, s, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, t, z_raw)
    imp_raw, exp_raw, fd_raw = g.x_raw.T
    tot_raw, va_raw = g.tot_raw, g.va_raw

    def _rng(x: Tensor) -> str:
        return f"{x.min():8.4g} → {x.max():8.4g}"

    print(f"\n── DEBUG ({tag}) ─────────────────────────")
    print(f"row_sum : {_rng(row)}")
    print(f"col_sum : {_rng(col)}")
    print(f"fd_raw  : {_rng(fd_raw)}")
    print(f"imp_raw : {_rng(imp_raw)}")
    print(f"exp_raw : {_rng(exp_raw)}")
    print(f"tot     : {_rng(tot_raw)}")
    print(f"va      : {_rng(va_raw)}")

    row_imb = row + fd_raw + exp_raw - imp_raw - tot_raw
    col_imb = col + va_raw - tot_raw
    net     = row + fd_raw + exp_raw - imp_raw - col - va_raw
    print(f"row_imb : {_rng(row_imb)}")
    print(f"col_imb : {_rng(col_imb)}")
    print(f"net mean(|·|) : {net.abs().mean():.4g}")
    print("────────────────────────────────────────\n")