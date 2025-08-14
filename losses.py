import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Callable, List
from constants import EPS

# Toggle this to True if you want one-time Z debug printouts
DEBUG_Z_PRINT: bool = False
_DEBUG_PRINTED_Z: int = 0
_DEBUG_MAX_Z: int = 1


def _rel_err(residual: Tensor, scale: Tensor, eps: float = EPS) -> Tensor:
    """
    Relative error: |residual| / (|scale| + eps)
    `scale` should be a positive baseline per node (e.g., g.tot_raw).
    """
    return residual.abs() / (scale.abs() + eps)


def pinn_single_z_raw(
    z_raw: Tensor,
    g: Data,
    *,
    w_row: float = 1.0,
    w_col: float = 0.0,
) -> Tensor:
    """
    Single-graph PINN term for Z on a relative (per-node) basis.
    - Primary term (IOIS-like): |row - col| / TOT_i
    - Optional net-balance term: |(row + FD + EXP - IMP) - (col + VA)| / TOT_i

    Returns a scalar loss averaged over nodes.
    """
    src, trg = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=z_raw.device).index_add_(0, src, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, trg, z_raw)

    # IOIS-style residual (relative per node)
    resid_io_rel = (row - col).abs() / (g.tot_raw.abs() + EPS)

    # Optional net residual (also relative)
    imp_raw, exp_raw, fd_raw = g.x_raw.T
    row_imb = row + fd_raw + exp_raw - imp_raw
    col_imb = col + g.va_raw
    resid_net_rel = (row_imb - col_imb).abs() / (g.tot_raw.abs() + EPS)

    if DEBUG_Z_PRINT and _should_print_z_debug():
        _print_z_debug(z_raw, g, tag="pinn_single_z_raw")

    # Mean over nodes (balanced across sectors)
    loss = w_row * resid_io_rel.mean() + w_col * resid_net_rel.mean()
    return loss


def pinn_single_va_raw(
    va_raw: Tensor,
    g: Data,
    *,
    w_col: float = 1.0,
) -> Tensor:
    """
    Single-graph PINN term for VA on a relative (per-node) basis.
    Enforces: Z·1 + VA ≈ TOT  (column balance)
    Uses relative residual: |(col_sum(Z) + VA - TOT)| / TOT
    """
    _, trg = g.edge_index
    n = g.num_nodes

    col_z = torch.zeros(n, device=va_raw.device).index_add_(0, trg, g.edge_attr)
    col_imb = col_z + va_raw - g.tot_raw

    col_res = _rel_err(col_imb, g.tot_raw)
    return w_col * col_res.mean()


def pinn_loss_z_batch_rel(
    z_cat: Tensor,
    batch: List[Data],
    *,
    include_exogenous: bool = True,
) -> Tensor:
    """
    Batch PINN loss for Z using node-wise relative residuals (recommended).

    For each graph:
      - Compute row/col sums from predicted z
      - If include_exogenous:
           row = Σ_j Z_ij + FD_i + EXP_i - IMP_i
           col = Σ_j Z_ji + VA_i
        else:
           row = Σ_j Z_ij
           col = Σ_j Z_ji
      - Node-wise relative residual: |row_i - col_i| / (TOT_i + eps)
      - Average over nodes per graph, then average over graphs
    """
    acc = z_cat.new_tensor(0.0)
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

        rel = (row - col).abs() / (g.tot_raw.abs() + EPS)
        acc = acc + rel.mean()
        num_graphs += 1

    return acc / max(1, num_graphs)


def pinn_loss_va_batch_raw(va_cat: Tensor, batch: List[Data]) -> Tensor:
    """
    Batch PINN loss for VA using relative column-balance residuals per graph.
    """
    off, losses = 0, []
    for g in batch:
        n = g.num_nodes
        va_slice = va_cat[off:off + n]
        off += n
        losses.append(pinn_single_va_raw(va_slice, g))
    return torch.stack(losses).mean()


def get_pinn_loss_function(kind: str) -> Callable:
    """
    Convenience selector if you want to fetch the PINN loss by kind.
    For 'Z', returns the relative (balanced) formulation by default.
    """
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
    """
    One-time diagnostic print for ranges and residuals in Z prediction.
    Useful when wiring datasets or checking units/scales.
    """
    src, trg = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=z_raw.device).index_add_(0, src, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, trg, z_raw)
    imp_raw, exp_raw, fd_raw = g.x_raw.T
    tot_raw, va_raw = g.tot_raw, g.va_raw

    def _rng(t: Tensor) -> str:
        return f"{t.min():8.4g} → {t.max():8.4g}"

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