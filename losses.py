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
    global _DEBUG_COUNTER
    src, trg = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=z_raw.device).index_add_(0, src, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, trg, z_raw)

    imp_raw, exp_raw, fd_raw = g.x_raw.T

    row_imb = row + fd_raw + exp_raw - imp_raw - g.tot_raw
    col_imb = col + g.va_raw - g.tot_raw
    net = row_imb - col_imb

    # ───── 디버그 출력 (처음 _DEBUG_MAX번) ─────
    if _DEBUG_COUNTER["Z"] < _DEBUG_MAX:
        _print_z_debug(z_raw, g, "pinn_single_z_raw", _DEBUG_COUNTER["Z"]+1)
        _DEBUG_COUNTER["Z"] += 1
    # -----------------------------------------

    return net.abs().mean()

def pinn_single_va_raw(
    va_raw: Tensor,
    g: Data,
    *,
    w_col: float = 1.0,
) -> Tensor:
    src, trg = g.edge_index
    n = g.num_nodes

    col_z = torch.zeros(n, device=va_raw.device).index_add_(0, trg, g.edge_attr)
    col_imb = col_z + va_raw - g.tot_raw  # ✅ 올바른 잔차

    col_res = _rel_err(col_imb, g.tot_raw)
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


# ------------------------------------------------------------
_DEBUG_COUNTER = {"Z": 0, "VA": 0}          # 전역 카운터
_DEBUG_MAX     = 5                          # 찍어볼 횟수

def _print_z_debug(z_raw: Tensor, g: Data, tag: str, step: int) -> None:
    """row/col/FD/IMP/EXP/TOT/VA 범위 & 통계 출력."""
    src, trg = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=z_raw.device).index_add_(0, src, z_raw)
    col = torch.zeros(n, device=z_raw.device).index_add_(0, trg, z_raw)
    imp_raw, exp_raw, fd_raw = g.x_raw.T           # 이미 /1e6 된 값
    tot, va = g.tot, g.va

    def _rng(t):       # 편의용
        return f"{t.min():8.4g} → {t.max():8.4g}"

    print(f"\n── DEBUG #{step} ({tag}) ─────────────────────────")
    print(f"row_sum : {_rng(row)}")
    print(f"col_sum : {_rng(col)}")
    print(f"fd_raw  : {_rng(fd_raw)}")
    print(f"imp_raw : {_rng(imp_raw)}")
    print(f"exp_raw : {_rng(exp_raw)}")
    print(f"tot     : {_rng(tot)}")
    print(f"va      : {_rng(va)}")

    # 잔차
    row_imb = row + fd_raw + exp_raw - imp_raw - tot
    col_imb = col + va - tot
    net     = row + fd_raw + exp_raw - imp_raw - col - va
    print(f"row_imb : {_rng(row_imb)}")
    print(f"col_imb : {_rng(col_imb)}")
    print(f"net mean(abs) : {net.abs().mean():.4g}")
    print("────────────────────────────────────────\n")


def get_pinn_loss_function(kind: str) -> Callable:
    if kind == "Z":
        return pinn_loss_z_batch_raw
    if kind == "VA":
        return pinn_loss_va_batch_raw
    raise ValueError(kind)

