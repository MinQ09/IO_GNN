# metrics.py
from __future__ import annotations
import torch
from torch_geometric.data import Data
from torch import Tensor

EPS: float = 1e-8  # 모듈 전역 상수로 통일

# ────────────────────────────────────────────────
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
        return float("nan")  # 분산 0 → 정의 불가
    ss_res = ((pred - true) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

@torch.inference_mode()
def mape(pred: Tensor, true: Tensor, eps: float = EPS) -> float:
    pred, true = pred.float(), true.float()
    mask = true.abs() > eps            # 0 분모 제외
    if mask.sum() == 0:
        return float("nan")
    return torch.mean((pred[mask] - true[mask]).abs() / true[mask].abs()).item()

@torch.inference_mode()
def smape(pred: Tensor, true: Tensor, eps: float = EPS) -> float:
    pred, true = pred.float(), true.float()
    denom = (pred.abs() + true.abs()).clamp(min=eps)
    return (2 * (pred - true).abs() / denom).mean().item()  # 0~2 범위

# ─────────── CVR (Constraint-Violation-Rate) ───────────
@torch.inference_mode()
def cvr_tensor(pred_raw: Tensor, g: Data, scale_node: float) -> float:
    """
    Parameters
    ----------
    pred_raw : Tensor[E]
        Predicted edge flows **in original scale** (already expm1 & *scale_Z).
    g : Data
        Target PyG graph containing x, va, tot, edge_index.
    scale_node : float
        Node-level scaling factor applied during pre-processing.

    Returns
    -------
    float
        Normalised constraint-violation rate, 0.0 ~ 1.0.
    """
    src, trg = g.edge_index
    n = g.num_nodes

    row = torch.zeros(n, device=pred_raw.device).index_add_(0, src, pred_raw)
    col = torch.zeros(n, device=pred_raw.device).index_add_(0, trg, pred_raw)

    x   = g.x * scale_node            # [Imports, Exports, Final_Demand]
    va  = g.va * scale_node
    tot = g.tot * scale_node

    imp, exp, fd = x[:, 0], x[:, 1], x[:, 2]

    row += fd + exp                   # 각 행 합계
    col += va + imp                   # 각 열 합계

    mismatch = (row - col).abs().sum()
    return (mismatch / tot.sum().clamp(min=EPS)).item()