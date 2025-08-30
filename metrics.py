# metrics.py — drop-in enhanced version

from __future__ import annotations
from typing import List
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from constants import EPS

# ------------------------------ Base metrics ------------------------------ #

@torch.inference_mode()
def rmse(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.float().reshape(-1), true.float().reshape(-1)
    return torch.sqrt(((pred - true) ** 2).mean()).item()

@torch.inference_mode()
def mae(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.float().reshape(-1), true.float().reshape(-1)
    return (pred - true).abs().mean().item()

@torch.inference_mode()
def r2(pred: Tensor, true: Tensor, *, zero_var_value: float = float("nan")) -> float:
    pred, true = pred.float().reshape(-1), true.float().reshape(-1)
    ss_tot = ((true - true.mean()) ** 2).sum()
    if ss_tot < EPS:
        return zero_var_value
    ss_res = ((pred - true) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()

@torch.inference_mode()
def smape(pred: Tensor, true: Tensor) -> float:
    pred, true = pred.float().reshape(-1), true.float().reshape(-1)
    denom = (pred.abs() + true.abs()).clamp(min=EPS)
    return (2.0 * (pred - true).abs() / denom).mean().item()

@torch.inference_mode()
def safe_pearson(x: Tensor, y: Tensor) -> float:
    x_np, y_np = x.detach().view(-1).cpu().numpy(), y.detach().view(-1).cpu().numpy()
    mask = np.isfinite(x_np) & np.isfinite(y_np)
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(np.corrcoef(x_np[mask], y_np[mask])[0, 1])
    except Exception:
        return float("nan")

def mean_ignore_nan(vals: List[float]) -> float:
    return float(np.nanmean(vals)) if vals else float("nan")

# -------------------------- helpers for IOIS -------------------------- #

@torch.inference_mode()
def _stable_den(tot: Tensor, *, q: float = 0.05, gamma: float = 1.0, floor: float = 1e-6) -> Tensor:
    """
    Robust per-node denominator:
      den = max(|TOT|, quantile_q(|TOT|)) ** gamma
    - gamma=1.0이면 기존 IOIS와 동일
    - gamma<1.0이면 큰 산업 가중을 완만하게 줄임
    """
    den = tot.abs()
    p = torch.quantile(den.detach(), q).clamp_min(floor)
    return torch.maximum(den, p).pow(gamma)

@torch.inference_mode()
def _row_col_from_pred(pred_raw: Tensor, g: Data, include_exogenous: bool) -> tuple[Tensor, Tensor]:
    s, t = g.edge_index
    n = g.num_nodes
    row_z = torch.zeros(n, device=pred_raw.device).index_add_(0, s, pred_raw)
    col_z = torch.zeros(n, device=pred_raw.device).index_add_(0, t, pred_raw)
    if include_exogenous:
        imp, exp, fd = g.x_raw.T  # [N]
        va = g.va_raw
        row = row_z + fd + exp - imp
        col = col_z + va
    else:
        row, col = row_z, col_z
    return row, col

# ------------------------------ IOIS metrics (RAW) ------------------------------ #

@torch.inference_mode()
def iois_z_raw(
    pred_raw: Tensor,
    g: Data,
    *,
    include_exogenous: bool = True,
) -> float:
    """
    Absolute IOIS on RAW scale:
      (Σ_i |row_i - col_i|) / (Σ_i TOT_i)
    """
    row, col = _row_col_from_pred(pred_raw, g, include_exogenous)
    mismatch = (row - col).abs().sum()
    total_output = g.tot_raw.sum().clamp(min=EPS)
    return (mismatch / total_output).item()

@torch.inference_mode()
def iois_z_rel_raw(
    pred_raw: Tensor,
    g: Data,
    *,
    include_exogenous: bool = True,
    robust: bool = False,
    q: float = 0.05,
    gamma: float = 1.0,
) -> float:
    """
    Node-wise relative IOIS on RAW scale:
      mean_i  |row_i - col_i| / den_i
    - robust=True면 den_i = max(|TOT_i|, q-quantile)|^gamma 로 안정화
    """
    row, col = _row_col_from_pred(pred_raw, g, include_exogenous)
    den = _stable_den(g.tot_raw, q=q, gamma=gamma) if robust else g.tot_raw.abs().clamp_min(EPS)
    return ((row - col).abs() / (den + EPS)).mean().item()

@torch.inference_mode()
def iois_z_rel_batch(
    pred_cat: Tensor,
    graphs: List[Data],
    *,
    include_exogenous: bool = True,
    robust: bool = False,
    q: float = 0.05,
    gamma: float = 1.0,
) -> float:
    """
    Average of graph-wise node-mean relative IOIS.
    """
    acc = 0.0
    off = 0
    num = 0
    for g in graphs:
        E = g.edge_index.size(1)
        pred = pred_cat[off:off + E]
        off += E
        acc += iois_z_rel_raw(
            pred, g,
            include_exogenous=include_exogenous,
            robust=robust, q=q, gamma=gamma
        )
        num += 1
    return acc / max(1, num)