import torch
from torch_geometric.data import Data
from typing import List

def smape(pred, true, eps=1e-8):
    denom = (pred.abs() + true.abs()).clamp(min=eps)
    return (2 * (pred - true).abs() / denom).mean().item()

def pinn_loss(pred_raw, g: Data, scale, tot_override=None):
    src, trg = g.edge_index
    n = g.num_nodes
    row = torch.zeros(n, device=pred_raw.device).index_add_(0, src, pred_raw)
    col = torch.zeros(n, device=pred_raw.device).index_add_(0, trg, pred_raw)
    x   = g.x * scale
    va  = g.va * scale
    tot = (tot_override if tot_override is not None else g.tot) * scale
    imp, exp, fd = x[:,0], x[:,1], x[:,2]
    row_res = (row + fd + exp - tot) / (tot + 1e-8)
    col_res = (col + va + imp - tot) / (tot + 1e-8)
    net_res = (row - col + fd + exp - va - imp) / (tot + 1e-8)
    return (row_res.pow(2).mean() + col_res.pow(2).mean() + net_res.pow(2).mean())/3

def pinn_loss_batch(pred_raw, tgts: List[Data], scale, x_override=None):
    e_offs = n_offs = 0
    losses = []
    for g in tgts:
        e = g.edge_attr.numel(); n = g.num_nodes
        losses.append(pinn_loss(pred_raw[e_offs:e_offs+e], g, scale,
                                tot_override=None if x_override is None else x_override[n_offs:n_offs+n]))
        e_offs += e; n_offs += n
    return torch.stack(losses).mean()