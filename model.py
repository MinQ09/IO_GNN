# model.py  ───────────────────────────────────────────────────────────────
"""
Graph-temporal model for IO-GNN.

* AttChebDirConv   : Directional Chebyshev filter + analysis-only attention
* GCLSTMCell       : LSTM cell whose gates are AttChebDirConv blocks
* IOGNN            : End-to-end model (W-step history → next-step edge flows)
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.utils import softmax


# ─────────────────────── AttChebDirConv ───────────────────────
class AttChebDirConv(nn.Module):
    """
    Directional Chebyshev convolution **with analysis-only attention**.

    * ``last_att_out`` : edge weights normalised over **out-going** edges (per source)
    * ``last_att_in``  : edge weights normalised over **in-coming** edges (per target)

    Both scores are computed inside a ``torch.no_grad()`` block
    → **they do not affect training**.
    """

    def __init__(
        self,
        fin: int,
        fout: int,
        k: int = 5,
        alpha_init: float = 0.5,
        learn_alpha: bool = True,
        att_hidden: int = 64,
    ):
        super().__init__()

        # Directional Chebyshev filters
        self.fwd_conv = ChebConv(fin, fout, k)
        self.bwd_conv = ChebConv(fin, fout, k)

        # Frozen Query / Key projections (analysis only)
        self.to_q = nn.Linear(fin, att_hidden, bias=False)
        self.to_k = nn.Linear(fin, att_hidden, bias=False)
        for p in (*self.to_q.parameters(), *self.to_k.parameters()):
            p.requires_grad = False

        # Mixing coefficient α   (learned in logit space)
        beta = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
        self._beta = nn.Parameter(beta) if learn_alpha else beta

        # Safe default tensors for first access
        self.register_buffer("last_att_out", torch.empty(0))
        self.register_buffer("last_att_in",  torch.empty(0))

    # ----------------------------------------------------------
    @property
    def alpha(self) -> Tensor:
        # Clamp for numerical stability
        return torch.sigmoid(self._beta.clamp(-5.0, 5.0))

    # ----------------------------------------------------------
    def forward(
        self,
        x: Tensor,            # [N, F_in]
        edge_index: Tensor,   # [2, E]  (source, target)
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        s, t = edge_index  # source, target

        # (1) Directional Chebyshev outputs
        out_fwd = self.fwd_conv(x, edge_index, edge_weight)
        out_bwd = self.bwd_conv(x, edge_index.flip(0), edge_weight)
        out = self.alpha * out_fwd + (1.0 - self.alpha) * out_bwd

        # (2) Attention scores (analysis only)
        with torch.no_grad():
            q = self.to_q(x)           # [N, H]
            k = self.to_k(x)           # [N, H]
            e = (q[s] * k[t]).sum(-1)  # dot-product score   [E]
            if edge_weight is not None:
                e = e * edge_weight
            # Normalise per source / per target
            att_out = softmax(e, index=s, num_nodes=x.size(0))  # out-going
            att_in  = softmax(e, index=t, num_nodes=x.size(0))  # in-coming
            self.last_att_out = att_out.detach()
            self.last_att_in  = att_in.detach()

        return out


# ─────────────────────── GCLSTMCell ───────────────────────
class GCLSTMCell(nn.Module):
    """
    Chebyshev-based Graph Convolutional LSTM cell.
    """

    def __init__(
        self,
        fin: int,
        hid: int,
        k: int,
        alpha_init: float,
        att_hidden: int = 32,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.hid = hid

        def conv(fi, fo):
            return AttChebDirConv(fi, fo, k, alpha_init, learn_alpha=True,
                                  att_hidden=att_hidden)

        self.Fx, self.Fh = conv(fin, hid), conv(hid, hid)
        self.Ix, self.Ih = conv(fin, hid), conv(hid, hid)
        self.Ox, self.Oh = conv(fin, hid), conv(hid, hid)
        self.Gx, self.Gh = conv(fin, hid), conv(hid, hid)

    # ------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        ei: Tensor,
        ew: Tensor,
        h: Tensor | None = None,
        c: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        n = x.size(0)
        if h is None or c is None:
            h = x.new_zeros(n, self.hid)
            c = x.new_zeros(n, self.hid)

        if self.use_attention:
            f = torch.sigmoid(self.Fx(x, ei, ew) + self.Fh(h, ei, ew))
            i = torch.sigmoid(self.Ix(x, ei, ew) + self.Ih(h, ei, ew))
            o = torch.sigmoid(self.Ox(x, ei, ew) + self.Oh(h, ei, ew))
            g = torch.tanh   (self.Gx(x, ei, ew) + self.Gh(h, ei, ew))
        else:
            # Ignore edge weights → substitute ones
            ew_dummy = torch.ones(ei.size(1), device=ei.device)
            f = torch.sigmoid(self.Fx(x, ei, ew_dummy) + self.Fh(h, ei, ew_dummy))
            i = torch.sigmoid(self.Ix(x, ei, ew_dummy) + self.Ih(h, ei, ew_dummy))
            o = torch.sigmoid(self.Ox(x, ei, ew_dummy) + self.Oh(h, ei, ew_dummy))
            g = torch.tanh   (self.Gx(x, ei, ew_dummy) + self.Gh(h, ei, ew_dummy))

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# ─────────────────────── helper MLP ───────────────────────
def mlp(
    d_in: int,
    d_hidden: int,
    depth: int = 3,
    dropout: float = 0.2,
    *,
    negative_slope: float = 1e-2
) -> nn.Sequential:
    
    layers: List[nn.Module] = []
    for i in range(depth):
        layers += [
            nn.Linear(d_in if i == 0 else d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
    
    layers.append(nn.Linear(d_hidden, 1))

    layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
    
    return nn.Sequential(*layers)


# ─────────────────────── 1) Edge model ───────────────────────
class IOGNN_Z(nn.Module):
    """
    Edge-level IO-GNN (predicts next-step inter-industry flows Ẑ_e).
    """

    def __init__(self, nfeat: int, cfg):
        super().__init__()
        # ─ backbone ─
        self.pre = nn.Sequential(
            nn.Linear(nfeat, cfg.hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden),
        )
        self.cell = GCLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden, k=cfg.k,
            alpha_init=cfg.alpha, att_hidden=cfg.att_hidden,
            use_attention=True,
        )
        # ─ edge decoder ─
        self.dec_edge = mlp(
            d_in=cfg.hidden * 2,
            d_hidden=cfg.hidden,
            depth=cfg.depth_edge,
            dropout=cfg.dropout,
        )

    # ----------------------------------------------------------
    def forward(
        self,
        seq_batch: List[List[Data]],
        tgt_batch: List[Data],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z_preds : Tensor[∑E]  edge-flow predictions
        att_out / att_in : Tensor[∑E]  analysis-only attention
        """
        z_outs, att_o, att_i = [], [], []
        for seq, tgt in zip(seq_batch, tgt_batch):
            h = c = None
            for g in seq:                      # W-step history
                h, c = self.cell(self.pre(g.x), g.edge_index, g.edge_attr, h, c)

            att_o.append(self.cell.Ox.last_att_out)
            att_i.append(self.cell.Ox.last_att_in)

            s, t = tgt.edge_index
            pairs = torch.cat([h[s], h[t]], dim=1)
            z_outs.append(self.dec_edge(pairs).squeeze(-1))

        return torch.cat(z_outs), torch.cat(att_o), torch.cat(att_i)


# ─────────────────────── 2) VA model ───────────────────────
class IOGNN_VA(nn.Module):
    """
    Node-level IO-GNN (predicts next-step Value Added VÂ_n).
    Same backbone; only the decoder changes.
    """

    def __init__(self, nfeat: int, cfg):
        super().__init__()
        # ─ backbone (shared design) ─
        self.pre = nn.Sequential(
            nn.Linear(nfeat, cfg.hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden),
        )
        self.cell = GCLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden, k=cfg.k,
            alpha_init=cfg.alpha, att_hidden=cfg.att_hidden,
            use_attention=True,
        )
        # ─ node decoder ─
        self.dec_node = mlp(
            d_in=cfg.hidden,
            d_hidden=cfg.hidden,
            depth=max(1, cfg.depth_edge - 1),  # slightly smaller by default
            dropout=cfg.dropout,
        )

    # ----------------------------------------------------------
    def forward(
        self,
        seq_batch: List[List[Data]],
        tgt_batch: List[Data],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        va_preds : Tensor[∑N]  node-level VA predictions
        att_out / att_in : Tensor[∑E]  attention scores (from output gate)
        """
        va_outs, att_o, att_i = [], [], []
        for seq, tgt in zip(seq_batch, tgt_batch):
            h = c = None
            for g in seq:
                h, c = self.cell(self.pre(g.x), g.edge_index, g.edge_attr, h, c)

            att_o.append(self.cell.Ox.last_att_out)
            att_i.append(self.cell.Ox.last_att_in)
            va_outs.append(self.dec_node(h).squeeze(-1))   # [N]

        return torch.cat(va_outs), torch.cat(att_o), torch.cat(att_i)