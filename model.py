# model.py  ───────────────────────────────────────────────────────────────
"""
Graph-temporal models for IO-GNN (Directional MPNN + GraphLSTMCell).

Key features
------------
- DirMPNN: explicit directional message passing (src→dst)
- GraphLSTMCell: LSTM-style recurrent cell using DirMPNN gates
  * forget-gate bias = +1.0 (long-range stability)
- Backward compatibility:
  * IOGNN_Z/VA __init__ supports (nfeat, cfg) and (nfeat, edge_feat_dim, cfg)
  * Provide alias: GCLSTMCell = GraphLSTMCell
- Flags: compute_attention, use_edge_weight, use_bwd_weights, alpha_mode('scalar'|'channel')
- Edge decoder features: [h_s, h_t, h_s⊙h_t, |h_s−h_t|]
- Optional non-negative VA head via cfg.va_nonneg (Softplus)
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math
# ───────────────────────── DirMPNN (Directional MPNN) ─────────────────────────
class DirMPNN(MessagePassing):
    """
    Explicit directional message passing:
      m_{i<-j} = phi_m([x_j, x_i, e_{j->i}])
      x'_i     = alpha * AGG_fwd(m_{i<-j}) + (1-alpha) * AGG_bwd(m_{i->j})

    We compute forward and backward paths separately; mixing via alpha happens
    in forward() if x_bwd_out is given.
    """

    def __init__(
        self,
        fin: int,
        fout: int,
        edge_dim: Optional[int] = None,
        att_hidden: int = 64,
        alpha_init: float = 0.5,
        learn_alpha: bool = True,
        alpha_mode: str = "scalar",   # 'scalar' | 'channel'
        compute_attention: bool = False,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.fout = fout
        self.edge_dim = edge_dim
        self.expect_edge = edge_dim is not None  # ← NEW: fix input dimensionality
        self.compute_attention = compute_attention
        self.alpha_mode = alpha_mode

        xin = fin + fin + (edge_dim or 0)  # [x_src, x_dst, e]
        hidden = max(fout, fin)

        self.phi_m = nn.Sequential(
            nn.Linear(xin, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, fout),
        )

        # alpha (mixing forward/backward paths)
        if alpha_mode == "scalar":
            beta = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
            self._beta = nn.Parameter(beta) if learn_alpha else beta
        elif alpha_mode == "channel":
            beta = torch.full((fout,), torch.logit(torch.tensor(alpha_init)), dtype=torch.float32)
            self._beta = nn.Parameter(beta) if learn_alpha else beta
        else:
            raise ValueError("alpha_mode must be 'scalar' or 'channel'")

        # Analysis-only attention (frozen Q/K)
        self.to_q = nn.Linear(fin, att_hidden, bias=False)
        self.to_k = nn.Linear(fin, att_hidden, bias=False)
        for p in (*self.to_q.parameters(), *self.to_k.parameters()):
            p.requires_grad = False

        self.register_buffer("last_att_out", torch.empty(0))
        self.register_buffer("last_att_in", torch.empty(0))

    @property
    def alpha(self) -> Tensor:
        a = torch.sigmoid(self._beta.clamp(-5.0, 5.0))
        if self.alpha_mode == "scalar":
            return a
        return a.view(1, -1)  # [1, F]

    def forward(
        self,
        x: Tensor,               # [N, F_in]
        edge_index: Tensor,      # [2, E] (src->dst)
        edge_attr: Optional[Tensor] = None,
        x_bwd_out: Optional[Tensor] = None,  # precomputed backward output to mix
    ) -> Tensor:
        src, dst = edge_index
        out_fwd = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

        out = out_fwd if x_bwd_out is None else (self.alpha * out_fwd + (1.0 - self.alpha) * x_bwd_out)

        # analysis-only attention
        if self.compute_attention:
            with torch.no_grad():
                q = self.to_q(x)  # [N, H]
                k = self.to_k(x)  # [N, H]
                e = (q[src] * k[dst]).sum(-1)  # [E]
                # Only apply scalar edge weight when the layer expects edges AND a 1-D edge vector is provided.
                if self.expect_edge and (edge_attr is not None) and (edge_attr.dim() == 1):
                    e = e * edge_attr
                self.last_att_out = softmax(e, index=src, num_nodes=x.size(0)).detach()
                self.last_att_in  = softmax(e, index=dst, num_nodes=x.size(0)).detach()
        else:
            self.last_att_out = x.new_empty(0)
            self.last_att_in  = x.new_empty(0)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        # x_j: src, x_i: dst
        if not self.expect_edge:
            # Edge features disabled at construction time: ignore runtime edge_attr
            z = torch.cat([x_j, x_i], dim=-1)
        else:
            # Edge features expected: always provide tensor of shape [E, edge_dim]
            if edge_attr is None:
                E = x_j.size(0)
                ea = x_j.new_zeros((E, int(self.edge_dim)))
            else:
                ea = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
                if ea.size(-1) != int(self.edge_dim):
                    raise RuntimeError(
                        f"[DirMPNN] edge_attr dim mismatch: got {ea.size(-1)} "
                        f"but layer was built with edge_dim={self.edge_dim}. "
                        "Set cfg.edge_feat_dim accordingly or disable edges via cfg.use_edge_weight=False."
                    )
            z = torch.cat([x_j, x_i, ea], dim=-1)
        return self.phi_m(z)

    def compute_att(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]):
        with torch.no_grad():
            s, t = edge_index
            q, k = self.to_q(x), self.to_k(x)
            e = (q[s] * k[t]).sum(-1)
            if self.expect_edge and (edge_attr is not None) and (edge_attr.dim() == 1):
                e = e * edge_attr
            att_out = softmax(e, index=s, num_nodes=x.size(0)).detach()
            att_in  = softmax(e, index=t, num_nodes=x.size(0)).detach()
        return att_out, att_in

# ─────────────────────────── GraphLSTMCell (DirMPNN 기반) ───────────────────────────
class GraphLSTMCell(nn.Module):
    """
    LSTM-style recurrent cell with directional MPNN gates.

    Flags:
      - compute_attention: only controls attention computation (analysis)
      - use_edge_weight  : controls edge_attr usage in message passing
      - use_bwd_weights  : whether to use edge_attr_bwd for backward pass (if provided)
      - alpha_mode       : 'scalar' or 'channel'
    """

    def __init__(
        self,
        fin: int,
        hid: int,
        edge_dim: Optional[int],
        alpha_init: float = 0.5,
        att_hidden: int = 64,
        compute_attention: bool = False,
        use_edge_weight: bool = True,
        use_bwd_weights: bool = False,
        alpha_mode: str = "scalar",
    ):
        super().__init__()
        self.hid = hid
        self.use_edge_weight = use_edge_weight
        self.use_bwd_weights = use_bwd_weights

        def conv(fi: int, fo: int) -> DirMPNN:
            return DirMPNN(
                fin=fi, fout=fo, edge_dim=edge_dim if self.use_edge_weight else None,
                att_hidden=att_hidden, alpha_init=alpha_init,
                learn_alpha=True, alpha_mode=alpha_mode,
                compute_attention=compute_attention,
            )

        self.Fx, self.Fh = conv(fin, hid), conv(hid, hid)
        self.Ix, self.Ih = conv(fin, hid), conv(hid, hid)
        self.Ox, self.Oh = conv(fin, hid), conv(hid, hid)
        self.Gx, self.Gh = conv(fin, hid), conv(hid, hid)

        self.ln_f = nn.LayerNorm(hid)
        self.ln_i = nn.LayerNorm(hid)
        self.ln_o = nn.LayerNorm(hid)
        self.ln_g = nn.LayerNorm(hid)
        self.ln_c = nn.LayerNorm(hid)

        # Residual on h (near-identity)
        self.hidden_proj = nn.Linear(hid, hid)
        with torch.no_grad():
            if self.hidden_proj.weight.shape[0] == self.hidden_proj.weight.shape[1]:
                nn.init.eye_(self.hidden_proj.weight)
            else:
                nn.init.kaiming_uniform_(self.hidden_proj.weight, a=math.sqrt(5))  # fallback
            nn.init.constant_(self.hidden_proj.bias, 0.0)

        # ── Gate biases (per-feature); forget bias = +1.0 ─────────────────
        self.b_f = nn.Parameter(torch.ones(hid))   # forget gate bias = +1
        self.b_i = nn.Parameter(torch.zeros(hid))  # input gate bias  = 0
        self.b_o = nn.Parameter(torch.zeros(hid))  # output gate bias = 0
        self.b_g = nn.Parameter(torch.zeros(hid))  # candidate bias   = 0

    def _dir_pass(
        self,
        conv_x: DirMPNN, conv_h: DirMPNN,
        x: Tensor, h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        edge_attr_bwd: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # forward path outputs
        fx = conv_x(x, edge_index, edge_attr=edge_attr)
        fh = conv_h(h, edge_index, edge_attr=edge_attr)

        # backward path outputs (mixed inside conv via alpha)
        bwd_index = edge_index.flip(0)
        edge_attr_b = edge_attr_bwd if self.use_bwd_weights else None

        fx_b = conv_x(x, bwd_index, edge_attr=edge_attr_b)
        fh_b = conv_h(h, bwd_index, edge_attr=edge_attr_b)

        fx_mixed = conv_x(x, edge_index, edge_attr=edge_attr, x_bwd_out=fx_b)
        fh_mixed = conv_h(h, edge_index, edge_attr=edge_attr, x_bwd_out=fh_b)
        return fx_mixed, fh_mixed

    def forward(
        self,
        x: Tensor,                  # [N, F_in]
        ei: Tensor,                 # [2, E]
        ew: Optional[Tensor],       # [E, D] or [E]
        h: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        ew_bwd: Optional[Tensor] = None,  # optional backward weights
    ) -> Tuple[Tensor, Tensor]:
        n = x.size(0)
        if h is None or c is None:
            h = x.new_zeros(n, self.hid)
            c = x.new_zeros(n, self.hid)

        # Gates with directional mixing inside
        xf, hf = self._dir_pass(self.Fx, self.Fh, x, h, ei, ew, ew_bwd)
        xi, hi = self._dir_pass(self.Ix, self.Ih, x, h, ei, ew, ew_bwd)
        xo, ho = self._dir_pass(self.Ox, self.Oh, x, h, ei, ew, ew_bwd)
        xg, hg = self._dir_pass(self.Gx, self.Gh, x, h, ei, ew, ew_bwd)

        # LayerNorm then add per-gate bias (forget bias = +1)
        f = torch.sigmoid(self.ln_f(xf + hf) + self.b_f)
        i = torch.sigmoid(self.ln_i(xi + hi) + self.b_i)
        o = torch.sigmoid(self.ln_o(xo + ho) + self.b_o)
        g = torch.tanh(   self.ln_g(xg + hg) + self.b_g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(self.ln_c(c_new)) + self.hidden_proj(h)
        return h_new, c_new


# Backward-compat alias
GCLSTMCell = GraphLSTMCell


# ───────────────────────────── helper MLP ─────────────────────────────
def mlp(
    d_in: int,
    d_hidden: int,
    depth: int = 4,
    dropout: float = 0.2,
    out_dim: int = 1,
    out_activation: Optional[nn.Module] = None,
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
    layers.append(nn.Linear(d_hidden, out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)

# ───────────────────────────── 1) Edge model ─────────────────────────────
class IOGNN_Z(nn.Module):
    """Edge-level IO-GNN (predicts next-step inter-industry flows Ẑ_e)."""

    def __init__(self, nfeat: int, *args, **kwargs):
        """
        Backward-compatible + keyword-friendly constructor.

        Accepted call patterns:
          - New style (positional):  IOGNN_Z(nfeat, edge_feat_dim, cfg)
          - Old style (positional):  IOGNN_Z(nfeat, cfg)  # edge_feat_dim inferred from cfg.edge_feat_dim or None
          - Keyword style:           IOGNN_Z(nfeat, cfg=..., edge_feat_dim=...)

        Args
        ----
        nfeat : int
            Node feature dimension.
        edge_feat_dim : Optional[int]
            Edge feature dimension (e.g., 1 for scalar weights). If None, edges are treated as unweighted.
        cfg : object
            Experiment/config object providing hyperparameters and flags, e.g.:
              - hidden, k/alpha/att_hidden (if used upstream), depth_edge, dropout
              - compute_attention, use_edge_weight, use_bwd_weights, alpha_mode
              - edge_feat_dim (optional default for edge_feat_dim)
        """
        super().__init__()

        # Unpack cfg / edge_feat_dim from args and kwargs (backward compatible).
        cfg = kwargs.get("cfg", None)
        edge_feat_dim_kw = kwargs.get("edge_feat_dim", None)
        if cfg is None:
            if len(args) == 2:
                edge_feat_dim, cfg = args
            elif len(args) == 1:
                (cfg,) = args
                edge_feat_dim = getattr(cfg, "edge_feat_dim", None)
            else:
                raise TypeError(
                    "IOGNN_Z.__init__ expected (nfeat, cfg) or (nfeat, edge_feat_dim, cfg) "
                    "or keyword form (nfeat, cfg=..., edge_feat_dim=...)"
                )
        else:
            edge_feat_dim = edge_feat_dim_kw if edge_feat_dim_kw is not None else getattr(cfg, "edge_feat_dim", None)

        if getattr(cfg, "use_edge_weight", True) and (edge_feat_dim is None):
            edge_feat_dim = 1  # default to scalar edge weights if enabled but not specified

        self.use_bwd_weights = getattr(cfg, "use_bwd_weights", False)

        # Backbone: simple linear projection with LN → GELU
        self.pre = nn.Sequential(
            nn.Linear(nfeat, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.GELU(),
        )

        # Recurrent graph-temporal cell (directional MPNN inside)
        self.cell = GraphLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden,
            edge_dim=edge_feat_dim,
            alpha_init=cfg.alpha,
            att_hidden=cfg.att_hidden,
            compute_attention=getattr(cfg, "compute_attention", False),
            use_edge_weight=getattr(cfg, "use_edge_weight", True),
            use_bwd_weights=getattr(cfg, "use_bwd_weights", False),
            alpha_mode=getattr(cfg, "alpha_mode", "scalar"),
        )

        # Edge decoder on pairwise node states:
        #   features = [h_s, h_t, h_s ⊙ h_t, |h_s − h_t|]
        self.dec_edge = mlp(
            d_in=cfg.hidden * 4,
            d_hidden=cfg.hidden,
            depth=cfg.depth_edge,
            dropout=cfg.dropout,
            out_dim=1,
            out_activation=None,
        )

    def forward(
        self,
        seq_batch: List[List[Data]],
        tgt_batch: List[Data],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass over a batch of graph sequences with respective target graphs.

        Parameters
        ----------
        seq_batch : List[List[Data]]
            For each sample, a list of `Data` objects (historical window).
        tgt_batch : List[Data]
            For each sample, the target `Data` (the graph to predict on).

        Returns
        -------
        z_concat : Tensor[∑E]
            Concatenated edge predictions across the mini-batch.
        att_out : Tensor[∑E]
            Analysis-only attention (outgoing) from the output gate (if enabled).
        att_in : Tensor[∑E]
            Analysis-only attention (incoming) from the output gate (if enabled).
        """
        z_outs, att_o, att_i = [], [], []

        for seq, tgt in zip(seq_batch, tgt_batch):
            # Run the recurrent cell over the historical sequence
            h = c = None
            for g in seq:
                ew_bwd = getattr(g, "edge_attr_bwd", None) if self.use_bwd_weights else None
                h, c = self.cell(self.pre(g.x), g.edge_index, g.edge_attr, h, c, ew_bwd=ew_bwd)

            # Analysis-only attention on the target graph (from Ox gate)
            ao, ai = self.cell.Ox.compute_att(h, tgt.edge_index, tgt.edge_attr)
            att_o.append(ao); att_i.append(ai)

            # Edge-wise decoding on the target graph
            s, t = tgt.edge_index
            hs, ht = h[s], h[t]
            pair_feat = torch.cat([hs, ht, hs * ht, torch.abs(hs - ht)], dim=1)
            z_outs.append(self.dec_edge(pair_feat).squeeze(-1))  # [E]

        return torch.cat(z_outs), torch.cat(att_o), torch.cat(att_i)

    def full_forward(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[Tensor, List[Data]]:
        """
        Convenience: run forward over ALL graphs in `loader` and concat predictions.

        Keeps autograd graph for downstream PINN loss computation.

        Returns
        -------
        z_cat_all : Tensor[∑E]
            Concatenated edge predictions.
        graphs    : List[Data]
            Target graphs in the same order (for slicing back).
        """
        outs: List[Tensor] = []
        graphs: List[Data] = []
        for seqs, tgts in loader:
            seqs = [[g.to(device) for g in s] for s in seqs]
            tgts = [g.to(device) for g in tgts]
            z_cat, *_ = self(seqs, tgts)
            outs.append(z_cat)
            graphs.extend(tgts)
        return (torch.cat(outs, dim=0) if outs else torch.tensor([], device=device)), graphs


# ───────────────────────────── 2) VA model ─────────────────────────────
class IOGNN_VA(nn.Module):
    """Node-level IO-GNN (predicts next-step Value Added VÂ_n)."""

    def __init__(self, nfeat: int, *args, **kwargs):
        """
        Backward-compatible + keyword-friendly constructor.

        Accepted call patterns:
          - New style (positional):  IOGNN_VA(nfeat, edge_feat_dim, cfg)
          - Old style (positional):  IOGNN_VA(nfeat, cfg)  # edge_feat_dim inferred from cfg.edge_feat_dim or None
          - Keyword style:           IOGNN_VA(nfeat, cfg=..., edge_feat_dim=...)

        Args
        ----
        nfeat : int
            Node feature dimension.
        edge_feat_dim : Optional[int]
            Edge feature dimension (used by the recurrent graph cell).
        cfg : object
            Experiment/config object with hyperparameters and flags.
        """
        super().__init__()

        # Unpack cfg / edge_feat_dim from args and kwargs (backward compatible).
        cfg = kwargs.get("cfg", None)
        edge_feat_dim_kw = kwargs.get("edge_feat_dim", None)
        if cfg is None:
            if len(args) == 2:
                edge_feat_dim, cfg = args
            elif len(args) == 1:
                (cfg,) = args
                edge_feat_dim = getattr(cfg, "edge_feat_dim", None)
            else:
                raise TypeError(
                    "IOGNN_VA.__init__ expected (nfeat, cfg) or (nfeat, edge_feat_dim, cfg) "
                    "or keyword form (nfeat, cfg=..., edge_feat_dim=...)"
                )
        else:
            edge_feat_dim = edge_feat_dim_kw if edge_feat_dim_kw is not None else getattr(cfg, "edge_feat_dim", None)

        if getattr(cfg, "use_edge_weight", True) and (edge_feat_dim is None):
            edge_feat_dim = 1

        self.use_bwd_weights = getattr(cfg, "use_bwd_weights", False)
        self.va_nonneg = getattr(cfg, "va_nonneg", False)

        # Backbone: LN → GELU
        self.pre = nn.Sequential(
            nn.Linear(nfeat, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.GELU(),
        )

        # Recurrent graph-temporal cell
        self.cell = GraphLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden,
            edge_dim=edge_feat_dim,
            alpha_init=cfg.alpha,
            att_hidden=cfg.att_hidden,
            compute_attention=getattr(cfg, "compute_attention", False),
            use_edge_weight=getattr(cfg, "use_edge_weight", True),
            use_bwd_weights=getattr(cfg, "use_bwd_weights", False),
            alpha_mode=getattr(cfg, "alpha_mode", "scalar"),
        )

        # Node decoder; optional non-negativity (Softplus) for VA
        out_act = nn.Softplus() if self.va_nonneg else None
        self.dec_node = mlp(
            d_in=cfg.hidden,
            d_hidden=cfg.hidden,
            depth=max(1, cfg.depth_edge - 1),
            dropout=cfg.dropout,
            out_dim=1,
            out_activation=out_act,
        )

    def forward(
        self,
        seq_batch: List[List[Data]],
        tgt_batch: List[Data],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        seq_batch : List[List[Data]]
            For each sample, a list of historical graphs.
        tgt_batch : List[Data]
            For each sample, the target graph to decode on.

        Returns
        -------
        va_concat : Tensor[∑N]
            Concatenated node predictions across the mini-batch.
        att_out / att_in : Tensor[∑E]
            Analysis-only attention from Ox gate (if enabled).
        """
        va_outs, att_o, att_i = [], [], []

        for seq, tgt in zip(seq_batch, tgt_batch):
            h = c = None
            for g in seq:
                ew_bwd = getattr(g, "edge_attr_bwd", None) if self.use_bwd_weights else None
                h, c = self.cell(self.pre(g.x), g.edge_index, g.edge_attr, h, c, ew_bwd=ew_bwd)

            va_outs.append(self.dec_node(h).squeeze(-1))  # [N]

            ao, ai = self.cell.Ox.compute_att(h, tgt.edge_index, tgt.edge_attr)
            att_o.append(ao); att_i.append(ai)

        return torch.cat(va_outs), torch.cat(att_o), torch.cat(att_i)

    def full_forward(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[Tensor, List[Data]]:
        """
        Convenience: run forward over ALL graphs in `loader` and concat predictions.

        Returns
        -------
        va_cat_all : Tensor[∑N]
            Concatenated node predictions.
        graphs     : List[Data]
            Target graphs in the same order.
        """
        outs: List[Tensor] = []
        graphs: List[Data] = []
        for seqs, tgts in loader:
            seqs = [[g.to(device) for g in s] for s in seqs]
            tgts = [g.to(device) for g in tgts]
            va_cat, *_ = self(seqs, tgts)
            outs.append(va_cat)
            graphs.extend(tgts)
        return (torch.cat(outs, dim=0) if outs else torch.tensor([], device=device)), graphs