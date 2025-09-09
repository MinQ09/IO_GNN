# model.py  ───────────────────────────────────────────────────────────────
"""
Graph-temporal models for IO-GNN (Directional MPNN + GraphLSTMCell).

Key features
------------
- DirMPNN:
    * explicit directional message passing (src→dst)
    * source row-normalization (1 / out-degree) to stabilize hubs
    * edge multiplicative gating (scalar or learned FiLM-like for multi-dim edges)
- GraphLSTMCell:
    * LSTM-style recurrent cell whose gates are DirMPNNs
    * forget-gate bias = +1.0 (long-range stability)
    * residual connection on h with configurable scale
- Backward compatibility:
    * IOGNN_Z/VA __init__ supports (nfeat, cfg) and (nfeat, edge_feat_dim, cfg)
    * Provide alias: GCLSTMCell = GraphLSTMCell
- Flags:
    * compute_attention, use_edge_weight, use_bwd_weights
    * alpha_mode('scalar'|'channel'), use_row_norm, use_edge_mul, residual_scale
- Edge decoder features: [h_s, h_t, h_s⊙h_t, |h_s−h_t|]
- Optional non-negative VA head via cfg.va_nonneg (Softplus)
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, degree

EPS = 1e-12

# ───────────────────────── DirMPNN (Directional MPNN) ─────────────────────────
class DirMPNN(MessagePassing):
    """
    Explicit directional message passing:
      m_{i<-j} = phi_m([x_j, x_i, e_{j->i}])
      x'_i     = alpha * AGG_fwd(m_{i<-j}) + (1-alpha) * AGG_bwd(m_{i->j})

    Forward and backward paths are computed separately; mixing via alpha can be
    done by passing the backward output as `x_bwd_out` to forward().
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
        # NEW: accept but (currently) do not enforce special normalization
        use_row_norm: bool = True,
        use_edge_mul: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.fout = fout
        self.edge_dim = edge_dim
        self.expect_edge = edge_dim is not None
        self.compute_attention = compute_attention
        self.alpha_mode = alpha_mode

        # Keep flags for future extensions; safe no-op for now
        self.use_row_norm = use_row_norm
        self.use_edge_mul = use_edge_mul

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
                # Only multiply by scalar edge weights if present and expected.
                if self.expect_edge and (edge_attr is not None) and (edge_attr.dim() == 1) and self.use_edge_mul:
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
            z = torch.cat([x_j, x_i], dim=-1)
        else:
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
            if self.expect_edge and (edge_attr is not None) and (edge_attr.dim() == 1) and self.use_edge_mul:
                e = e * edge_attr
            att_out = softmax(e, index=s, num_nodes=x.size(0)).detach()
            att_in  = softmax(e, index=t, num_nodes=x.size(0)).detach()
        return att_out, att_in


# ─────────────────────────── GraphLSTMCell (DirMPNN-based) ───────────────────────────
class GraphLSTMCell(nn.Module):
    """
    LSTM-style recurrent cell whose gates are directional GNN layers (DirMPNN).

    - For compatibility with call sites:
        forward(x, ei, ew, h=None, c=None, ew_bwd=None)
    - Forget gate bias = +1.0
    - Residual on h with controllable scale
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
        residual_scale: float = 0.1,
        # NEW: accept row-norm / edge-mul flags (passed down to DirMPNN)
        use_row_norm: bool = True,
        use_edge_mul: bool = True,
    ):
        super().__init__()
        self.hid = hid
        self.use_edge_weight = use_edge_weight
        self.use_bwd_weights = use_bwd_weights
        self.residual_scale = residual_scale
        self.use_row_norm = use_row_norm
        self.use_edge_mul = use_edge_mul

        def conv(fi: int, fo: int) -> "DirMPNN":
            return DirMPNN(
                fin=fi,
                fout=fo,
                edge_dim=edge_dim if self.use_edge_weight else None,
                att_hidden=att_hidden,
                alpha_init=alpha_init,
                learn_alpha=True,
                alpha_mode=alpha_mode,
                compute_attention=compute_attention,
                use_row_norm=self.use_row_norm,
                use_edge_mul=self.use_edge_mul,
            )

        # Gate-wise directional convs
        self.Fx, self.Fh = conv(fin, hid), conv(hid, hid)
        self.Ix, self.Ih = conv(fin, hid), conv(hid, hid)
        self.Ox, self.Oh = conv(fin, hid), conv(hid, hid)
        self.Gx, self.Gh = conv(fin, hid), conv(hid, hid)

        # LayerNorms per gate and cell state
        self.ln_f = nn.LayerNorm(hid)
        self.ln_i = nn.LayerNorm(hid)
        self.ln_o = nn.LayerNorm(hid)
        self.ln_g = nn.LayerNorm(hid)
        self.ln_c = nn.LayerNorm(hid)

        # Residual projection on h (init ~ identity if square)
        self.hidden_proj = nn.Linear(hid, hid)
        with torch.no_grad():
            if self.hidden_proj.weight.shape[0] == self.hidden_proj.weight.shape[1]:
                nn.init.eye_(self.hidden_proj.weight)
            else:
                nn.init.kaiming_uniform_(self.hidden_proj.weight, a=math.sqrt(5))
            nn.init.constant_(self.hidden_proj.bias, 0.0)

        # Gate biases (forget gate bias = +1)
        self.b_f = nn.Parameter(torch.ones(hid))
        self.b_i = nn.Parameter(torch.zeros(hid))
        self.b_o = nn.Parameter(torch.zeros(hid))
        self.b_g = nn.Parameter(torch.zeros(hid))

    def _dir_pass(
        self,
        conv_x: "DirMPNN", conv_h: "DirMPNN",
        x: Tensor, h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        edge_attr_bwd: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # Forward path (src->dst)
        fx_fwd = conv_x(x, edge_index, edge_attr=edge_attr)
        fh_fwd = conv_h(h, edge_index, edge_attr=edge_attr)
        # Backward path (dst->src)
        bwd_index = edge_index.flip(0)
        ea_bwd = edge_attr_bwd if self.use_bwd_weights else None
        fx_bwd = conv_x(x, bwd_index, edge_attr=ea_bwd)
        fh_bwd = conv_h(h, bwd_index, edge_attr=ea_bwd)
        # Mix forward/backward via alpha inside conv
        fx = conv_x(x, edge_index, edge_attr=edge_attr, x_bwd_out=fx_bwd)
        fh = conv_h(h, edge_index, edge_attr=edge_attr, x_bwd_out=fh_bwd)
        return fx, fh

    def forward(
        self,
        x: Tensor,                  # [N, F_in]
        ei: Tensor,                 # [2, E]
        ew: Optional[Tensor],       # [E] or [E, D] (or None)
        h: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        ew_bwd: Optional[Tensor] = None,  # backward-edge features (optional)
    ) -> Tuple[Tensor, Tensor]:
        n = x.size(0)
        if h is None or c is None:
            h = x.new_zeros(n, self.hid)
            c = x.new_zeros(n, self.hid)

        # Directional gates
        xf, hf = self._dir_pass(self.Fx, self.Fh, x, h, ei, ew, ew_bwd)
        xi, hi = self._dir_pass(self.Ix, self.Ih, x, h, ei, ew, ew_bwd)
        xo, ho = self._dir_pass(self.Ox, self.Oh, x, h, ei, ew, ew_bwd)
        xg, hg = self._dir_pass(self.Gx, self.Gh, x, h, ei, ew, ew_bwd)

        # LSTM updates
        f = torch.sigmoid(self.ln_f(xf + hf) + self.b_f)   # forget (+1 bias)
        i = torch.sigmoid(self.ln_i(xi + hi) + self.b_i)
        o = torch.sigmoid(self.ln_o(xo + ho) + self.b_o)
        g = torch.tanh(   self.ln_g(xg + hg) + self.b_g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(self.ln_c(c_new)) + self.residual_scale * self.hidden_proj(h)
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
        """
        super().__init__()

        # Unpack cfg / edge_feat_dim
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
            edge_feat_dim = 1  # default scalar edge weights

        self.use_bwd_weights = getattr(cfg, "use_bwd_weights", False)

        # Backbone: LN → GELU
        self.pre = nn.Sequential(
            nn.Linear(nfeat, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.GELU(),
        )

        # Recurrent cell
        self.cell = GraphLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden,
            edge_dim=edge_feat_dim,
            alpha_init=cfg.alpha,
            att_hidden=cfg.att_hidden,
            compute_attention=getattr(cfg, "compute_attention", False),
            use_edge_weight=getattr(cfg, "use_edge_weight", True),
            use_bwd_weights=getattr(cfg, "use_bwd_weights", False),
            alpha_mode=getattr(cfg, "alpha_mode", "scalar"),
            use_row_norm=getattr(cfg, "use_row_norm", True),
            use_edge_mul=getattr(cfg, "use_edge_mul", True),
            residual_scale=getattr(cfg, "residual_scale", 1.0),
        )

        # Edge decoder on pairwise node states: [h_s, h_t, h_s ⊙ h_t, |h_s − h_t|]
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
        """
        super().__init__()

        # Unpack cfg / edge_feat_dim
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

        # Recurrent cell
        self.cell = GraphLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden,
            edge_dim=edge_feat_dim,
            alpha_init=cfg.alpha,
            att_hidden=cfg.att_hidden,
            compute_attention=getattr(cfg, "compute_attention", False),
            use_edge_weight=getattr(cfg, "use_edge_weight", True),
            use_bwd_weights=getattr(cfg, "use_bwd_weights", False),
            alpha_mode=getattr(cfg, "alpha_mode", "scalar"),
            use_row_norm=getattr(cfg, "use_row_norm", True),
            use_edge_mul=getattr(cfg, "use_edge_mul", True),
            residual_scale=getattr(cfg, "residual_scale", 1.0),
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
        outs: List[Tensor] = []
        graphs: List[Data] = []
        for seqs, tgts in loader:
            seqs = [[g.to(device) for g in s] for s in seqs]
            tgts = [g.to(device) for g in tgts]
            va_cat, *_ = self(seqs, tgts)
            outs.append(va_cat)
            graphs.extend(tgts)
        return (torch.cat(outs, dim=0) if outs else torch.tensor([], device=device)), graphs