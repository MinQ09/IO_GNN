# model.py  ───────────────────────────────────────────────────────────────
"""
Graph-temporal models for IO-GNN (Directional MPNN + GraphLSTMCell).

What’s improved (drop-in compatible)
------------------------------------
- DirMPNN:
  * (NEW) warmup-safe edge gating: can disable edge_mul early (cfg.edge_mul_warmup)
  * (SAFE) edge feature auto-shape: works even if edge_attr is None or wrong dim
  * (SAFE) attention callable even if att_mlp is None
- GraphLSTMCell:
  * (NEW) learnable forward/backward mixing per gate (α = σ(β))  → stability↑/expressivity↑
  * (NEW) optional 2-hop reinforcement per gate (cfg.two_hop, cfg.hop_residual)
  * (KEEP) forget-bias + residual on h
- Decoders:
  * Z: [h_s, h_t, h_s⊙h_t, |h_s−h_t|]
  * VA: optional Softplus via cfg.va_nonneg
- Utils:
  * get_config_summary() for quick logging
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


# ───────────────────────── DirMPNN (Attention/Row-norm) ─────────────────────────
class DirMPNN(MessagePassing):
    """
    Directional MPNN with usable attention (or row-norm fallback).

    If compute_attention=True:
        α_{ij} = softmax_j( a([x_j, x_i, e_ji]) / T ) over src node
        message = φ_m([x_j, x_i, e_ji]) * α_{ij} * (optional scalar edge gate)
    else:
        message = φ_m([...]) * (1 / outdeg(src)) * (optional scalar edge gate)
    """

    def __init__(
        self,
        *,
        fin: int,
        fout: int,
        edge_dim: Optional[int] = None,
        att_hidden: int = 128,
        compute_attention: bool = True,
        use_row_norm: bool = True,
        use_edge_mul: bool = True,
        att_dropout: float = 0.0,
        att_temperature: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.fin = fin
        self.fout = fout
        self.edge_dim = edge_dim
        self.compute_attention = compute_attention
        self.use_row_norm = use_row_norm
        self.use_edge_mul = use_edge_mul
        self.eps = eps
        self.att_temperature = max(1e-3, float(att_temperature))
        self.att_drop = nn.Dropout(att_dropout) if att_dropout > 0 else nn.Identity()

        in_dim = fin + fin + (edge_dim or 0)
        hidden_m = max(64, min(4 * fout, 512))
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_m),
            nn.GELU(),
            nn.Linear(hidden_m, fout),
        )

        if self.compute_attention:
            hidden_a = max(32, min(att_hidden, 256))
            self.att_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_a),
                nn.GELU(),
                nn.Linear(hidden_a, 1),
            )
        else:
            self.att_mlp = None

        self.out_proj = nn.Identity()

        # For analysis/logging
        self.last_alpha: Optional[Tensor] = None
        self._edge_src: Optional[Tensor] = None
        self._num_nodes: Optional[int] = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ---- NEW: safe edge feature normalizer ---------------------------------
    def _edge_feat_safe(self, edge_attr: Optional[Tensor], E: int, like: Tensor) -> Optional[Tensor]:
        if self.edge_dim is None:
            return None
        if edge_attr is None:
            return torch.zeros(E, self.edge_dim, device=like.device, dtype=like.dtype)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        if edge_attr.dim() == 2:
            d = edge_attr.size(1)
            if d == self.edge_dim:
                return edge_attr
            if d < self.edge_dim:
                pad = torch.zeros(edge_attr.size(0), self.edge_dim - d,
                                  device=edge_attr.device, dtype=edge_attr.dtype)
                return torch.cat([edge_attr, pad], dim=1)
            return edge_attr[:, :self.edge_dim]
        return torch.zeros(E, self.edge_dim, device=like.device, dtype=like.dtype)

    # ---- (2) in_features 실시간 보정(부족하면 0패딩, 넘치면 트렁크)
    @staticmethod
    def _auto_match_in_features(h: Tensor, expected_in: int) -> Tensor:
        cur = h.size(1)
        if cur == expected_in:
            return h
        if cur < expected_in:
            pad = h.new_zeros(h.size(0), expected_in - cur)
            return torch.cat([h, pad], dim=1)
        # cur > expected_in
        return h[:, :expected_in]

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, *_):
        row, _ = edge_index
        self._edge_src = row
        self._num_nodes = int(x.size(0))

        if self.compute_attention:
            E = edge_index.size(1)
            e = self._edge_feat_safe(edge_attr, E, like=x)
            xj, xi = x[edge_index[0]], x[edge_index[1]]
            parts = [xj, xi]
            if e is not None:
                parts.append(e)
            att_in = torch.cat(parts, dim=-1)
            # <<< 여기가 핵심: att_mlp in_features 에 맞추어 보정 >>>
            if self.att_mlp is not None:
                expected = self.att_mlp[0].in_features
                att_in = self._auto_match_in_features(att_in, expected)
                logits = self.att_mlp(att_in).squeeze(-1) / self.att_temperature
            else:
                logits = x.new_zeros(E)
            alpha = softmax(logits, row, num_nodes=self._num_nodes)
            alpha = self.att_drop(alpha)
        else:
            alpha = None

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, alpha=alpha)
        out = self.out_proj(out)
        self.last_alpha = alpha
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Optional[Tensor], alpha: Optional[Tensor]) -> Tensor:
        E = x_j.size(0)
        e = self._edge_feat_safe(edge_attr, E, like=x_j)
        parts = [x_j, x_i]
        if e is not None:
            parts.append(e)
        h = torch.cat(parts, dim=-1)
        # <<< 여기가 핵심: msg_mlp in_features 에 맞추어 보정 >>>
        expected = self.msg_mlp[0].in_features
        h = self._auto_match_in_features(h, expected)

        msg = self.msg_mlp(h)

        # 스칼라 게이트는 원본 edge_attr가 1D일 때만
        if self.use_edge_mul and edge_attr is not None and edge_attr.dim() == 1:
            msg = msg * edge_attr.view(-1, 1)

        if alpha is not None:
            msg = msg * alpha.view(-1, 1)
        elif self.use_row_norm and (self._edge_src is not None) and (self._num_nodes is not None):
            src = self._edge_src
            outdeg = degree(src, num_nodes=self._num_nodes, dtype=msg.dtype).clamp_min(1.0)
            msg = msg * (1.0 / outdeg[src]).view(-1, 1)
        return msg

    def compute_att_analysis(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        row, _ = edge_index
        with torch.no_grad():
            if self.att_mlp is None:
                logits = x.new_zeros(edge_index.size(1))
            else:
                E = edge_index.size(1)
                e = self._edge_feat_safe(edge_attr, E, like=x)
                xj, xi = x[edge_index[0]], x[edge_index[1]]
                parts = [xj, xi]
                if e is not None:
                    parts.append(e)
                att_in = torch.cat(parts, dim=-1)
                expected = self.att_mlp[0].in_features
                att_in = self._auto_match_in_features(att_in, expected)
                logits = self.att_mlp(att_in).squeeze(-1)
            alpha = softmax(logits, row, num_nodes=x.size(0))
        return alpha

# ─────────────────────────── GraphLSTMCell (DirMPNN-based) ───────────────────────────
class GraphLSTMCell(nn.Module):
    """
    LSTM-style recurrent cell whose gates are directional GNN layers (DirMPNN).

    forward(x, ei, ew, h=None, c=None, ew_bwd=None) -> (h_new, c_new)
    """

    def __init__(
        self,
        fin: int,
        hid: int,
        edge_dim: Optional[int],
        att_hidden: int = 128,
        compute_attention: bool = True,
        use_edge_weight: bool = True,
        use_bwd_weights: bool = False,
        residual_scale: float = 0.1,
        use_row_norm: bool = True,
        use_edge_mul: bool = True,
        # NEW: stability/expressivity
        learn_mix: bool = True,
        mix_init: float = 0.0,          # α=σ(β), β init
        two_hop: bool = False,          # 2-hop reinforcement per gate
        hop_residual: float = 0.2,      # residual from 1st hop into 2nd hop output
        edge_mul_warmup: int = 0,       # disable edge_mul for first N steps (set via module attr)
    ):
        super().__init__()
        self.hid = hid
        self.use_edge_weight = use_edge_weight
        self.use_bwd_weights = use_bwd_weights
        self.residual_scale = residual_scale
        self.use_row_norm = use_row_norm
        self.use_edge_mul = use_edge_mul
        self.learn_mix = learn_mix
        self.two_hop = two_hop
        self.hop_residual = hop_residual
        self.edge_mul_warmup = edge_mul_warmup  # public attr; trainer can update step counter
        self._global_step = 0

        def conv(fi: int, fo: int) -> "DirMPNN":
            return DirMPNN(
                fin=fi, fout=fo,
                edge_dim=edge_dim if self.use_edge_weight else None,
                att_hidden=att_hidden,
                compute_attention=compute_attention,
                use_row_norm=self.use_row_norm,
                use_edge_mul=self.use_edge_mul,
            )

        # Gate-wise directional convs
        self.Fx, self.Fh = conv(fin, hid), conv(hid, hid)
        self.Ix, self.Ih = conv(fin, hid), conv(hid, hid)
        self.Ox, self.Oh = conv(fin, hid), conv(hid, hid)
        self.Gx, self.Gh = conv(fin, hid), conv(hid, hid)

        # Learnable forward/backward mixing per gate (β → α=σ(β))
        beta_init = torch.tensor(mix_init, dtype=torch.float32)
        def p(): return nn.Parameter(beta_init.clone()) if self.learn_mix else beta_init.clone().detach()
        self.beta_fx, self.beta_fh = p(), p()
        self.beta_ix, self.beta_ih = p(), p()
        self.beta_ox, self.beta_oh = p(), p()
        self.beta_gx, self.beta_gh = p(), p()

        # LayerNorms per gate and cell state
        self.ln_f = nn.LayerNorm(hid)
        self.ln_i = nn.LayerNorm(hid)
        self.ln_o = nn.LayerNorm(hid)
        self.ln_g = nn.LayerNorm(hid)
        self.ln_c = nn.LayerNorm(hid)

        # Residual projection on h (near identity)
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

    @staticmethod
    def _mix(fwd: Tensor, bwd: Tensor, beta: Tensor) -> Tensor:
        alpha = torch.sigmoid(beta).clamp(0.01, 0.99)
        return alpha * fwd + (1.0 - alpha) * bwd

    def _maybe_two_hop(self, conv: "DirMPNN", x: Tensor, ei: Tensor, ea: Optional[Tensor],
                       bwd_ei: Tensor, ea_bwd: Optional[Tensor]) -> Tensor:
        """Optional 2-hop reinforcement with small residual from first hop."""
        if not self.two_hop:
            return conv(x, ei, edge_attr=ea)
        h1_fwd = conv(x, ei, edge_attr=ea)
        h1_bwd = conv(x, bwd_ei, edge_attr=ea_bwd)
        h1 = 0.5 * (h1_fwd + h1_bwd)
        # second pass
        h2_fwd = conv(h1, ei, edge_attr=ea)
        h2_bwd = conv(h1, bwd_ei, edge_attr=ea_bwd)
        h2 = 0.5 * (h2_fwd + h2_bwd) + self.hop_residual * h1
        return h2

    def step(self):  # call this from trainer each batch if you want warmup tracking
        self._global_step += 1

    def _dir_pass(
        self,
        conv_x: "DirMPNN", conv_h: "DirMPNN",
        x: Tensor, h: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        edge_attr_bwd: Optional[Tensor],
        beta_x: Tensor, beta_h: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Apply temporary edge_mul disable during warmup if set
        if self.edge_mul_warmup > 0 and self._global_step < self.edge_mul_warmup:
            old = (conv_x.use_edge_mul, conv_h.use_edge_mul)
            conv_x.use_edge_mul = False
            conv_h.use_edge_mul = False
        else:
            old = None

        bwd_index = edge_index.flip(0)
        ea_bwd = edge_attr_bwd if self.use_bwd_weights else None

        # x-path
        x_fwd = self._maybe_two_hop(conv_x, x, edge_index, edge_attr, bwd_index, ea_bwd)
        x_bwd = self._maybe_two_hop(conv_x, x, bwd_index, ea_bwd, edge_index, edge_attr)
        fx = self._mix(x_fwd, x_bwd, beta_x) if self.learn_mix else 0.5 * (x_fwd + x_bwd)

        # h-path
        h_fwd = self._maybe_two_hop(conv_h, h, edge_index, edge_attr, bwd_index, ea_bwd)
        h_bwd = self._maybe_two_hop(conv_h, h, bwd_index, ea_bwd, edge_index, edge_attr)
        fh = self._mix(h_fwd, h_bwd, beta_h) if self.learn_mix else 0.5 * (h_fwd + h_bwd)

        # restore edge_mul flags if altered
        if old is not None:
            conv_x.use_edge_mul, conv_h.use_edge_mul = old
        return fx, fh

    def forward(
        self,
        x: Tensor,                  # [N, F_in]
        ei: Tensor,                 # [2, E]
        ew: Optional[Tensor],       # [E] or [E, D] (or None)
        h: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        ew_bwd: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        n = x.size(0)
        if h is None or c is None:
            h = x.new_zeros(n, self.hid)
            c = x.new_zeros(n, self.hid)

        # Directional gates (with learnable mixing and optional 2-hop)
        xf, hf = self._dir_pass(self.Fx, self.Fh, x, h, ei, ew, ew_bwd, self.beta_fx, self.beta_fh)
        xi, hi = self._dir_pass(self.Ix, self.Ih, x, h, ei, ew, ew_bwd, self.beta_ix, self.beta_ih)
        xo, ho = self._dir_pass(self.Ox, self.Oh, x, h, ei, ew, ew_bwd, self.beta_ox, self.beta_oh)
        xg, hg = self._dir_pass(self.Gx, self.Gh, x, h, ei, ew, ew_bwd, self.beta_gx, self.beta_gh)

        # LSTM updates
        f = torch.sigmoid(self.ln_f(xf + hf) + self.b_f)   # forget (+1 bias)
        i = torch.sigmoid(self.ln_i(xi + hi) + self.b_i)
        o = torch.sigmoid(self.ln_o(xo + ho) + self.b_o)
        g = torch.tanh(   self.ln_g(xg + hg) + self.b_g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(self.ln_c(c_new)) + self.residual_scale * self.hidden_proj(h)

        # step counter for warmup scheduling
        self._global_step += 1
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
        super().__init__()
        cfg = kwargs.get("cfg", None)
        edge_feat_dim_kw = kwargs.get("edge_feat_dim", None)
        if cfg is None:
            if len(args) == 2:
                edge_feat_dim, cfg = args
            elif len(args) == 1:
                (cfg,) = args
                edge_feat_dim = getattr(cfg, "edge_feat_dim", None)
            else:
                raise TypeError("IOGNN_Z expected (nfeat, cfg) or (nfeat, edge_feat_dim, cfg)")
        else:
            edge_feat_dim = edge_feat_dim_kw if edge_feat_dim_kw is not None else getattr(cfg, "edge_feat_dim", None)
        if getattr(cfg, "use_edge_weight", True) and (edge_feat_dim is None):
            edge_feat_dim = 1

        self.use_bwd_weights = getattr(cfg, "use_bwd_weights", False)

        self.pre = nn.Sequential(
            nn.Linear(nfeat, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.GELU(),
        )

        self.cell = GraphLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden,
            edge_dim=edge_feat_dim,
            att_hidden=cfg.att_hidden,
            compute_attention=getattr(cfg, "compute_attention", True),
            use_edge_weight=getattr(cfg, "use_edge_weight", True),
            use_bwd_weights=getattr(cfg, "use_bwd_weights", False),
            residual_scale=getattr(cfg, "residual_scale", 1.0),
            use_row_norm=getattr(cfg, "use_row_norm", True),
            use_edge_mul=getattr(cfg, "use_edge_mul", True),
            learn_mix=getattr(cfg, "learn_mix", True),
            mix_init=getattr(cfg, "mix_init", 0.0),
            two_hop=getattr(cfg, "two_hop", False),
            hop_residual=getattr(cfg, "hop_residual", 0.2),
            edge_mul_warmup=int(getattr(cfg, "edge_mul_warmup", 0)),
        )

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
            h = c = None
            for g in seq:
                ew_bwd = getattr(g, "edge_attr_bwd", None) if self.use_bwd_weights else None
                h, c = self.cell(self.pre(g.x), g.edge_index, g.edge_attr, h, c, ew_bwd=ew_bwd)

            # analysis attention on target graph using one gate (Ox)
            ao = self.cell.Ox.compute_att_analysis(h, tgt.edge_index, tgt.edge_attr)
            ai = self.cell.Ox.compute_att_analysis(h, tgt.edge_index.flip(0), tgt.edge_attr)  # in-att (dst-based)
            att_o.append(ao); att_i.append(ai)

            s, t = tgt.edge_index
            hs, ht = h[s], h[t]
            pair_feat = torch.cat([hs, ht, hs * ht, torch.abs(hs - ht)], dim=1)
            z_outs.append(self.dec_edge(pair_feat).squeeze(-1))  # [E]

        return torch.cat(z_outs), torch.cat(att_o), torch.cat(att_i)

    # Convenience for full-forward PINN
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

    def get_config_summary(self) -> str:
        cm = self.cell
        return (f"[IOGNN_Z] hid={cm.hid} two_hop={cm.two_hop} learn_mix={cm.learn_mix} "
                f"residual_scale={cm.residual_scale} use_edge_mul={cm.use_edge_mul} "
                f"use_row_norm={cm.use_row_norm} use_bwd_weights={self.use_bwd_weights}")


# ───────────────────────────── 2) VA model ─────────────────────────────
class IOGNN_VA(nn.Module):
    """Node-level IO-GNN (predicts next-step Value Added VÂ_n)."""

    def __init__(self, nfeat: int, *args, **kwargs):
        super().__init__()
        cfg = kwargs.get("cfg", None)
        edge_feat_dim_kw = kwargs.get("edge_feat_dim", None)
        if cfg is None:
            if len(args) == 2:
                edge_feat_dim, cfg = args
            elif len(args) == 1:
                (cfg,) = args
                edge_feat_dim = getattr(cfg, "edge_feat_dim", None)
            else:
                raise TypeError("IOGNN_VA expected (nfeat, cfg) or (nfeat, edge_feat_dim, cfg)")
        else:
            edge_feat_dim = edge_feat_dim_kw if edge_feat_dim_kw is not None else getattr(cfg, "edge_feat_dim", None)
        if getattr(cfg, "use_edge_weight", True) and (edge_feat_dim is None):
            edge_feat_dim = 1

        self.use_bwd_weights = getattr(cfg, "use_bwd_weights", False)
        self.va_nonneg = getattr(cfg, "va_nonneg", False)

        self.pre = nn.Sequential(
            nn.Linear(nfeat, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.GELU(),
        )

        self.cell = GraphLSTMCell(
            fin=cfg.hidden, hid=cfg.hidden,
            edge_dim=edge_feat_dim,
            att_hidden=cfg.att_hidden,
            compute_attention=getattr(cfg, "compute_attention", True),
            use_edge_weight=getattr(cfg, "use_edge_weight", True),
            use_bwd_weights=getattr(cfg, "use_bwd_weights", False),
            residual_scale=getattr(cfg, "residual_scale", 1.0),
            use_row_norm=getattr(cfg, "use_row_norm", True),
            use_edge_mul=getattr(cfg, "use_edge_mul", True),
            learn_mix=getattr(cfg, "learn_mix", True),
            mix_init=getattr(cfg, "mix_init", 0.0),
            two_hop=getattr(cfg, "two_hop", False),
            hop_residual=getattr(cfg, "hop_residual", 0.2),
            edge_mul_warmup=int(getattr(cfg, "edge_mul_warmup", 0)),
        )

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

            ao = self.cell.Ox.compute_att_analysis(h, tgt.edge_index, tgt.edge_attr)
            ai = self.cell.Ox.compute_att_analysis(h, tgt.edge_index.flip(0), tgt.edge_attr)
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

    def get_config_summary(self) -> str:
        cm = self.cell
        return (f"[IOGNN_VA] hid={cm.hid} two_hop={cm.two_hop} learn_mix={cm.learn_mix} "
                f"residual_scale={cm.residual_scale} use_edge_mul={cm.use_edge_mul} "
                f"use_row_norm={cm.use_row_norm} use_bwd_weights={self.use_bwd_weights}")