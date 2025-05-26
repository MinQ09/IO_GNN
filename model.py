import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data

class ChebDirConv(nn.Module):
    def __init__(self, fin: int, fout: int, k: int, a: float):
        super().__init__()
        self.oc = ChebConv(fin, fout, k)
        self.ic = ChebConv(fin, fout, k)
        self.a  = a
    def forward(self, x, ei, ew=None):
        return self.a * self.oc(x, ei, ew) + (1 - self.a) * self.ic(x, ei.flip(0), ew)

class GCLSTMCell(nn.Module):
    def __init__(self, fin: int, hid: int, k: int, a: float):
        super().__init__()
        self.hid = hid
        def c(fi, fo): return ChebDirConv(fi, fo, k, a)
        self.Fx, self.Fh = c(fin, hid), c(hid, hid)
        self.Ix, self.Ih = c(fin, hid), c(hid, hid)
        self.Ox, self.Oh = c(fin, hid), c(hid, hid)
        self.Gx, self.Gh = c(fin, hid), c(hid, hid)
    def forward(self, x, ei, ew, h=None, c=None):
        n = x.size(0)
        if h is None:
            h = x.new_zeros(n, self.hid)
            c = x.new_zeros(n, self.hid)
        f = torch.sigmoid(self.Fx(x, ei, ew) + self.Fh(h, ei, ew))
        i = torch.sigmoid(self.Ix(x, ei, ew) + self.Ih(h, ei, ew))
        o = torch.sigmoid(self.Ox(x, ei, ew) + self.Oh(h, ei, ew))
        g = torch.tanh   (self.Gx(x, ei, ew) + self.Gh(h, ei, ew))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

def mlp(d: int, h: int, depth: int, dp: float):
    layers = []
    for i in range(depth):
        layers += [nn.Linear(d if i == 0 else h, h), nn.LayerNorm(h), nn.GELU()]
        if dp: layers.append(nn.Dropout(dp))
    layers.append(nn.Linear(h, 1))
    return nn.Sequential(*layers)

class IOGNN(nn.Module):
    def __init__(self, nfeat: int, cfg):
        super().__init__()
        self.pre   = nn.Sequential(nn.Linear(nfeat, cfg.hidden), nn.GELU())
        self.cell  = GCLSTMCell(cfg.hidden, cfg.hidden, cfg.k, 0.5)
        self.dec_e = mlp(cfg.hidden*2, cfg.hidden, 2, cfg.dropout)
        self.dec_n = mlp(cfg.hidden   , cfg.hidden, 1, cfg.dropout)
    def forward(self, seq_batch: List[List[Data]], tgt_batch: List[Data]) -> Tuple[Tensor, Tensor]:
        z_out, x_out = [], []
        for seq, tgt in zip(seq_batch, tgt_batch):
            h = c = None
            for g in seq:
                h, c = self.cell(self.pre(g.x), g.edge_index, g.edge_attr, h, c)
            s, t  = tgt.edge_index
            z_out.append(self.dec_e(torch.cat([h[s], h[t]], 1)).squeeze(-1))
            x_out.append(self.dec_n(h).squeeze(-1))
        return torch.cat(z_out), torch.cat(x_out)