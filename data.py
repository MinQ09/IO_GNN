from pathlib import Path
from typing import List, Tuple
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

NODE_COLS = ["Imports", "Exports", "Final_Demand"]

def load_node(p: Path, scale: float) -> Tuple[Tensor, Tensor, Tensor]:
    df = pd.read_csv(p, thousands=",")
    x   = torch.tensor(df[NODE_COLS].values, dtype=torch.float32) / scale
    va  = torch.tensor(df["Value_Added"].values, dtype=torch.float32) / scale
    tot = torch.tensor(df["Total"].values,       dtype=torch.float32) / scale
    return x, va, tot

def load_edges(p: Path):
    df = pd.read_csv(p, thousands=",")
    ei = torch.tensor(df.iloc[:, :2].values.T, dtype=torch.long)
    ew = torch.log1p(torch.tensor(df.iloc[:, 2].values, dtype=torch.float32))
    return ei, ew

def make_graph(x_csv: Path, e_csv: Path, scale_node: float) -> Data:
    x, va, tot = load_node(x_csv, scale_node)
    ei, ew     = load_edges(e_csv)
    return Data(x=x, edge_index=ei, edge_attr=ew, va=va, tot=tot)

class GraphWindowDataset(Dataset):
    def __init__(self, years: List[int], cfg):
        self.graphs_A = [make_graph(Path(cfg.data_dir)/f"X_{y}.csv",
                                    Path(cfg.data_dir)/f"Af_{y}.csv",
                                    cfg.scale_node) for y in years]
        self.graphs_Z = [make_graph(Path(cfg.data_dir)/f"X_{y}.csv",
                                    Path(cfg.data_dir)/f"Zf_{y}.csv",
                                    cfg.scale_node) for y in years]
        self.win = cfg.window
    def __len__(self): return len(self.graphs_A) - self.win
    def __getitem__(self, idx):
        seq = self.graphs_A[idx : idx + self.win]
        tgt = self.graphs_Z[idx + self.win]
        return seq, tgt

def collate_window(batch):
    seqs, tgts = zip(*batch)
    return list(seqs), list(tgts)