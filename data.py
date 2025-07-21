# data.py ───────────────────────────────────────────────────────────────
"""
Data I/O utilities
------------------

* Convert .csv files to PyG ``Data`` objects
* Provide ``GraphWindowDataset`` for fixed-window sequences
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

# ────────────────────────── constants ──────────────────────────
NODE_COLS: Sequence[str] = ("Imports", "Exports", "Final_Demand")

# ────────────────────────── loader functions ──────────────────
def load_node(csv_path: Path, scale: float = 1e6) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Read node-level features and scale them.

    Parameters
    ----------
    csv_path : Path
        CSV expected to contain columns
        [Imports, Exports, Final_Demand, Value_Added, Total].
    scale : float, default=1e6
        Divide all numeric values by this factor.

    Returns
    -------
    x   : (N, 3) tensor
        Imports / Exports / Final Demand
    va  : (N,) tensor
        Value Added
    tot : (N,) tensor
        Total Output (used as row/col target in PINN loss)
    """
    try:
        df = pd.read_csv(csv_path, thousands=",")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[load_node] File not found: {csv_path}") from e

    x   = torch.tensor(df[list(NODE_COLS)].values, dtype=torch.float32) / scale
    va  = torch.tensor(df["Value_Added"].values,  dtype=torch.float32) / scale
    tot = torch.tensor(df["Total"].values,        dtype=torch.float32) / scale
    return x, va, tot


def load_edges(
    csv_path: Path,
    value_col: str | int = 2,
    *,
    ensure_symmetric: bool = False,
    default_rev: float = 0.0,
    log_scale: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Read edge list CSV and return edge_index / edge_weight tensors.

    Parameters
    ----------
    csv_path : Path
        Edge list CSV. The first two columns must be (source, target)
        indices; the third column is the edge value.
    value_col : str | int, default=2
        Column containing edge values. If int, interpreted as position;
        if str, interpreted as column name.
    ensure_symmetric : bool, default=False
        If True, add a reverse edge (j,i) with *default_rev* value
        when (i,j) exists but (j,i) does not.
    default_rev : float, default=0.0
        Value assigned to automatically added reverse edges.
    log_scale : bool, default=True
        If True, apply ``log1p`` to edge weights.

    Returns
    -------
    edge_index : (2, E) long tensor
    edge_weight: (E,)   float tensor
    """
    try:
        df = pd.read_csv(csv_path, thousands=",")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[load_edges] File not found: {csv_path}") from e

    if ensure_symmetric:
        # Create reversed edge list and merge to ensure bidirectionality
        rev = df.rename(columns={df.columns[0]: "target", df.columns[1]: "source"})[df.columns]
        merged = rev.merge(df, on=list(df.columns[:2]), how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"].iloc[:, :3].copy()
        missing.iloc[:, 2] = default_rev
        df = pd.concat([df, missing], ignore_index=True)

    edge_idx = torch.tensor(df.iloc[:, :2].values.T, dtype=torch.long)
    vals = df.iloc[:, value_col] if isinstance(value_col, int) else df[value_col]
    edge_wt = torch.tensor(vals.values, dtype=torch.float32)
    if log_scale:
        edge_wt = torch.log1p(edge_wt)

    return edge_idx, edge_wt


def make_graph(x_csv: Path, e_csv: Path, scale_node: float = 1e6) -> Data:
    """
    Convert two CSV files (nodes, edges) to ``torch_geometric.data.Data``.
    """
    x, va, tot = load_node(x_csv, scale_node)
    edge_idx, edge_wt = load_edges(e_csv)
    return Data(x=x, edge_index=edge_idx, edge_attr=edge_wt, va=va, tot=tot)


# ─────────────────────── Dataset class ───────────────────────
class GraphWindowDataset(Dataset):
    """
    Return (W-step history, next-step target) tuples.

    * seq  : List[Data]  length == window size
    * tgt  : Data        graph for the next year
    """

    def __init__(self, years: List[int], cfg) -> None:
        base = Path(cfg.data_dir)
        self.graphs_A = [
            make_graph(base / f"X_{y}.csv", base / f"Af_{y}.csv", cfg.scale_node)
            for y in years
        ]
        self.graphs_Z = [
            make_graph(base / f"X_{y}.csv", base / f"Zf_{y}.csv", cfg.scale_node)
            for y in years
        ]
        self.window = cfg.window

    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.graphs_A) - self.window

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        seq : List[Data]
            History window (length = self.window)
        tgt : Data
            Z graph of the year immediately after the window
        """
        return (
            self.graphs_A[idx : idx + self.window],
            self.graphs_Z[idx + self.window],
        )


# ─────────────────────── collate function ───────────────────────
def collate_window(batch):
    """
    Custom collate_fn for DataLoader:
    splits a list of (seq, tgt) into separate seqs and tgts lists.
    """
    seqs, tgts = zip(*batch)
    return list(seqs), list(tgts)