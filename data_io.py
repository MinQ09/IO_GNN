# data_io.py  ──────────────────────────────────────────────────────────────
"""
Data I/O utilities (StandardScaler version, zero-NaN, identity-Af support).

• CSV → PyG `Data` objects with optional standardisation
• `GraphWindowDataset`  – fixed-window dataloader for temporal GNNs
• All scalers stored in a *simple* dict: {"node", "edge_A", "edge_Z"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
import pickle
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from constants import EPS  # global constant

# ──────────────── constants ────────────────
NODE_COLS: Sequence[str] = ("Import", "Export", "Final_Demand")

# ──────────────── helpers ────────────────
def _safe_read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, thousands=",").replace(["–", "-", ""], np.nan)
    return df.fillna(0.0)


def _protect_zero_variance(scaler: StandardScaler) -> None:
    """Set scale_=1 where variance is zero to avoid +/-inf later."""
    scaler.scale_[scaler.scale_ == 0.0] = 1.0


def _identity_scaler() -> StandardScaler:
    """A do-nothing scaler (mean=0, scale=1)."""
    s = StandardScaler()
    s.fit(np.array([[0.0], [1.0]]))
    _protect_zero_variance(s)
    return s


# ──────────────── node loader ────────────────
def load_nodes(
    csv_path: Path,
    scalers: Optional[Dict[str, StandardScaler]] = None,
    fit: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, StandardScaler]]:
    df = _safe_read_csv(csv_path)

    if scalers is None:
        scalers = {
            "node_features": StandardScaler(),
            "value_added" : StandardScaler(),
            "total"       : StandardScaler(),
        }

    x_np  = df[list(NODE_COLS)].values.astype(np.float32)
    va_np = df["Value_Added"].values.reshape(-1, 1).astype(np.float32)
    tot_np= df["Total"].values.reshape(-1, 1).astype(np.float32)

    if fit:
        x_std  = scalers["node_features"].fit_transform(x_np)
        va_std = scalers["value_added"].fit_transform(va_np)
        tot_std= scalers["total"].fit_transform(tot_np)
        for s in scalers.values():
            _protect_zero_variance(s)
    else:
        x_std  = scalers["node_features"].transform(x_np)
        va_std = scalers["value_added"].transform(va_np)
        tot_std= scalers["total"].transform(tot_np)

    x   = torch.from_numpy(x_std).float()
    va  = torch.from_numpy(va_std.squeeze()).float()
    tot = torch.from_numpy(tot_std.squeeze()).float()
    return x, va, tot, scalers


# ──────────────── edge loader ────────────────
def load_edges(
    csv_path: Path,
    value_col: int | str = 2,
    *,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
    ensure_symmetric: bool = False,
    default_rev: float = 0.0,
    apply_scaler: bool = True,          # <-- NEW flag
) -> Tuple[Tensor, Tensor, StandardScaler]:
    df = _safe_read_csv(csv_path)

    if ensure_symmetric:
        rev = df.rename(columns={df.columns[0]: "target",
                                 df.columns[1]: "source"})[df.columns]
        merged  = rev.merge(df, on=list(df.columns[:2]),
                            how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"].iloc[:, :3].copy()
        missing.iloc[:, 2] = default_rev
        df = pd.concat([df, missing], ignore_index=True)

    edge_idx = torch.tensor(df.iloc[:, :2].values.T, dtype=torch.long)
    edge_val = (df.iloc[:, value_col]
                if isinstance(value_col, int)
                else df[value_col]).values.reshape(-1, 1).astype(np.float32)

    if scaler is None:
        scaler = _identity_scaler() if not apply_scaler else StandardScaler()

    if apply_scaler:
        edge_std = scaler.fit_transform(edge_val) if fit else scaler.transform(edge_val)
        _protect_zero_variance(scaler)
        edge_wt = torch.from_numpy(edge_std.squeeze()).float()
    else:
        edge_wt = torch.from_numpy(edge_val.squeeze()).float()

    return edge_idx, edge_wt, scaler


# ──────────────── graph factory ────────────────
def make_graph(
    x_csv: Path,
    e_csv: Path,
    *,
    node_scalers: Optional[Dict[str, StandardScaler]],
    edge_scaler: Optional[StandardScaler],
    fit_scalers: bool,
    apply_edge_scaler: bool,
) -> Tuple[Data, Dict[str, StandardScaler], StandardScaler]:
    x, va, tot, node_scalers = load_nodes(x_csv, node_scalers, fit_scalers)
    ei, ew, edge_scaler      = load_edges(
        e_csv,
        scaler=edge_scaler,
        fit=fit_scalers,
        apply_scaler=apply_edge_scaler,
    )
    graph = Data(x=x, edge_index=ei, edge_attr=ew, va=va, tot=tot)
    return graph, node_scalers, edge_scaler


# ──────────────── dataset class ────────────────
class GraphWindowDataset(Dataset):
    """
    Returns (W-step history, target_graph) pairs.
    `scalers` structure  ───────────────────────────
    {
        "node"   : {StandardScaler, ...},
        "edge_A" : StandardScaler | identity,
        "edge_Z" : StandardScaler
    }
    """

    def __init__(
        self,
        years: List[int],
        cfg: Any,
        scalers: Optional[Dict[str, Any]] = None,
        fit_scalers: bool = True,
    ):
        self.window = cfg.window
        base        = Path(cfg.data_dir)

        if scalers is None:
            scalers = {"node": None, "edge_A": None, "edge_Z": None}
        self.scalers = scalers

        self.graphs_A, self.graphs_Z = [], []

        for y in years:
            g_A, self.scalers["node"], self.scalers["edge_A"] = make_graph(
                base / f"X_{y}.csv",
                base / f"Af_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_A"],
                fit_scalers=fit_scalers,
                apply_edge_scaler=False,   # Af stays 0-1
            )
            g_Z, _, self.scalers["edge_Z"] = make_graph(
                base / f"X_{y}.csv",
                base / f"Zf_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_Z"],
                fit_scalers=fit_scalers,
                apply_edge_scaler=True,    # Z needs scaling
            )

            self.graphs_A.append(g_A)
            self.graphs_Z.append(g_Z)

    # -------------- public API --------------
    def get_scalers(self) -> Dict[str, Any]:
        return self.scalers.copy()

    def __len__(self) -> int:
        return len(self.graphs_A) - self.window

    def __getitem__(self, idx: int):
        return (
            self.graphs_A[idx : idx + self.window],
            self.graphs_Z[idx + self.window],
        )


# ──────────────── (de)serialisation helpers ────────────────
def save_scalers(scalers: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)


def load_scalers(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────── simple inverse helper ────────────────
def inverse_transform_1d(pred_std: Tensor, scaler: StandardScaler) -> Tensor:
    """Torch-native inverse transform for a 1-d feature."""
    scale = torch.tensor(scaler.scale_, device=pred_std.device, dtype=pred_std.dtype)
    mean  = torch.tensor(scaler.mean_,  device=pred_std.device, dtype=pred_std.dtype)
    return pred_std * scale + mean


# ---------------- collate ----------------
def collate_window(batch):
    seqs, tgts = zip(*batch)
    return list(seqs), list(tgts)