# data_io.py ──────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pickle
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

from constants import EPS  # unchanged

# ───────────────────────────── constants ────────────────────────────────
NODE_COLS: Sequence[str] = ("Imports", "Exports", "Final_Demand")
SCALE_FACTOR = 1e3  # Z, VA, Total scaled down by 1e6

# ───────────────────────────── utils ────────────────────────────────────

def _safe_read_csv(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path, thousands=",")
          .replace(["–", "-", ""], np.nan)
          .fillna(0.0)
    )

def _protect_zero_variance(scaler: StandardScaler) -> None:
    scaler.scale_[scaler.scale_ == 0.0] = 1.0

def _identity_scaler() -> StandardScaler:
    s = StandardScaler()
    s.mean_  = np.zeros(1, dtype=np.float32)
    s.scale_ = np.ones (1, dtype=np.float32)
    s.var_   = np.ones (1, dtype=np.float32)
    return s

# single helper kept – all inverse transforms route here

def _inverse_1d(std: Tensor, scaler: StandardScaler) -> Tensor:
    """Inverse transform a 1-D tensor with *scaler* (handles identity fast)."""
    if abs(scaler.scale_[0] - 1.0) < 1e-6 and abs(scaler.mean_[0]) < 1e-6:
        return std
    scale = torch.as_tensor(scaler.scale_,  dtype=std.dtype, device=std.device)
    mean  = torch.as_tensor(scaler.mean_,   dtype=std.dtype, device=std.device)
    return std * scale + mean

# ----------------------------------------------------------------------
# node table loader
# ----------------------------------------------------------------------

def load_nodes(
    csv_path: Path,
    scalers: Optional[Dict[str, StandardScaler]] = None,
    *,
    fit: bool = True,
    scale_node_feats: bool = True,
    scale_va_tot: bool = False,  # <- now honoured
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, StandardScaler]]:
    """Load node-level features / targets and return (x_std, x_raw, va_std, tot_std)."""
    df = _safe_read_csv(csv_path).astype(np.float32)

    # scalers dict initialisation -------------------------------------------------
    if scalers is None:
        scalers = {
            "node_features": StandardScaler(),
            "value_added" : StandardScaler(),
            "total"       : StandardScaler(),
        }

    # raw numpy arrays ------------------------------------------------------------
    x_np   = df[list(NODE_COLS)].values / SCALE_FACTOR
    va_np  = df["Value_Added"].values.reshape(-1, 1) / SCALE_FACTOR
    tot_np = df["Total"      ].values.reshape(-1, 1) / SCALE_FACTOR

    # (1) node-feature standardisation -------------------------------------------
    if scale_node_feats:
        if fit:
            x_std = scalers["node_features"].fit_transform(x_np)
            _protect_zero_variance(scalers["node_features"])
        else:
            x_std = scalers["node_features"].transform(x_np)
    else:
        scalers["node_features"] = _identity_scaler(); x_std = x_np

    # (2) VA / Total scaling (optional) ------------------------------------------
    if scale_va_tot:
        if fit:
            va_std = scalers["value_added"].fit_transform(va_np)
            tot_std = scalers["total"].fit_transform(tot_np)
            _protect_zero_variance(scalers["value_added"])
            _protect_zero_variance(scalers["total"])
        else:
            va_std = scalers["value_added"].transform(va_np)
            tot_std = scalers["total"].transform(tot_np)
    else:
        if scalers["value_added"] is None:
            scalers["value_added"] = _identity_scaler()
        if scalers["total"] is None:
            scalers["total"] = _identity_scaler()
        va_std, tot_std = va_np, tot_np

    # tensors --------------------------------------------------------------------
    x     = torch.from_numpy(x_std).float()
    x_raw = torch.from_numpy(x_np ).float()
    va    = torch.from_numpy(va_std.squeeze()).float()
    tot   = torch.from_numpy(tot_std.squeeze()).float()
    va_raw = torch.from_numpy(va_np.squeeze()).float()
    tot_raw = torch.from_numpy(tot_np.squeeze()).float()
    return x, x_raw, va, tot, va_raw, tot_raw, scalers

# ----------------------------------------------------------------------
# edge table loader
# ----------------------------------------------------------------------

def load_edges(
    csv_path: Path,
    value_col: int | str = 2,
    *,
    apply_scaler: bool = True,
    ensure_symmetric: bool = False,
) -> Tuple[Tensor, Tensor, StandardScaler]:
    """Return (edge_index, edge_weight, scaler). scaler is identity by default."""
    df = (
        pd.read_csv(csv_path, thousands=",")
          .replace(["–", "-", ""], 0.0)
          .fillna(0.0)
          .astype(np.float32)
    )

    # symmetric completion (optional) -------------------------------------------
    if ensure_symmetric:
        src_col, tgt_col = df.columns[:2]
        rev = df.rename(columns={src_col: tgt_col, tgt_col: src_col})
        df  = pd.concat([df, rev]).drop_duplicates(subset=df.columns[:3], ignore_index=True)

    edge_idx = torch.tensor(df.iloc[:, :2].values.T, dtype=torch.long)
    vals = df.iloc[:, value_col] if isinstance(value_col, int) else df[value_col]
    edge_val = vals.to_numpy(dtype=np.float32).reshape(-1, 1)
    if apply_scaler:
        edge_val /= SCALE_FACTOR
    scaler = _identity_scaler()  # placeholder; could swap for real scaler if needed
    edge_wt = torch.from_numpy(edge_val.squeeze()).float()
    return edge_idx, edge_wt, scaler

# ----------------------------------------------------------------------
# graph constructor
# ----------------------------------------------------------------------

def make_graph(
    x_csv: Path,
    e_csv: Path,
    *,
    node_scalers: Optional[Dict[str, StandardScaler]] = None,
    edge_scaler: Optional[StandardScaler] = None,
    fit_scalers: bool = True,
    scale_node_feats: bool = True,
    scale_va_tot: bool = False,
    apply_edge_scaler: bool = True,
) -> Tuple[Data, Dict[str, StandardScaler], StandardScaler]:
    x, x_raw, va, tot, va_raw, tot_raw, node_scalers = load_nodes(
        x_csv,
        node_scalers,
        fit=fit_scalers,
        scale_node_feats=scale_node_feats,
        scale_va_tot=scale_va_tot,
    )
    ei, ew, edge_scaler = load_edges(
        e_csv,
        value_col=2,
        apply_scaler=apply_edge_scaler,
        ensure_symmetric=False,
    )
    return Data(x=x, x_raw=x_raw, edge_index=ei, edge_attr=ew, va=va, tot=tot, va_raw=va_raw, tot_raw=tot_raw), node_scalers, edge_scaler

# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------

class GraphWindowDataset(Dataset):
    """Returns (history_graphs_A, target_graph_Z)."""
    def __init__(
        self,
        years: List[int],
        cfg: Any,
        scalers: Optional[Dict[str, Any]] = None,
        *,
        fit_scalers: bool = True,
        scale_targets: bool = False,  # kept for interface compatibility
    ):
        self.window = cfg.window
        base = Path(cfg.data_dir)
        self.scalers = scalers or {"node": None, "edge_A": None, "edge_Z": None}
        if self.scalers["edge_Z"] is None:
            self.scalers["edge_Z"] = _identity_scaler()
        self.graphs_A: List[Data] = []
        self.graphs_Z: List[Data] = []

        for y in years:
            g_A, self.scalers["node"], self.scalers["edge_A"] = make_graph(
                base / f"X_{y}.csv",
                base / f"Af_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_A"],
                fit_scalers=fit_scalers,
                scale_node_feats=True,
                scale_va_tot=True,
                apply_edge_scaler=False,
            )
            g_Z, _, _ = make_graph(
                base / f"X_{y}.csv",
                base / f"Zf_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_Z"],
                fit_scalers=False,
                scale_node_feats=True,
                scale_va_tot= True,
                apply_edge_scaler=True,
            )
            self.graphs_A.append(g_A)
            self.graphs_Z.append(g_Z)

        # expose feature dim
        self.nfeat = len(NODE_COLS)

    # --------------------------------------------------
    def get_scalers(self) -> Dict[str, Any]:
        """Deep copy to prevent accidental mutation by caller."""
        return deepcopy(self.scalers)

    # Torch Dataset protocol ---------------------------
    def __len__(self) -> int:
        return len(self.graphs_A) - self.window

    def __getitem__(self, idx: int):
        return self.graphs_A[idx:idx + self.window], self.graphs_Z[idx + self.window]

# ----------------------------------------------------------------------
# convenience I/O helpers
# ----------------------------------------------------------------------

def save_scalers(scalers: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)


def load_scalers(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------------------------------------------------------
# high-level inverse helpers (used in run_single)
# ----------------------------------------------------------------------

def inverse_transform_predictions(pred_std: Tensor, scalers: Dict[str, Any], kind: str) -> Tensor:
    if kind == "Z":
        return _inverse_1d(pred_std, scalers["edge_Z"])
    else:  # VA
        return _inverse_1d(pred_std, scalers["node"]["value_added"])


def inverse_transform_targets(tgt_std: Tensor, scalers: Dict[str, Any], kind: str) -> Tensor:
    return inverse_transform_predictions(tgt_std, scalers, kind)

# ----------------------------------------------------------------------
# simple collate fn
# ----------------------------------------------------------------------

def collate_window(batch):
    seqs, tgts = zip(*batch)
    return list(seqs), list(tgts)
