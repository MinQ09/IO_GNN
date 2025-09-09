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
NODE_COLS: Sequence[str] = ("Import", "Export", "Final_Demand")
SCALE_FACTOR = 1e6  # Z, VA, Total scaled down by 1e6 for numerical stability

# ───────────────────────────── utils ────────────────────────────────────

def _safe_read_csv(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path, thousands=",")
          .replace(["–", "-", ""], np.nan)
          .fillna(0.0)
    ).astype(np.float32)

def _protect_zero_variance(scaler: StandardScaler) -> None:
    # Avoid division by zero when a column is constant
    if hasattr(scaler, "scale_"):
        scaler.scale_[scaler.scale_ == 0.0] = 1.0

def _identity_scaler(n_features: int = 1) -> StandardScaler:
    """
    Return a StandardScaler that behaves like a fitted identity scaler
    for a given feature dimension.
    """
    s = StandardScaler()
    s.mean_  = np.zeros(n_features, dtype=np.float32)
    s.scale_ = np.ones (n_features, dtype=np.float32)
    s.var_   = np.ones (n_features, dtype=np.float32)
    # 일부 sklearn 버전에서 요구하는 속성들:
    s.n_samples_seen_ = int(1)                 # 또는 np.array([1]*n_features)도 가능
    s.n_features_in_  = int(n_features)
    return s

def _ensure_node_scalers_dict(
    scalers: Optional[Dict[str, Any]],
    *,
    n_node_features: int = len(NODE_COLS)
) -> Dict[str, Any]:
    """
    Normalize scalers dict structure:
      scalers = {
        "node": {"features": SS, "value_added": SS, "total": SS},
        "edge_A": SS,
        "edge_Z": SS
      }
    Missing keys are filled with identity scalers with correct dimensions.
    """
    s = scalers or {}
    node = s.get("node", {})
    node = dict(node) if isinstance(node, dict) else {}
    node.setdefault("features", _identity_scaler(n_node_features))
    node.setdefault("value_added", _identity_scaler(1))
    node.setdefault("total", _identity_scaler(1))
    s["node"] = node
    s.setdefault("edge_A", _identity_scaler(1))
    s.setdefault("edge_Z", _identity_scaler(1))
    return s

# Single helper – all inverse transforms route here (torch-friendly)
def _inverse_1d(std: Tensor, scaler: StandardScaler) -> Tensor:
    """
    Inverse transform a 1-D (or flattened) tensor with *scaler*; identity fast-path.
    """
    scale_np = getattr(scaler, "scale_", np.array([1.0], dtype=np.float32))
    mean_np  = getattr(scaler, "mean_",  np.array([0.0], dtype=np.float32))
    if np.allclose(scale_np, 1.0, atol=1e-6) and np.allclose(mean_np, 0.0, atol=1e-6):
        return std
    scale = torch.as_tensor(scale_np, dtype=std.dtype, device=std.device)
    mean  = torch.as_tensor(mean_np,  dtype=std.dtype, device=std.device)
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
    scale_targets: bool = False,  # ← VA / TOTAL 표준화 여부
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, StandardScaler]]:
    """
    Load node-level features / targets.

    Returns
    -------
    x      : standardized node features (or raw if scale_node_feats=False)
    x_raw  : raw node features (scaled only by SCALE_FACTOR)
    va     : value added (standardized if scale_targets=True, else raw)
    tot    : total (standardized if scale_targets=True, else raw)
    va_raw : raw value added (scaled only by SCALE_FACTOR)
    tot_raw: raw total (scaled only by SCALE_FACTOR)
    scalers: dict with keys {"features", "value_added", "total"}
    """
    df = _safe_read_csv(csv_path)

    # scalers dict initialisation -------------------------------------------------
    if scalers is None:
        scalers = {
            "features": StandardScaler(),
            "value_added": StandardScaler(),
            "total": StandardScaler(),
        }

    # raw numpy arrays ------------------------------------------------------------
    x_np   = df[list(NODE_COLS)].values / SCALE_FACTOR
    va_np  = df["Value_Added"].values.reshape(-1, 1) / SCALE_FACTOR
    tot_np = df["Total"      ].values.reshape(-1, 1) / SCALE_FACTOR

    # (1) node-feature standardisation -------------------------------------------
    if scale_node_feats:
        if fit:
            x_std = scalers["features"].fit_transform(x_np)
            _protect_zero_variance(scalers["features"])
        else:
            x_std = scalers["features"].transform(x_np)
    else:
        scalers["features"] = _identity_scaler(x_np.shape[1])
        x_std = x_np

    # (2) VA / Total scaling (optional) ------------------------------------------
    if scale_targets:
        if fit:
            va_std  = scalers["value_added"].fit_transform(va_np)
            tot_std = scalers["total"].fit_transform(tot_np)
            _protect_zero_variance(scalers["value_added"])
            _protect_zero_variance(scalers["total"])
        else:
            va_std  = scalers["value_added"].transform(va_np)
            tot_std = scalers["total"].transform(tot_np)
    else:
        # Always override with identity scalers to ensure mean_/scale_ exist
        scalers["value_added"] = _identity_scaler(1)
        scalers["total"]       = _identity_scaler(1)
        va_std, tot_std = va_np, tot_np

    # tensors --------------------------------------------------------------------
    x       = torch.from_numpy(x_std).float()
    x_raw   = torch.from_numpy(x_np ).float()
    va      = torch.from_numpy(va_std.squeeze()).float()
    tot     = torch.from_numpy(tot_std.squeeze()).float()
    va_raw  = torch.from_numpy(va_np.squeeze()).float()
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
    """Return (edge_index, edge_weight, scaler). 'scaler' is identity by default."""
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
    scaler = _identity_scaler(1)  # placeholder; keep identity unless you add a true edge scaler
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
    edge_scaler: Optional[StandardScaler] = None,   # kept for signature parity
    fit_scalers: bool = True,
    scale_node_feats: bool = True,
    scale_targets: bool = False,
    apply_edge_scaler: bool = True,
) -> Tuple[Data, Dict[str, StandardScaler], StandardScaler]:
    x, x_raw, va, tot, va_raw, tot_raw, node_scalers = load_nodes(
        x_csv,
        node_scalers,
        fit=fit_scalers,
        scale_node_feats=scale_node_feats,
        scale_targets=scale_targets,
    )
    ei, ew, edge_scaler = load_edges(
        e_csv,
        value_col=2,
        apply_scaler=apply_edge_scaler,
        ensure_symmetric=False,
    )
    data = Data(
        x=x, x_raw=x_raw,
        edge_index=ei, edge_attr=ew,
        va=va, tot=tot,
        va_raw=va_raw, tot_raw=tot_raw
    )
    return data, node_scalers, edge_scaler

# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------

class GraphWindowDataset(Dataset):
    """Returns (history_graphs_A, target_graph_Z).
       Nowcasting 모드: 히스토리의 마지막 시점(t) 자체를 예측."""
    def __init__(
        self,
        years: List[int],
        cfg: Any,
        scalers: Optional[Dict[str, Any]] = None,
        *,
        fit_scalers: bool = True,
        scale_targets: bool = False,
    ):
        self.window = int(getattr(cfg, "window", 1))
        base = Path(cfg.data_dir)

        # normalize scalers layout (dim 반영)
        self.scalers = _ensure_node_scalers_dict(scalers, n_node_features=len(NODE_COLS))

        # 효과적인 표준화 플래그: 인자로 True이거나 cfg.scale_targets=True이면 적용
        self.scale_targets = bool(scale_targets or getattr(cfg, "scale_targets", False))
        self.scale_node_feats = bool(getattr(cfg, "scale_node_feats", True))

        self.graphs_A: List[Data] = []
        self.graphs_Z: List[Data] = []

        for y in years:
            # A-graph (history inputs)
            g_A, self.scalers["node"], self.scalers["edge_A"] = make_graph(
                base / f"X_{y}.csv",
                base / f"Af_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_A"],
                fit_scalers=fit_scalers,
                scale_node_feats=self.scale_node_feats,
                scale_targets=self.scale_targets,   # ★ VA/TOT 표준화 적용
                apply_edge_scaler=False,
            )
            # Z-graph (target)
            g_Z, _, self.scalers["edge_Z"] = make_graph(
                base / f"X_{y}.csv",
                base / f"Zf_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_Z"],
                fit_scalers=False,                  # node_scalers는 A에서 이미 fit됨
                scale_node_feats=self.scale_node_feats,
                scale_targets=self.scale_targets,   # ★ VA/TOT 표준화 적용
                apply_edge_scaler=True,
            )
            self.graphs_A.append(g_A)
            self.graphs_Z.append(g_Z)

        self.nfeat = len(NODE_COLS)

    def get_scalers(self) -> Dict[str, Any]:
        return deepcopy(self.scalers)

    def __len__(self) -> int:
        return max(0, len(self.graphs_A) - self.window + 1)

    def __getitem__(self, idx: int):
        seq = self.graphs_A[idx : idx + self.window]
        tgt = self.graphs_Z[idx + self.window - 1]
        return seq, tgt

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