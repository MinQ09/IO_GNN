# data_io.py ──────────────────────────────────────────────────────────────
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
from constants import EPS

NODE_COLS: Sequence[str] = ("Import", "Export", "Final_Demand")
SCALE_FACTOR = 1e6  # Z, VA, Total에 적용할 스케일 팩터 (백만 단위)

def _safe_read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, thousands=",").replace(["–", "-", ""], np.nan)
    return df.fillna(0.0)

def _protect_zero_variance(scaler: StandardScaler) -> None:
    scaler.scale_[scaler.scale_ == 0.0] = 1.0

def _identity_scaler() -> StandardScaler:
    s = StandardScaler()
    s.fit(np.array([[0.0], [1.0]]))
    _protect_zero_variance(s)
    return s

def load_nodes(
    csv_path: Path,
    scalers: Optional[Dict[str, StandardScaler]] = None,
    *,
    fit: bool = True,
    scale_node_feats: bool = True,
    scale_va_tot: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, StandardScaler]]:
    df = _safe_read_csv(csv_path)
    # 1) 강제 타입 캐스팅: 모든 컬럼을 float32
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)

    if scalers is None:
        scalers = {
            "node_features": StandardScaler(),
            "value_added"  : StandardScaler(),
            "total"        : StandardScaler(),
        }

    # 2) Raw numpy arrays
    x_np   = df[list(NODE_COLS)].values  # float32
    va_np  = df["Value_Added"].values.reshape(-1, 1) / SCALE_FACTOR
    tot_np = df["Total"].values.reshape(-1, 1)       / SCALE_FACTOR

    # 3) Node feature scaling
    if scale_node_feats:
        if fit:
            x_std = scalers["node_features"].fit_transform(x_np)
            _protect_zero_variance(scalers["node_features"])
        else:
            x_std = scalers["node_features"].transform(x_np)
    else:
        scalers["node_features"] = _identity_scaler()
        x_std = x_np

    # 4) VA & Total: always raw-scaled by factor, use identity scaler
    scalers["value_added"] = _identity_scaler()
    scalers["total"]       = _identity_scaler()
    va_std, tot_std = va_np, tot_np

    # 5) To tensors
    x     = torch.from_numpy(x_std).float()
    x_raw = torch.from_numpy(x_np).float()
    va    = torch.from_numpy(va_std.squeeze()).float()
    tot   = torch.from_numpy(tot_std.squeeze()).float()
    return x, x_raw, va, tot, scalers

def load_edges(
    csv_path: Path,
    value_col: int | str = 2,
    *,
    apply_scaler: bool = True,
    ensure_symmetric: bool = False,
    default_rev: float = 0.0,
) -> Tuple[Tensor, Tensor, StandardScaler]:
    # 1) CSV 읽기 + NaN → 0, thousands 처리
    df = pd.read_csv(csv_path, thousands=",").replace(["–","-",""], 0.0).fillna(0.0)
    # 2) 모든 컬럼 float32 강제
    for col in df.columns:
        df[col] = df[col].astype(np.float32)
    # 3) ensure_symmetric
    if ensure_symmetric:
        rev = df.rename(columns={df.columns[0]:"target", df.columns[1]:"source"})[df.columns]
        df = pd.concat([df, rev[~rev.set_index(df.columns[:2]).index.isin(df.set_index(df.columns[:2]).index)]], ignore_index=True)
        for col in df.columns:
            df[col] = df[col].astype(np.float32)
    # 4) edge_index
    edge_idx = torch.tensor(df.iloc[:, :2].values.T, dtype=torch.long)
    # 5) edge_val as float32 numpy
    vals = df.iloc[:, value_col] if isinstance(value_col, int) else df[value_col]
    edge_val = np.asarray(vals.values, dtype=np.float32).reshape(-1,1)
    # 6) scale by 1e6 if requested
    if apply_scaler:
        edge_val /= SCALE_FACTOR
    # 7) always use identity scaler
    scaler = _identity_scaler()
    edge_wt = torch.from_numpy(edge_val.squeeze()).float()
    return edge_idx, edge_wt, scaler

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
    x, x_raw, va, tot, node_scalers = load_nodes(
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
        default_rev=0.0,
    )
    graph = Data(x=x, x_raw=x_raw, edge_index=ei, edge_attr=ew, va=va, tot=tot)
    return graph, node_scalers, edge_scaler

class GraphWindowDataset(Dataset):
    """
    Returns (history_graphs_A, target_graph_Z)
    scalers keys: "node", "edge_A", "edge_Z"
    """
    def __init__(
        self,
        years: List[int],
        cfg: Any,
        scalers: Optional[Dict[str, Any]] = None,
        fit_scalers: bool = True,
        scale_targets: bool = False,
    ):
        self.window = cfg.window
        base = Path(cfg.data_dir)
        self.scalers = scalers or {"node": None, "edge_A": None, "edge_Z": None}
        self.graphs_A, self.graphs_Z = [], []

        for y in years:
            # A matrix (0-1 range)
            g_A, self.scalers["node"], self.scalers["edge_A"] = make_graph(
                base / f"X_{y}.csv",
                base / f"Af_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_A"],
                fit_scalers=fit_scalers,
                scale_node_feats=True,
                scale_va_tot=False,
                apply_edge_scaler=False,
            )
            # Z matrix (scaled by 1e6)
            g_Z, _, self.scalers["edge_Z"] = make_graph(
                base / f"X_{y}.csv",
                base / f"Zf_{y}.csv",
                node_scalers=self.scalers["node"],
                edge_scaler=self.scalers["edge_Z"],
                fit_scalers=fit_scalers,
                scale_node_feats=True,
                scale_va_tot=False,
                apply_edge_scaler= True,
            )
            self.graphs_A.append(g_A)
            self.graphs_Z.append(g_Z)

    def get_scalers(self) -> Dict[str, Any]:
        return self.scalers.copy()

    def __len__(self) -> int:
        return len(self.graphs_A) - self.window

    def __getitem__(self, idx: int):
        return self.graphs_A[idx : idx + self.window], self.graphs_Z[idx + self.window]

def save_scalers(scalers: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)

def load_scalers(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)

def inverse_transform_1d(pred_std: Tensor, scaler: StandardScaler) -> Tensor:
    if abs(scaler.scale_[0] - 1.0) < 1e-6 and abs(scaler.mean_[0] - 0.0) < 1e-6:
        return pred_std
    scale = torch.tensor(scaler.scale_, device=pred_std.device, dtype=pred_std.dtype)
    mean  = torch.tensor(scaler.mean_,  device=pred_std.device, dtype=pred_std.dtype)
    return pred_std * scale + mean

def inverse_scale_predictions(pred: Tensor, scale_factor: float = SCALE_FACTOR) -> Tensor:
    return pred * scale_factor

def inverse_scale_targets(targets: Tensor, scale_factor: float = SCALE_FACTOR) -> Tensor:
    return targets * scale_factor

def collate_window(batch):
    seqs, tgts = zip(*batch)
    return list(seqs), list(tgts)
