from pathlib import Path
from typing import List, Any, Dict, Literal, Optional

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
import pickle
from sklearn.preprocessing import StandardScaler


# ----------------------------- utilities -----------------------------

def _is_identity_scaler(scaler: Optional[StandardScaler]) -> bool:
    """
    Return True if the given StandardScaler is effectively an identity map.
    Safe-guards when scalers are absent or not sklearn scalers.
    """
    if scaler is None:
        return True
    return (
        hasattr(scaler, "scale_")
        and hasattr(scaler, "mean_")
        and float(scaler.scale_[0]) == 1.0
        and float(scaler.mean_[0]) == 0.0
    )


def _edges_to_matrix(
    values: Tensor,
    src: Tensor,
    trg: Tensor,
    n_nodes: int,
    *,
    reduce: Literal["sum", "mean"] = "sum",
) -> Tensor:
    """
    Convert a 1-D edge vector to an (n×n) dense matrix with proper handling
    of duplicate (src, trg) pairs.

    Parameters
    ----------
    values : Tensor[E]
        Edge values.
    src, trg : Tensor[E]
        Source and target node indices.
    n_nodes : int
        Number of nodes in the graph.
    reduce : {"sum", "mean"}, default "sum"
        How to merge duplicates (accumulate or average).

    Returns
    -------
    Tensor
        Dense (n×n) matrix on CPU.
    """
    mat = torch.zeros(n_nodes, n_nodes, dtype=values.dtype, device=values.device)
    mat.index_put_((src, trg), values, accumulate=True)

    if reduce == "mean":
        counts = torch.zeros_like(mat)
        counts.index_put_((src, trg), torch.ones_like(values), accumulate=True)
        mat = torch.where(counts > 0, mat / counts.clamp(min=1), mat)

    return mat.cpu()


# ---------------------------- main dump ----------------------------

def dump_pred_matrices(
    model: torch.nn.Module,      # IOGNN_Z or IOGNN_VA
    scalers_path: Path,
    years: List[int],
    save_dir: Path,
    cfg: Any | None = None,
    *,
    kind: str = "Z",             # "Z" or "VA"
    save_x: bool = True,
    float_fmt: str = "%.6g",
    reduce: Literal["sum", "mean"] = "sum",
) -> None:
    """
    Save model predictions & targets as CSV files.
    - Robust to parallel edges.
    - Avoids unnecessary inverse-transforms when scalers are identity.
    """
    assert kind in {"Z", "VA"}, "`kind` must be 'Z' or 'VA'"

    # Load scalers
    with open(scalers_path, "rb") as f:
        scalers: Dict[str, Any] = pickle.load(f)

    # Local import to avoid circular reference
    from data_io import GraphWindowDataset

    ds = GraphWindowDataset(years, cfg, scalers=scalers, fit_scalers=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # Pick scalers
    edge_scaler: Optional[StandardScaler] = scalers.get("edge_Z")
    node_scalers: Dict[str, Any] = scalers.get("node", {})
    va_scaler: Optional[StandardScaler] = node_scalers.get("value_added")
    feat_scaler: Optional[StandardScaler] = node_scalers.get("node_features")

    for idx, (seq, tgt) in enumerate(ds):
        seq = [g.to(device) for g in seq]
        tgt = tgt.to(device)

        with torch.no_grad():
            # Forward once for this graph
            pred_std, att_out, att_in = model([seq], [tgt])

            if kind == "Z":
                # Predictions
                if _is_identity_scaler(edge_scaler):
                    p_raw = pred_std.detach()
                    t_raw = tgt.edge_attr.detach()
                else:
                    p_raw = torch.from_numpy(
                        edge_scaler.inverse_transform(pred_std.cpu().numpy().reshape(-1, 1))
                    ).flatten().to(device)
                    t_raw = torch.from_numpy(
                        edge_scaler.inverse_transform(tgt.edge_attr.cpu().numpy().reshape(-1, 1))
                    ).flatten().to(device)

            else:  # kind == "VA"
                if _is_identity_scaler(va_scaler):
                    p_raw = pred_std.detach()
                    t_raw = tgt.va.detach()
                else:
                    p_raw = torch.from_numpy(
                        va_scaler.inverse_transform(pred_std.cpu().numpy().reshape(-1, 1))
                    ).flatten().to(device)
                    t_raw = torch.from_numpy(
                        va_scaler.inverse_transform(tgt.va.cpu().numpy().reshape(-1, 1))
                    ).flatten().to(device)

        s, t = tgt.edge_index.cpu()
        n = tgt.num_nodes

        # -------- predictions / targets --------
        if kind == "Z":
            pd.DataFrame(
                _edges_to_matrix(p_raw, s, t, n, reduce=reduce).numpy()
            ).to_csv(save_dir / f"pred_Z_{idx:03d}.csv", index=False, float_format=float_fmt)

            pd.DataFrame(
                _edges_to_matrix(t_raw, s, t, n, reduce=reduce).numpy()
            ).to_csv(save_dir / f"true_Z_{idx:03d}.csv", index=False, float_format=float_fmt)
        else:  # VA
            pd.DataFrame(p_raw.cpu().numpy(), columns=["VA_pred"]) \
              .to_csv(save_dir / f"pred_VA_{idx:03d}.csv", index=False, float_format=float_fmt)
            pd.DataFrame(t_raw.cpu().numpy(), columns=["VA_true"]) \
              .to_csv(save_dir / f"true_VA_{idx:03d}.csv", index=False, float_format=float_fmt)

        # -------- attention heat-maps --------
        # Outgoing attention uses (src, trg); incoming attention is mirrored
        # over (trg, src) to reflect target-normalized scores.
        suffix_out = f"attn_out_{kind}_{idx:03d}.csv"
        suffix_in  = f"attn_in_{kind}_{idx:03d}.csv"

        pd.DataFrame(
            _edges_to_matrix(att_out.cpu(), s, t, n, reduce=reduce).numpy()
        ).to_csv(save_dir / suffix_out, index=False, float_format=float_fmt)

        pd.DataFrame(
            _edges_to_matrix(att_in.cpu(), t, s, n, reduce=reduce).numpy()
        ).to_csv(save_dir / suffix_in,  index=False, float_format=float_fmt)

        # -------- optional node features --------
        if save_x and feat_scaler is not None:
            x_std = tgt.x.cpu().numpy()
            x_raw = feat_scaler.inverse_transform(x_std)

            # Column names by convention: typically [IMP, EXP, FD]
            cols = [f"X{i}" for i in range(x_raw.shape[1])]
            if x_raw.shape[1] == 3:
                cols = ["IMP", "EXP", "FD"]
            elif x_raw.shape[1] == 2:
                cols = ["F1", "F2"]  # fallback when only two features are present

            pd.DataFrame(x_raw, columns=cols) \
              .to_csv(save_dir / f"X_{idx:03d}.csv", index=False, float_format=float_fmt)


def save_edge_attention(
    att_out: torch.Tensor,
    att_in: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
    file_prefix: str,
    save_dir: Path,
    *,
    float_fmt: str = "%.6g",
) -> None:
    """
    Store edge attention vectors as two n×n CSVs (outgoing / incoming).
    Provided as a standalone helper when you already have attention tensors.
    """
    s, t = edge_index
    heat_out = torch.zeros(n_nodes, n_nodes, device=att_out.device)
    heat_out[s, t] = att_out

    heat_in = torch.zeros_like(heat_out)
    heat_in[t, s] = att_in

    pd.DataFrame(heat_out.cpu().numpy()) \
      .to_csv(save_dir / f"{file_prefix}_attn_out.csv", index=False, float_format=float_fmt)
    pd.DataFrame(heat_in.cpu().numpy()) \
      .to_csv(save_dir / f"{file_prefix}_attn_in.csv",  index=False, float_format=float_fmt)


def inverse_transform_predictions(
    pred_std: Tensor,
    scalers: Dict[str, Any],
    kind: str
) -> Tensor:
    """
    Convert standardized predictions (edges or VA) back to original scale.
    Skips work if the corresponding scaler is identity.
    """
    if kind == "Z":
        scaler: Optional[StandardScaler] = scalers.get("edge_Z")
    else:
        scaler = scalers.get("node", {}).get("value_added")

    if _is_identity_scaler(scaler):
        return pred_std.detach()

    arr = pred_std.detach().cpu().numpy().reshape(-1, 1)
    orig = scaler.inverse_transform(arr).flatten()  # type: ignore[arg-type]
    return torch.from_numpy(orig).to(pred_std.device)


def inverse_transform_targets(
    tgt_std: Tensor,
    scalers: Dict[str, Any],
    kind: str
) -> Tensor:
    """
    Convert standardized targets back to original scale.
    Skips work if the corresponding scaler is identity.
    """
    if kind == "Z":
        scaler: Optional[StandardScaler] = scalers.get("edge_Z")
    else:
        scaler = scalers.get("node", {}).get("value_added")

    if _is_identity_scaler(scaler):
        return tgt_std.detach()

    arr = tgt_std.detach().cpu().numpy().reshape(-1, 1)
    orig = scaler.inverse_transform(arr).flatten()  # type: ignore[arg-type]
    return torch.from_numpy(orig).to(tgt_std.device)