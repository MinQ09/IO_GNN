from pathlib import Path
from typing import List, Any, Dict
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ───────────────────────── helper ─────────────────────────
def _edges_to_matrix(pred: Tensor, src: Tensor, trg: Tensor, n: int) -> Tensor:
    """1-D edge vector → (n×n) dense matrix (CPU)."""
    mat = torch.zeros(n, n, dtype=pred.dtype, device=pred.device)
    mat[src, trg] = pred
    return mat.cpu()


# ──────────────────────── main dump ────────────────────────
def dump_pred_matrices(
    model: torch.nn.Module,      # IOGNN_Z  or  IOGNN_VA
    scalers_path: Path,
    years: List[int],
    save_dir: Path,
    cfg,
    *,
    kind: str = "Z",             # "Z"  or  "VA"
    save_x: bool = True,
    float_fmt: str = "%.6g",     # reduce csv size / sci-notation
) -> None:
    """
    Save model predictions & targets as CSV files using StandardScaler.
    Expects scalers saved via pickle at scalers_path.

    kind == "Z"
        pred_Z_###.csv , true_Z_###.csv      (n×n)
    kind == "VA"
        pred_VA_###.csv, true_VA_###.csv     (n,)
    Both kinds:
        attn_out_<kind>_###.csv , attn_in_<kind>_###.csv (n×n)
        X_###.csv (FD, VA)  if save_x
    """
    assert kind in {"Z", "VA"}, "`kind` must be 'Z' or 'VA'"

    # load scalers
    with open(scalers_path, "rb") as f:
        scalers: Dict[str, Any] = pickle.load(f)

    # local import to avoid circular reference
    from data_io import GraphWindowDataset

    ds = GraphWindowDataset(years, cfg, scalers=scalers, fit_scalers=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for idx, (seq, tgt) in enumerate(ds):
        seq = [g.to(next(model.parameters()).device) for g in seq]
        tgt = tgt.to(next(model.parameters()).device)

        with torch.no_grad():
            if kind == "Z":
                p_z_std, att_out, att_in = model([seq], [tgt])
                # inverse-transform edges
                edge_scaler: StandardScaler = scalers["edge_Z"]
                p_raw = torch.from_numpy(
                    edge_scaler.inverse_transform(p_z_std.cpu().numpy().reshape(-1,1))
                ).flatten().to(tgt.edge_attr.device)
                # true edges: apply inverse of log1p then scaler if used similarly
                t_std = tgt.edge_attr.cpu().numpy().reshape(-1,1)
                t_raw = torch.from_numpy(
                    edge_scaler.inverse_transform(t_std)
                ).flatten().to(tgt.edge_attr.device)
            else:  # VA
                p_va_std, att_out, att_in = model([seq], [tgt])
                # inverse-transform VA per node
                va_scaler: StandardScaler = scalers["node"]["value_added"]
                p_raw = torch.from_numpy(
                    va_scaler.inverse_transform(p_va_std.cpu().numpy().reshape(-1,1))
                ).flatten().to(tgt.va.device)
                # true VA
                t_raw = torch.from_numpy(
                    va_scaler.inverse_transform(tgt.va.cpu().numpy().reshape(-1,1))
                ).flatten().to(tgt.va.device)

        s, t = tgt.edge_index.cpu()
        n = tgt.num_nodes

        # ─────── predictions / targets ───────
        if kind == "Z":
            pd.DataFrame(_edges_to_matrix(p_raw, s, t, n).numpy()) \
              .to_csv(save_dir / f"pred_Z_{idx:03d}.csv",
                      index=False, float_format=float_fmt)
            pd.DataFrame(_edges_to_matrix(t_raw, s, t, n).numpy()) \
              .to_csv(save_dir / f"true_Z_{idx:03d}.csv",
                      index=False, float_format=float_fmt)
        else:  # VA
            pd.DataFrame(p_raw.cpu().numpy(), columns=["VA_pred"]) \
              .to_csv(save_dir / f"pred_VA_{idx:03d}.csv",
                      index=False, float_format=float_fmt)
            pd.DataFrame(t_raw.cpu().numpy(), columns=["VA_true"]) \
              .to_csv(save_dir / f"true_VA_{idx:03d}.csv",
                      index=False, float_format=float_fmt)

        # ─────── attention heat-maps ───────
        suffix_out = f"attn_out_{kind}_{idx:03d}.csv"
        suffix_in  = f"attn_in_{kind}_{idx:03d}.csv"

        pd.DataFrame(_edges_to_matrix(att_out.cpu(), s, t, n).numpy()) \
          .to_csv(save_dir / suffix_out, index=False, float_format=float_fmt)
        pd.DataFrame(_edges_to_matrix(att_in.cpu(), t, s, n).numpy()) \
          .to_csv(save_dir / suffix_in,  index=False, float_format=float_fmt)

        # ─────── optional node features ───────
        if save_x:
            # inverse-transform node features FD, VA
            feat_scaler: StandardScaler = scalers["node"]["node_features"]
            x_std = tgt.x[:, :2].cpu().numpy()
            x_raw = feat_scaler.inverse_transform(x_std)
            pd.DataFrame(x_raw, columns=["FD", "VA"]) \
              .to_csv(save_dir / f"X_{idx:03d}.csv",
                      index=False, float_format=float_fmt)


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
    Store edge attention vectors as two n×n CSVs (out / in).
    """
    s, t = edge_index
    heat_out = torch.zeros(n_nodes, n_nodes, device=att_out.device)
    heat_out[s, t] = att_out

    heat_in = torch.zeros_like(heat_out)
    heat_in[t, s] = att_in

    pd.DataFrame(heat_out.cpu().numpy()) \
      .to_csv(save_dir / f"{file_prefix}_attn_out.csv",
              index=False, float_format=float_fmt)
    pd.DataFrame(heat_in.cpu().numpy()) \
      .to_csv(save_dir / f"{file_prefix}_attn_in.csv",
              index=False, float_format=float_fmt)


def inverse_transform_predictions(
    pred_std: Tensor,
    scalers: Dict[str, Any],
    kind: str
) -> Tensor:
    """
    Convert standardized predictions (edges or VA) back to original scale.
    - kind="Z": use scalers["Z_edge"]
    - kind="VA": use scalers["A_node"]["value_added"]
    """
    arr = pred_std.detach().cpu().numpy().reshape(-1, 1)
    if kind == "Z":
        scaler = scalers["edge_Z"]
    else:
        scaler = scalers["node"]["value_added"]
    orig = scaler.inverse_transform(arr).flatten()
    return torch.from_numpy(orig).to(pred_std.device)

def inverse_transform_targets(
    tgt_std: Tensor,
    scalers: Dict[str, Any],
    kind: str
) -> Tensor:
    """
    Convert standardized targets back to original scale.
    Same logic as for predictions.
    """
    arr = tgt_std.detach().cpu().numpy().reshape(-1, 1)
    if kind == "Z":
        scaler = scalers["edge_Z"]
    else:
        scaler = scalers["node"]["value_added"]
    orig = scaler.inverse_transform(arr).flatten()
    return torch.from_numpy(orig).to(tgt_std.device)