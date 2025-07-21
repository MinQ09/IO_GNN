from pathlib import Path
from typing import List, Any
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
import numpy as np

# ───────────────────────── helper ─────────────────────────
def _edges_to_matrix(pred: Tensor, src: Tensor, trg: Tensor, n: int) -> Tensor:
    """1-D edge vector → (n×n) dense matrix (CPU)."""
    mat = torch.zeros(n, n, dtype=pred.dtype, device=pred.device)
    mat[src, trg] = pred
    return mat.cpu()


# ──────────────────────── main dump ────────────────────────
def dump_pred_matrices(
    model: torch.nn.Module,      # IOGNN_Z  or  IOGNN_VA
    cfg: Any,
    years: List[int],
    save_dir: Path,
    *,
    kind: str = "Z",             # "Z"  or  "VA"
    to_won: bool = False,
    save_x: bool = True,
    float_fmt: str = "%.6g",     # NEW: reduce csv size / sci-notation
) -> None:
    """
    Save model predictions & targets as CSV files.

    kind == "Z"
        pred_Z_###.csv , true_Z_###.csv      (n×n)
    kind == "VA"
        pred_VA_###.csv, true_VA_###.csv     (n,)
    Both kinds:
        attn_out_<kind>_###.csv , attn_in_<kind>_###.csv (n×n)
        X_###.csv (FD, VA)  if save_x
    """
    assert kind in {"Z", "VA"}, "`kind` must be 'Z' or 'VA'"

    # local import to avoid circular reference
    try:
        from data import GraphWindowDataset
    except ImportError as e:
        raise ImportError("Could not import GraphWindowDataset; check PYTHONPATH") from e

    ds = GraphWindowDataset(years, cfg)
    save_dir.mkdir(parents=True, exist_ok=True)

    factor = 1e6 if to_won else 1.0          # NEW: unified scaling

    model.eval()
    for idx, (seq, tgt) in enumerate(ds):
        seq = [g.to(cfg.device) for g in seq]
        tgt = tgt.to(cfg.device)

        with torch.no_grad():
            if kind == "Z":
                p_z_log, att_out, att_in = model([seq], [tgt])
                p_raw = torch.expm1(p_z_log) * factor
                t_raw = torch.expm1(tgt.edge_attr) * factor
            else:  # VA
                p_va, att_out, att_in = model([seq], [tgt])
                p_raw = p_va * factor
                t_raw = tgt.va * factor

        s, t = tgt.edge_index.cpu()
        n = tgt.num_nodes

        # ─────── predictions / targets ───────
        if kind == "Z":
            pd.DataFrame(_edges_to_matrix(p_raw, s, t, n).numpy())\
              .to_csv(save_dir / f"pred_Z_{idx:03d}.csv",
                      index=False, float_format=float_fmt)
            pd.DataFrame(_edges_to_matrix(t_raw, s, t, n).numpy())\
              .to_csv(save_dir / f"true_Z_{idx:03d}.csv",
                      index=False, float_format=float_fmt)
        else:  # VA
            pd.DataFrame(p_raw.cpu().numpy(), columns=["VA_pred"])\
              .to_csv(save_dir / f"pred_VA_{idx:03d}.csv",
                      index=False, float_format=float_fmt)
            pd.DataFrame(t_raw.cpu().numpy(), columns=["VA_true"])\
              .to_csv(save_dir / f"true_VA_{idx:03d}.csv",
                      index=False, float_format=float_fmt)

        # ─────── attention heat-maps ───────
        suffix_out = f"attn_out_{kind}_{idx:03d}.csv"   # NEW: kind-aware filename
        suffix_in  = f"attn_in_{kind}_{idx:03d}.csv"

        pd.DataFrame(_edges_to_matrix(att_out.cpu(), s, t, n).numpy())\
          .to_csv(save_dir / suffix_out, index=False, float_format=float_fmt)

        pd.DataFrame(_edges_to_matrix(att_in.cpu(), t, s, n).numpy())\
          .to_csv(save_dir / suffix_in,  index=False, float_format=float_fmt)

        # ─────── optional node features ───────
        if save_x:
            x = tgt.x[:, :2].cpu().numpy() * factor  # FD, VA
            pd.DataFrame(x, columns=["FD", "VA"])\
              .to_csv(save_dir / f"X_{idx:03d}.csv",
                      index=False, float_format=float_fmt)


# ──────────────────── (legacy) edge attention ────────────────────
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
    Retained for backward-compat; main dump already writes attention files.
    """
    s, t = edge_index
    heat_out = torch.zeros(n_nodes, n_nodes, device=att_out.device)
    heat_out[s, t] = att_out

    heat_in = torch.zeros_like(heat_out)
    heat_in[t, s] = att_in

    pd.DataFrame(heat_out.cpu().numpy())\
      .to_csv(save_dir / f"{file_prefix}_attn_out.csv",
              index=False, float_format=float_fmt)
    pd.DataFrame(heat_in.cpu().numpy())\
      .to_csv(save_dir / f"{file_prefix}_attn_in.csv",
              index=False, float_format=float_fmt)