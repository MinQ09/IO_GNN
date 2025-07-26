# run_single.py ────────────────────────────────────────────────
"""
Single‑run trainer for the IO‑GNN models.

Supported tasks
---------------
* ``kind="Z"``  –   predict **edge flows** (inter‑industry transactions)
* ``kind="VA"`` –   predict **node value‑added**

Key changes vs. your original script
------------------------------------
1. **Mini‑batch λₜ is now computed with an inline helper** for clarity.
2. **Scaler inversion** uses the new utility
   ``inverse_transform_predictions()`` so you never mix up Z / VA scalers.
3. **Cosmetic clean‑ups** (typing, f‑string alignment, black‑style imports).
4. The core training logic and early‑stopping semantics are **unchanged**.

Feel free to diff this against the original – only ~60 lines moved / edited.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

import torch_geometric.data as pyg

from data_io import GraphWindowDataset, collate_window
from helper import (
    dump_pred_matrices,
    inverse_transform_predictions,
    inverse_transform_targets,
    save_edge_attention,
)
from losses import get_pinn_loss_function, pinn_single_z_std, pinn_single_va_std
from metrics import (
    cvr_tensor_standardized,
    mae,
    mean_ignore_nan,
    rmse,
    r2,
    safe_pearson,
    smape,
)
from model import IOGNN_VA, IOGNN_Z
from utils import set_seed

# ───────────────────────── helpers ──────────────────────────

def _slice_batch(cat: torch.Tensor, graphs: List[pyg.data.Data], edge: bool):
    """Yield *cat* slices belonging to individual *graphs*."""

    offs = 0
    for g in graphs:
        span = g.edge_attr.numel() if edge else g.num_nodes
        yield cat[offs : offs + span], g
        offs += span


def _adaptive_lambda(
    *, mse: torch.Tensor, pinn: torch.Tensor, cfg, global_step: int
) -> torch.Tensor:
    """Equation (7) in the paper – with warm‑up and numerical guard."""

    scale = (
        1.0
        if cfg.warmup == 0
        else min(1.0, float(global_step) / cfg.warmup)
    )
    return (
        scale
        * cfg.lambda_max
        * (mse.detach() / (pinn.detach() + 1e-12)).clamp(max=1.0)
    )


# ─────────────────────────── main ───────────────────────────

def run_single(
    cfg: Any, seed: int, *, kind: str = "Z"
) -> Tuple[torch.nn.Module, Dict[str, List[float]], None, Dict[str, float]]:
    """Train once with *cfg* and *seed*, return the fitted model & metrics."""

    if kind not in {"Z", "VA"}:
        raise ValueError("kind must be either 'Z' or 'VA'")

    set_seed(seed)
    edge_mode = kind == "Z"

    # ─── I/O dirs ──────────────────────────────────────────
    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # ─── DATASETS & SCALERS ───────────────────────────────
    years = list(range(1, 56))  # 1..72 inclusive
    tr_y, vl_y, ts_y = years[:-10], years[-10:-5], years[-5:]

    tr_ds = GraphWindowDataset(tr_y, cfg, scalers=None, fit_scalers=True)
    scalers = tr_ds.get_scalers()
    if cfg.save_scalers and not (Path(cfg.scalers_path).exists()):
        with open(cfg.scalers_path, "wb") as f:
            pickle.dump(scalers, f)

    def _mk_loader(Y, shuffle: bool, bs: int):
        return DataLoader(
            GraphWindowDataset(Y, cfg, scalers=scalers, fit_scalers=False),
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_window,
            pin_memory=True,
        )

    tr_ld, vl_ld, ts_ld = (
        _mk_loader(tr_y, True, cfg.batch_size),
        _mk_loader(vl_y, False, cfg.batch_size),
        _mk_loader(ts_y, False, 1),
    )

    # ─── MODEL / OPTIMISER ────────────────────────────────
    model_cls = IOGNN_Z if edge_mode else IOGNN_VA
    model = model_cls(nfeat=3, cfg=cfg).to(cfg.device)

    pinn_fn = get_pinn_loss_function(kind)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    get_scaler = (
        lambda: scalers["edge_Z"] if edge_mode else scalers["node"]["value_added"]
    )

    # ─── LOGGERS ──────────────────────────────────────────
    keys = (
        "train_tot",
        "train_mse",
        "train_pinn",
        "train_R2",
        "val_tot",
        "val_mse",
        "val_pinn",
        "val_RMSE",
        "val_MAE",
        "val_SMAPE",
        "val_R2",
        "val_RHO",
        "val_CVR",
        "lambda_t",
    )
    hist: Dict[str, List[float]] = {k: [] for k in keys}

    best_metric, best_state, bad_epochs = float("inf"), None, 0
    global_step = 0

    # ════════════════════ TRAIN / VAL LOOP ════════════════════
    for ep in trange(1, cfg.epochs + 1, desc=f"{kind}-seed{seed}"):
        # ─── TRAIN ──────────────────────────────────────
        model.train()
        tot = mse_acc = pinn_acc = r2_acc = lam_acc = 0.0

        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, *_ = model(seqs, tgts)
            tgt_std = torch.cat(
                [g.edge_attr if edge_mode else g.va for g in tgts]
            )

            pinn = pinn_fn(pred_std, tgts, scalers)
            mse = F.mse_loss(pred_std, tgt_std)
            lam_t = _adaptive_lambda(
                mse=mse, pinn=pinn, cfg=cfg, global_step=global_step
            )

            loss = mse + lam_t * pinn
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            # ─── logging on‑the‑fly
            tot += loss.item(); mse_acc += mse.item(); pinn_acc += pinn.item()
            lam_acc += lam_t.item()

            pred_o = inverse_transform_predictions(pred_std, scalers, kind)
            tgt_o = inverse_transform_targets(tgt_std, scalers, kind)
            r2_acc += r2(pred_o, tgt_o)

            global_step += 1

        nb = len(tr_ld)
        lam_avg = lam_acc / nb
        hist["train_tot"].append(tot / nb)
        hist["train_mse"].append(mse_acc / nb)
        hist["train_pinn"].append(pinn_acc / nb)
        hist["train_R2"].append(r2_acc / nb)
        hist["lambda_t"].append(lam_avg)

        tqdm.write(
            f"[EP {ep:03d}] train: loss {tot/nb:.4f} | MSE {mse_acc/nb:.4f} | "
            f"PINN {pinn_acc/nb:.4f} | λ̄ {lam_avg:.3e} | R² {r2_acc/nb:.3f}"
        )

        # ─── VALIDATION ─────────────────────────────────
        model.eval(); v_tot = v_mse = v_pinn = 0.0
        acc = {k: [] for k in ("rmse", "mae", "smape", "r2", "RHO", "cvr")}

        with torch.no_grad():
            for seqs, tgts in vl_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [g.to(cfg.device) for g in tgts]

                pred_std, *_ = model(seqs, tgts)
                
                print("PINN batch:", pinn_fn(pred_std, tgts, scalers).item())
                
                '''if ep == 1:
                    m = pred_std.mean().item()
                    s = pred_std.std().item()
                    print(f"[VAL ep{ep}] pred_std mean={m:.4f}, std={s:.4f}")
                    print("pred_std.shape:", pred_std[:10].cpu())'''
                
                tgt_std = torch.cat(
                    [g.edge_attr if edge_mode else g.va for g in tgts]
                )

                pinn = pinn_fn(pred_std, tgts, scalers)
                mse = F.mse_loss(pred_std, tgt_std)

                v_tot += (mse + lam_avg * pinn).item()
                v_mse += mse.item(); v_pinn += pinn.item()

                pred_o = inverse_transform_predictions(pred_std, scalers, kind)
                tgt_o = inverse_transform_targets(tgt_std, scalers, kind)

                acc["rmse"].append(rmse(pred_o, tgt_o))
                acc["mae"].append(mae(pred_o, tgt_o))
                acc["smape"].append(smape(pred_o, tgt_o))
                acc["r2"].append(r2(pred_o, tgt_o))
                acc["RHO"].append(safe_pearson(pred_o, tgt_o))

                if edge_mode:
                    for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                        acc["cvr"].append(cvr_tensor_standardized(p_slice, g, scalers))

        nh = len(vl_ld)
        hist["val_tot"].append(v_tot / nh)
        hist["val_mse"].append(v_mse / nh)
        hist["val_pinn"].append(v_pinn / nh)
        hist["val_RMSE"].append(mean_ignore_nan(acc["rmse"]))
        hist["val_MAE"].append(mean_ignore_nan(acc["mae"]))
        hist["val_SMAPE"].append(mean_ignore_nan(acc["smape"]))
        hist["val_R2"].append(mean_ignore_nan(acc["r2"]))
        hist["val_RHO"].append(mean_ignore_nan(acc["RHO"]))
        hist["val_CVR"].append(mean_ignore_nan(acc["cvr"]) if edge_mode else np.nan)

        if ep % cfg.log_every == 0:
            extra = f"  CVR {hist['val_CVR'][-1]:.3e}" if edge_mode else ""
            tqdm.write(
                f"[VAL {ep:03d}] tot {hist['val_tot'][-1]:.4f} | "
                f"RMSE {hist['val_RMSE'][-1]:.2f}  MAE {hist['val_MAE'][-1]:.2f}  "
                f"SMAPE {hist['val_SMAPE'][-1]:.3f}  R² {hist['val_R2'][-1]:.3f}{extra}"
            )

        monitor = hist["val_SMAPE"][-1] if edge_mode else hist["val_MAE"][-1]
        if monitor < best_metric - 1e-8:
            best_metric, best_state, bad_epochs = (
                monitor,
                {k: v.cpu() for k, v in model.state_dict().items()},
                0,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stop @ {ep} (best={best_metric:.4f})"); break

    if best_state:
        model.load_state_dict(best_state)

    # ─── TEST ────────────────────────────────────────────
    model.eval(); res = {k: [] for k in ("rmse", "mae", "smape", "r2", "RHO", "cvr")}

    with torch.no_grad():
        for seqs, tgts in ts_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, att_out, att_in = model(seqs, tgts)
            tgt_std = torch.cat(
                [g.edge_attr if edge_mode else g.va for g in tgts]
            )

            pred_o = inverse_transform_predictions(pred_std, scalers, kind)
            tgt_o = inverse_transform_targets(tgt_std, scalers, kind)

            res["rmse"].append(rmse(pred_o, tgt_o))
            res["mae"].append(mae(pred_o, tgt_o))
            res["smape"].append(smape(pred_o, tgt_o))
            res["r2"].append(r2(pred_o, tgt_o))
            res["RHO"].append(safe_pearson(pred_o, tgt_o))

            if edge_mode:
                for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                    res["cvr"].append(cvr_tensor_standardized(p_slice, g, scalers))

                save_edge_attention(
                    att_out,
                    att_in,
                    tgts[0].edge_index,
                    tgts[0].num_nodes,
                    kind,
                    save_dir,
                )

    metrics = {k.upper(): mean_ignore_nan(v) for k, v in res.items()}
    if not edge_mode:
        metrics.pop("CVR", None)

    # ─── SAVE ARTEFACTS ──────────────────────────────────
    scalers_path = save_dir / "scalers.pkl"
    if not scalers_path.exists():
        pickle.dump(scalers, open(scalers_path, "wb"))

    dump_pred_matrices(
        model,
        scalers_path,
        years=ts_y,
        save_dir=save_dir,
        cfg=cfg,
        kind=kind,
        save_x=False,
    )

    torch.save(model.cpu().state_dict(), save_dir / "model.pth")
    (save_dir / "alpha.txt").write_text(f"{model.cell.Ox.alpha.item():.6f}")
    (save_dir / "val_history.json").write_text(
        json.dumps({k: list(map(float, v)) for k, v in hist.items()}, indent=2)
    )

    print("\n[Test]")
    for k, v in metrics.items():
        print(f"{k:<5}: {v:.4f}")

    return model, hist, None, metrics
