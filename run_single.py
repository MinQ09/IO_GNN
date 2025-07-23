# run_single.py ─────────────────────────────────────────────────────────
"""
Single-run trainer for IO-GNN (“Z”: edge flows, “VA”: node value-added)

추가 사항
---------
• mini-batch λₜ = scale(t) · λ_max · MSE /(PINN+ε)
  └ scale(t)=min(1, global_step / warmup)   (warm-up 스케일러)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json, pickle, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm
import torch_geometric.data as pyg

from data_io import GraphWindowDataset, collate_window, inverse_transform_1d
from metrics  import rmse, mae, smape, r2, cvr_tensor_standardized, \
                     mean_ignore_nan, safe_pearson
from losses   import get_pinn_loss_function
from model    import IOGNN_Z, IOGNN_VA
from utils    import set_seed
from helper   import dump_pred_matrices, save_edge_attention

# ───────────────────────── helpers ─────────────────────────
def _slice_batch(cat: torch.Tensor, graphs: List[pyg.data.Data], edge: bool):
    off = 0
    for g in graphs:
        span = g.edge_attr.numel() if edge else g.num_nodes
        yield cat[off : off + span], g
        off += span


EPS_LAM = 1e-12               # for adaptive λₜ denominator

# ─────────────────────────── main ──────────────────────────
def run_single(
    cfg: Any, seed: int, *, kind: str = "Z"
) -> Tuple[torch.nn.Module, Dict[str, list], None, Dict[str, float]]:

    assert kind in {"Z", "VA"}
    set_seed(seed)
    edge_mode = kind == "Z"

    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) DATA ───────────────────────────────────────────────
    years          = list(range(1, 67))
    tr_y, vl_y, ts_y = years[:-12], years[-12:-6], years[-6:]

    tr_ds  = GraphWindowDataset(tr_y, cfg, None, fit_scalers=True)
    scalers = tr_ds.get_scalers()
    if cfg.save_scalers:
        pickle.dump(scalers, open(cfg.scalers_path, "wb"))

    mk_loader = lambda Y, shuf, bs: DataLoader(
        GraphWindowDataset(Y, cfg, scalers, fit_scalers=False),
        batch_size=bs, shuffle=shuf, collate_fn=collate_window, pin_memory=True
    )
    tr_ld, vl_ld, ts_ld = (mk_loader(tr_y, True, cfg.batch_size),
                           mk_loader(vl_y, False, cfg.batch_size),
                           mk_loader(ts_y, False, 1))

    # 2) MODEL / LOSS ───────────────────────────────────────
    model   = (IOGNN_Z if edge_mode else IOGNN_VA)(nfeat=3, cfg=cfg).to(cfg.device)
    pinn_fn = get_pinn_loss_function(kind)
    optim   = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    scaler_Z  = scalers["edge_Z"]
    scaler_VA = scalers["node"]["value_added"]
    get_scaler = lambda: scaler_Z if edge_mode else scaler_VA

    # 3) LOGGING SETUP ──────────────────────────────────────
    keys = ("train_tot","train_mse","train_pinn","train_R2",
            "val_tot","val_mse","val_pinn",
            "val_RMSE","val_MAE","val_SMAPE","val_R2","val_RHO","val_CVR",
            "lambda_t")
    hist: Dict[str, List[float]] = {k: [] for k in keys}
    best_metric, best_state, bad_epochs = float("inf"), None, 0

    # ───── warm-up state ─────
    global_step = 0

    # ═══════════════ Train / Validate ═══════════════
    for ep in trange(1, cfg.epochs + 1, desc=f"{kind}-seed{seed}"):
        # ───────── TRAIN ─────────
        model.train()
        tot = mse_sum = pinn_sum = r2_sum = lam_sum = 0.0

        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, *_ = model(seqs, tgts)
            tgt_std      = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

            pinn = pinn_fn(pred_std, tgts, scalers)
            mse  = F.mse_loss(pred_std, tgt_std)

            # ─── adaptive λₜ with warm-up ───
            scale = 1.0
            if cfg.warmup:                     # 0 → warm-up off
                scale = min(1.0, global_step / cfg.warmup)
            lam_t = scale * cfg.lambda_max * (mse.detach() /
                    (pinn.detach() + EPS_LAM)).clamp(max=1.0)

            loss = mse + lam_t * pinn
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            # logging
            tot += loss.item(); mse_sum += mse.item()
            pinn_sum += pinn.item(); lam_sum += lam_t.item()

            pred_o = inverse_transform_1d(pred_std, get_scaler())
            tgt_o  = inverse_transform_1d(tgt_std,  get_scaler())
            r2_sum += r2(pred_o, tgt_o)

            global_step += 1                   # ↑ warm-up step advance

        nb = len(tr_ld); lam_avg = lam_sum / nb
        hist["train_tot"].append(tot/nb); hist["train_mse"].append(mse_sum/nb)
        hist["train_pinn"].append(pinn_sum/nb); hist["train_R2"].append(r2_sum/nb)
        hist["lambda_t"].append(lam_avg)

        tqdm.write(f"[EP {ep:03d}] train: "
                   f"loss {tot/nb:.4f} | MSE {mse_sum/nb:.4f} | "
                   f"PINN {pinn_sum/nb:.4f} | λ̄ {lam_avg:.3e} | R² {r2_sum/nb:.3f}")

        # ───────── VALIDATION ─────────
        model.eval(); v_tot=v_mse=v_pinn=0.0
        acc = {k: [] for k in ("rmse","mae","smape","r2","rho","cvr")}
        with torch.no_grad():
            for seqs, tgts in vl_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [g.to(cfg.device) for g in tgts]

                pred_std, *_ = model(seqs, tgts)
                tgt_std      = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

                pinn = pinn_fn(pred_std, tgts, scalers); mse = F.mse_loss(pred_std, tgt_std)
                v_tot += (mse + lam_avg * pinn).item()
                v_mse += mse.item(); v_pinn += pinn.item()

                pred_o = inverse_transform_1d(pred_std, get_scaler())
                tgt_o  = inverse_transform_1d(tgt_std,  get_scaler())
                acc["rmse"].append(rmse(pred_o, tgt_o)); acc["mae"].append(mae(pred_o, tgt_o))
                acc["smape"].append(smape(pred_o, tgt_o)); acc["r2"].append(r2(pred_o, tgt_o))
                acc["rho"].append(safe_pearson(pred_o, tgt_o))

                if edge_mode:
                    for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                        acc["cvr"].append(cvr_tensor_standardized(p_slice, g, scalers))

        nh = len(vl_ld)
        hist["val_tot"].append(v_tot/nh); hist["val_mse"].append(v_mse/nh)
        hist["val_pinn"].append(v_pinn/nh)
        hist["val_RMSE"].append(mean_ignore_nan(acc["rmse"]))
        hist["val_MAE"].append(mean_ignore_nan(acc["mae"]))
        hist["val_SMAPE"].append(mean_ignore_nan(acc["smape"]))
        hist["val_R2"].append(mean_ignore_nan(acc["r2"]))
        hist["val_RHO"].append(mean_ignore_nan(acc["rho"]))
        hist["val_CVR"].append(mean_ignore_nan(acc["cvr"]) if edge_mode else np.nan)

        if ep % cfg.log_every == 0:
            extra = f"  CVR {hist['val_CVR'][-1]:.3e}" if edge_mode else ""
            tqdm.write(f"[VAL {ep:03d}] tot {hist['val_tot'][-1]:.4f} | "
                       f"RMSE {hist['val_RMSE'][-1]:.2f}  MAE {hist['val_MAE'][-1]:.2f}  "
                       f"SMAPE {hist['val_SMAPE'][-1]:.3f}  R² {hist['val_R2'][-1]:.3f}{extra}")

        # early-stop
        monitor = hist["val_SMAPE"][-1] if edge_mode else hist["val_MAE"][-1]
        if monitor < best_metric - 1e-8:
            best_metric, best_state, bad_epochs = monitor, \
                {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stop @ {ep} (best={best_metric:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    # 4) TEST ──────────────────────────────────────────────
    model.eval()
    res = {k: [] for k in ("rmse", "mae", "smape", "r2", "rho", "cvr")}

    with torch.no_grad():
        for seqs, tgts in test_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, att_out, att_in = model(seqs, tgts)
            tgt_std = torch.cat(
                [g.edge_attr if edge_mode else g.va for g in tgts]
            )

            pred_orig = inverse_transform_1d(pred_std, get_scaler())
            tgt_orig  = inverse_transform_1d(tgt_std,  get_scaler())

            res["rmse"].append(rmse(pred_orig, tgt_orig))
            res["mae" ].append(mae (pred_orig, tgt_orig))
            res["smape"].append(smape(pred_orig, tgt_orig))
            res["r2"   ].append(r2   (pred_orig, tgt_orig))
            res["rho"  ].append(safe_pearson(pred_orig, tgt_orig))

            if edge_mode:
                for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                    res["cvr"].append(
                        cvr_tensor_standardized(p_slice, g, scalers)
                    )
                save_edge_attention(
                    att_out, att_in,
                    tgts[0].edge_index, tgts[0].num_nodes,
                    kind, save_dir,
                )

    metrics = {k.upper(): mean_ignore_nan(v) for k, v in res.items()}
    if not edge_mode:
        metrics.pop("CVR", None)

    # 5) SAVE ARTEFACTS ────────────────────────────────────
    scalers_path = save_dir / "scalers.pkl"
    if not scalers_path.exists():
        pickle.dump(scalers, open(scalers_path, "wb"))

    dump_pred_matrices(
        model,
        scalers_path,      # pass path, not dict
        years=test_y,
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