# run_single.py  ──────────────────────────────────────────────────────────
"""
Train / validate / test a single run of IO-GNN.

Kind switch
-----------
kind = "Z"  :  edge-flow prediction  (IOGNN_Z + PINN_Z)
kind = "VA" :  node Value-Added      (IOGNN_VA + PINN_VA)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict, Any

import json, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from scipy.stats import pearsonr

from data   import GraphWindowDataset, collate_window
from losses import pinn_loss_z_batch, pinn_loss_va_batch
from metrics import rmse, mae, smape, r2, cvr_tensor
from utils   import set_seed
from model      import IOGNN_Z, IOGNN_VA
from helper import save_edge_attention, dump_pred_matrices   
# ──────────────────────────────────────────────────────────
def run_single(cfg: Any, seed: int, *, kind: str = "Z"
               ) -> Tuple[torch.nn.Module, Dict[str, list], None, Dict[str, float]]:
    """Train one model (edge or VA) and return trained model & metrics."""
    assert kind in {"Z", "VA"}, "`kind` must be 'Z' or 'VA'"
    set_seed(seed)

    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / f"{kind}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ───── data loaders ─────
    years       = list(range(1, 67))
    train_years = years[:-12]   # 1–54
    val_years   = years[-12:-6] # 55–60
    test_years  = years[-6:]    # 61–66

    def make_loader(y, shuffle, bs):
        return DataLoader(GraphWindowDataset(y, cfg),
                          batch_size=bs, shuffle=shuffle,
                          collate_fn=collate_window, pin_memory=False)

    train_ld = make_loader(train_years, True,  cfg.batch_size)
    val_ld   = make_loader(val_years,   False, cfg.batch_size)
    test_ld  = make_loader(test_years,  False, 1)

    # ───── model / loss helpers ─────
    if kind == "Z":
        model  = IOGNN_Z(nfeat=3, cfg=cfg).to(cfg.device)
        pinn   = pinn_loss_z_batch
        raw_fn = lambda x: torch.expm1(x) * cfg.scale_Z
    else:  # VA
        model  = IOGNN_VA(nfeat=3, cfg=cfg).to(cfg.device)
        pinn   = pinn_loss_va_batch
        raw_fn = lambda x: x * cfg.scale_node

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # ───── history dict ─────
    hist: Dict[str, list] = {k: [] for k in (
        "train_tot","train_mse","train_pinn","train_R2",
        "val_tot","val_mse","val_pinn",
        "val_RMSE","val_MAE","val_SMAPE","val_R2","val_RHO","val_CVR",
        "lam_t"
    )}

    best_metric = float("inf")
    best_state, bad_epochs = None, 0

    # ───────────── training loop ─────────────
    for ep in trange(1, cfg.epochs + 1, desc=f"{kind}-seed{seed}"):
        lam_t = cfg.lambda_max * min(ep / cfg.warmup, 1.0)
        model.train()

        tot_epoch = mse_epoch = pinn_epoch = r2_sum = 0.0
        for seqs, tgts in train_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [t.to(cfg.device) for t in tgts]

            preds, *_ = model(seqs, tgts)
            targets = torch.cat([t.edge_attr if kind=="Z" else t.va
                                 for t in tgts])

            mse    = F.mse_loss(preds, targets)
            pinn_l = pinn(raw_fn(preds), tgts, cfg.scale_node)
            loss   = mse + lam_t * pinn_l

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            tot_epoch += loss.item()
            mse_epoch += mse.item()
            pinn_epoch+= pinn_l.item()

            r2_sum += r2(raw_fn(preds), raw_fn(targets))

        hist["train_tot"].append(tot_epoch / len(train_ld))
        hist["train_mse"].append(mse_epoch / len(train_ld))
        hist["train_pinn"].append(pinn_epoch / len(train_ld))
        hist["train_R2"].append(r2_sum / len(train_ld))
        hist["lam_t"].append(lam_t)

        tqdm.write(
            f"[{kind}] Ep {ep:03d}  "
            f"loss {tot_epoch/len(train_ld):.4f}  "
            f"MSE {mse_epoch/len(train_ld):.4f}  "
            f"PINN {pinn_epoch/len(train_ld):.4f}  "
            f"R2 {r2_sum/len(train_ld):.3f}"
        )


        # ───── validation ─────
        model.eval()
        val_tot = val_mse = val_pinn = 0.0
        metrics_acc = {k: [] for k in ("rmse","mae","smape","r2","rho","cvr")}
        with torch.no_grad():
            for seqs, tgts in val_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [t.to(cfg.device) for t in tgts]

                preds, *_ = model(seqs, tgts)
                targets = torch.cat([t.edge_attr if kind=="Z" else t.va
                                     for t in tgts])

                mse    = F.mse_loss(preds, targets)
                pinn_l = pinn(raw_fn(preds), tgts, cfg.scale_node)

                val_tot  += (mse + lam_t * pinn_l).item()
                val_mse  += mse.item()
                val_pinn += pinn_l.item()

                p_raw = raw_fn(preds)
                t_raw = raw_fn(targets)
                metrics_acc["rmse"].append(rmse(p_raw, t_raw))
                metrics_acc["mae"].append(mae(p_raw, t_raw))
                metrics_acc["smape"].append(smape(p_raw, t_raw))
                metrics_acc["r2"].append(r2(p_raw, t_raw))
                metrics_acc["rho"].append(
                    pearsonr(p_raw.cpu().numpy().ravel(),
                             t_raw.cpu().numpy().ravel())[0])

                if kind == "Z":
                    off = 0
                    for g in tgts:
                        e = g.edge_attr.numel()
                        metrics_acc["cvr"].append(cvr_tensor(p_raw[off:off+e], g,
                                                             cfg.scale_node))
                        off += e

        # log averages
        hist["val_tot"].append(val_tot / len(val_ld))
        hist["val_mse"].append(val_mse / len(val_ld))
        hist["val_pinn"].append(val_pinn / len(val_ld))
        hist["val_RMSE"].append(np.mean(metrics_acc["rmse"]))
        hist["val_MAE"].append(np.mean(metrics_acc["mae"]))
        hist["val_SMAPE"].append(np.mean(metrics_acc["smape"]))
        hist["val_R2"].append(np.mean(metrics_acc["r2"]))
        hist["val_RHO"].append(np.mean(metrics_acc["rho"]))
        hist["val_CVR"].append(np.mean(metrics_acc["cvr"]) if kind=="Z" else np.nan)

        # simple early-stopping based on SMAPE / MAE
        key_metric = hist["val_SMAPE"][-1] if kind=="Z" else hist["val_MAE"][-1]
        if key_metric < best_metric - 1e-8:
            best_metric, best_state, bad_epochs = key_metric, \
                {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stop at ep {ep} (best metric={best_metric:.4f})")
                break

    # ───── restore best weights ─────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ───── test loop (same pattern) ─────
    model.eval()
    rmse_t, mae_t, smape_t, r2_t, rho_t, cvr_t = ([] for _ in range(6))
    with torch.no_grad():
        for year_idx, (seqs, tgts) in enumerate(test_ld, 1):
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [t.to(cfg.device) for t in tgts]

            preds, att_out, att_in = model(seqs, tgts)
            targets = torch.cat([t.edge_attr if kind=="Z" else t.va
                                 for t in tgts])

            p_raw = raw_fn(preds); t_raw = raw_fn(targets)

            rmse_t.append(rmse(p_raw, t_raw))
            mae_t.append(mae (p_raw, t_raw))
            smape_t.append(smape(p_raw, t_raw))
            r2_t.append(r2(p_raw, t_raw))
            rho_t.append(pearsonr(p_raw.cpu().numpy().ravel(),
                                  t_raw.cpu().numpy().ravel())[0])
            if kind=="Z":
                off=0
                for g in tgts:
                    e=g.edge_attr.numel()
                    cvr_t.append(cvr_tensor(p_raw[off:off+e], g, cfg.scale_node))
                    off+=e
                save_edge_attention(att_out.cpu(), att_in.cpu(),
                                    tgts[0].edge_index.cpu(), tgts[0].num_nodes,
                                    f"{kind}_{year_idx:03d}", save_dir)

    metrics = {
        "RMSE":  np.mean(rmse_t),
        "MAE":   np.mean(mae_t),
        "SMAPE": np.mean(smape_t),
        "R2":    np.mean(r2_t),
        "rho":   np.mean(rho_t),
        "CVR":   np.mean(cvr_t) if kind=="Z" else np.nan
    }

    # ───── dump artefacts ─────
    dump_pred_matrices(model, cfg, years=test_years,
                       save_dir=save_dir, kind=kind, save_x=False)
    torch.save(model.cpu().state_dict(), save_dir / "model.pth")

    # alpha value (attention mixing coeff)
    alpha = model.cell.Ox.alpha.item()
    (save_dir / "alpha.txt").write_text(f"{alpha:.6f}")

    with open(save_dir / "val_history.json", "w") as f:
        json.dump({k: list(map(float, v)) for k, v in hist.items()},
                  f, indent=2)

    # ───── summary printout ─────
    print(f"\n[Test {kind}]")
    for k, v in metrics.items():
        print(f"{k:<5s}: {v:.4f}")

    return model, hist, None, metrics
