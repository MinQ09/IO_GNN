# run_single.py ───────────────────────────────────────────────────────────
"""
IO-GNN single-run trainer (Z / VA)

핵심 특징
---------
1.  PINN 은 *표준화 공간* 예측을 그대로 받아 계산합니다.  <--- 중요
2.  미니배치마다 ∧자율 λ_t = λ_max · MSE / (PINN+ε) 로 조절.
3.  학습 로그:   train  ➜  loss | MSE | PINN | λ̄ | R²
                val    ➜  tot  | RMSE | MAE | SMAPE | R² | (CVR)
4.  모든 메트릭은 **원 스케일**로 환산 후 계산.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json, pickle, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

from data_io  import GraphWindowDataset, collate_window, inverse_transform_1d
from metrics   import (rmse, mae, smape, r2, cvr_tensor_standardized,
                       mean_ignore_nan, safe_pearson)
from losses    import get_pinn_loss_function
from model     import IOGNN_Z, IOGNN_VA
from utils     import set_seed
from helper    import dump_pred_matrices, save_edge_attention
import torch_geometric.data as pyg

# ───────────── helper ──────────────
def _slice_batch(arr: torch.Tensor,
                 graphs: List[pyg.data.Data],
                 edge_mode: bool):
    """배치 concat-tensor ➜ (slice, graph) 반복자"""
    off = 0
    for g in graphs:
        span = g.edge_attr.numel() if edge_mode else g.num_nodes
        yield arr[off: off+span], g
        off += span

EPS_LAM = 1e-12        # λ 분모 안정화

# ───────────── main ────────────────
def run_single(cfg: Any, seed: int, *, kind: str = "Z"
               ) -> Tuple[torch.nn.Module, Dict[str, list], None, Dict[str, float]]:

    assert kind in {"Z", "VA"}
    set_seed(seed)
    edge_mode = (kind == "Z")

    save_dir = Path(cfg.out_dir) / f"seed_{seed}" / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) DATA ──────────────────────────────────────
    yrs = list(range(1, 67))
    tr_yrs, vl_yrs, ts_yrs = yrs[:-12], yrs[-12:-6], yrs[-6:]

    tr_ds   = GraphWindowDataset(tr_yrs, cfg, scalers=None, fit_scalers=True)
    scalers = tr_ds.get_scalers()          # 학습 세트에서 fit 완료
    if cfg.save_scalers:
        pickle.dump(scalers, open(cfg.scalers_path, "wb"))

    mk_loader = lambda Y, shuf, bs: DataLoader(
        GraphWindowDataset(Y, cfg, scalers, fit_scalers=False),
        batch_size=bs, shuffle=shuf,
        collate_fn=collate_window, pin_memory=True
    )

    tr_ld = mk_loader(tr_yrs, True,  cfg.batch_size)
    vl_ld = mk_loader(vl_yrs, False, cfg.batch_size)
    ts_ld = mk_loader(ts_yrs, False, 1)

    # 2) MODEL / LOSS ─────────────────────────────
    model   = (IOGNN_Z if edge_mode else IOGNN_VA)(nfeat=3, cfg=cfg).to(cfg.device)
    pinn_fn = get_pinn_loss_function(kind)      # <-- 표준화 PINN
    optim   = torch.optim.AdamW(model.parameters(),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler_edgeZ = scalers["edge_Z"]
    scaler_VA    = scalers["node"]["value_added"]
    get_scaler   = lambda: scaler_edgeZ if edge_mode else scaler_VA

    # 3) HISTORY ──────────────────────────────────
    hist_keys = ("train_tot","train_mse","train_pinn","train_R2",
                 "val_tot","val_mse","val_pinn",
                 "val_RMSE","val_MAE","val_SMAPE","val_R2","val_RHO","val_CVR",
                 "lambda_t")
    hist: Dict[str, List[float]] = {k: [] for k in hist_keys}
    best_metric, best_state, bad_epochs = float("inf"), None, 0

    # ════════════════ TRAIN / VAL LOOP ════════════════
    for ep in trange(1, cfg.epochs+1, desc=f"{kind}-seed{seed}"):
        model.train()
        tot=mse_sum=pinn_sum=r2_sum=lam_sum=0.0

        # ─── TRAIN ───
        for seqs, tgts in tr_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, *_ = model(seqs, tgts)          # (E) or (N), std-space
            tgt_std      = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

            # PINN 계산: **표준화 공간 예측 그대로 사용**
            pinn = pinn_fn(pred_std, tgts, scalers)     # grad X
            mse  = F.mse_loss(pred_std, tgt_std)

            # Adaptive λ_t (미니배치)
            lam_t = cfg.lambda_max * (mse.detach() /
                     (pinn.detach() + EPS_LAM)).clamp(max=1.0)

            loss = mse + lam_t * pinn
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            # ---- logging ----
            tot += loss.item(); mse_sum += mse.item()
            pinn_sum += pinn.item(); lam_sum += lam_t.item()

            # R² (원 스케일) ---------------------------------
            pred_orig = inverse_transform_1d(pred_std, get_scaler())
            tgt_orig  = inverse_transform_1d(tgt_std,  get_scaler())
            r2_sum   += r2(pred_orig, tgt_orig)

        nb = len(tr_ld); lam_avg = lam_sum/nb
        hist["train_tot"].append(tot/nb);   hist["train_mse"].append(mse_sum/nb)
        hist["train_pinn"].append(pinn_sum/nb); hist["train_R2"].append(r2_sum/nb)
        hist["lambda_t"].append(lam_avg)

        tqdm.write(f"[EP {ep:03d}] train: "
                   f"loss {tot/nb:.4f} | MSE {mse_sum/nb:.4f} | "
                   f"PINN {pinn_sum/nb:.4f} | λ̄ {lam_avg:.3e} | "
                   f"R² {r2_sum/nb:.3f}")

        # ─── VALIDATE ───
        model.eval(); v_tot=v_mse=v_pinn=0.0
        acc = {k: [] for k in ("rmse","mae","smape","r2","rho","cvr")}
        with torch.no_grad():
            for seqs, tgts in vl_ld:
                seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                tgts = [g.to(cfg.device) for g in tgts]

                pred_std, *_ = model(seqs, tgts)
                tgt_std      = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

                pinn = pinn_fn(pred_std, tgts, scalers)
                mse  = F.mse_loss(pred_std, tgt_std)

                v_tot += (mse + lam_avg * pinn).item()
                v_mse += mse.item(); v_pinn += pinn.item()

                pred_orig = inverse_transform_1d(pred_std, get_scaler())
                tgt_orig  = inverse_transform_1d(tgt_std,  get_scaler())

                acc["rmse"].append(rmse(pred_orig, tgt_orig))
                acc["mae" ].append(mae (pred_orig, tgt_orig))
                acc["smape"].append(smape(pred_orig, tgt_orig))
                acc["r2"   ].append(r2   (pred_orig, tgt_orig))
                acc["rho"  ].append(safe_pearson(pred_orig, tgt_orig))

                if edge_mode:
                    for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                        acc["cvr"].append(cvr_tensor_standardized(p_slice, g, scalers))

        nh = len(vl_ld)
        hist["val_tot" ].append(v_tot/nh);   hist["val_mse" ].append(v_mse/nh)
        hist["val_pinn"].append(v_pinn/nh)
        hist["val_RMSE"].append(mean_ignore_nan(acc["rmse"]))
        hist["val_MAE" ].append(mean_ignore_nan(acc["mae" ]))
        hist["val_SMAPE"].append(mean_ignore_nan(acc["smape"]))
        hist["val_R2"  ].append(mean_ignore_nan(acc["r2"  ]))
        hist["val_RHO" ].append(mean_ignore_nan(acc["rho" ]))
        hist["val_CVR" ].append(mean_ignore_nan(acc["cvr" ]) if edge_mode else np.nan)

        if ep % cfg.log_every == 0:
            add = f"  CVR {hist['val_CVR'][-1]:.3e}" if edge_mode else ""
            tqdm.write(f"[VAL {ep:03d}] "
                       f"tot {hist['val_tot'][-1]:.4f} | "
                       f"RMSE {hist['val_RMSE'][-1]:.2f}  "
                       f"MAE {hist['val_MAE'][-1]:.2f}  "
                       f"SMAPE {hist['val_SMAPE'][-1]:.3f}  "
                       f"R² {hist['val_R2'][-1]:.3f}{add}")

        # Early-stop
        monitor = hist["val_SMAPE"][-1] if edge_mode else hist["val_MAE"][-1]
        if monitor < best_metric - 1e-8:
            best_metric, best_state, bad_epochs = monitor, \
                {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stop @ {ep} (best={best_metric:.4f})")
                break

    # best 가중치 복원
    if best_state:
        model.load_state_dict(best_state)

    # 4) TEST ───────────────────────────────────────
    model.eval(); res = {k: [] for k in ("rmse","mae","smape","r2","rho","cvr")}
    with torch.no_grad():
        for seqs, tgts in ts_ld:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [g.to(cfg.device) for g in tgts]

            pred_std, att_out, att_in = model(seqs, tgts)
            tgt_std  = torch.cat([g.edge_attr if edge_mode else g.va for g in tgts])

            pred_orig = inverse_transform_1d(pred_std, get_scaler())
            tgt_orig  = inverse_transform_1d(tgt_std,  get_scaler())

            res["rmse"].append(rmse(pred_orig,tgt_orig))
            res["mae" ].append(mae (pred_orig,tgt_orig))
            res["smape"].append(smape(pred_orig,tgt_orig))
            res["r2"   ].append(r2   (pred_orig,tgt_orig))
            res["rho"  ].append(safe_pearson(pred_orig,tgt_orig))

            if edge_mode:
                for p_slice, g in _slice_batch(pred_std.cpu(), tgts, True):
                    res["cvr"].append(cvr_tensor_standardized(p_slice, g, scalers))
                save_edge_attention(att_out, att_in,
                                    tgts[0].edge_index, tgts[0].num_nodes,
                                    kind, save_dir)

    metrics = {k.upper(): mean_ignore_nan(v) for k, v in res.items()}
    if not edge_mode:
        metrics.pop("CVR", None)

    # 5) SAVE ───────────────────────────────────────
    dump_pred_matrices(model, scalers, years=ts_yrs,
                       save_dir=save_dir, kind=kind, save_x=False)
    torch.save(model.cpu().state_dict(), save_dir / "model.pth")
    (save_dir / "alpha.txt").write_text(f"{model.cell.Ox.alpha.item():.6f}")
    (save_dir / "val_history.json").write_text(
        json.dumps({k:[float(x) for x in v] for k,v in hist.items()},
                   indent=2)
    )

    print("\n[Test]")
    for k, v in metrics.items():
        print(f"{k:<5}: {v:.4f}")

    return model, hist, None, metrics