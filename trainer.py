import numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from utils import set_seed
from data import GraphWindowDataset, collate_window
from model import IOGNN
from losses import smape, pinn_loss_batch

def split_years(first, last, window, val_len, test_len):
    years = list(range(first, last + 1))
    test = years[-test_len:]
    val  = years[-test_len-window-val_len:-test_len]
    train = years[: -test_len-window-val_len-window]
    return train, val, test

def evaluate(loader, model, cfg):
    model.eval(); sm = []
    with torch.no_grad():
        for seqs, tgts in loader:
            seqs = [[g.to(cfg.device) for g in s] for s in seqs]
            tgts = [t.to(cfg.device) for t in tgts]
            p, _ = model(seqs, tgts)
            t = torch.cat([g.edge_attr for g in tgts])
            p_raw = torch.expm1(p) * cfg.scale_Z
            t_raw = torch.expm1(t) * cfg.scale_Z
            sm.append(smape(p_raw, t_raw))
    return float(np.mean(sm))

def run(cfg, seed):
    set_seed(seed)
    best_model, best_val, best_pair = None, 1e9, None
    for lam in cfg.lambda_candidates:
        for beta in cfg.beta_candidates:
            train_years, val_years, test_years = split_years(1, 66, cfg.window, 5, 5)
            train_ds = GraphWindowDataset(train_years, cfg)
            val_ds   = GraphWindowDataset(val_years,   cfg)
            train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  collate_fn=collate_window)
            val_ld   = DataLoader(val_ds,  batch_size=cfg.batch_size, shuffle=False,
                                  collate_fn=collate_window)
            model = IOGNN(3, cfg).to(cfg.device)
            opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            best, wait = 1e9, 0
            for ep in range(1, cfg.epochs+1):
                lam_t = lam * min(ep/cfg.warmup, 1.0)
                model.train()
                for seqs, tgts in train_ld:
                    seqs = [[g.to(cfg.device) for g in s] for s in seqs]
                    tgts = [t.to(cfg.device) for t in tgts]
                    p_z, p_x = model(seqs, tgts)
                    t_z = torch.cat([g.edge_attr for g in tgts])
                    t_x = torch.cat([g.tot for g in tgts])
                    loss = (torch.nn.functional.mse_loss(p_z, t_z) +
                            beta*torch.nn.functional.mse_loss(p_x, t_x) +
                            lam_t*pinn_loss_batch(torch.expm1(p_z)*cfg.scale_Z,
                                                  tgts, cfg.scale_node,x_override=p_x))
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                val_sm = evaluate(val_ld, model, cfg)
                if val_sm < best-1e-6:
                    best, wait = val_sm, 0
                    best_state = {k: v.cpu() for k,v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= cfg.patience: break
            if best < best_val:
                best_val, best_pair = best, (lam, beta)
                best_model = model
                best_model.load_state_dict(best_state)
    return best_model, best_pair, best_val