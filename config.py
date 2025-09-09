# config.py — improved (patched)
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional
import itertools
import yaml
import torch


@dataclass
class Config:
    """Top-level configuration for all IO-GNN runs."""

    # ───────── Base hyper-params ─────────
    batch_size: int = 64
    lr: float = 0.005                         # Effective lr in run_single is lr*0.2; keep this and use scheduler to decay
    weight_decay: float = 5e-4
    epochs: int = 1000                        # Longer training for stability (was 500)
    patience: int = 200                       # Tighter early stop (was 300)
    seeds: List[int] = field(default_factory=lambda: [123])

    # ───────── Paths ─────────
    data_dir: Path = Path("/Users/mingyu/Desktop/IO_GNN-main/Data")
    out_dir: Path = Path("/Users/mingyu/Desktop/IO_GNN-main/Results/V71")
    scalers_fname: str = "scalers.pkl"

    # ───────── Scaling & window ─────────
    window: int = 2
    scale_node_feats: bool = True
    scale_targets: bool = True
    save_scalers: bool = True

    # ───────── Rolling-Window CV ─────────
    rolling_val: bool = False
    rolling_splits: int = 5
    rolling_train_size: int | None = None
    rolling_test_size: int = 1
    rolling_gap: int = 0
    fold_epochs: int = 5  # reserved for RW-CV inner training epochs

    # ───────── PINN / multi-task ─────────
    lambda_max: float = 2.0                   # Slightly larger cap to avoid vanishing adaptive λ
    warmup: int = 50
    beta_x: float = 0.0
    beta_init: float = 0.1

    # Full-forward PINN cadence (1 = every batch). Larger = faster/approximate.
    pinn_full_every: int = 5

    # ───────── LR scheduler (recommended) ─────────
    # These are read/used by run_single.py when you add ReduceLROnPlateau.
    use_plateau_scheduler: bool = True
    plateau_factor: float = 0.5
    plateau_patience: int = 20
    plateau_min_lr: float = 1e-5
    plateau_metric: str = "val_tot"          # Which metric to watch (matches run_single logging keys)

    # ───────── Model structure ─────────
    hidden: int = 512
    k: int = 5
    dropout: float = 0.3
    alpha: float = 0.7
    att_hidden: int = 256
    depth_edge: int = 3

    # ───────── Graph/edge options (DirMPNN / GraphLSTMCell) ─────────
    # If use_edge_weight=True and edge_feat_dim is None, the model assumes scalar edges (dim=1).
    edge_feat_dim: Optional[int] = None       # None -> infer (usually 1 if use_edge_weight=True)
    use_edge_weight: bool = True              # Include edge_attr in message passing
    use_bwd_weights: bool = False             # Use edge_attr_bwd on backward pass
    compute_attention: bool = True            # Compute analysis-only attention scores (no-grad)
    alpha_mode: str = "scalar"                # 'scalar' or 'channel' mixing of fwd/bwd
    va_nonneg: bool = True                    # Apply Softplus head for VA prediction

    # Directional normalization & gating
    use_row_norm: bool = True                 # Source row-normalization (1/out-degree) per edge
    use_edge_mul: bool = True                 # Multiplicative edge gating (scalar multiply or learned gate)
    residual_scale: float = 0.1               # Residual skip scale for h in GraphLSTMCell (e.g., 0.1 ~ 1.0)

    # ───────── Sweep (non-grid) ─────────
    lambda_candidates: List[float] = field(default_factory=lambda: [0, 1])
    beta_candidates:   List[float] = field(default_factory=lambda: [0.0])

    # ───────── Grid-search flags/axes ─────────
    grid_search: bool = False
    batch_size_candidates:   List[int]   = field(default_factory=lambda: [32, 64])
    lr_candidates:           List[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4])
    weight_decay_candidates: List[float] = field(default_factory=lambda: [1e-5, 5e-5])
    hidden_candidates:       List[int]   = field(default_factory=lambda: [256, 512])
    k_candidates:            List[int]   = field(default_factory=lambda: [3, 5])
    dropout_candidates:      List[float] = field(default_factory=lambda: [0.1, 0.3])
    lambda_grid_candidates:  List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5])
    scale_targets_candidates: List[bool] = field(default_factory=lambda: [False, True])

    # ───────── Misc ─────────
    log_every: int = 10
    device: str = field(init=False, repr=False)

    # ================= Initialization =================
    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._validate()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.grid_search:
            n = self.count_grid_combinations()
            print(f"▶ Grid Search: {n:,} combos")
            if n > 10_000:
                print("⚠️  Warning: very large grid; consider pruning axes or using random search.")

    # ================= Derived paths ==================
    @property
    def scalers_path(self) -> Path:
        return self.out_dir / self.scalers_fname

    # ================= Validation =====================
    def _validate(self) -> None:
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.lr > 0, "lr must be > 0"
        assert self.weight_decay >= 0, "weight_decay must be ≥ 0"
        assert self.epochs > 0 and self.patience >= 0, "epochs>0 and patience≥0 required"
        assert self.hidden > 0, "hidden must be > 0"
        assert self.k >= 1, "k (Chebyshev order) must be ≥ 1"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
        assert self.lambda_max >= 0, "lambda_max must be ≥ 0"
        assert self.warmup >= 0, "warmup must be ≥ 0"
        assert self.att_hidden > 0 and self.depth_edge >= 1, "invalid attention/depth settings"
        assert self.window >= 1, "window length must be ≥ 1"
        assert self.alpha_mode in {"scalar", "channel"}, "alpha_mode must be 'scalar' or 'channel'"
        assert self.residual_scale >= 0.0, "residual_scale must be ≥ 0.0"

        if self.edge_feat_dim is not None:
            assert isinstance(self.edge_feat_dim, int) and self.edge_feat_dim >= 1, \
                "edge_feat_dim must be None or an integer ≥ 1"

        if self.use_plateau_scheduler:
            assert self.plateau_factor > 0 and self.plateau_factor < 1.0, "plateau_factor must be in (0,1)"
            assert self.plateau_patience >= 1, "plateau_patience must be ≥ 1"
            assert self.plateau_min_lr > 0, "plateau_min_lr must be > 0"
            assert self.plateau_metric in {
                "val_tot", "val_RMSE", "val_MAE", "val_SMAPE", "val_R2", "val_RHO", "val_IOIS"
            }, "plateau_metric must match a logged validation key"

        if self.rolling_val:
            assert self.rolling_splits >= 2, "rolling_splits must be ≥ 2 when rolling_val=True"
            assert self.rolling_test_size >= 1, "rolling_test_size must be ≥ 1"
            if self.rolling_train_size is not None:
                assert self.rolling_train_size >= 1, "rolling_train_size must be None or ≥ 1"
            assert self.rolling_gap >= 0, "rolling_gap must be ≥ 0"

    # ================= Pretty identifiers =============
    def get_param_string(self) -> str:
        lr_s = f"{float(self.lr):.0e}"
        wd_s = f"{float(self.weight_decay):.0e}"
        lam_s = f"{float(self.lambda_max):.3g}"
        raw = int(self.scale_targets is False)
        return f"bs{self.batch_size}_lr{lr_s}_wd{wd_s}_h{self.hidden}_k{self.k}_dp{self.dropout}_lam{lam_s}_raw{raw}"

    def run_tag(self) -> str:
        """Stable short tag for filesystem/logging."""
        return self.get_param_string()

    # ================= Grid utilities =================
    def count_grid_combinations(self) -> int:
        return (
            len(self.batch_size_candidates)
            * len(self.lr_candidates)
            * len(self.weight_decay_candidates)
            * len(self.hidden_candidates)
            * len(self.k_candidates)
            * len(self.dropout_candidates)
            * len(self.lambda_grid_candidates)
            * len(self.scale_targets_candidates)
            * len(self.seeds)
        )

    def generate_grid_configs(self) -> List["Config"]:
        """Return a list of immutable Config objects – one per grid point."""
        if not self.grid_search:
            # For non-grid runs, keep a single normalized copy (device will be recomputed in __post_init__)
            base = asdict(self)
            base.pop("device", None)
            return [Config(**base)]

        combos = itertools.product(
            self.batch_size_candidates,
            self.lr_candidates,
            self.weight_decay_candidates,
            self.hidden_candidates,
            self.k_candidates,
            self.dropout_candidates,
            self.lambda_grid_candidates,
            self.scale_targets_candidates,
            self.seeds,
        )

        configs: List[Config] = []
        for i, (bs, lr, wd, h, k, dp, lam, scale_tg, seed) in enumerate(combos):
            cfg = replace(
                self,
                grid_search=False,              # materialize each config
                batch_size=bs,
                lr=lr,
                weight_decay=wd,
                hidden=h,
                k=k,
                dropout=dp,
                lambda_max=lam,
                scale_targets=scale_tg,
                seeds=[seed],
                out_dir=(self.out_dir / f"grid_{i:03d}_{h}h_{k}k_dp{dp}_{self._lr_tag(lr)}_lam{lam}"),
            )
            # Re-run post init validations and ensure dir exists
            cfg.__post_init__()
            configs.append(cfg)
        return configs

    @staticmethod
    def _lr_tag(lr: float) -> str:
        try:
            return f"{float(lr):.0e}"
        except Exception:
            return str(lr)

    # ================= YAML I/O =======================
    def to_dict_for_save(self) -> Dict:
        """Drop transient fields (e.g., device) and stringify Paths for YAML."""
        d = {k: v for k, v in asdict(self).items() if k != "device"}
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
        return d

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict_for_save(), sort_keys=False))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        for p in ("data_dir", "out_dir"):
            if p in raw and not isinstance(raw[p], Path):
                raw[p] = Path(raw[p])
        cfg = cls(**raw)
        return cfg

    # (optional) domain metadata
    industry_id_order = list(range(1, 34))  # 1..33
    node_feature_names = ["Import", "Export", "Final_Demand"]