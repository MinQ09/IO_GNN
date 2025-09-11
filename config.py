# config.py — drop-in compatible with the patched model/run_single
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
    batch_size: int = 128
    lr: float = 0.0025                      # run_single에서 실제는 lr*0.2 사용
    weight_decay: float = 1e-3
    epochs: int = 500
    patience: int = 500
    seeds: List[int] = field(default_factory=lambda: [95])

    # ───────── Paths ─────────
    data_dir: Path = Path("/Users/mingyu/Desktop/IO_GNN-main/Data")
    out_dir: Path = Path("/Users/mingyu/Desktop/IO_GNN-main/Results/V72")
    scalers_fname: str = "scalers.pkl"

    # ───────── Scaling & window ─────────
    window: int = 2
    scale_node_feats: bool = True
    scale_targets: bool = True
    save_scalers: bool = True
    save_val_preds: bool = True            # run_single에서 VAL 저장 토글

    # ───────── Validation cadence ─────────
    val_every: int = 10                    # run_single이 주기적으로 출력

    # ───────── Rolling-Window CV ─────────
    rolling_val: bool = False
    rolling_splits: int = 5
    rolling_train_size: int | None = None
    rolling_test_size: int = 1
    rolling_gap: int = 0
    fold_epochs: int = 5                   # RW-CV 내부 에폭

    # ───────── PINN / multi-task ─────────
    lambda_max: float = 2.0                # adaptive λ 상한
    warmup: int = 50                       # adaptive λ warmup step
    # (아래 둘은 구버전 호환용; 쓰지 않아도 무방)
    beta_x: float = 0.0
    beta_init: float = 0.1

    # Full-forward PINN cadence (1 = every batch). Larger = faster/approximate.
    pinn_full_every: int = 5

    # ───────── LR scheduler (optional, run_single에서 사용 가능) ─────────
    use_plateau_scheduler: bool = True
    plateau_factor: float = 0.5
    plateau_patience: int = 20
    plateau_min_lr: float = 1e-5
    plateau_metric: str = "val_tot"        # run_single의 기록 키 중 하나

    # ───────── Model structure ─────────
    hidden: int = 256
    k: int = 5                              # (ChebConv용, DirMPNN에선 무시됨)
    dropout: float = 0.4
    alpha: float = 0.7                      # (ChebConv용, DirMPNN에선 무시됨)
    att_hidden: int = 256
    depth_edge: int = 3

    # ───────── Graph/edge options (DirMPNN / GraphLSTMCell) ─────────
    # If use_edge_weight=True and edge_feat_dim is None, the model assumes scalar edges (dim=1).
    edge_feat_dim: Optional[int] = None     # None → 보통 1로 추정됨
    use_edge_weight: bool = True            # edge_attr 메시지패싱 포함
    use_bwd_weights: bool = False           # 역방향 edge_attr_bwd 사용
    compute_attention: bool = True          # 분석용 attention 점수 계산(무미분)
    alpha_mode: str = "scalar"              # (예전 API 호환)
    va_nonneg: bool = True                  # VA 헤드 Softplus

    # Directional normalization & gating
    use_row_norm: bool = True               # 1/out-degree 가중
    use_edge_mul: bool = True               # 스칼라 edge 곱 게이팅
    residual_scale: float = 0.1             # LSTM h 잔차 스케일

    # ───────── New model options (patched) ─────────
    learn_mix: bool = True                  # fwd/bwd 혼합 가중 학습 (β → α=σ(β))
    mix_init: float = 0.0                   # 혼합 가중 초기 β
    two_hop: bool = False                   # 2-hop reinforcement
    hop_residual: float = 0.2               # 1→2 hop residual 비율
    edge_mul_warmup: int = 0                # 초반 N스텝 edge_mul 비활성화

    # ───────── Sweep (non-grid) ─────────
    lambda_candidates: List[float] = field(default_factory=lambda: [0, 0.1])
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
        assert self.val_every >= 1, "val_every must be ≥ 1"
        assert self.hop_residual >= 0.0, "hop_residual must be ≥ 0.0"
        assert self.edge_mul_warmup >= 0, "edge_mul_warmup must be ≥ 0"

        if self.edge_feat_dim is not None:
            assert isinstance(self.edge_feat_dim, int) and self.edge_feat_dim >= 1, \
                "edge_feat_dim must be None or an integer ≥ 1"

        if self.use_plateau_scheduler:
            assert 0 < self.plateau_factor < 1.0, "plateau_factor must be in (0,1)"
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
            cfg.__post_init__()                # re-validate & ensure directories
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