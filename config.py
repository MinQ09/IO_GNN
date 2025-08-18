# config.py  ──────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Union
import itertools
import yaml
import torch
from copy import deepcopy


@dataclass
class Config:
    """Top-level configuration for all IO-GNN runs."""

    # ─────────────────────────── Base hyper-params ──────────────────────────
    batch_size: int = 32
    lr: float = 0.005
    weight_decay: float = 0.0001
    epochs: int = 500
    patience: int =  500
    seeds: List[int] = field(default_factory=lambda: [123])

    # ─────────────────────────── Paths ──────────────────────────────────────
    data_dir: Path = Path("./Data")
    out_dir: Path = Path("./Results/V45")
    scalers_fname: str = "scalers.pkl"

    # ─────────────────────────── Scaling & window ───────────────────────────
    window: int = 1
    scale_node_feats: bool = True
    scale_targets: bool = False        # ← single-run default
    save_scalers: bool = True
    
    # ───────── Rolling-Window CV ─────────
    rolling_val: bool = False
    rolling_splits: int = 5
    rolling_train_size: int | None = None
    rolling_test_size: int = 1
    rolling_gap: int = 0
    fold_epochs: int = 5

    # ─────────────────────────── PINN / multi-task ─────────────────────────
    lambda_max: float = 0.5
    warmup: int = 20
    beta_x: float = 0.0
    beta_init: float = 0.1

    # ─────────────────────────── Model structure ───────────────────────────
    hidden: int = 512
    k: int = 5
    dropout: float = 0.2
    alpha: float = 0.5
    att_hidden: int = 64
    depth_edge: int = 3

    # ─────────────────────────── Sweep (non-grid) ──────────────────────────
    lambda_candidates: List[float] = field(default_factory=lambda: [0,0.05])
    beta_candidates:   List[float] = field(default_factory=lambda: [0.0])

    # ─────────────────────────── Grid-search flags ─────────────────────────
    grid_search: bool = False

    # main search axes
    batch_size_candidates:   List[int]   = field(default_factory=lambda: [32, 64])
    lr_candidates:           List[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4])
    weight_decay_candidates: List[float] = field(default_factory=lambda: [1e-5, 5e-5])
    hidden_candidates:       List[int]   = field(default_factory=lambda: [256, 512])
    k_candidates:            List[int]   = field(default_factory=lambda: [3, 5])
    dropout_candidates:      List[float] = field(default_factory=lambda: [0.1, 0.3])
    lambda_grid_candidates:  List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5])
    scale_targets_candidates: List[bool] = field(default_factory=lambda: [False, True])

    # ─────────────────────────── Misc ──────────────────────────────────────
    log_every: int = 10
    device: str = field(init=False)

    # ======================================================================
    # initialiser
    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.grid_search:
            print(f"▶ Grid Search: {self.count_grid_combinations():,} combos")

    # ----------------------------------------------------------------------
    # helpers
    @property
    def scalers_path(self) -> Path:
        return self.out_dir / self.scalers_fname

    # ----------------------------------------------------------------------
    # grid-search utilities
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
        """Return a list of Config objects – one per grid point."""
        if not self.grid_search:
            return [self]

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

        for i, (
            bs,
            lr,
            wd,
            h,
            k,
            dp,
            lam,
            scale_tg,
            seed,
        ) in enumerate(combos):
            cfg_dict = deepcopy(asdict(self))

            # mutate with grid values
            cfg_dict.update(
                dict(
                    grid_search=False,         
                    batch_size=bs,
                    lr=lr,
                    weight_decay=wd,
                    hidden=h,
                    k=k,
                    dropout=dp,
                    lambda_max=lam,
                    scale_targets=scale_tg,
                    seeds=[seed],
                    out_dir=self.out_dir / f"grid_{i:03d}",  # seed/lam dir는 run_single에서 추가
                )
            )

            # ensure Path objects survive YAML round-trip
            cfg_dict["data_dir"] = Path(cfg_dict["data_dir"])
            cfg_dict["out_dir"]  = Path(cfg_dict["out_dir"])

            cfg_dict.pop("device", None)
            configs.append(Config(**cfg_dict))

        return configs

    # ----------------------------------------------------------------------
    # pretty id string
    def get_param_string(self) -> str:
        lr_s = f"{float(self.lr):.0e}"
        wd_s = f"{float(self.weight_decay):.0e}"
        lam_s = f"{float(self.lambda_max):.3g}"
        return (
            f"bs{self.batch_size}_lr{lr_s}_wd{wd_s}_"
            f"h{self.hidden}_k{self.k}_dp{self.dropout}_lam{lam_s}_"
            f"raw{int(self.scale_targets is False)}"
        )

    # ----------------------------------------------------------------------
    # YAML helpers
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            yaml.safe_dump(
                {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()},
                sort_keys=False,
            )
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        for p in ("data_dir", "out_dir"):
            if p in raw:
                raw[p] = Path(raw[p])
        return cls(**raw)
