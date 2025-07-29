# config.py ─────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Union
import yaml, torch
import itertools

@dataclass
class Config:
    """IO-GNN 전체 설정"""

    # ────────── 필수(기본값 O) ──────────
    batch_size: int = 32

    # ────────── Paths ──────────
    data_dir: Path = Path("/Users/mingyu/Downloads/KR6/Data")
    out_dir: Path  = Path("/Users/mingyu/Downloads/KR6/Results/V23")
    scalers_fname: str = "scalers.pkl"

    # ────────── Window & Scaling ──────────
    window: int = 4
    use_standard_scaler: bool = True
    save_scalers: bool = True

    # ────────── Training ──────────
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 30

    # ────────── PINN / Multitask ──────────
    lambda_max: float = 0.5
    warmup: int = 100
    beta_x: float = 0.0
    beta_init: float = 0.1

    # ────────── Model ──────────
    hidden: int = 512
    k: int = 3
    alpha: float = 0.5
    dropout: float = 0.3
    att_hidden: int = 64
    depth_edge: int = 2

    # ────────── Experiment sweep ──────────
    seeds: List[int] = field(default_factory=lambda: [19])
    lambda_candidates: List[float] = field(
        default_factory=lambda: [0.0, 0.1]
    )
    beta_candidates: List[float] = field(default_factory=lambda: [0.0])

    # ────────── Grid Search Parameters ──────────
    grid_search: bool = False
    batch_size_candidates: List[int] = field(default_factory=lambda: [32, 64])
    lr_candidates: List[float] = field(default_factory=lambda: [5e-5, 1e-4])
    weight_decay_candidates: List[float] = field(default_factory=lambda: [1e-5])
    hidden_candidates: List[int] = field(default_factory=lambda: [256, 512])
    k_candidates: List[int] = field(default_factory=lambda: [3, 5])
    dropout_candidates: List[float] = field(default_factory=lambda: [0.1, 0.3])

    # ────────── Misc ──────────
    log_every: int = 10
    device: str = field(init=False)

    # --------------------------------------------------
    def __post_init__(self):
        # --------------------------------------------------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.grid_search:
            print(f"▶ Grid Search: {self.count_grid_combinations():,} combos")

    # ---------- Helper ----------
    @property
    def scalers_path(self) -> Path:
        return self.out_dir / self.scalers_fname

    # ---------- Grid Search ----------
    def count_grid_combinations(self) -> int:
        return (len(self.batch_size_candidates)
                * len(self.lr_candidates)
                * len(self.weight_decay_candidates)
                * len(self.hidden_candidates)
                * len(self.k_candidates)
                * len(self.dropout_candidates)
                * len(self.lambda_candidates)
                * len(self.seeds))

    def generate_grid_configs(self) -> List["Config"]:
        if not self.grid_search:
            return [self]

        configs = []
        for i, (bs, lr, wd, h, k, dp, lam, seed) in enumerate(
            itertools.product(
                self.batch_size_candidates,
                self.lr_candidates,
                self.weight_decay_candidates,
                self.hidden_candidates,
                self.k_candidates,
                self.dropout_candidates,
                self.lambda_candidates,
                self.seeds,
            )
        ):
            cfg_dict = asdict(self)
            cfg_dict.update(
                dict(
                    grid_search=False,  # 무한 재귀 방지
                    out_dir=self.out_dir
                    / f"seed_{seed}"
                    / f"lam_{lam:.4g}"
                    / f"grid_{i:03d}",
                    batch_size=bs,
                    lr=lr,
                    weight_decay=wd,
                    hidden=h,
                    k=k,
                    dropout=dp,
                    lambda_max=lam,
                    seeds=[seed],
                    lambda_candidates=[lam],
                )
            )
            cfg_dict["data_dir"] = Path(cfg_dict["data_dir"])
            cfg_dict["out_dir"] = Path(cfg_dict["out_dir"])
            configs.append(Config(**{k: v for k, v in cfg_dict.items() if k != "device"}))
        return configs

    # ---------- Convenience ----------
    def get_param_string(self) -> str:
        lr_s  = f"{float(self.lr):.0e}"
        wd_s  = f"{float(self.weight_decay):.0e}"
        lam_s = f"{float(self.lambda_max):.3g}"
        return (f"bs{self.batch_size}_lr{lr_s}_wd{wd_s}_"
                f"h{self.hidden}_k{self.k}_dp{self.dropout}_lam{lam_s}")

    # ---------- YAML I/O ----------
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            yaml.safe_dump({k: (str(v) if isinstance(v, Path) else v)
                            for k, v in asdict(self).items()}, sort_keys=False)
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        for p in ("data_dir", "out_dir"):
            if p in raw:
                raw[p] = Path(raw[p])
        return cls(**raw)
