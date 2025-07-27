# config.py  ─────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Union
import yaml, torch


@dataclass
class Config:
    # ────────── Paths ──────────
    data_dir: Path = Path("./Data")
    out_dir:  Path = Path("./Results/V10")
    scalers_fname: str = "scalers.pkl"

    # ────────── Window & Scaling ──────────
    window: int = 3                       
    use_standard_scaler: bool = True
    save_scalers: bool = True  # ← Fixed: Added missing attribute

    # ────────── Training ──────────
    batch_size: int = 16
    epochs: int = 500
    lr: float = 5e-4
    weight_decay: float = 5e-4
    patience: int = 50             
    
    # ────────── PINN / Multitask ──────────
    lambda_max: float = 0.5
    warmup: int = 100
    beta_x: float = 0.0
    beta_init: float = 0.1

    # ────────── Model ──────────
    hidden: int = 768
    k: int = 5
    alpha: float = 0.5
    dropout: float = 0.15
    att_hidden: int = 64
    depth_edge: int = 4

    # ────────── Experiment sweep ──────────
    seeds: List[int] = field(default_factory=lambda: [17])
    lambda_candidates: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7]
    )
    beta_candidates: List[float] = field(
        default_factory=lambda: [0.0]
    )

    # ────────── Misc ──────────
    log_every: int = 10
    device: str = field(init=False)

    # ----------------------------------------------------------
    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        assert self.window > 0, "window must be positive"
        assert self.warmup >= 0, "warmup must be non-negative"

        print("▶ Using StandardScaler" if self.use_standard_scaler
              else "▶ Using *deprecated* legacy scaling")

    # ---------- Helper ----------
    @property
    def scalers_path(self) -> Path:
        return self.out_dir / self.scalers_fname

    # ---------- YAML I/O ----------
    def save(self, path: Union[str, Path]) -> None:
        data = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(self).items()
        }
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        for k in ("data_dir", "out_dir"):
            if k in raw:
                raw[k] = Path(raw[k])
        return cls(**raw)

    # ---------- Convenience ----------
    def to_dict(self) -> dict:
        return {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(self).items()
        }
