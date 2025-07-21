from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import List
import yaml, torch, os

@dataclass
class Config:
    """
    Experiment-wide hyperparameters & paths.
    Save / load to YAML for reproducibility.
    """

    # ── Paths ─────────────────────────────────────────
    data_dir: Path = Path("/Users/mingyukim/Desktop/JP/data")
    out_dir : Path = Path("/Users/mingyukim/Desktop/JP/results")
    
    # ── Window & Scaling ─────────────────────────────
    window: int = 5
    scale_node: float = 1e6
    scale_Z:    float = 1.0
    scale_edge: float = 1e6

    # ── Training params ─────────────────────────────
    batch_size: int = 8
    epochs:     int = 300
    lr:         float = 5e-4
    weight_decay: float = 1e-4
    patience:     int = 50

    # ── PINN / multitask ────────────────────────────
    lambda_max: float = 1.0
    warmup:     int   = 100
    beta_x:     float = 0.0
    beta_init:  float = 0.0     # <-- 추가

    # ── Model architecture ─────────────────────────
    hidden: int = 512
    k: int = 5
    alpha: float = 0.5
    dropout: float = 0.3
    att_hidden: int = 64
    depth_edge: int = 3

    # ── Seeds & sweep ───────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [17])
    lambda_candidates: List[float] = field(default_factory=lambda: [1.0])
    beta_candidates:   List[float] = field(default_factory=lambda: [0.0])

    # ── Misc ────────────────────────────────────────
    log_every: int = 10
    device: str = field(init=False)  # set in __post_init__

    # ────────────────────────────────────────────────
    def __post_init__(self):
        # device auto-detect
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # basic validation
        assert self.window > 0, "window must be positive"
        assert self.warmup >= 0, "warmup must be >= 0"

        # ensure output dir exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- YAML I/O ----------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)
