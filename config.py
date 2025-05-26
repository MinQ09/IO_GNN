import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class Config:
    data_dir: str
    out_dir: str
    window: int = 4
    scale_node: float = 1e6
    scale_Z: float = 1.0
    hidden: int = 512
    k: int = 5
    dropout: float = 0.3
    lambda_candidates: list[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3])
    beta_candidates: list[float]   = field(default_factory=lambda: [0.05, 0.1, 0.2])
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    warmup: int = 20
    batch_size: int = 8
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    def save(self, path: Path) -> None: path.write_text(yaml.dump(asdict(self)))
