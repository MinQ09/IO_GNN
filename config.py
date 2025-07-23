# config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Union
import yaml, torch

@dataclass
class Config:
    """
    Experiment-wide hyperparameters (StandardScaler 버전).
    YAML ↔ dataclass 직렬화 시 Path / 숫자형 모두 안전 변환.
    """

    # ────────── Paths ──────────
    data_dir : Path = Path("/Users/mingyukim/Desktop/JP/data")
    out_dir  : Path = Path("/Users/mingyukim/Desktop/JP/results")
    scalers_fname: str = "scalers.pkl"        # 파일명만 보관

    # ────────── Window & Scaling ──────────
    window : int  = 300
    use_standard_scaler: bool = True
    save_scalers       : bool = True         # 학습 후 저장 여부

    # ────────── Training ──────────
    batch_size   : int = 8
    epochs       : int = 5
    lr           : float = 5e-4
    weight_decay : float = 1e-4
    patience     : int = 30

    # ────────── PINN / Multitask ──────────
    lambda_max : float = 0.1
    warmup     : int   = 100
    beta_x     : float = 0.0
    beta_init  : float = 0.0

    # ────────── Model ──────────
    hidden      : int   = 512
    k           : int   = 5
    alpha       : float = 0.5
    dropout     : float = 0.3
    att_hidden  : int   = 64
    depth_edge  : int   = 3

    # ────────── Experiment sweep ──────────
    seeds            : List[int]   = field(default_factory=lambda: [17])
    lambda_candidates: List[float] = field(default_factory=lambda: [0.1])
    beta_candidates  : List[float] = field(default_factory=lambda: [0.0])

    # ────────── Misc ──────────
    log_every: int = 10
    device   : str = field(init=False)

    # ----------------------------------------------------------
    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        assert self.window > 0, "window must be positive"
        assert self.warmup >= 0, "warmup must be non-negative"

        print("▶ Using StandardScaler" if self.use_standard_scaler
              else "▶ Using *deprecated* legacy scaling")

    # ---------- Helper 프로퍼티 ----------
    @property
    def scalers_path(self) -> Path:
        """ out_dir 변경 시 자동 반영되는 Scaler 경로 """
        return self.out_dir / self.scalers_fname

    # ---------- YAML I/O ----------
    def save(self, path: Union[str, Path]) -> None:
        # Path 객체 → str 로 변환 후 덤프
        data = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self).items()}
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        # 경로형 필드를 다시 Path 로 캐스팅
        for k in ("data_dir", "out_dir"):
            if k in raw:
                raw[k] = Path(raw[k])
        return cls(**raw)

    # ---------- Convenience ----------
    def to_dict(self) -> dict:
        """ JSON / 로그용 dict (Path → str) """
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self).items()}
