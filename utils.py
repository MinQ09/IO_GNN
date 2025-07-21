# utils.py
import os, random
from typing import Final

import numpy as np
import torch

# Ensures deterministic CuBLAS kernels (PyTorch >= 1.10)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Fixes random seeds for Python, NumPy, and PyTorch (CPU + all GPUs).

    Parameters
    ----------
    seed : int
        Seed value.
    deterministic : bool, default=True
        If True, enables PyTorch deterministic algorithms & disables CUDNN benchmark.
        Set False for speed-oriented training where exact reproducibility is not required.
    """
    # ─ Python & NumPy ──────────────────────
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ─ PyTorch ─────────────────────────────
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # multi-GPU

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Disable TF32 for bit-wise reproducibility (A100/H100 etc.)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = True