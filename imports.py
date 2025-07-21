from __future__ import annotations

# ------------------------- imports --------------------------
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Sequence, Tuple, Dict
import os, json, yaml, random, math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn import ChebConv

from tqdm.auto import tqdm, trange

from scipy.stats import pearsonr


from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange
from scipy.stats import pearsonr

import json