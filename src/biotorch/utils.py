"""Seeding, device selection and tensor movement helpers."""

from __future__ import annotations

# Gaurav Sablok
# gsablok@proton.me
import contextlib
import copy
import csv
import gzip
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

__all__ = [
    "get_device",
    "move_to",
    "seed_everything",
]


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def move_to(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors / dicts / lists / GraphBatch to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to(v, device) for v in obj)
    if hasattr(obj, "to") and callable(obj.to):  # GraphBatch and other containers
        return obj.to(device)
    return obj


def _open(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)
