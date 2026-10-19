"""Task losses and numpy metrics (AUROC, AP, F1, R2, Pearson, Spearman, ...)."""

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
    "BioModel",
    "StepOutput",
    "auroc",
    "average_precision",
    "compute_metrics",
    "macro_f1",
    "task_loss",
    "task_pred",
]


@dataclass
class StepOutput:
    loss: torch.Tensor
    pred: torch.Tensor  # probabilities (classification) or values (regression)
    target: torch.Tensor


class BioModel(nn.Module):
    """Base class: models implement ``step(batch) -> StepOutput`` and set ``task``."""

    task: str = "multiclass"

    def step(self, batch) -> StepOutput:
        raise NotImplementedError


def task_loss(task, logits, target, class_weight=None, ignore_index=-100):
    if task in ("multiclass", "token_multiclass", "mlm"):
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target.reshape(-1),
            weight=class_weight,
            ignore_index=ignore_index,
        )
    if task == "binary":
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1).float(), target.reshape(-1).float()
        )
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(logits.float(), target.float())
    if task == "regression":
        return F.mse_loss(logits.float().reshape(target.shape), target.float())
    raise ValueError(task)


def task_pred(task, logits, target=None):
    if task in ("multiclass", "token_multiclass", "mlm"):
        return logits.float().softmax(-1)
    if task == "binary":
        return logits.float().reshape(-1).sigmoid()
    if task == "multilabel":
        return logits.float().sigmoid()
    return (
        logits.float().reshape(target.shape) if target is not None else logits.float()
    )


def _rank(a: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return ((cum - cnt) + cum + 1)[inv] / 2.0


def auroc(y, s) -> float:
    y, s = np.asarray(y).astype(bool), np.asarray(s, dtype=float)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((_rank(s)[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(y, s) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-np.asarray(s), kind="mergesort")
    y = y[o]
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float(prec[y].mean())


def macro_f1(t, p) -> float:
    f = []
    for c in np.unique(np.r_[t, p]):
        tp = ((p == c) & (t == c)).sum()
        fp = ((p == c) & (t != c)).sum()
        fn = ((p != c) & (t == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def _corr(a, b):
    return (
        float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    )


def compute_metrics(
    task: str, preds: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    m: Dict[str, float] = {}
    if task in ("multiclass", "token_multiclass", "mlm"):
        t = targets.astype(int)
        pc = preds.argmax(1) if preds.ndim == 2 else preds.astype(int)
        m["acc"] = float((pc == t).mean())
        if task == "mlm" and preds.ndim == 2:
            m["ppl"] = float(
                np.exp(-np.log(np.clip(preds[np.arange(len(t)), t], 1e-9, 1)).mean())
            )
            return m
        m["f1"] = macro_f1(t, pc)
        if preds.ndim == 2:
            if preds.shape[1] == 2:
                m["auroc"] = auroc(t == 1, preds[:, 1])
            elif preds.shape[1] > 2:
                a = [
                    auroc(t == c, preds[:, c])
                    for c in np.unique(t)
                    if 0 < (t == c).sum() < len(t)
                ]
                if a:
                    m["auroc"] = float(np.mean(a))
    elif task == "binary":
        p, t = preds.reshape(-1), targets.reshape(-1).astype(int)
        pc = (p > 0.5).astype(int)
        tp = ((pc == 1) & (t == 1)).sum()
        m["acc"] = float((pc == t).mean())
        m["f1"] = float(
            2
            * tp
            / max(
                1, 2 * tp + ((pc == 1) & (t == 0)).sum() + ((pc == 0) & (t == 1)).sum()
            )
        )
        m["auroc"], m["auprc"] = auroc(t, p), average_precision(t, p)
    elif task == "multilabel":
        t = targets.astype(int)
        pc = (preds > 0.5).astype(int)
        tp = ((pc == 1) & (t == 1)).sum()
        m["acc"] = float((pc == t).mean())
        m["f1"] = float(
            2
            * tp
            / max(
                1, 2 * tp + ((pc == 1) & (t == 0)).sum() + ((pc == 0) & (t == 1)).sum()
            )
        )
        a = [auroc(t[:, k], preds[:, k]) for k in range(t.shape[1])]
        a = [x for x in a if not math.isnan(x)]
        if a:
            m["auroc"] = float(np.mean(a))
    elif task == "regression":
        p, t = preds.reshape(-1).astype(float), targets.reshape(-1).astype(float)
        m["mse"] = float(((p - t) ** 2).mean())
        m["mae"] = float(np.abs(p - t).mean())
        m["r2"] = float(
            1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)
        )
        m["pearson"], m["spearman"] = _corr(p, t), _corr(_rank(p), _rank(t))
    return m
