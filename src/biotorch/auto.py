"""Dataset automation and the AutoML loop (auto_split, auto_fit, load_dataset)."""

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

from .models import *  # noqa: F401,F403
from .networks import *  # noqa: F401,F403
from .sequences import *  # noqa: F401,F403
from .training import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

__all__ = [
    "AutoResult",
    "auto_fit",
    "auto_split",
    "load_dataset",
    "make_loader",
    "sample_params",
]


def _get_collate(ds):
    base = ds
    while isinstance(base, Subset):
        base = base.dataset
    return getattr(base, "collate", None)


def make_loader(
    ds,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_get_collate(ds),
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _eval_view(sub: Subset) -> Subset:
    base = sub.dataset
    if getattr(base, "augment", False):
        base = copy.copy(base)
        base.augment = False
    return Subset(base, sub.indices)


def auto_split(dataset, ratios=(0.8, 0.1, 0.1), seed=0, stratify=None):
    """Random (optionally label-stratified) train/val/test split. Val/test have augmentation disabled."""
    n, rng = len(dataset), np.random.default_rng(seed)
    if stratify is not None:
        y = np.asarray(stratify)
        groups = [np.where(y == c)[0] for c in np.unique(y)]
    else:
        groups = [np.arange(n)]
    parts: List[List[int]] = [[], [], []]
    for g in groups:
        g = rng.permutation(g)
        a, b = (
            int(round(len(g) * ratios[0])),
            int(round(len(g) * (ratios[0] + ratios[1]))),
        )
        parts[0] += g[:a].tolist()
        parts[1] += g[a:b].tolist()
        parts[2] += g[b:].tolist()
    tr, va, te = (Subset(dataset, p) for p in parts)
    return tr, _eval_view(va), _eval_view(te)


def sample_params(space: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """list -> choice ; (lo, hi) -> uniform ; (lo, hi, 'log') -> log-uniform ; (lo, hi, 'int') -> integer."""
    out = {}
    for k, v in space.items():
        if isinstance(v, list):
            out[k] = rng.choice(v)
        elif isinstance(v, tuple):
            lo, hi, kind = (v + ("lin",))[:3]
            if kind == "log":
                out[k] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            elif kind == "int":
                out[k] = rng.randint(int(lo), int(hi))
            else:
                out[k] = rng.uniform(lo, hi)
        else:
            out[k] = v
    return out


@dataclass
class AutoResult:
    best_params: Dict[str, Any]
    best_score: float
    trials: List[Dict[str, Any]]
    test_metrics: Dict[str, float]
    model: nn.Module
    trainer: Trainer
    out_dir: Optional[str] = None


def auto_fit(
    model_factory: Callable[..., BioModel],
    dataset=None,
    task: str = "multiclass",
    search_space: Optional[Dict[str, Any]] = None,
    n_trials: int = 6,
    trial_epochs: int = 6,
    final_epochs: int = 30,
    ratios=(0.8, 0.1, 0.1),
    base_cfg: Optional[TrainConfig] = None,
    loaders: Optional[Tuple] = None,
    stratify_labels=None,
    monitor: str = "val_loss",
    mode: str = "min",
    seed: int = 0,
    out_dir: Optional[str] = "runs/auto",
    time_budget: Optional[float] = None,
    defaults: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> AutoResult:
    """End-to-end loop: split -> random search (short runs) -> retrain best -> test -> report.json.

    Keys of ``search_space`` that are TrainConfig fields (lr, weight_decay, batch_size, ...) configure the
    trainer; every other key is passed to ``model_factory(**params)``.
    """
    base_cfg = copy.deepcopy(base_cfg) if base_cfg else TrainConfig()
    base_cfg.monitor, base_cfg.mode = monitor, mode
    cfg_names = {f.name for f in fields(TrainConfig)}
    rng, space = random.Random(seed), (search_space or {})
    if loaders is None:
        splits = auto_split(dataset, ratios, seed, stratify_labels)

    def get_loaders(bs):
        if loaders is not None:
            return loaders
        return tuple(make_loader(d, bs, shuffle=(i == 0)) for i, d in enumerate(splits))

    def run(params, epochs, tag, save):
        seed_everything(seed)
        tp = {k: v for k, v in params.items() if k in cfg_names}
        mp = {k: v for k, v in params.items() if k not in cfg_names}
        cfg = replace(
            base_cfg,
            epochs=epochs,
            verbose=verbose and save,
            out_dir=os.path.join(out_dir, tag) if (save and out_dir) else None,
            **tp,
        )
        tl, vl, sl = get_loaders(cfg.batch_size)
        tr = Trainer(model_factory(**mp), task, cfg)
        tr.fit(tl, vl)
        return tr, tr.best_score, sl

    trials: List[Dict[str, Any]] = []
    t0, best = time.time(), None
    for t in range(n_trials if space else 1):
        if time_budget and trials and time.time() - t0 > time_budget:
            break
        params = {**(defaults or {}), **sample_params(space, rng)}
        try:
            _, score, _ = run(params, trial_epochs, f"trial_{t}", False)
            rec = {"trial": t, "params": params, "score": score}
        except Exception as e:  # e.g. OOM / invalid combo -> skip
            rec = {"trial": t, "params": params, "score": None, "error": repr(e)}
        trials.append(rec)
        if verbose:
            print(f"[auto] trial {t}: {monitor}={rec['score']} params={params}")
        ok = rec["score"] is not None and not math.isnan(rec["score"])
        if ok and (
            best is None
            or (
                rec["score"] < best["score"]
                if mode == "min"
                else rec["score"] > best["score"]
            )
        ):
            best = rec
    if best is None:
        raise RuntimeError(
            "all trials failed: " + str([t.get("error") for t in trials])
        )
    if verbose:
        print(
            f"[auto] best params: {best['params']} -> retraining for {final_epochs} epochs"
        )
    tr, score, test_loader = run(best["params"], final_epochs, "final", True)
    test = tr.evaluate(test_loader, "test")
    if verbose:
        print("[auto] test:", {k: round(v, 4) for k, v in test.items()})
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(
                {
                    "best_params": best["params"],
                    "best_score": score,
                    "trials": trials,
                    "test": test,
                },
                f,
                indent=1,
                default=str,
            )
    return AutoResult(best["params"], score, trials, test, tr.model, tr, out_dir)


def load_dataset(path: str, alphabet: str = "dna", task: str = "multiclass", **kw):
    """Auto-detect: .fa/.fasta/.fna/.faa -> SequenceDataset ; .csv/.tsv with a sequence column -> SequenceDataset ;
    otherwise treated as a network edge list -> BioNetwork."""
    p = path.lower().replace(".gz", "")
    alph = {"dna": DNA, "rna": RNA, "protein": PROTEIN}[alphabet]
    if p.endswith((".fa", ".fasta", ".fna", ".faa")):
        return SequenceDataset.from_fasta(path, alphabet=alph, task=task, **kw)
    if p.endswith((".csv", ".tsv")):
        with _open(path) as f:
            head = f.readline().lower()
        if "sequence" in head or "seq" in head:
            return SequenceDataset.from_csv(path, alphabet=alph, task=task, **kw)
    return BioNetwork.from_tsv(path, sep="," if p.endswith(".csv") else "\t", **kw)
