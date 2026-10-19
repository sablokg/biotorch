"""Command line interface."""

from __future__ import annotations

# Gaurav Sablok
# gsablok@proton.me
import argparse
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

from .auto import *  # noqa: F401,F403
from .demos import *  # noqa: F401,F403
from .graphs import *  # noqa: F401,F403
from .layers import *  # noqa: F401,F403
from .loaders import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from .networks import *  # noqa: F401,F403
from .sequences import *  # noqa: F401,F403
from .synthetic import *  # noqa: F401,F403
from .training import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

__all__ = [
    "main",
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="biotorch: PyTorch for biological sequences, networks and graphs"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("demo")
    d.add_argument("name", choices=list(DEMOS) + ["all"])
    s = sub.add_parser(
        "train-seq", help="train on a CSV/TSV/FASTA of sequences with automatic HPO"
    )
    s.add_argument("--csv")
    s.add_argument("--fasta")
    s.add_argument("--seq-col", default="sequence")
    s.add_argument("--label-col", default="label")
    s.add_argument("--alphabet", default="dna", choices=["dna", "rna", "protein"])
    s.add_argument(
        "--task",
        default="multiclass",
        choices=["multiclass", "binary", "regression", "multilabel"],
    )
    s.add_argument(
        "--arch", default="cnn", choices=["cnn", "transformer", "gru", "lstm"]
    )
    s.add_argument("--max-len", type=int, default=1000)
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--trials", type=int, default=4)
    s.add_argument("--out", default="runs/seq")
    l = sub.add_parser("train-link", help="link prediction on an edge-list file")
    l.add_argument("--edges", required=True)
    l.add_argument("--epochs", type=int, default=100)
    l.add_argument("--layer", default="sage", choices=list(GNN_LAYERS))
    l.add_argument("--out", default="runs/link")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "demo":
        for n in list(DEMOS) if a.name == "all" else [a.name]:
            print(f"\n=== demo: {n} ===")
            print("result:", {k: round(v, 4) for k, v in DEMOS[n](True).items()})
    elif a.cmd == "train-seq":
        alph = {"dna": DNA, "rna": RNA, "protein": PROTEIN}[a.alphabet]
        if a.fasta:
            ds = SequenceDataset.from_fasta(
                a.fasta,
                label_fn=lambda n: int(n.split("label=")[-1].split()[0]),
                alphabet=alph,
                task=a.task,
                max_len=a.max_len,
            )
        else:
            ds = SequenceDataset.from_csv(
                a.csv,
                a.seq_col,
                a.label_col,
                task=a.task,
                alphabet=alph,
                max_len=a.max_len,
                augment=a.alphabet != "protein",
            )
        strat = ds.labels if a.task in ("multiclass", "binary") else None
        space = {
            "lr": (3e-4, 1e-2, "log"),
            "dim": [64, 128],
            "num_layers": (2, 5, "int"),
            "dropout": (0.0, 0.3),
            "batch_size": [32, 64],
        }
        res = auto_fit(
            lambda dim=128, num_layers=4, dropout=0.1: SeqModel(
                alph, ds.n_out, a.task, a.arch, dim, num_layers, dropout
            ),
            ds,
            a.task,
            space,
            a.trials,
            max(3, a.epochs // 5),
            a.epochs,
            stratify_labels=strat,
            out_dir=a.out,
        )
        torch.save(res.model.state_dict(), os.path.join(a.out, "model.pt"))
        print("saved:", os.path.join(a.out, "model.pt"))
    elif a.cmd == "train-link":
        net = load_dataset(a.edges)
        g = net.to_graph(feature="random")
        tr, va, te = link_split_loaders(g)
        res = auto_fit(
            lambda hidden=64, num_layers=2, dropout=0.2: LinkModel(
                g.x.size(1), hidden, num_layers, a.layer, dropout
            ),
            None,
            "binary",
            {
                "lr": (1e-3, 2e-2, "log"),
                "hidden": [32, 64, 128],
                "num_layers": (1, 3, "int"),
            },
            4,
            20,
            a.epochs,
            loaders=(tr, va, te),
            monitor="val_auroc",
            mode="max",
            out_dir=a.out,
            base_cfg=TrainConfig(scheduler="none", warmup_epochs=0),
        )
