"""Synthetic biological data for demos and tests."""

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

from .graphs import *  # noqa: F401,F403
from .networks import *  # noqa: F401,F403
from .sequences import *  # noqa: F401,F403

__all__ = [
    "synthetic_graphs",
    "synthetic_network",
    "synthetic_seq_network",
    "synthetic_sequences",
]


def synthetic_sequences(n=600, length=80, alphabet="dna", task="binary", seed=0):
    """Planted-motif sequences. binary / multiclass(3) / regression (motif copies) / token_multiclass."""
    rng = np.random.default_rng(seed)
    letters = list("ACGT") if alphabet == "dna" else list("ACDEFGHIKLMNPQRSTVWY")
    motifs = (
        ["TATAAT", "GGCCAA"]
        if alphabet == "dna"
        else ["KDELKD", "HEXXHE".replace("X", "G")]
    )
    seqs, labels = [], []
    for _ in range(n):
        s = rng.choice(letters, length)
        if task == "regression":
            k = int(rng.integers(0, 4))
            for _ in range(k):
                p = int(rng.integers(0, length - 6))
                s[p : p + 6] = list(motifs[0])
            labels.append(float(k) + float(rng.normal(0, 0.1)))
        elif task == "token_multiclass":
            lab = np.zeros(length, dtype=int)
            if rng.random() < 0.7:
                p = int(rng.integers(0, length - 6))
                s[p : p + 6] = list(motifs[0])
                lab[p : p + 6] = 1
            labels.append(lab.tolist())
        elif task == "multiclass":
            c = int(rng.integers(0, 3))
            if c > 0:
                p = int(rng.integers(0, length - 6))
                s[p : p + 6] = list(motifs[c - 1])
            labels.append(c)
        else:
            c = int(rng.random() < 0.5)
            if c:
                p = int(rng.integers(0, length - 6))
                s[p : p + 6] = list(motifs[0])
            labels.append(c)
        seqs.append("".join(s))
    return seqs, labels


def synthetic_graphs(n=240, nodes=(14, 26), seed=0) -> List[BioGraph]:
    """Class 1 = random graph with a planted 5-clique ; class 0 = density-matched random graph."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        N, c = int(rng.integers(*nodes)), i % 2
        A = np.triu(rng.random((N, N)) < (0.12 if c else 0.18), 1)
        if c:
            idx = rng.choice(N, 5, replace=False)
            for a, b in itertools.combinations(idx, 2):
                A[min(a, b), max(a, b)] = True
        u, v = np.nonzero(A | A.T)
        ei = torch.tensor(np.stack([u, v]), dtype=torch.long)
        deg = np.bincount(u, minlength=N)
        x = torch.tensor(np.stack([np.log1p(deg), np.ones(N)], 1), dtype=torch.float32)
        out.append(BioGraph(x=x, edge_index=ei, y=torch.tensor(c)))
    return out


def synthetic_network(
    n_nodes=300, n_comm=3, p_in=0.08, p_out=0.005, feat_dim=16, feat_signal=0.6, seed=0
):
    """Stochastic-block-model 'PPI' with community labels. Returns (BioNetwork, labels)."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, n_comm, n_nodes)
    P = np.where(lab[:, None] == lab[None, :], p_in, p_out)
    A = np.triu(rng.random((n_nodes, n_nodes)) < P, 1)
    u, v = np.nonzero(A)
    names = [f"P{i:04d}" for i in range(n_nodes)]
    centers = rng.normal(0, 1, (n_comm, feat_dim))
    feats = feat_signal * centers[lab] + rng.normal(0, 1, (n_nodes, feat_dim))
    net = BioNetwork(
        names, np.stack([u, v]), directed=False, node_features=feats.astype(np.float32)
    )
    return net, lab


def synthetic_seq_network(n_nodes=240, n_comm=3, length=40, seed=0):
    """Network where node class is only visible from its sequence motif (graph helps via homophily)."""
    net, lab = synthetic_network(n_nodes, n_comm, feat_signal=0.0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    motifs = ["TATAAT", "GGCCAA", "CAGCTG"][:n_comm]
    toks = []
    for c in lab:
        s = rng.choice(list("ACGT"), length)
        p = int(rng.integers(0, length - 6))
        s[p : p + 6] = list(motifs[c])
        toks.append(DNA.encode("".join(s)))
    g = net.to_graph(feature="given", node_y=torch.tensor(lab))
    g.node_seq = torch.tensor(toks, dtype=torch.long)
    return g
