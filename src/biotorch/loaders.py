"""Full-graph loaders for node classification and link prediction."""

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

__all__ = [
    "FullBatchLoader",
    "LinkLoader",
    "link_split_loaders",
    "node_split_loaders",
]


class FullBatchLoader:
    """Yields one full-graph batch per epoch with a node mask (node classification splits)."""

    def __init__(self, base: GraphBatch, mask: torch.Tensor):
        self.base, self.mask = base, mask

    def __len__(self):
        return 1

    def __iter__(self):
        b = copy.copy(self.base)
        b.mask = self.mask
        yield b


def node_split_loaders(graph: BioGraph, ratios=(0.6, 0.2, 0.2), seed=0, stratify=True):
    N = graph.num_nodes
    y = graph.node_y.cpu().numpy()
    rng = np.random.default_rng(seed)
    groups = (
        [np.where(y == c)[0] for c in np.unique(y)]
        if stratify and y.dtype.kind in "iu"
        else [np.arange(N)]
    )
    parts = [[], [], []]
    for g in groups:
        g = rng.permutation(g)
        a, b = (
            int(round(len(g) * ratios[0])),
            int(round(len(g) * (ratios[0] + ratios[1]))),
        )
        parts[0] += g[:a].tolist()
        parts[1] += g[a:b].tolist()
        parts[2] += g[b:].tolist()
    base = collate_graphs([graph])
    loaders = []
    for p in parts:
        m = torch.zeros(N, dtype=torch.bool)
        m[torch.tensor(p, dtype=torch.long)] = True
        loaders.append(FullBatchLoader(base, m))
    return tuple(loaders)


class LinkLoader:
    def __init__(
        self,
        base,
        pos_edge,
        known,
        resample=False,
        batch_size=None,
        neg_ratio=1.0,
        shuffle=False,
    ):
        self.base, self.pos, self.known = base, pos_edge, known
        self.resample, self.bs, self.ratio, self.shuffle = (
            resample,
            batch_size,
            neg_ratio,
            shuffle,
        )
        self.N = base.x.size(0)
        self.fixed_neg = (
            None
            if resample
            else negative_sampling(
                known, self.N, max(1, int(pos_edge.size(1) * neg_ratio))
            )
        )

    def __len__(self):
        return math.ceil(self.pos.size(1) / self.bs) if self.bs else 1

    def __iter__(self):
        idx = (
            torch.randperm(self.pos.size(1))
            if self.shuffle
            else torch.arange(self.pos.size(1))
        )
        for chunk in idx.split(self.bs) if self.bs else [idx]:
            b = copy.copy(self.base)
            b.pos_edge = self.pos[:, chunk]
            if self.resample:
                b.neg_edge = negative_sampling(
                    self.known, self.N, max(1, int(len(chunk) * self.ratio))
                )
            else:
                k = max(1, int(len(chunk) * self.ratio))
                b.neg_edge = (
                    self.fixed_neg[:, torch.randperm(self.fixed_neg.size(1))[:k]]
                    if self.bs
                    else self.fixed_neg
                )
            yield b


def link_split_loaders(
    graph: BioGraph,
    val_ratio=0.1,
    test_ratio=0.1,
    edge_batch_size=None,
    neg_ratio=1.0,
    seed=0,
    undirected=True,
):
    """Train (dynamic negatives) / val / test loaders. Test message-passing uses train+val edges."""
    N = graph.num_nodes
    sp = split_edges(graph.edge_index, N, val_ratio, test_ratio, undirected, seed)
    und = (lambda e: to_undirected(e, None, N)) if undirected else (lambda e: e)
    known = und(coalesce(graph.edge_index, None, N)[0])

    def base(e):
        return collate_graphs(
            [
                BioGraph(
                    x=graph.x,
                    edge_index=e,
                    node_type=graph.node_type,
                    node_seq=graph.node_seq,
                )
            ]
        )

    tr = LinkLoader(
        base(und(sp["train"])),
        sp["train"],
        known,
        True,
        edge_batch_size,
        neg_ratio,
        True,
    )
    va = LinkLoader(base(und(sp["train"])), sp["val"], known, False, None, neg_ratio)
    te = LinkLoader(
        base(und(torch.cat([sp["train"], sp["val"]], 1))),
        sp["test"],
        known,
        False,
        None,
        neg_ratio,
    )
    return tr, va, te
