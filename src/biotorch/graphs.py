"""Graph primitives: BioGraph, GraphBatch, collation, edge utilities, structure/molecule graphs."""

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

from .sequences import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

__all__ = [
    "BioGraph",
    "GraphBatch",
    "GraphDataset",
    "add_self_loops",
    "coalesce",
    "collate_graphs",
    "contact_map",
    "knn_graph",
    "negative_sampling",
    "protein_graph",
    "rbf",
    "smiles_to_graph",
    "split_edges",
    "to_undirected",
]


@dataclass
class BioGraph:
    """A single graph. Optional fields cover typed (heterogeneous) and structural data."""

    x: torch.Tensor  # [N, F] node features
    edge_index: torch.Tensor  # [2, E] (source -> target)
    edge_attr: Optional[torch.Tensor] = None  # [E, D]
    edge_type: Optional[torch.Tensor] = None  # [E] relation ids
    node_type: Optional[torch.Tensor] = None  # [N] node-type ids
    y: Optional[torch.Tensor] = None  # graph-level target
    node_y: Optional[torch.Tensor] = None  # node-level target
    pos: Optional[torch.Tensor] = None  # [N, 3] coordinates
    node_seq: Optional[torch.Tensor] = None  # [N, L] token ids (sequence per node)
    node_names: Optional[List[str]] = None

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)


class GraphBatch:
    """Disjoint-union batch of graphs. Missing attributes read as None."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def to(self, device) -> "GraphBatch":
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)


def collate_graphs(graphs: Sequence[BioGraph]) -> GraphBatch:
    keys = ("x", "edge_attr", "edge_type", "node_type", "node_y", "pos", "node_seq")
    cat: Dict[str, list] = {k: [] for k in keys}
    eis, batch, ptr, off = [], [], [0], 0
    for i, g in enumerate(graphs):
        n = g.num_nodes
        eis.append(g.edge_index + off)
        batch.append(torch.full((n,), i, dtype=torch.long))
        off += n
        ptr.append(off)
        for k in keys:
            v = getattr(g, k)
            if v is not None:
                cat[k].append(v)
    out: Dict[str, Any] = {
        "edge_index": torch.cat(eis, 1),
        "batch": torch.cat(batch),
        "ptr": torch.tensor(ptr),
        "num_graphs": len(graphs),
    }
    for k, v in cat.items():
        if len(v) == len(graphs):
            if k == "node_seq":
                L = max(t.size(1) for t in v)
                v = [F.pad(t, (0, L - t.size(1))) for t in v]
            out[k] = torch.cat(v, 0)
    if all(g.y is not None for g in graphs):
        out["y"] = torch.stack([torch.as_tensor(g.y) for g in graphs])
    return GraphBatch(**out)


class GraphDataset(Dataset):
    def __init__(
        self, graphs: Sequence[BioGraph], transform: Optional[Callable] = None
    ):
        self.graphs, self.transform = list(graphs), transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        g = self.graphs[i]
        return self.transform(g) if self.transform else g

    @staticmethod
    def collate(items):
        return collate_graphs(items)


def coalesce(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
):
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    n = num_nodes if num_nodes is not None else int(edge_index.max()) + 1
    key = edge_index[0] * n + edge_index[1]
    uniq, inv = torch.unique(key, return_inverse=True)
    perm = torch.empty(uniq.size(0), dtype=torch.long, device=edge_index.device)
    perm[inv] = torch.arange(inv.size(0), device=edge_index.device)
    return edge_index[:, perm], (edge_attr[perm] if edge_attr is not None else None)


def to_undirected(edge_index, edge_attr=None, num_nodes=None):
    ei = torch.cat([edge_index, edge_index.flip(0)], 1)
    ea = torch.cat([edge_attr, edge_attr], 0) if edge_attr is not None else None
    return coalesce(ei, ea, num_nodes)[0] if ea is None else coalesce(ei, ea, num_nodes)


def add_self_loops(edge_index, num_nodes, edge_attr=None):
    loops = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    ei = torch.cat([edge_index, loops], 1)
    if edge_attr is not None:
        edge_attr = torch.cat(
            [edge_attr, edge_attr.new_zeros(num_nodes, edge_attr.size(1))], 0
        )
    return ei, edge_attr


def negative_sampling(edge_index, num_nodes, num_samples, max_tries=10):
    """Uniform negative edges that are not in edge_index (pass both directions for undirected)."""
    dev = edge_index.device
    pos_key = torch.unique(edge_index[0] * num_nodes + edge_index[1])
    out, got = [], 0
    for _ in range(max_tries):
        need = num_samples - got
        if need <= 0:
            break
        cand = torch.randint(0, num_nodes, (2, int(need * 1.5) + 10), device=dev)
        keep = (cand[0] != cand[1]) & ~torch.isin(
            cand[0] * num_nodes + cand[1], pos_key
        )
        out.append(cand[:, keep])
        got += int(keep.sum())
    neg = torch.cat(out, 1) if out else torch.zeros(2, 0, dtype=torch.long, device=dev)
    return neg[:, :num_samples]


def split_edges(
    edge_index, num_nodes, val_ratio=0.1, test_ratio=0.1, undirected=True, seed=0
):
    ei = edge_index
    if undirected:
        ei = torch.stack([ei.min(0).values, ei.max(0).values])
    ei = ei[:, ei[0] != ei[1]]
    ei, _ = coalesce(ei, None, num_nodes)
    g = torch.Generator().manual_seed(seed)
    ei = ei[:, torch.randperm(ei.size(1), generator=g)]
    nv, nt = int(val_ratio * ei.size(1)), int(test_ratio * ei.size(1))
    return {"val": ei[:, :nv], "test": ei[:, nv : nv + nt], "train": ei[:, nv + nt :]}


def knn_graph(pos: torch.Tensor, k: int = 10) -> torch.Tensor:
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    k = min(k, pos.size(0) - 1)
    nbr = d.topk(k, largest=False).indices  # [N, k]
    col = torch.arange(pos.size(0)).unsqueeze(1).expand_as(nbr)
    return torch.stack([nbr.reshape(-1), col.reshape(-1)])


def rbf(d: torch.Tensor, n: int = 16, dmax: float = 20.0) -> torch.Tensor:
    mu = torch.linspace(0, dmax, n, device=d.device)
    return torch.exp(-(((d.unsqueeze(-1) - mu) / (dmax / n)) ** 2))


def contact_map(coords: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    return (torch.cdist(coords, coords) < threshold).float()


def protein_graph(
    coords, seq: str, k: int = 10, y=None, node_y=None, alphabet: Alphabet = PROTEIN
) -> BioGraph:
    """Residue graph from C-alpha coordinates: kNN edges with RBF-distance + log sequence-separation features."""
    coords = torch.as_tensor(coords, dtype=torch.float32)
    x = one_hot(seq, alphabet)
    ei = knn_graph(coords, k)
    dist = (coords[ei[0]] - coords[ei[1]]).norm(dim=-1)
    sep = torch.log1p((ei[0] - ei[1]).abs().float()).unsqueeze(-1)
    return BioGraph(
        x=x,
        edge_index=ei,
        edge_attr=torch.cat([rbf(dist), sep], -1),
        pos=coords,
        y=None if y is None else torch.as_tensor(y),
        node_y=node_y,
    )


def smiles_to_graph(smiles: str, y=None) -> BioGraph:
    """Molecular graph (requires `pip install rdkit`)."""
    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError("smiles_to_graph needs rdkit: pip install rdkit") from e
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    hyb = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
    ]
    x = []
    for a in mol.GetAtoms():
        f = [float(a.GetAtomicNum() == z) for z in atoms] + [
            float(a.GetAtomicNum() not in atoms)
        ]
        f += [
            a.GetDegree() / 4,
            a.GetFormalCharge() / 2,
            float(a.GetIsAromatic()),
            a.GetTotalNumHs() / 4,
        ]
        f += [float(a.GetHybridization() == h) for h in hyb]
        x.append(f)
    bt = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = [float(b.GetBondType() == t) for t in bt] + [
            float(b.GetIsConjugated()),
            float(b.IsInRing()),
        ]
        src += [i, j]
        dst += [j, i]
        ea += [f, f]
    ei = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.zeros(2, 0, dtype=torch.long)
    )
    ea = torch.tensor(ea, dtype=torch.float32) if ea else torch.zeros(0, 6)
    return BioGraph(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=ei,
        edge_attr=ea,
        y=None if y is None else torch.as_tensor(y),
    )
