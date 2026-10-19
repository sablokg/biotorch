"""Named biological networks (PPI, regulatory, metabolic, typed) and de Bruijn graphs."""

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
from .sequences import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .utils import _open  # noqa: F401

__all__ = [
    "BioNetwork",
    "debruijn_graph",
]


class BioNetwork:
    """Named, optionally weighted / directed / typed biological network."""

    def __init__(
        self,
        names: List[str],
        edges: np.ndarray,
        weights: Optional[np.ndarray] = None,
        directed: bool = False,
        edge_types: Optional[np.ndarray] = None,
        relation_names: Optional[List[str]] = None,
        node_types: Optional[np.ndarray] = None,
        node_type_names: Optional[List[str]] = None,
        node_features: Optional[np.ndarray] = None,
    ):
        self.names = list(names)
        self.name2id = {n: i for i, n in enumerate(self.names)}
        self.edges = np.asarray(edges, dtype=np.int64).reshape(2, -1)
        self.weights, self.directed = weights, directed
        self.edge_types, self.relation_names = edge_types, relation_names
        self.node_types, self.node_type_names = node_types, node_type_names
        self.node_features = node_features

    # ---- construction ----
    @classmethod
    def from_edge_list(
        cls,
        edge_list: Iterable[tuple],
        directed: bool = False,
        weighted: bool = False,
        typed: bool = False,
        drop_self_loops: bool = True,
        dedupe: bool = True,
        node_types: Optional[Dict[str, str]] = None,
        node_features: Optional[Dict[str, Sequence[float]]] = None,
    ) -> "BioNetwork":
        """Items are (u, v[, weight][, relation]) ; `weighted`/`typed` say which extras are present."""
        idx: Dict[str, int] = {}
        names: List[str] = []

        def nid(n):
            if n not in idx:
                idx[n] = len(names)
                names.append(n)
            return idx[n]

        U, V, W, R, rel_idx, rel_names = [], [], [], [], {}, []
        for t in edge_list:
            u, v, j, w, r = nid(t[0]), nid(t[1]), 2, 1.0, 0
            if weighted:
                w = float(t[j])
                j += 1
            if typed:
                if t[j] not in rel_idx:
                    rel_idx[t[j]] = len(rel_names)
                    rel_names.append(t[j])
                r = rel_idx[t[j]]
            if drop_self_loops and u == v:
                continue
            U.append(u)
            V.append(v)
            W.append(w)
            R.append(r)
        e = np.array([U, V], dtype=np.int64).reshape(2, -1)
        w, r = np.array(W, dtype=np.float32), np.array(R, dtype=np.int64)
        if dedupe and e.shape[1]:
            k = e if directed else np.sort(e, axis=0)
            _, keep = np.unique(k[0] * len(names) + k[1], return_index=True)
            keep = np.sort(keep)
            e, w, r = e[:, keep], w[keep], r[keep]
        nt = ntn = nf = None
        if node_types:
            ntn = sorted(set(node_types.values()))
            nt = np.array(
                [
                    ntn.index(node_types.get(n, "unknown"))
                    if node_types.get(n) in ntn
                    else 0
                    for n in names
                ],
                dtype=np.int64,
            )
        if node_features:
            dim = len(next(iter(node_features.values())))
            nf = np.array(
                [node_features.get(n, [0.0] * dim) for n in names], dtype=np.float32
            )
        return cls(
            names,
            e,
            w if weighted else None,
            directed,
            r if typed else None,
            rel_names if typed else None,
            nt,
            ntn,
            nf,
        )

    @classmethod
    def from_tsv(
        cls,
        path: str,
        sep: str = "\t",
        header: bool = False,
        weight_col: Optional[int] = None,
        type_col: Optional[int] = None,
        directed: bool = False,
        **kw,
    ) -> "BioNetwork":
        edges = []
        with _open(path) as f:
            for i, line in enumerate(f):
                if (header and i == 0) or not line.strip() or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split(sep)
                t = [p[0], p[1]]
                if weight_col is not None:
                    t.append(p[weight_col])
                if type_col is not None:
                    t.append(p[type_col])
                edges.append(tuple(t))
        return cls.from_edge_list(
            edges,
            directed=directed,
            weighted=weight_col is not None,
            typed=type_col is not None,
            **kw,
        )

    # ---- properties / queries ----
    @property
    def num_nodes(self) -> int:
        return len(self.names)

    @property
    def num_edges(self) -> int:
        return self.edges.shape[1]

    def csr(self) -> Tuple[np.ndarray, np.ndarray]:
        u, v = self.edges
        if not self.directed:
            u, v = np.r_[u, v], np.r_[v, u]
        o = np.argsort(u, kind="stable")
        u, v = u[o], v[o]
        indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        np.add.at(indptr, u + 1, 1)
        return np.cumsum(indptr), v

    def degree(self) -> np.ndarray:
        return np.diff(self.csr()[0])

    def adjacency(self) -> np.ndarray:
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        w = (
            self.weights
            if self.weights is not None
            else np.ones(self.num_edges, dtype=np.float32)
        )
        A[self.edges[0], self.edges[1]] = w
        if not self.directed:
            A[self.edges[1], self.edges[0]] = w
        return A

    def neighbors(self, name: str) -> List[str]:
        ip, ix = self.csr()
        i = self.name2id[name]
        return [self.names[j] for j in ix[ip[i] : ip[i + 1]]]

    def k_hop(self, name: str, k: int = 2) -> List[str]:
        ip, ix = self.csr()
        seen, frontier = {self.name2id[name]}, [self.name2id[name]]
        for _ in range(k):
            nxt = []
            for i in frontier:
                for j in ix[ip[i] : ip[i + 1]]:
                    if int(j) not in seen:
                        seen.add(int(j))
                        nxt.append(int(j))
            frontier = nxt
        return [self.names[i] for i in sorted(seen)]

    def connected_components(self) -> List[List[str]]:
        parent = list(range(self.num_nodes))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in self.edges.T:
            parent[find(int(a))] = find(int(b))
        comps: Dict[int, List[str]] = {}
        for i in range(self.num_nodes):
            comps.setdefault(find(i), []).append(self.names[i])
        return sorted(comps.values(), key=len, reverse=True)

    def subgraph(self, keep_names: Sequence[str]) -> "BioNetwork":
        keep = np.array(sorted(self.name2id[n] for n in keep_names))
        remap = -np.ones(self.num_nodes, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        m = (remap[self.edges[0]] >= 0) & (remap[self.edges[1]] >= 0)
        sel = lambda a: None if a is None else a[m]
        return BioNetwork(
            [self.names[i] for i in keep],
            remap[self.edges[:, m]],
            sel(self.weights),
            self.directed,
            sel(self.edge_types),
            self.relation_names,
            None if self.node_types is None else self.node_types[keep],
            self.node_type_names,
            None if self.node_features is None else self.node_features[keep],
        )

    def random_walks(
        self, walks_per_node: int = 5, length: int = 20, seed: int = 0
    ) -> np.ndarray:
        """Uniform random walks, vectorised. Returns int array [N*walks_per_node, length]."""
        rng = np.random.default_rng(seed)
        ip, ix = self.csr()
        cur = np.tile(np.arange(self.num_nodes), walks_per_node)
        walks = [cur]
        for _ in range(length - 1):
            deg = ip[cur + 1] - ip[cur]
            off = (rng.random(len(cur)) * np.maximum(deg, 1)).astype(np.int64)
            pos = np.minimum(ip[cur] + off, max(len(ix) - 1, 0))
            cur = np.where(deg > 0, ix[pos] if len(ix) else cur, cur)
            walks.append(cur)
        return np.stack(walks, 1)

    # ---- export ----
    def to_graph(
        self, feature: str = "auto", dim: int = 16, node_y=None, y=None, seed: int = 0
    ) -> BioGraph:
        N = self.num_nodes
        if feature == "auto":
            feature = "given" if self.node_features is not None else "degree"
        if feature == "given":
            x = torch.tensor(self.node_features, dtype=torch.float32)
        elif feature == "degree":
            d = self.degree()
            x = torch.tensor(
                np.stack([np.log1p(d), np.ones(N)], 1), dtype=torch.float32
            )
        elif feature == "onehot":
            x = torch.eye(N)
        elif feature == "random":
            x = torch.randn(N, dim, generator=torch.Generator().manual_seed(seed))
        else:
            raise ValueError(feature)
        u, v = self.edges
        w, et = self.weights, self.edge_types
        if not self.directed:
            u, v = np.r_[u, v], np.r_[v, u]
            w = None if w is None else np.r_[w, w]
            et = None if et is None else np.r_[et, et]
        return BioGraph(
            x=x,
            edge_index=torch.tensor(np.stack([u, v]), dtype=torch.long),
            edge_attr=None
            if w is None
            else torch.tensor(w, dtype=torch.float32).unsqueeze(-1),
            edge_type=None if et is None else torch.tensor(et, dtype=torch.long),
            node_type=None
            if self.node_types is None
            else torch.tensor(self.node_types, dtype=torch.long),
            y=None if y is None else torch.as_tensor(y),
            node_y=None if node_y is None else torch.as_tensor(node_y),
            node_names=self.names,
        )


def debruijn_graph(sequences: Iterable[str], k: int = 4) -> BioNetwork:
    """Directed (k-1)-mer graph; edge weight = k-mer multiplicity (genome-assembly style)."""
    counts: Dict[Tuple[str, str], int] = {}
    for s in sequences:
        for km in kmers(s.upper(), k):
            e = (km[:-1], km[1:])
            counts[e] = counts.get(e, 0) + 1
    return BioNetwork.from_edge_list(
        [(a, b, c) for (a, b), c in counts.items()],
        directed=True,
        weighted=True,
        drop_self_loops=False,
    )
