"""Pure-PyTorch scatter ops and GNN layers (GCN, SAGE, GAT, GIN, RGCN)."""

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
    "GATLayer",
    "GCNLayer",
    "GINLayer",
    "GNNEncoder",
    "GNN_LAYERS",
    "RGCNLayer",
    "SAGELayer",
    "global_pool",
    "scatter_max",
    "scatter_mean",
    "scatter_softmax",
    "scatter_sum",
]


def scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    return out.index_add_(0, index, src)


def scatter_mean(src, index, dim_size):
    s = scatter_sum(src, index, dim_size)
    cnt = scatter_sum(
        torch.ones(src.size(0), device=src.device, dtype=src.dtype), index, dim_size
    ).clamp(min=1)
    return s / cnt.view(-1, *([1] * (src.dim() - 1)))


def scatter_max(src, index, dim_size):
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return src.new_zeros((dim_size,) + tuple(src.shape[1:])).scatter_reduce(
        0, idx, src, "amax", include_self=False
    )


def scatter_softmax(src, index, dim_size):
    mx = scatter_max(src, index, dim_size)[index]
    ex = (src - mx).exp()
    return ex / (scatter_sum(ex, index, dim_size)[index] + 1e-16)


class GCNLayer(nn.Module):
    def __init__(self, i, o, **_):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False)
        self.bias = nn.Parameter(torch.zeros(o))

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        N = x.size(0)
        (row, col), _ = add_self_loops(edge_index, N)
        deg = scatter_sum(
            torch.ones(row.size(0), device=x.device, dtype=x.dtype), col, N
        )
        norm = (deg[row] * deg[col]).rsqrt()
        return scatter_sum(self.lin(x)[row] * norm.unsqueeze(-1), col, N) + self.bias


class SAGELayer(nn.Module):
    def __init__(self, i, o, **_):
        super().__init__()
        self.l, self.r = nn.Linear(i, o), nn.Linear(i, o, bias=False)

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        return self.l(x) + self.r(
            scatter_mean(x[edge_index[0]], edge_index[1], x.size(0))
        )


class GATLayer(nn.Module):
    def __init__(self, i, o, heads=4, edge_dim=0, attn_dropout=0.0, **_):
        super().__init__()
        assert o % heads == 0, "hidden must be divisible by heads"
        self.H, self.C, self.drop = heads, o // heads, attn_dropout
        self.lin = nn.Linear(i, o, bias=False)
        self.a_src = nn.Parameter(torch.randn(1, heads, self.C) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(1, heads, self.C) * 0.1)
        self.edge_lin = nn.Linear(edge_dim, o, bias=False) if edge_dim else None
        self.a_edge = (
            nn.Parameter(torch.randn(1, heads, self.C) * 0.1) if edge_dim else None
        )
        self.bias = nn.Parameter(torch.zeros(o))

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        N = x.size(0)
        ei, ea = add_self_loops(
            edge_index, N, edge_attr if self.edge_lin is not None else None
        )
        row, col = ei
        h = self.lin(x).view(N, self.H, self.C)
        e = (h * self.a_src).sum(-1)[row] + (h * self.a_dst).sum(-1)[col]
        if self.edge_lin is not None and ea is not None:
            e = e + (self.edge_lin(ea).view(-1, self.H, self.C) * self.a_edge).sum(-1)
        alpha = F.dropout(
            scatter_softmax(F.leaky_relu(e, 0.2), col, N), self.drop, self.training
        )
        return (
            scatter_sum(h[row] * alpha.unsqueeze(-1), col, N).reshape(N, -1) + self.bias
        )


class GINLayer(nn.Module):
    def __init__(self, i, o, edge_dim=0, **_):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(i, o), nn.GELU(), nn.Linear(o, o))
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_lin = nn.Linear(edge_dim, i) if edge_dim else None

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        m = x[edge_index[0]]
        if self.edge_lin is not None and edge_attr is not None:
            m = m + self.edge_lin(edge_attr)
        return self.mlp(
            (1 + self.eps) * x + scatter_sum(F.relu(m), edge_index[1], x.size(0))
        )


class RGCNLayer(nn.Module):
    """Relational GCN for typed networks (e.g. drug-target-disease, regulatory +/- edges)."""

    def __init__(self, i, o, num_relations=1, **_):
        super().__init__()
        self.R = max(1, num_relations)
        self.weight = nn.Parameter(torch.empty(self.R, i, o))
        nn.init.xavier_uniform_(self.weight.view(self.R * i, o))
        self.root = nn.Linear(i, o)

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        out = self.root(x)
        if edge_type is None:
            edge_type = torch.zeros(
                edge_index.size(1), dtype=torch.long, device=x.device
            )
        for r in range(self.R):
            m = edge_type == r
            if m.any():
                hr = x @ self.weight[r]
                out = out + scatter_mean(
                    hr[edge_index[0, m]], edge_index[1, m], x.size(0)
                )
        return out


GNN_LAYERS = {
    "gcn": GCNLayer,
    "sage": SAGELayer,
    "gat": GATLayer,
    "gin": GINLayer,
    "rgcn": RGCNLayer,
}


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        num_layers: int = 3,
        layer: str = "gcn",
        dropout: float = 0.1,
        edge_dim: int = 0,
        heads: int = 4,
        num_relations: int = 0,
        num_node_types: int = 0,
        residual: bool = True,
        norm: str = "layer",
    ):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.type_emb = (
            nn.Embedding(num_node_types, hidden) if num_node_types > 0 else None
        )
        cls = GNN_LAYERS[layer]
        self.convs = nn.ModuleList(
            [
                cls(
                    hidden,
                    hidden,
                    heads=heads,
                    edge_dim=edge_dim,
                    num_relations=num_relations,
                )
                for _ in range(num_layers)
            ]
        )
        mk = {
            "layer": lambda: nn.LayerNorm(hidden),
            "batch": lambda: nn.BatchNorm1d(hidden),
        }.get(norm, nn.Identity)
        self.norms = nn.ModuleList([mk() for _ in range(num_layers)])
        self.dropout, self.residual, self.out_dim = dropout, residual, hidden

    def forward(self, x, edge_index, edge_attr=None, edge_type=None, node_type=None):
        h = self.inp(x)
        if self.type_emb is not None and node_type is not None:
            h = h + self.type_emb(node_type)
        for conv, norm in zip(self.convs, self.norms):
            z = F.dropout(
                F.gelu(norm(conv(h, edge_index, edge_attr, edge_type))),
                self.dropout,
                self.training,
            )
            h = h + z if self.residual else z
        return h


def global_pool(x, batch, num_graphs, mode: str = "meanmax"):
    if mode == "mean":
        return scatter_mean(x, batch, num_graphs)
    if mode == "sum":
        return scatter_sum(x, batch, num_graphs)
    if mode == "max":
        return scatter_max(x, batch, num_graphs)
    return torch.cat(
        [scatter_mean(x, batch, num_graphs), scatter_max(x, batch, num_graphs)], -1
    )
