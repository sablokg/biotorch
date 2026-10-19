"""Sequence, graph, node, link, MLM, Node2Vec and sequence+network models."""

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
from .layers import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403
from .networks import *  # noqa: F401,F403
from .sequences import *  # noqa: F401,F403

__all__ = [
    "CNNEncoder",
    "GraphModel",
    "LinkModel",
    "Node2Vec",
    "NodeModel",
    "RNNEncoder",
    "SeqGraphNodeModel",
    "SeqMLM",
    "SeqModel",
    "SkipGramDataset",
    "TransformerEncoderSeq",
    "make_seq_encoder",
    "masked_pool",
    "sinusoidal",
]


def masked_pool(h, mask, mode="meanmax"):
    if mode == "cls":
        return h[:, 0]
    m = mask.unsqueeze(-1).to(h.dtype)
    mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
    if mode == "mean":
        return mean
    mx = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
    return mx if mode == "max" else torch.cat([mean, mx], -1)


class CNNEncoder(nn.Module):
    """Dilated residual 1-D CNN (DeepSEA/Basset/ProtCNN-style)."""

    def __init__(
        self,
        vocab,
        dim=128,
        num_layers=4,
        kernel=9,
        dropout=0.1,
        dilation_growth=2,
        **_,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    dim,
                    dim,
                    kernel,
                    padding=(dilation_growth**i) * (kernel - 1) // 2,
                    dilation=dilation_growth**i,
                )
                for i in range(num_layers)
            ]
        )
        self.drop, self.out_dim = nn.Dropout(dropout), dim

    def forward(self, ids, mask):
        h = self.emb(ids) * mask.unsqueeze(-1)
        for norm, conv in zip(self.norms, self.convs):
            z = conv(norm(h).transpose(1, 2)).transpose(1, 2)
            h = (h + self.drop(F.gelu(z))) * mask.unsqueeze(-1)
        return h


def sinusoidal(L, D, device):
    pos = torch.arange(L, device=device).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D)
    )
    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : D // 2]
    return pe


class TransformerEncoderSeq(nn.Module):
    def __init__(
        self, vocab, dim=128, num_layers=4, heads=4, ff_mult=4, dropout=0.1, **_
    ):
        super().__init__()
        assert dim % 2 == 0
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * ff_mult,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm, self.drop, self.dim, self.out_dim = (
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            dim,
            dim,
        )

    def forward(self, ids, mask):
        h = self.emb(ids) * math.sqrt(self.dim) + sinusoidal(
            ids.size(1), self.dim, ids.device
        )
        h = self.enc(self.drop(h), src_key_padding_mask=~mask)
        return self.norm(h) * mask.unsqueeze(-1)


class RNNEncoder(nn.Module):
    def __init__(self, vocab, dim=128, num_layers=2, dropout=0.1, cell="gru", **_):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(
            dim,
            dim // 2,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = dim

    def forward(self, ids, mask):
        L = ids.size(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(ids),
            mask.sum(1).clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out, _ = self.rnn(packed)
        return nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)[
            0
        ]


def make_seq_encoder(
    arch: str,
    vocab: int,
    dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.1,
    **kw,
):
    arch = arch.lower()
    if arch == "cnn":
        return CNNEncoder(vocab, dim, num_layers, dropout=dropout, **kw)
    if arch == "transformer":
        return TransformerEncoderSeq(vocab, dim, num_layers, dropout=dropout, **kw)
    if arch in ("gru", "lstm"):
        return RNNEncoder(vocab, dim, max(1, num_layers // 2), dropout, cell=arch)
    raise ValueError(f"unknown arch {arch}")


class SeqModel(BioModel):
    """Sequence -> label(s). Works for every task in TASKS except 'mlm' (see SeqMLM)."""

    def __init__(
        self,
        alphabet: Alphabet,
        n_out: int = 1,
        task: str = "multiclass",
        arch: str = "cnn",
        dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        pool: str = "meanmax",
        class_weight: Optional[torch.Tensor] = None,
        **enc_kw,
    ):
        super().__init__()
        self.task = task
        n_out = 1 if task in ("binary", "regression") and n_out < 1 else n_out
        if task == "binary":
            n_out = 1
        self.pool = pool
        self.enc = make_seq_encoder(
            arch, len(alphabet), dim, num_layers, dropout, **enc_kw
        )
        pd = dim * (2 if pool == "meanmax" else 1)
        if task == "token_multiclass":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_out))
        else:
            self.head = nn.Sequential(
                nn.Linear(pd, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, n_out),
            )
        self.register_buffer("class_weight", class_weight, persistent=False)

    def forward(self, batch):
        h = self.enc(batch["input_ids"], batch["mask"])
        if self.task == "token_multiclass":
            return self.head(h)
        return self.head(masked_pool(h, batch["mask"], self.pool))

    def step(self, batch):
        logits, y = self(batch), batch["labels"]
        return StepOutput(
            task_loss(self.task, logits, y, self.class_weight),
            task_pred(self.task, logits, y),
            y,
        )


class SeqMLM(BioModel):
    """Masked-language-model pre-training for any alphabet (BERT-style 80/10/10 corruption)."""

    task = "mlm"

    def __init__(
        self,
        alphabet: Alphabet,
        arch: str = "transformer",
        dim: int = 128,
        num_layers: int = 4,
        mask_prob: float = 0.15,
        dropout: float = 0.1,
        **enc_kw,
    ):
        super().__init__()
        self.alphabet, self.mask_prob = alphabet, mask_prob
        self.enc = make_seq_encoder(
            arch, len(alphabet), dim, num_layers, dropout, **enc_kw
        )
        self.head = nn.Linear(dim, len(alphabet))

    def embed(self, batch, pool="mean"):
        return masked_pool(
            self.enc(batch["input_ids"], batch["mask"]), batch["mask"], pool
        )

    def step(self, batch):
        ids, mask, a = batch["input_ids"], batch["mask"], self.alphabet
        cand = mask & (ids >= a.n_special)
        sel = (torch.rand(ids.shape, device=ids.device) < self.mask_prob) & cand
        if not sel.any():
            i = cand.nonzero(as_tuple=False)[0]
            sel[i[0], i[1]] = True
        labels = torch.full_like(ids, -100)
        labels[sel] = ids[sel]
        r = torch.rand(ids.shape, device=ids.device)
        inp = ids.clone()
        inp[sel & (r < 0.8)] = a.mask_idx
        rnd = sel & (r >= 0.8) & (r < 0.9)
        inp[rnd] = torch.randint(
            a.n_special, len(a), (int(rnd.sum()),), device=ids.device
        )
        logits = self.head(self.enc(inp, mask))
        return StepOutput(
            task_loss("mlm", logits, labels), task_pred("mlm", logits), labels
        )


class GraphModel(BioModel):
    """Graph-level prediction (molecules, protein structures, pathway graphs, ...)."""

    def __init__(
        self,
        in_dim,
        n_out=1,
        task="multiclass",
        hidden=64,
        num_layers=3,
        layer="gin",
        pool="meanmax",
        dropout=0.1,
        edge_dim=0,
        heads=4,
        num_relations=0,
        num_node_types=0,
    ):
        super().__init__()
        self.task, self.pool = task, pool
        n_out = 1 if task == "binary" else n_out
        self.enc = GNNEncoder(
            in_dim,
            hidden,
            num_layers,
            layer,
            dropout,
            edge_dim,
            heads,
            num_relations,
            num_node_types,
        )
        pd = hidden * (2 if pool == "meanmax" else 1)
        self.head = nn.Sequential(
            nn.Linear(pd, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, b: GraphBatch):
        h = self.enc(b.x, b.edge_index, b.edge_attr, b.edge_type, b.node_type)
        return self.head(global_pool(h, b.batch, b.num_graphs, self.pool))

    def step(self, b):
        out = self(b)
        return StepOutput(
            task_loss(self.task, out, b.y), task_pred(self.task, out, b.y), b.y
        )


class NodeModel(BioModel):
    """Node classification / regression on one (possibly large) network. Uses ``batch.mask``."""

    def __init__(
        self,
        in_dim,
        n_out,
        task="multiclass",
        hidden=64,
        num_layers=2,
        layer="gcn",
        dropout=0.3,
        edge_dim=0,
        heads=4,
        num_relations=0,
        num_node_types=0,
    ):
        super().__init__()
        self.task = task
        n_out = 1 if task == "binary" else n_out
        self.enc = GNNEncoder(
            in_dim,
            hidden,
            num_layers,
            layer,
            dropout,
            edge_dim,
            heads,
            num_relations,
            num_node_types,
        )
        self.head = nn.Linear(hidden, n_out)

    def forward(self, b):
        return self.head(
            self.enc(b.x, b.edge_index, b.edge_attr, b.edge_type, b.node_type)
        )

    def step(self, b):
        out, y = self(b), b.node_y
        if b.mask is not None:
            out, y = out[b.mask], y[b.mask]
        return StepOutput(task_loss(self.task, out, y), task_pred(self.task, out, y), y)


class LinkModel(BioModel):
    """Link prediction (PPI, drug-target, gene regulation). Batch must carry pos_edge / neg_edge."""

    task = "binary"

    def __init__(
        self,
        in_dim,
        hidden=64,
        num_layers=2,
        layer="sage",
        dropout=0.2,
        decoder="mlp",
        edge_dim=0,
        heads=4,
        num_relations=0,
        num_node_types=0,
    ):
        super().__init__()
        self.decoder = decoder
        self.enc = GNNEncoder(
            in_dim,
            hidden,
            num_layers,
            layer,
            dropout,
            edge_dim,
            heads,
            num_relations,
            num_node_types,
        )
        self.rel = nn.Parameter(torch.ones(hidden))
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def encode(self, b):
        return self.enc(b.x, b.edge_index, b.edge_attr, b.edge_type, b.node_type)

    def decode(self, z, e):
        zu, zv = z[e[0]], z[e[1]]
        if self.decoder == "dot":
            return (zu * zv).sum(-1)
        if self.decoder == "distmult":
            return (zu * self.rel * zv).sum(-1)
        return self.mlp(torch.cat([zu * zv, (zu - zv).abs()], -1)).squeeze(-1)

    def step(self, b):
        z = self.encode(b)
        s = torch.cat([self.decode(z, b.pos_edge), self.decode(z, b.neg_edge)])
        y = torch.cat(
            [torch.ones(b.pos_edge.size(1)), torch.zeros(b.neg_edge.size(1))]
        ).to(s.device)
        return StepOutput(
            F.binary_cross_entropy_with_logits(s.float(), y), s.float().sigmoid(), y
        )


class SeqGraphNodeModel(BioModel):
    """Fuses per-node sequences (e.g. protein sequence) with network context (e.g. PPI) -> node labels."""

    def __init__(
        self,
        alphabet: Alphabet,
        n_out,
        task="multiclass",
        hidden=64,
        seq_arch="cnn",
        in_dim=0,
        num_layers=2,
        layer="sage",
        dropout=0.2,
        seq_layers=3,
    ):
        super().__init__()
        self.task, self.pad = task, alphabet.pad_idx
        self.seq_enc = make_seq_encoder(
            seq_arch, len(alphabet), hidden, seq_layers, dropout
        )
        self.fuse = nn.Linear(hidden + in_dim, hidden)
        self.gnn = GNNEncoder(hidden, hidden, num_layers, layer, dropout)
        self.head = nn.Linear(hidden, 1 if task == "binary" else n_out)

    def forward(self, b):
        ids = b.node_seq
        mask = ids != self.pad
        mask[:, 0] |= ~mask.any(1)
        s = masked_pool(self.seq_enc(ids, mask), mask, "mean")
        if b.x is not None and b.x.size(1) > 0 and self.fuse.in_features > s.size(1):
            s = torch.cat([s, b.x], -1)
        return self.head(self.gnn(self.fuse(s), b.edge_index, b.edge_attr, b.edge_type))

    def step(self, b):
        out, y = self(b), b.node_y
        if b.mask is not None:
            out, y = out[b.mask], y[b.mask]
        return StepOutput(task_loss(self.task, out, y), task_pred(self.task, out, y), y)


class SkipGramDataset(Dataset):
    """(center, context) pairs from random walks, for DeepWalk / node2vec-style embeddings."""

    def __init__(
        self,
        net: BioNetwork,
        walk_length=20,
        walks_per_node=5,
        window=3,
        num_neg=5,
        seed=0,
    ):
        walks = net.random_walks(walks_per_node, walk_length, seed)
        cs, xs = [], []
        for o in range(1, window + 1):
            a, b = walks[:, :-o].reshape(-1), walks[:, o:].reshape(-1)
            cs += [a, b]
            xs += [b, a]
        self.c, self.x, self.num_neg = np.concatenate(cs), np.concatenate(xs), num_neg
        p = (net.degree().astype(float) + 1) ** 0.75
        self.noise = torch.tensor(p / p.sum(), dtype=torch.float)

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        return int(self.c[i]), int(self.x[i])

    def collate(self, items):
        c = torch.tensor([a for a, _ in items])
        x = torch.tensor([b for _, b in items])
        neg = torch.multinomial(
            self.noise, len(items) * self.num_neg, replacement=True
        ).view(len(items), -1)
        return {"center": c, "context": x, "neg": neg}


class Node2Vec(BioModel):
    task = "binary"

    def __init__(self, num_nodes: int, dim: int = 64):
        super().__init__()
        self.emb_in, self.emb_out = (
            nn.Embedding(num_nodes, dim),
            nn.Embedding(num_nodes, dim),
        )
        nn.init.normal_(self.emb_in.weight, std=0.1)
        nn.init.normal_(self.emb_out.weight, std=0.1)

    def embedding(self) -> torch.Tensor:
        return self.emb_in.weight.detach()

    def step(self, b):
        u, v, n = (
            self.emb_in(b["center"]),
            self.emb_out(b["context"]),
            self.emb_out(b["neg"]),
        )
        pos = (u * v).sum(-1)
        neg = torch.bmm(n, u.unsqueeze(-1)).squeeze(-1)
        loss = -F.logsigmoid(pos).mean() - F.logsigmoid(-neg).sum(1).mean()
        s = torch.cat([pos, neg.reshape(-1)])
        y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg.reshape(-1))])
        return StepOutput(loss, s.sigmoid(), y)
