"""Runnable demos and the built-in self-test."""

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

from ._version import __version__
from .auto import *  # noqa: F401,F403
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
    "DEMOS",
    "demo_auto",
    "demo_graph",
    "demo_link",
    "demo_mlm",
    "demo_node",
    "demo_node2vec",
    "demo_seq",
    "demo_seqgraph",
    "selftest",
]


def _cfg(epochs, **kw):
    return TrainConfig(
        epochs=epochs, verbose=kw.pop("verbose", True), out_dir=None, **kw
    )


def demo_seq(task="binary", arch="cnn", epochs=12, verbose=True):
    alph = DNA
    seqs, labels = synthetic_sequences(task=task)
    ds = SequenceDataset(seqs, labels, alph, task=task, augment=False)
    tr, va, te = auto_split(
        ds, stratify=labels if task in ("binary", "multiclass") else None
    )
    n_out = 3 if task == "multiclass" else (2 if task == "token_multiclass" else 1)
    model = SeqModel(alph, n_out, task, arch=arch, dim=64, num_layers=3)
    t = Trainer(model, task, _cfg(epochs, lr=2e-3, patience=0, verbose=verbose))
    t.fit(make_loader(tr, 32, True), make_loader(va, 64))
    return t.evaluate(make_loader(te, 64), "test")


def demo_mlm(epochs=3, verbose=True):
    seqs, _ = synthetic_sequences(n=200)
    ds = SequenceDataset(seqs, None, DNA, task="mlm", add_special=True)
    model = SeqMLM(DNA, "transformer", dim=64, num_layers=2)
    t = Trainer(model, "mlm", _cfg(epochs, lr=1e-3, patience=0, verbose=verbose))
    tr, va, te = auto_split(ds)
    t.fit(make_loader(tr, 32, True), make_loader(va, 64))
    return t.evaluate(make_loader(te, 64), "test")


def demo_graph(epochs=25, layer="gin", verbose=True):
    ds = GraphDataset(synthetic_graphs())
    tr, va, te = auto_split(ds, stratify=[int(g.y) for g in ds.graphs])
    model = GraphModel(2, 2, "multiclass", hidden=48, layer=layer)
    t = Trainer(model, "multiclass", _cfg(epochs, lr=3e-3, patience=0, verbose=verbose))
    t.fit(make_loader(tr, 32, True), make_loader(va, 64))
    return t.evaluate(make_loader(te, 64), "test")


def demo_node(epochs=120, layer="gcn", verbose=True):
    net, lab = synthetic_network()
    g = net.to_graph(node_y=torch.tensor(lab))
    tr, va, te = node_split_loaders(g)
    model = NodeModel(g.x.size(1), 3, "multiclass", hidden=32, layer=layer)
    t = Trainer(
        model,
        "multiclass",
        _cfg(
            epochs,
            lr=1e-2,
            patience=0,
            scheduler="none",
            warmup_epochs=0,
            verbose=verbose,
        ),
    )
    t.fit(tr, va)
    return t.evaluate(te, "test")


def demo_link(epochs=60, verbose=True):
    net, _ = synthetic_network(p_in=0.1)
    g = net.to_graph(feature="given")
    tr, va, te = link_split_loaders(g)
    model = LinkModel(g.x.size(1), hidden=32, layer="sage")
    t = Trainer(
        model,
        "binary",
        _cfg(
            epochs,
            lr=5e-3,
            patience=0,
            scheduler="none",
            warmup_epochs=0,
            verbose=verbose,
        ),
    )
    t.fit(tr, va)
    return t.evaluate(te, "test")


def demo_node2vec(epochs=2, verbose=True):
    net, lab = synthetic_network(n_nodes=200)
    ds = SkipGramDataset(net, walk_length=10, walks_per_node=3, window=2)
    model = Node2Vec(net.num_nodes, 32)
    t = Trainer(
        model,
        "binary",
        _cfg(
            epochs,
            lr=1e-2,
            patience=0,
            scheduler="none",
            warmup_epochs=0,
            verbose=verbose,
        ),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=True, collate_fn=ds.collate)
    t.fit(loader)
    Z = model.embedding().numpy()
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    same = (lab[:, None] == lab[None, :]) & ~np.eye(len(lab), dtype=bool)
    sim = Z @ Z.T
    return {
        "sim_same_comm": float(sim[same].mean()),
        "sim_diff_comm": float(sim[~same & ~np.eye(len(lab), dtype=bool)].mean()),
    }


def demo_seqgraph(epochs=60, verbose=True):
    g = synthetic_seq_network()
    tr, va, te = node_split_loaders(g)
    model = SeqGraphNodeModel(DNA, 3, hidden=32, in_dim=0)
    t = Trainer(
        model,
        "multiclass",
        _cfg(
            epochs,
            lr=5e-3,
            patience=0,
            scheduler="none",
            warmup_epochs=0,
            verbose=verbose,
        ),
    )
    t.fit(tr, va)
    return t.evaluate(te, "test")


def demo_auto(verbose=True):
    seqs, labels = synthetic_sequences(n=400)
    ds = SequenceDataset(seqs, labels, DNA, task="binary")
    res = auto_fit(
        lambda dim=64, num_layers=3, dropout=0.1, arch="cnn": SeqModel(
            DNA, 1, "binary", arch=arch, dim=dim, num_layers=num_layers, dropout=dropout
        ),
        ds,
        "binary",
        search_space={
            "lr": (1e-3, 1e-2, "log"),
            "dim": [32, 64],
            "num_layers": (2, 4, "int"),
            "arch": ["cnn", "gru"],
            "batch_size": [32, 64],
        },
        n_trials=3,
        trial_epochs=3,
        final_epochs=8,
        stratify_labels=labels,
        out_dir=None,
        monitor="val_loss",
        verbose=verbose,
    )
    return res.test_metrics


DEMOS = {
    "seq": lambda v: demo_seq("binary", "cnn", 12, v),
    "multiclass": lambda v: demo_seq("multiclass", "cnn", 15, v),
    "regression": lambda v: demo_seq("regression", "cnn", 20, v),
    "token": lambda v: demo_seq("token_multiclass", "cnn", 12, v),
    "transformer": lambda v: demo_seq("binary", "transformer", 12, v),
    "mlm": lambda v: demo_mlm(verbose=v),
    "graph": lambda v: demo_graph(verbose=v),
    "node": lambda v: demo_node(verbose=v),
    "link": lambda v: demo_link(verbose=v),
    "node2vec": lambda v: demo_node2vec(verbose=v),
    "seqgraph": lambda v: demo_seqgraph(verbose=v),
    "auto": lambda v: demo_auto(verbose=v),
}


def selftest() -> None:
    print(f"biotorch {__version__} self-test (torch {torch.__version__})")
    # primitives
    assert (
        reverse_complement("ATGC") == "GCAT"
        and translate("ATGGCCTAA", to_stop=True) == "MA"
    )
    assert abs(gc_content("GGCC") - 1.0) < 1e-9 and kmer_counts("ACGT", 1).sum() == 1.0
    net = BioNetwork.from_edge_list([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])
    assert (
        net.num_nodes == 5
        and len(net.connected_components()) == 2
        and net.k_hop("A", 1) == ["A", "B", "C"]
    )
    assert debruijn_graph(["ACGTACGT"], 3).num_nodes > 0
    g = net.to_graph(feature="random")
    assert collate_graphs([g, g]).num_graphs == 2
    # models
    checks = [
        ("seq", 0.85, "test_acc"),
        ("multiclass", 0.7, "test_acc"),
        ("regression", 0.5, "test_r2"),
        ("token", 0.9, "test_acc"),
        ("graph", 0.6, "test_acc"),
        ("node", 0.7, "test_acc"),
        ("link", 0.62, "test_auroc"),
        ("seqgraph", 0.7, "test_acc"),
    ]
    for name, thr, key in checks:
        m = DEMOS[name](False)
        status = "ok " if m[key] >= thr else "LOW"
        print(f"  {status} {name:<11} {key}={m[key]:.3f} (>= {thr})")
        assert not math.isnan(m[key])
    m = demo_mlm(2, False)
    print(f"  ok  mlm         ppl={m['test_ppl']:.2f}")
    m = demo_node2vec(2, False)
    print(f"  ok  node2vec    {m}")
    m = demo_auto(False)
    print(f"  ok  auto_fit    { ({k: round(v, 3) for k, v in m.items()}) }")
    print("self-test finished")
