"""Biological sequences: alphabets, tokenisation, k-mers, FASTA/FASTQ I/O, SequenceDataset."""

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

from .utils import *  # noqa: F401,F403
from .utils import _open  # noqa: F401

__all__ = [
    "Alphabet",
    "CODON_TABLE",
    "DNA",
    "PROTEIN",
    "RNA",
    "SequenceDataset",
    "TASKS",
    "gc_content",
    "kmer_counts",
    "kmers",
    "one_hot",
    "read_fasta",
    "read_fastq",
    "reverse_complement",
    "transcribe",
    "translate",
    "write_fasta",
]


class Alphabet:
    """Token vocabulary. Indices 0-4 are always <pad>,<unk>,<cls>,<eos>,<mask>."""

    SPECIALS = ("<pad>", "<unk>", "<cls>", "<eos>", "<mask>")

    def __init__(self, residues: Iterable[str], name: str = "custom"):
        self.name = name
        self.tokens = list(self.SPECIALS) + list(residues)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.pad_idx, self.unk_idx, self.cls_idx, self.eos_idx, self.mask_idx = range(5)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def n_special(self) -> int:
        return len(self.SPECIALS)

    def encode(self, seq: str, add_special: bool = False) -> List[int]:
        ids = [self.stoi.get(c, self.unk_idx) for c in seq.strip().upper()]
        if add_special:
            ids = [self.cls_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        out = []
        for i in ids:
            t = self.tokens[int(i)]
            if strip_special and t in self.SPECIALS:
                continue
            out.append(t)
        return "".join(out)


DNA = Alphabet("ACGTN", "dna")
RNA = Alphabet("ACGUN", "rna")
PROTEIN = Alphabet("ACDEFGHIKLMNPQRSTVWYXBZUO", "protein")

_COMP = str.maketrans(
    "ACGTUNacgtunRYKMBDHVSWrykmbdhvsw", "TGCAANtgcaanYRMKVHDBSWyrmkvhdbsw"
)
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {
    "".join(c): _AAS[i] for i, c in enumerate(itertools.product(_BASES, repeat=3))
}


def reverse_complement(seq: str, rna: bool = False) -> str:
    out = seq.translate(_COMP)[::-1]
    return out.replace("T", "U").replace("t", "u") if rna else out


def transcribe(dna: str) -> str:
    return dna.upper().replace("T", "U")


def translate(seq: str, to_stop: bool = False, frame: int = 0) -> str:
    seq = seq.upper().replace("U", "T")[frame:]
    aa = []
    for i in range(0, len(seq) - 2, 3):
        a = CODON_TABLE.get(seq[i : i + 3], "X")
        if a == "*" and to_stop:
            break
        aa.append(a)
    return "".join(aa)


def gc_content(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(1, len(s))


def kmers(seq: str, k: int, stride: int = 1) -> List[str]:
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1, stride)]


def kmer_counts(
    seq: str, k: int = 3, letters: str = "ACGT", normalize: bool = True
) -> np.ndarray:
    idx = {c: i for i, c in enumerate(letters)}
    v = np.zeros(len(letters) ** k, dtype=np.float32)
    for km in kmers(seq.upper(), k):
        if all(c in idx for c in km):
            j = 0
            for c in km:
                j = j * len(letters) + idx[c]
            v[j] += 1
    if normalize and v.sum() > 0:
        v /= v.sum()
    return v


def one_hot(seq: str, alphabet: Alphabet = DNA) -> torch.Tensor:
    """[L, n_residues] one-hot (special/unknown characters -> all zeros)."""
    ids = torch.tensor(alphabet.encode(seq)) - alphabet.n_special
    out = torch.zeros(len(ids), len(alphabet) - alphabet.n_special)
    ok = ids >= 0
    out[ok.nonzero(as_tuple=True)[0], ids[ok]] = 1.0
    return out


# ---- file I/O ---------------------------------------------------------------
def read_fasta(path: str) -> List[Tuple[str, str]]:
    records, name, buf = [], None, []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def write_fasta(records: Iterable[Tuple[str, str]], path: str, width: int = 80) -> None:
    with _open(path, "wt") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            for i in range(0, len(seq), width):
                f.write(seq[i : i + width] + "\n")


def read_fastq(path: str) -> List[Tuple[str, str, str]]:
    out = []
    with _open(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            seq, _, qual = f.readline().strip(), f.readline(), f.readline().strip()
            out.append((h[1:].strip(), seq, qual))
    return out


# ---- dataset ----------------------------------------------------------------
TASKS = ("multiclass", "binary", "multilabel", "regression", "token_multiclass", "mlm")


class SequenceDataset(Dataset):
    """Generic biological-sequence dataset.

    labels per task:
      multiclass -> int; binary -> 0/1; multilabel -> list of 0/1; regression -> float(s);
      token_multiclass -> list of ints, one per residue; None -> unlabeled (e.g. MLM).
    """

    def __init__(
        self,
        sequences: Sequence[str],
        labels: Optional[Sequence] = None,
        alphabet: Alphabet = DNA,
        task: str = "multiclass",
        max_len: Optional[int] = None,
        add_special: bool = False,
        augment: bool = False,
        rc_prob: float = 0.5,
        names: Optional[Sequence[str]] = None,
        classes: Optional[List[str]] = None,
    ):
        assert task in TASKS, f"task must be one of {TASKS}"
        self.sequences = list(sequences)
        self.labels = None if labels is None else list(labels)
        self.alphabet, self.task, self.max_len = alphabet, task, max_len
        self.add_special, self.augment, self.rc_prob = add_special, augment, rc_prob
        self.names, self.classes = names, classes

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def n_out(self) -> int:
        if self.task == "multiclass" or self.task == "token_multiclass":
            if self.classes:
                return len(self.classes)
            flat = np.concatenate([np.atleast_1d(l) for l in self.labels])
            return int(flat.max()) + 1
        if self.task == "multilabel":
            return len(self.labels[0])
        return 1

    def __getitem__(self, i: int):
        seq = self.sequences[i]
        lab = None if self.labels is None else self.labels[i]
        token = self.task == "token_multiclass"
        if token:
            lab = list(lab)
        if self.max_len and len(seq) > self.max_len:
            s = random.randint(0, len(seq) - self.max_len) if self.augment else 0
            seq = seq[s : s + self.max_len]
            if token:
                lab = lab[s : s + self.max_len]
        if (
            self.augment
            and self.alphabet.name in ("dna", "rna")
            and random.random() < self.rc_prob
        ):
            seq = reverse_complement(seq, rna=self.alphabet.name == "rna")
            if token:
                lab = lab[::-1]
        ids = self.alphabet.encode(seq, self.add_special)
        if token and self.add_special:
            lab = [-100] + lab + [-100]
        return ids, lab

    def collate(self, items) -> Dict[str, torch.Tensor]:
        B, L = len(items), max(len(x) for x, _ in items)
        ids = torch.full((B, L), self.alphabet.pad_idx, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)
        for b, (x, _) in enumerate(items):
            ids[b, : len(x)] = torch.tensor(x, dtype=torch.long)
            mask[b, : len(x)] = True
        batch = {"input_ids": ids, "mask": mask}
        if items[0][1] is not None:
            if self.task == "token_multiclass":
                lab = torch.full((B, L), -100, dtype=torch.long)
                for b, (_, l) in enumerate(items):
                    lab[b, : len(l)] = torch.tensor(l, dtype=torch.long)
            else:
                arr = np.asarray([l for _, l in items])
                lab = torch.as_tensor(
                    arr,
                    dtype=torch.long if self.task == "multiclass" else torch.float32,
                )
            batch["labels"] = lab
        return batch

    # ---- constructors ----
    @classmethod
    def from_records(
        cls, records: Sequence[Tuple[str, str]], **kw
    ) -> "SequenceDataset":
        return cls([s for _, s in records], names=[n for n, _ in records], **kw)

    @classmethod
    def from_fasta(
        cls, path: str, label_fn: Optional[Callable[[str], Any]] = None, **kw
    ) -> "SequenceDataset":
        recs = read_fasta(path)
        labels = [label_fn(n) for n, _ in recs] if label_fn else None
        return cls(
            [s for _, s in recs], labels=labels, names=[n for n, _ in recs], **kw
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        seq_col: str = "sequence",
        label_col: Optional[str] = "label",
        task: str = "multiclass",
        **kw,
    ) -> "SequenceDataset":
        delim = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
        with _open(path) as f:
            rows = list(csv.DictReader(f, delimiter=delim))
        seqs = [r[seq_col] for r in rows]
        labels, classes = None, None
        if label_col and label_col in rows[0]:
            raw = [r[label_col] for r in rows]
            if task in ("multiclass", "binary"):
                try:
                    labels = [int(float(x)) for x in raw]
                    if task == "multiclass":
                        classes = [str(i) for i in range(max(labels) + 1)]
                except ValueError:
                    classes = sorted(set(raw))
                    labels = [classes.index(x) for x in raw]
            elif task == "regression":
                labels = [float(x) for x in raw]
            elif task == "multilabel":
                labels = [[int(t) for t in x.replace(";", ",").split(",")] for x in raw]
        return cls(seqs, labels=labels, task=task, classes=classes, **kw)
