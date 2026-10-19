"""Trainer: AMP, accumulation, clipping, schedulers, early stopping, checkpoints, resume."""

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
from .metrics import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

__all__ = [
    "Callback",
    "TrainConfig",
    "Trainer",
]


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 32
    optimizer: str = "adamw"  # adamw | adam | sgd
    scheduler: str = "cosine"  # cosine | linear | plateau | none
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True  # only active on CUDA
    accum_steps: int = 1
    patience: int = 10  # early stopping (epochs without improvement); 0 disables
    monitor: str = "val_loss"
    mode: str = "min"  # min | max
    out_dir: Optional[str] = None
    seed: int = 0
    device: str = "auto"
    resume: bool = False
    train_metrics: bool = True
    verbose: bool = True


class Callback:
    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None: ...


class Trainer:
    def __init__(
        self,
        model: BioModel,
        task: Optional[str] = None,
        cfg: Optional[TrainConfig] = None,
        callbacks: Sequence[Callback] = (),
    ):
        self.cfg = cfg or TrainConfig()
        self.model, self.task = model, task or getattr(model, "task", "multiclass")
        self.device = get_device(self.cfg.device)
        self.model.to(self.device)
        self.use_amp = self.cfg.amp and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.callbacks = list(callbacks)
        self.history: List[Dict[str, float]] = []
        self.best_score: Optional[float] = None
        self.best_state = None
        self.optimizer = self.sched = None
        self._per_step = False
        if self.cfg.out_dir:
            os.makedirs(self.cfg.out_dir, exist_ok=True)

    # ---- setup ----
    def _build(self, steps_per_epoch: int):
        c = self.cfg
        decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [
            p for p in self.model.parameters() if p.requires_grad and p.ndim < 2
        ]
        groups = [
            {"params": decay, "weight_decay": c.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if c.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(groups, lr=c.lr)
        elif c.optimizer == "adam":
            self.optimizer = torch.optim.Adam(groups, lr=c.lr)
        else:
            self.optimizer = torch.optim.SGD(groups, lr=c.lr, momentum=0.9)
        total, warm = c.epochs * steps_per_epoch, int(c.warmup_epochs * steps_per_epoch)
        if c.scheduler in ("cosine", "linear", "none"):

            def lam(s):
                if s < warm:
                    return (s + 1) / max(1, warm)
                prog = min(1.0, (s - warm) / max(1, total - warm))
                if c.scheduler == "cosine":
                    return c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (
                        1 + math.cos(math.pi * prog)
                    )
                if c.scheduler == "linear":
                    return c.min_lr_ratio + (1 - c.min_lr_ratio) * (1 - prog)
                return 1.0

            self.sched, self._per_step = (
                torch.optim.lr_scheduler.LambdaLR(self.optimizer, lam),
                True,
            )
        elif c.scheduler == "plateau":
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=c.mode,
                factor=0.5,
                patience=max(1, c.patience // 3),
            )

    def _optim_step(self):
        c = self.cfg
        if c.grad_clip:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self._per_step:
            self.sched.step()

    @staticmethod
    def _collect(out: StepOutput, P: list, T: list):
        p, t = out.pred.detach().float().cpu(), out.target.detach().cpu()
        if p.dim() == 3:
            p, t = p.reshape(-1, p.size(-1)), t.reshape(-1)
        if (
            t.dtype == torch.long
            and t.dim() == p.dim() - 1
            or (t.dtype == torch.long and p.dim() == 2)
        ):
            keep = t != -100
            p, t = p[keep], t[keep]
        P.append(p)
        T.append(t)

    # ---- loops ----
    def _run_epoch(
        self, loader, train: bool, with_metrics: bool = True
    ) -> Dict[str, float]:
        self.model.train(train)
        losses, P, T, bad, pending = [], [], [], 0, 0
        accum = self.cfg.accum_steps
        amp_ctx = (
            (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
            if self.use_amp
            else contextlib.nullcontext
        )
        with torch.enable_grad() if train else torch.no_grad():
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            for batch in loader:
                batch = move_to(batch, self.device)
                with amp_ctx():
                    out = self.model.step(batch)
                if not torch.isfinite(out.loss):
                    bad += 1
                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                        pending = 0
                    if bad > 10:
                        raise RuntimeError(
                            "loss is not finite (diverged?) - try a lower lr"
                        )
                    continue
                if train:
                    self.scaler.scale(out.loss / accum).backward()
                    pending += 1
                    if pending == accum:
                        self._optim_step()
                        pending = 0
                losses.append(float(out.loss.detach()))
                if with_metrics:
                    self._collect(out, P, T)
            if train and pending:
                self._optim_step()
        stats = {"loss": float(np.mean(losses)) if losses else float("nan")}
        if with_metrics and P:
            try:
                stats.update(
                    compute_metrics(
                        self.task, torch.cat(P).numpy(), torch.cat(T).numpy()
                    )
                )
            except Exception as e:  # metrics must never kill training
                print(f"[biotorch] metric warning: {e}", file=sys.stderr)
        return stats

    def evaluate(self, loader, prefix: str = "val") -> Dict[str, float]:
        return {f"{prefix}_{k}": v for k, v in self._run_epoch(loader, False).items()}

    @torch.no_grad()
    def predict(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        P, T = [], []
        for batch in loader:
            self._collect(self.model.step(move_to(batch, self.device)), P, T)
        return torch.cat(P).numpy(), torch.cat(T).numpy()

    def _better(self, v: float) -> bool:
        if self.best_score is None or math.isnan(self.best_score):
            return True
        return v < self.best_score if self.cfg.mode == "min" else v > self.best_score

    def save(self, name: str = "last") -> Optional[str]:
        if not self.cfg.out_dir:
            return None
        path = os.path.join(self.cfg.out_dir, f"{name}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": asdict(self.cfg),
                "history": self.history,
                "best_score": self.best_score,
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "sched": self.sched.state_dict() if self.sched else None,
            },
            path,
        )
        return path

    def load(self, path: str, strict: bool = True) -> dict:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"], strict=strict)
        return ck

    def fit(self, train_loader, val_loader=None) -> List[Dict[str, float]]:
        c = self.cfg
        seed_everything(c.seed)
        self._build(max(1, math.ceil(len(train_loader) / c.accum_steps)))
        start, bad_epochs = 0, 0
        last = os.path.join(c.out_dir, "last.pt") if c.out_dir else None
        if c.resume and last and os.path.exists(last):
            ck = self.load(last)
            self.history, self.best_score = ck["history"], ck["best_score"]
            self.optimizer.load_state_dict(ck["optimizer"])
            self.sched.load_state_dict(ck["sched"])
            start = len(self.history)
            self._log(f"resumed from epoch {start}")
        for ep in range(start, c.epochs):
            t0 = time.time()
            logs = {"epoch": ep + 1}
            logs.update(
                {
                    f"train_{k}": v
                    for k, v in self._run_epoch(
                        train_loader, True, c.train_metrics
                    ).items()
                }
            )
            if val_loader is not None:
                logs.update(self.evaluate(val_loader, "val"))
            logs["lr"] = self.optimizer.param_groups[0]["lr"]
            logs["time"] = time.time() - t0
            key = c.monitor if c.monitor in logs else "train_loss"
            score = logs[key]
            if isinstance(
                self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau
            ) and not math.isnan(score):
                self.sched.step(score)
            if self._better(score):
                self.best_score, bad_epochs = score, 0
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                self.save("best")
            else:
                bad_epochs += 1
            self.history.append(logs)
            self.save("last")
            self._log(logs)
            for cb in self.callbacks:
                cb.on_epoch_end(self, ep, logs)
            if c.patience and bad_epochs >= c.patience:
                self._log(
                    f"early stopping at epoch {ep + 1} (best {key}={self.best_score:.4f})"
                )
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if c.out_dir:
            with open(os.path.join(c.out_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=1)
        return self.history

    def _log(self, logs):
        if not self.cfg.verbose:
            return
        if isinstance(logs, str):
            print(f"[biotorch] {logs}")
            return
        skip = {"epoch", "time"}
        body = " ".join(
            f"{k}={v:.4f}"
            for k, v in logs.items()
            if k not in skip and isinstance(v, float)
        )
        print(
            f"[{int(logs['epoch']):03d}/{self.cfg.epochs}] {body} ({logs['time']:.1f}s)"
        )
