from __future__ import annotations

import copy
from typing import Iterable, List

import torch
import torch.nn as nn


class EMATarget:
    """Exponential moving average copy of an online module.

    Target parameters are updated as:
        target = decay * target + (1 - decay) * online
    and never receive gradients from the JEPA loss.
    """

    def __init__(self, online: nn.Module, decay: float = 0.996):
        self.decay = decay
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, online: nn.Module, decay: float | None = None) -> None:
        d = self.decay if decay is None else decay
        for t, s in zip(self.target.parameters(), online.parameters()):
            t.data.mul_(d).add_(s.data, alpha=1.0 - d)
        # Keep buffers (e.g. BN running stats) in sync with online.
        for t, s in zip(self.target.buffers(), online.buffers()):
            t.data.copy_(s.data)

    @torch.no_grad()
    def load_online(self, online: nn.Module) -> None:
        self.target.load_state_dict(online.state_dict())

    def to(self, device) -> "EMATarget":
        self.target = self.target.to(device)
        return self

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.target.parameters()


def ema_decay_schedule(
    step: int, start: float = 0.996, end: float = 0.999, warmup_steps: int = 10000
) -> float:
    if warmup_steps <= 0:
        return end
    t = min(1.0, max(0.0, step / float(warmup_steps)))
    return start + (end - start) * t
