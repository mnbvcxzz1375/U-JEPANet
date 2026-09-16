from __future__ import annotations

import copy
from typing import Iterable, Optional

import torch
import torch.nn as nn


class EMATarget(nn.Module):
    """EMA copy of the online encoder path.

    Registered as a submodule so ``state_dict()`` includes EMA weights for
    checkpoint/resume. Parameters never receive JEPA gradients.
    """

    def __init__(self, online: nn.Module, decay: float = 0.996):
        super().__init__()
        self.decay = decay
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, online: nn.Module, decay: Optional[float] = None) -> None:
        d = self.decay if decay is None else decay
        for t, s in zip(self.target.parameters(), online.parameters()):
            t.data.mul_(d).add_(s.data, alpha=1.0 - d)
        for t, s in zip(self.target.buffers(), online.buffers()):
            # Skip num_batches_tracked-like integer buffers if any.
            if t.dtype.is_floating_point:
                t.data.copy_(s.data)
            else:
                t.data.copy_(s.data)

    @torch.no_grad()
    def load_online(self, online: nn.Module) -> None:
        self.target.load_state_dict(online.state_dict())

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
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
