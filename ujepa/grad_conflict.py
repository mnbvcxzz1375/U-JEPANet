from __future__ import annotations

"""Gradient conflict diagnostic: cos(g_seg, g_J) and norm ratio on shared encoder."""

from typing import Dict, List

import torch
import torch.nn as nn

from .losses import dice_ce_loss, jepa_smooth_l1
from .model import UJEPATrainer


def _grad_of(loss: torch.Tensor, params: List[nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flats = []
    for g in grads:
        if g is None:
            continue
        flats.append(g.reshape(-1))
    if not flats:
        return torch.zeros(1, device=loss.device)
    return torch.cat(flats)


def _grads_paired(
    loss_a: torch.Tensor, loss_b: torch.Tensor, params: List[nn.Parameter]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients restricted to params that receive signal from BOTH losses."""
    ga = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, retain_graph=True, allow_unused=True)
    fa, fb = [], []
    for a, b in zip(ga, gb):
        if a is None or b is None:
            continue
        fa.append(a.reshape(-1))
        fb.append(b.reshape(-1))
    if not fa:
        z = torch.zeros(1, device=loss_a.device)
        return z, z
    return torch.cat(fa), torch.cat(fb)


@torch.no_grad()
def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x).item())


def measure_grad_conflict(
    model: UJEPATrainer,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    x_j: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """One-step diagnostic. Does not update weights.

    Shared params = online encoder path (including dualpath if present).
    Predictor is excluded from g_seg (it never gets seg grads).
    """
    model.train()
    if model.target is not None:
        model.target.train(False)
        model.target.target.eval()

    shared = [p for p in model.online.parameters() if p.requires_grad]
    logits = model.forward_seg(x_l)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    l_seg = dice_ce_loss(logits, y_l, num_classes)
    out = model.jepa_step(x_j)
    l_j = jepa_smooth_l1(out["pred_tokens"], out["tgt_tokens"], out["visible"])
    g_seg, g_j = _grads_paired(l_seg, l_j, shared)

    n_s = max(_norm(g_seg), 1e-12)
    n_j = max(_norm(g_j), 1e-12)
    cos = float(torch.dot(g_seg, g_j).item() / (n_s * n_j))
    return {
        "l_seg": float(l_seg.detach().item()),
        "l_jepa": float(l_j.detach().item()),
        "norm_g_seg": n_s,
        "norm_g_jepa": n_j,
        "norm_ratio_j_over_s": n_j / n_s,
        "cos_gseg_gjepa": cos,
        "conflict": float(cos < 0),
    }


def run_grad_conflict_scan(
    model: UJEPATrainer,
    loader_l,
    loader_j,
    device: torch.device,
    num_classes: int,
    n_batches: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    it_l = iter(loader_l)
    it_j = iter(loader_j)
    rows = []
    for i in range(n_batches):
        try:
            bl = next(it_l)
        except StopIteration:
            it_l = iter(loader_l)
            bl = next(it_l)
        try:
            bj = next(it_j)
        except StopIteration:
            it_j = iter(loader_j)
            bj = next(it_j)
        x_l = bl["image"].to(device)
        y_l = bl["label"].to(device)
        x_j = bj["image"].to(device)
        m = measure_grad_conflict(model, x_l, y_l, x_j, num_classes)
        rows.append(m)
        print(
            f"batch={i:03d} cos={m['cos_gseg_gjepa']:+.4f} "
            f"ratio={m['norm_ratio_j_over_s']:.3f} conflict={int(m['conflict'])}",
            flush=True,
        )
    cos = [r["cos_gseg_gjepa"] for r in rows]
    ratio = [r["norm_ratio_j_over_s"] for r in rows]
    return {
        "n": len(rows),
        "cos_mean": sum(cos) / len(cos),
        "cos_min": min(cos),
        "cos_max": max(cos),
        "frac_conflict": sum(r["conflict"] for r in rows) / len(rows),
        "ratio_mean": sum(ratio) / len(ratio),
        "rows": rows,
    }
