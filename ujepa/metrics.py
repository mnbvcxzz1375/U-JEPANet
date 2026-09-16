from __future__ import annotations

"""Validation metrics: per-organ Dice and aggregate means.

Evaluation is model-only (no labels used as input). Callers must pass GT.
Does not read imagesVal from disk by itself — loader/experiment harness owns I/O.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> Dict[int, float]:
    """pred/target: (D,H,W) or (B,D,H,W) integer maps. Returns class→Dice."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    out: Dict[int, float] = {}
    start = 1 if ignore_background else 0
    for c in range(start, num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        if denom == 0:
            out[c] = float("nan")
        else:
            out[c] = float(2.0 * inter / denom)
    return out


def aggregate_dice(per_class: Dict[int, float]) -> Dict[str, float]:
    vals = [v for v in per_class.values() if v == v]  # drop NaN
    mean_fg = float(sum(vals) / len(vals)) if vals else float("nan")
    return {"mean_fg_dice": mean_fg, "n_scored": float(len(vals))}


@torch.no_grad()
def evaluate_segmentation_batch(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """logits (B,C,D,H,W) or primary head; target (B,D,H,W)."""
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = torch.argmax(logits, dim=1)
    metrics: Dict[str, float] = {}
    all_per: Dict[int, List[float]] = {c: [] for c in range(1, num_classes)}
    for b in range(pred.shape[0]):
        pc = dice_per_class(pred[b], target[b], num_classes)
        for c, v in pc.items():
            if v == v:
                all_per[c].append(v)
    per_mean = {
        c: (float(sum(vs) / len(vs)) if vs else float("nan"))
        for c, vs in all_per.items()
    }
    for c, v in per_mean.items():
        metrics[f"dice_c{c:02d}"] = v
    vals = [v for v in per_mean.values() if v == v]
    metrics["mean_fg_dice"] = float(sum(vals) / len(vals)) if vals else float("nan")
    return metrics


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Average per-organ Dice over a labeled validation loader."""
    was_training = model.training
    model.eval()
    acc: Dict[int, List[float]] = {c: [] for c in range(1, num_classes)}
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            x, y = batch
            labeled = torch.ones(x.shape[0], dtype=torch.bool)
        else:
            x = batch["image"]
            y = batch["label"]
            labeled = batch.get("is_labeled")
            if labeled is None:
                labeled = torch.ones(x.shape[0], dtype=torch.bool)
        # Validation should only use labeled rows with real GT.
        if not bool(labeled.all().item()):
            x = x[labeled]
            y = y[labeled]
            if x.shape[0] == 0:
                continue
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pred = torch.argmax(logits, dim=1)
        for b in range(pred.shape[0]):
            pc = dice_per_class(pred[b], y[b], num_classes)
            for c, v in pc.items():
                if v == v:
                    acc[c].append(v)
            n += 1
    if was_training:
        model.train()
        # Re-assert EMA eval invariant after model.train().
        tgt = getattr(model, "target", None)
        if tgt is not None and hasattr(tgt, "target"):
            tgt.target.eval()

    metrics: Dict[str, float] = {}
    for c, vs in acc.items():
        metrics[f"dice_c{c:02d}"] = float(sum(vs) / len(vs)) if vs else float("nan")
    scored = [metrics[f"dice_c{c:02d}"] for c in acc if metrics[f"dice_c{c:02d}"] == metrics[f"dice_c{c:02d}"]]
    metrics["mean_fg_dice"] = float(sum(scored) / len(scored)) if scored else float("nan")
    metrics["n_val_cases"] = float(n)
    return metrics
