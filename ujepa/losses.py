from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Multi-class soft Dice on one-hot targets. logits/target: (B, C, ...) / (B, ...)."""
    probs = torch.softmax(logits, dim=1)
    if target.dtype != torch.long:
        target = target.long()
    onehot = F.one_hot(target, num_classes=num_classes).permute(0, -1, *range(1, target.dim())).float()
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        valid = (target != ignore_index).unsqueeze(1).float()
        probs = probs * valid
        onehot = onehot * valid
    dims = tuple(range(2, probs.dim()))
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    # Skip absent classes in the batch for stability.
    present = onehot.sum(dims) > 0
    if present.any():
        return 1.0 - dice[present].mean()
    return 1.0 - dice.mean()


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long(), weight=class_weights)
    dice = soft_dice_loss(logits, target, num_classes)
    return dice_weight * dice + ce_weight * ce


def deep_supervision_loss(
    outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    target: torch.Tensor,
    num_classes: int,
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """outputs[0] is full resolution; later entries may be lower resolution."""
    if isinstance(outputs, torch.Tensor):
        return dice_ce_loss(outputs, target, num_classes)
    if weights is None:
        weights = [1.0 / (2**i) for i in range(len(outputs))]
        s = sum(weights)
        weights = [w / s for w in weights]
    total = 0.0
    for w, out in zip(weights, outputs):
        if out.shape[-3:] != target.shape[-3:]:
            tgt = F.interpolate(
                target.float().unsqueeze(1), size=out.shape[-3:], mode="nearest"
            ).squeeze(1).long()
        else:
            tgt = target
        total = total + w * dice_ce_loss(out, tgt, num_classes)
    return total


def jepa_smooth_l1(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 on masked positions only.

    pred_tokens / target_tokens: (B, N, C)
    visible: (B, N) bool, True = visible (not predicted against).
    """
    masked = ~visible
    if not masked.any():
        return pred_tokens.sum() * 0.0
    # Layer-normalize targets (stop-grad is handled by caller via no_grad / detach).
    tar = F.layer_norm(target_tokens, (target_tokens.shape[-1],))
    pred_m = pred_tokens[masked]
    tar_m = tar[masked]
    return F.smooth_l1_loss(pred_m, tar_m, beta=1.0)


def lambda_j_schedule(
    step: int,
    seg_only_steps: int = 3000,
    ramp_steps: int = 3000,
    lambda_max: float = 0.3,
) -> float:
    if step < seg_only_steps:
        return 0.0
    if step < seg_only_steps + ramp_steps:
        return lambda_max * (step - seg_only_steps + 1) / float(ramp_steps)
    return lambda_max
