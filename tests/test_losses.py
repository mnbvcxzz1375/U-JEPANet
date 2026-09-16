from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ujepa.losses import dice_ce_loss, jepa_smooth_l1, lambda_j_schedule


def test_lambda_schedule():
    assert lambda_j_schedule(0, 100, 100, 0.3) == 0.0
    assert lambda_j_schedule(99, 100, 100, 0.3) == 0.0
    mid = lambda_j_schedule(150, 100, 100, 0.3)
    assert 0 < mid < 0.3
    assert abs(lambda_j_schedule(300, 100, 100, 0.3) - 0.3) < 1e-9


def test_dice_ce_finite():
    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8, 8))
    loss = dice_ce_loss(logits, target, 4)
    loss.backward()
    assert torch.isfinite(loss)


def test_jepa_smooth_l1_only_masked():
    pred = torch.zeros(1, 4, 8)
    tgt = torch.zeros(1, 4, 8)
    pred[0, 1] = 10.0
    tgt[0, 0] = 10.0
    visible = torch.zeros(1, 4, dtype=torch.bool)
    visible[0, 0] = True  # position 0 is visible; its error must be ignored
    loss = jepa_smooth_l1(pred, tgt, visible)
    assert torch.isfinite(loss)
    # Changing visible position content should not change loss
    pred2 = pred.clone()
    pred2[0, 0] = 999.0
    loss2 = jepa_smooth_l1(pred2, tgt, visible)
    assert torch.allclose(loss, loss2)
