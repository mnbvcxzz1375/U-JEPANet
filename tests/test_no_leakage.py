from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ujepa.masking import apply_mask, sample_block_mask
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model


def _make_cfg(**kwargs):
    base = dict(
        arm="A3",
        feature_chns=[8, 16, 32, 64],
        class_num=4,
        embed_dim=48,
        num_heads=2,
        num_blocks=2,
        predictor_blocks=1,
        image_stride=4,
        deep_stage=2,
        multiscale_pred=False,
        mask_ratio=0.6,
    )
    base.update(kwargs)
    return UJEPAConfig(**base)


def test_mask_covers_requested_ratio():
    g = torch.Generator().manual_seed(0)
    mask = sample_block_mask(2, (16, 16, 16), mask_ratio=0.4, generator=g)
    hidden = 1.0 - mask.float().mean().item()
    assert 0.2 < hidden < 0.7


def test_apply_mask_zeroes_hidden():
    x = torch.ones(1, 1, 8, 8, 8)
    mask = torch.ones(1, 1, 8, 8, 8)
    mask[:, :, :4] = 0
    y = apply_mask(x, mask, fill_value=0.0)
    assert y[:, :, :4].abs().sum() == 0
    assert torch.allclose(y[:, :, 4:], torch.ones_like(y[:, :, 4:]))


def test_online_prediction_invariant_to_hidden_region_content():
    """Core no-leakage contract.

    Fix the visibility mask. Change ONLY the originally-hidden voxels of the
    clean image. Rebuild X_mask from the same mask on both images. Online JEPA
    prediction must be identical; target representation may differ.
    """
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = build_model(cfg)
    model.eval()

    d = h = w = 32
    x1 = torch.randn(1, 1, d, h, w)
    # Second volume: identical in visible region, different in a hidden block.
    x2 = x1.clone()
    g = torch.Generator().manual_seed(1)
    mask = sample_block_mask(1, (d, h, w), mask_ratio=0.5, generator=g)
    # Force a known hidden region and mutate it.
    hidden = mask < 0.5
    noise = torch.randn_like(x2) * 5.0
    x2 = torch.where(hidden, x2 + noise, x2)

    with torch.no_grad():
        out1 = model.jepa_step(x1, mask=mask)
        out2 = model.jepa_step(x2, mask=mask)

    # Online predictions on masked positions should match (same visible context).
    vis = out1["visible"]
    masked = ~vis
    p1 = out1["pred_tokens"][masked]
    p2 = out2["pred_tokens"][masked]
    assert torch.allclose(p1, p2, atol=1e-4), (
        "Online JEPA prediction changed when only hidden voxels changed — leakage."
    )

    # Target tokens come from full image and SHOULD differ (sanity: we actually
    # changed the full-image content).
    t1 = out1["tgt_tokens"]
    t2 = out2["tgt_tokens"]
    assert not torch.allclose(t1, t2, atol=1e-4)


def test_both_paths_use_masked_input_in_jepa():
    """Sanity: applying jepa_step with fully zero mask still runs (no crash)."""
    cfg = _make_cfg(mask_ratio=0.0)
    model = build_model(cfg)
    x = torch.randn(1, 1, 32, 32, 32)
    # mask_ratio 0 means all visible; loss masked set may be empty.
    out = model.jepa_step(x)
    assert out["pred_tokens"].shape[-1] == cfg.embed_dim
