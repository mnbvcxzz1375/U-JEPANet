from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ujepa.masking import mask_hidden_fraction, sample_token_mask, token_mask_to_volume_mask
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model


def _make_cfg(**kwargs):
    base = dict(
        arm="A3",
        feature_chns=[8, 16, 32, 64],
        class_num=17,
        embed_dim=48,
        num_heads=2,
        num_blocks=2,
        predictor_blocks=1,
        jepa_token_stride=8,
        deep_stage=2,
        multiscale_pred=False,
        mask_ratio=0.4,
    )
    base.update(kwargs)
    return UJEPAConfig(**base)


def test_token_mask_ratio_small_grid():
    g = torch.Generator().manual_seed(0)
    mask = sample_token_mask(2, (8, 8, 8), mask_ratio=0.4, generator=g)
    frac = mask_hidden_fraction(mask)
    assert 0.35 <= frac <= 0.55, frac


def test_token_mask_ratio_word_like_grids():
    """Real WORD sizes: ratio must actually approach mask_ratio."""
    for grid, mr in [((8, 8, 6), 0.4), ((32, 32, 24), 0.4), ((16, 16, 12), 0.35)]:
        g = torch.Generator().manual_seed(1)
        mask = sample_token_mask(1, grid, mask_ratio=mr, generator=g)
        frac = mask_hidden_fraction(mask)
        assert abs(frac - mr) < 0.08, (grid, mr, frac)


def test_token_mask_upsample_to_volume():
    token_mask = sample_token_mask(1, (8, 8, 6), mask_ratio=0.4, generator=torch.Generator().manual_seed(2))
    vol = token_mask_to_volume_mask(token_mask, (128, 128, 96))
    assert vol.shape == (1, 1, 128, 128, 96)
    # nearest upsample preserves the token-level ratio approximately
    frac = mask_hidden_fraction(vol)
    assert 0.30 <= frac <= 0.55, frac


def test_apply_mask_zeroes_hidden():
    x = torch.ones(1, 1, 8, 8, 8)
    mask = torch.ones(1, 1, 8, 8, 8)
    mask[:, :, :4] = 0
    from ujepa.masking import apply_mask

    y = apply_mask(x, mask, fill_value=0.0)
    assert y[:, :, :4].abs().sum() == 0


def test_online_prediction_invariant_to_hidden_region_content():
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = build_model(cfg)
    model.eval()

    x1 = torch.randn(1, 1, 32, 32, 32)
    x2 = x1.clone()
    g = torch.Generator().manual_seed(1)
    token_mask = sample_token_mask(1, (4, 4, 4), mask_ratio=0.5, generator=g)
    # stride 8 on 32^3 → token grid 4^3
    from ujepa.masking import token_mask_to_volume_mask

    vol_mask = token_mask_to_volume_mask(token_mask, (32, 32, 32))
    hidden = vol_mask < 0.5
    noise = torch.randn_like(x2) * 5.0
    x2 = torch.where(hidden, x2 + noise, x2)

    with torch.no_grad():
        out1 = model.jepa_step(x1, mask=token_mask)
        out2 = model.jepa_step(x2, mask=token_mask)

    vis = out1["visible"]
    masked = ~vis
    assert masked.any()
    p1 = out1["pred_tokens"][masked]
    p2 = out2["pred_tokens"][masked]
    assert torch.allclose(p1, p2, atol=1e-4), "online pred changed when only hidden voxels changed"

    t1 = out1["tgt_tokens"]
    t2 = out2["tgt_tokens"]
    assert not torch.allclose(t1, t2, atol=1e-4)


def test_predictor_uses_position():
    """Different masked positions must not produce identical predictions."""
    torch.manual_seed(0)
    cfg = _make_cfg(mask_ratio=0.5)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model.jepa_step(x)
    pred = out["pred_tokens"]
    masked = ~out["visible"]
    if masked.sum() >= 2:
        preds = pred[masked]
        # pairwise distances among masked predictions should not all be ~0
        d = torch.cdist(preds, preds)
        off = d[~torch.eye(d.shape[0], dtype=torch.bool, device=d.device)]
        assert float(off.mean()) > 1e-4, "masked predictions are identical — missing target PE"


def test_jepa_token_count_in_range():
    cfg = _make_cfg(jepa_token_stride=8)
    model = build_model(cfg)
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model.jepa_step(x)
    n = out["pred_tokens"].shape[1]
    assert 256 <= n <= 1024 or n == 4 * 4 * 4, n  # 64 tokens on 4^3 is ok for tiny smoke
    assert n == 4 * 4 * 4
