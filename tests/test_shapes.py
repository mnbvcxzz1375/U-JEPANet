from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ujepa.model import UJEPAConfig, UJEPATrainer, build_model, build_online_for_arm
from ujepa.unet3d import UNet3D


def test_a0_forward_shapes():
    cfg = UJEPAConfig(
        arm="A0",
        feature_chns=[8, 16, 32, 64],
        class_num=4,
        embed_dim=48,
        multiscale_pred=False,
    )
    model = build_model(cfg)
    x = torch.randn(2, 1, 32, 32, 32)
    y = model(x)
    assert y.shape == (2, 4, 32, 32, 32)


def test_a1_dualpath_forward():
    cfg = UJEPAConfig(
        arm="A1",
        feature_chns=[8, 16, 32, 64],
        class_num=4,
        embed_dim=48,
        num_heads=2,
        num_blocks=2,
        image_stride=4,
        deep_stage=2,
        multiscale_pred=False,
    )
    model = build_model(cfg)
    x = torch.randn(1, 1, 32, 32, 32)
    out = model.forward_seg(x) if isinstance(model, UJEPATrainer) else model(x)
    assert out.shape == (1, 4, 32, 32, 32)


def test_a3_jepa_step_shapes_and_ema():
    cfg = UJEPAConfig(
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
        mask_ratio=0.5,
    )
    model = build_model(cfg)
    assert isinstance(model, UJEPATrainer)
    x = torch.randn(2, 1, 32, 32, 32)
    out = model.jepa_step(x)
    n = out["pred_tokens"].shape[1]
    assert out["pred_tokens"].shape == out["tgt_tokens"].shape
    assert out["visible"].shape == (2, n)
    # Some tokens must be masked.
    assert (~out["visible"]).any()
    # EMA update runs
    model.update_ema(0.9)
    logits = model.forward_seg(x)
    assert logits.shape == (1, 4, 32, 32, 32) or logits.shape[0] == 2


def test_a2_inference_drops_predictor():
    cfg = UJEPAConfig(
        arm="A2",
        feature_chns=[8, 16, 32, 64],
        class_num=4,
        embed_dim=48,
        num_heads=2,
        num_blocks=2,
        predictor_blocks=1,
        deep_stage=2,
        multiscale_pred=False,
    )
    trainer = build_model(cfg)
    online = build_online_for_arm(cfg)
    online.load_state_dict(trainer.online.state_dict())
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        a = trainer.forward_seg(x)
        b = online(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_alpha_nonzero_init():
    cfg = UJEPAConfig(arm="A1", feature_chns=[8, 16, 32, 64], embed_dim=48, num_heads=2, num_blocks=1, class_num=4)
    model = build_model(cfg)
    dual = model.online.dualpath if isinstance(model, UJEPATrainer) else model.dualpath
    assert dual is not None
    assert abs(float(dual.alpha.detach()) - 0.1) < 1e-6
