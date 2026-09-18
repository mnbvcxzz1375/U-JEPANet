"""Sanity tests for V1.2 PredictiveUNet (P0-1 / P0-2 contracts)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.predictive_unet import PredictiveUNet, target_grid_384


def _model():
    return PredictiveUNet(
        feature_chns=(16, 32, 64, 128),
        class_num=17,
        deep_stage=2,
        embed_dim=64,
        num_heads=2,
        predictor_blocks=1,
        mask_ratio=0.4,
        use_residual=True,
        multiscale_pred=False,
    )


def test_target_grid_384():
    assert target_grid_384((32, 32, 24)) == (8, 8, 6)
    n = 8 * 8 * 6
    assert n == 384


def test_deterministic_eval():
    m = _model().eval()
    x = torch.randn(1, 1, 64, 64, 48)
    with torch.no_grad():
        y1 = m(x)
        y2 = m(x)
    assert torch.allclose(y1, y2, atol=1e-5), "eval() must be deterministic full-context"


def test_token_contract_128():
    m = _model()
    x = torch.randn(1, 1, 128, 128, 96)
    # don't run full decode if too heavy — just encode stages
    bb = m.backbone
    x0 = bb.in_conv(x)
    x1 = bb.down1(x0)
    f2 = bb.down2(x1)
    assert tuple(f2.shape[-3:]) == (32, 32, 24), f"F2 spatial {tuple(f2.shape)}"
    tokens, grid, _ = m.bottleneck.encode_tokens(f2)
    assert grid == (8, 8, 6)
    assert tokens.shape[1] == 384, f"N={tokens.shape[1]}"


def test_deep_path_f3_sees_f2_star():
    """P0-1: changing bottleneck output must change F3."""
    m = _model()
    x = torch.randn(1, 1, 64, 64, 48)
    m.eval()
    with torch.no_grad():
        _ = m(x)
        f3_a = m._last_f3.clone()
        f2s_a = m._last_f2_star.clone()
        # force different F2*
        m.bottleneck.merge.weight.data.zero_()
        m.bottleneck.merge.bias.data.fill_(10.0)
        _ = m(x)
        f3_b = m._last_f3.clone()
        f2s_b = m._last_f2_star.clone()
    assert not torch.allclose(f2s_a, f2s_b), "F2* should change when merge changes"
    assert not torch.allclose(f3_a, f3_b), (
        "P0-1 FAIL: F3 unchanged → bottleneck not on deep path (E3 used old F2)"
    )


def _first_grad(module: nn.Module):
    for p in module.parameters():
        if p.grad is not None:
            return p.grad
    return None


def test_aux_grad_reaches_encoder_proj_predictor():
    """P0-2: L_P backward must hit E2/down2, proj, and predictor."""
    m = _model()
    m.train()
    x = torch.randn(1, 1, 64, 64, 48)
    _ = m(x)
    lp = m.pred_loss
    assert float(lp.detach()) == float(lp.detach()), "pred loss finite"
    lp.backward()
    g_pred = m.bottleneck.predictor.mask_token.grad
    assert g_pred is not None and g_pred.abs().sum() > 0, "predictor grad"
    g_proj = m.bottleneck.proj.weight.grad
    assert g_proj is not None and g_proj.abs().sum() > 0, (
        "P0-2 FAIL: proj got no grad from L_P"
    )
    g_e2 = _first_grad(m.backbone.down2)
    assert g_e2 is not None and g_e2.abs().sum() > 0, (
        "P0-2 FAIL: encoder down2 (E2) got no grad from L_P"
    )


def test_masked_only_fraction():
    m = _model()
    m.train()
    f2 = torch.randn(2, 64, 16, 16, 12)  # stage-like
    b = m.bottleneck
    tokens, grid, _ = b.encode_tokens(f2)
    n = tokens.shape[1]
    n_mask = max(1, int(round(b.mask_ratio * n)))
    assert abs(n_mask / n - 0.4) < 0.05, f"mask frac {n_mask}/{n}"
    # loss should be finite and not require all-token mean identity
    lp = b.masked_pred_loss(f2)
    assert torch.isfinite(lp)


def test_seg_loss_still_works():
    m = _model()
    x = torch.randn(1, 1, 64, 64, 48)
    m.train()
    out = m(x)
    loss = out.sum() + m.pred_loss
    loss.backward()
    g_in = _first_grad(m.backbone.in_conv)
    assert g_in is not None and g_in.abs().sum() > 0, "in_conv grad from total loss"


if __name__ == "__main__":
    tests = [
        test_target_grid_384,
        test_deterministic_eval,
        test_token_contract_128,
        test_deep_path_f3_sees_f2_star,
        test_aux_grad_reaches_encoder_proj_predictor,
        test_masked_only_fraction,
        test_seg_loss_still_works,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:
            print("FAIL", t.__name__, e)
            failed.append(t.__name__)
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print("ALL_PRED_V12_TESTS_PASS")
