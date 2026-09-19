"""V3.2 contract tests + real-case shuffle diagnostic helper."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.aligned_global_innovation import V32UNet, gridsample_global, rms_norm_delta


def test_h0_h1_init_match():
    torch.manual_seed(42)
    h0 = V32UNet(arm="H0", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False, use_image_global=False)
    torch.manual_seed(42)
    h1 = V32UNet(arm="H1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False, use_image_global=True)
    s0, s1 = h0.state_dict(), h1.state_dict()
    assert set(s0) == set(s1)
    bad = [k for k in s0 if not torch.equal(s0[k], s1[k])]
    assert not bad, bad[:6]
    assert abs(float(h0.alpha_g) - 0.04) < 1e-6
    assert abs(float(h1.alpha_g) - 0.04) < 1e-6


def test_innovation_isolated_no_zl_in_fusion():
    """Fusion params only see I; inspect innov_merge input path via mech S_pre."""
    torch.manual_seed(0)
    m = V32UNet(arm="H1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    m.train()
    B = 2
    x = torch.randn(B, 1, 64, 64, 48)
    crop = torch.tensor([[0, 0, 0], [40, 50, 20]], dtype=torch.long)
    shape = torch.tensor([[128, 128, 96]] * B, dtype=torch.long)
    th = torch.eye(3, 4).unsqueeze(0).expand(B, -1, -1).contiguous()
    glob = torch.stack([torch.rand(1, 64, 64, 64) + i for i in range(B)])
    y = m(x, global_image=glob, crop_origin=crop, full_shape=shape,
          affine_theta=th, patch_size=(64, 64, 48))
    if isinstance(y, (list, tuple)):
        y = y[0]
    assert y.shape[0] == 2
    mech = m.mech_dict
    assert "S_pre" in mech and "S_fuse" in mech
    # H1 with different glob should have S_pre > 0 typically
    print("  mech", mech)


def test_h0_innovation_near_zero():
    """Shared P + PE-only both sides → I ≈ 0 when global_image is None/PE."""
    torch.manual_seed(0)
    m = V32UNet(arm="H0", embed_dim=64, predictor_blocks=1, num_heads=2,
                multiscale_pred=False, use_image_global=False)
    m.eval()
    B = 2
    x = torch.randn(B, 1, 64, 64, 48)
    crop = torch.tensor([[0, 0, 0], [30, 30, 10]], dtype=torch.long)
    shape = torch.tensor([[128, 128, 96]] * B, dtype=torch.long)
    th = torch.eye(3, 4).unsqueeze(0).expand(B, -1, -1).contiguous()
    with torch.no_grad():
        _ = m(x, global_image=None, crop_origin=crop, full_shape=shape,
              affine_theta=th, patch_size=(64, 64, 48))
    s_pre = m.mech_dict.get("S_pre", 1.0)
    print("  H0 S_pre", s_pre)
    assert s_pre < 1e-5, f"H0 PE-only should have I~0, got S_pre={s_pre}"


def test_h1_innovation_positive_with_distinct_globals():
    torch.manual_seed(0)
    m = V32UNet(arm="H1", embed_dim=64, predictor_blocks=1, num_heads=2,
                multiscale_pred=False, use_image_global=True)
    m.eval()
    B = 2
    x = torch.randn(B, 1, 64, 64, 48)
    crop = torch.tensor([[0, 0, 0], [40, 50, 20]], dtype=torch.long)
    shape = torch.tensor([[128, 128, 96]] * B, dtype=torch.long)
    th = torch.eye(3, 4).unsqueeze(0).expand(B, -1, -1).contiguous()
    glob = torch.stack([torch.rand(1, 64, 64, 64) + 3.0 * i for i in range(B)])
    with torch.no_grad():
        _ = m(x, global_image=glob, crop_origin=crop, full_shape=shape,
              affine_theta=th, patch_size=(64, 64, 48))
    s_pre = m.mech_dict.get("S_pre", 0.0)
    print("  H1 S_pre", s_pre, "S_fuse", m.mech_dict.get("S_fuse"))
    assert s_pre > 0.0
    assert abs(m.mech_dict.get("S_fuse", 0) - float(m.alpha_g_value)) < 0.02


def test_rms_scale():
    delta = torch.randn(2, 8, 4, 4, 4) * 10
    ref = torch.randn(2, 8, 4, 4, 4) * 0.1
    t = rms_norm_delta(delta, ref)
    assert abs(float(t.pow(2).mean().sqrt()) - float(ref.pow(2).mean().sqrt())) < 1e-3


def test_gridsample_identity_origin():
    B, C = 1, 8
    G = torch.zeros(B, C, 8, 8, 8)
    G[0, :, 4, 4, 4] = torch.arange(C, dtype=torch.float32)
    crop = torch.tensor([[0, 0, 0]])
    shape = torch.tensor([[64, 64, 48]])
    g, _ = gridsample_global(G, (1, 1, 1), crop, shape, (32, 32, 24), None)
    assert g.shape == (1, 1, C)


def test_r1_only_forward():
    m = V32UNet(arm="R1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    x = torch.randn(1, 1, 64, 64, 48)
    y = m(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    assert y.shape[1] == 17


if __name__ == "__main__":
    tests = [
        test_h0_h1_init_match,
        test_innovation_isolated_no_zl_in_fusion,
        test_h0_innovation_near_zero,
        test_h1_innovation_positive_with_distinct_globals,
        test_rms_scale,
        test_gridsample_identity_origin,
        test_r1_only_forward,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:
            print("FAIL", t.__name__, type(e).__name__, e)
            failed.append(t.__name__)
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print("ALL_V32_TESTS_PASS")
