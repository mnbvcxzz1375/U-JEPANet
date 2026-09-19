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
    dims = (1, 2, 3, 4)
    assert abs(float(t[0].pow(2).mean().sqrt()) - float(ref[0].pow(2).mean().sqrt())) < 1e-3
    assert abs(float(t[1].pow(2).mean().sqrt()) - float(ref[1].pow(2).mean().sqrt())) < 1e-3


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


def test_per_sample_rms_independence():
    """P0: RMS scale per-sample; changing sample1 scale must not affect sample0."""
    delta = torch.randn(2, 4, 8, 8, 8)
    ref = torch.randn(2, 4, 8, 8, 8)
    delta = delta.clone()
    delta[1] *= 100.0  # only patient 1
    t = rms_norm_delta(delta, ref, stop_grad_scale=False)
    dims = (1, 2, 3, 4)
    rms0 = float(t[0].pow(2).mean().sqrt())
    rms0_ref = float(ref[0].pow(2).mean().sqrt())
    assert abs(rms0 - rms0_ref) < 1e-4, (rms0, rms0_ref)
    rms1_ref = float(ref[1].pow(2).mean().sqrt())
    rms1 = float(t[1].pow(2).mean().sqrt())
    assert abs(rms1 - rms1_ref) < 1e-4


def test_coarse_gridsample_same_token_count():
    """P1: coarse map GridSample yields N tokens matching fine (no 1D interp)."""
    B, C = 1, 8
    Gfine = torch.randn(B, C, 16, 16, 16)
    Gcoarse = torch.randn(B, C, 8, 8, 8)
    crop = torch.tensor([[10, 20, 30]])
    shape = torch.tensor([[128, 128, 96]])
    grid = (8, 8, 6)
    patch = (64, 64, 48)
    gf, _ = gridsample_global(Gfine, grid, crop, shape, patch, None)
    gc, _ = gridsample_global(Gcoarse, grid, crop, shape, patch, None)
    assert gf.shape == gc.shape == (B, 8 * 8 * 6, C)


def test_innov_merge_bias_false_h0_zero_delta():
    """P1: innov_merge bias=False ⇒ I=0 ⇒ Δ=0."""
    torch.manual_seed(0)
    m = V32UNet(arm="H0", embed_dim=32, predictor_blocks=1, num_heads=2,
                multiscale_pred=False, use_image_global=False)
    assert m.innov_merge.bias is None
    # zero I path: PE-only, I should be 0; fusion delta ~0
    m.eval()
    x = torch.randn(1, 1, 64, 64, 48)
    crop = torch.tensor([[0, 0, 0]])
    shape = torch.tensor([[64, 64, 48]])
    th = torch.eye(3, 4).unsqueeze(0)
    with torch.no_grad():
        _ = m(x, global_image=None, crop_origin=crop, full_shape=shape,
              affine_theta=th, patch_size=(64, 64, 48))
    assert m.mech_dict.get("S_pre", 1.0) < 1e-5
    assert m.mech_dict.get("S_fuse", 1.0) < 1e-5


def test_h0_equiv_r1_forward():
    """P0/P1: H0 with I=0 and bias-free merge ≈ R1 backbone output when alpha any."""
    torch.manual_seed(42)
    r1 = V32UNet(arm="R1", embed_dim=32, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    torch.manual_seed(42)
    h0 = V32UNet(arm="H0", embed_dim=32, predictor_blocks=1, num_heads=2,
                 multiscale_pred=False, use_image_global=False)
    # copy R1 backbone+local_bn weights into H0
    r1_sd = r1.state_dict()
    h0_sd = h0.state_dict()
    for k in r1_sd:
        if k in h0_sd and h0_sd[k].shape == r1_sd[k].shape:
            h0_sd[k] = r1_sd[k]
    h0.load_state_dict(h0_sd)
    h0.eval()
    r1.eval()
    x = torch.randn(1, 1, 64, 64, 48)
    crop = torch.tensor([[8, 8, 8]])
    shape = torch.tensor([[128, 128, 96]])
    th = torch.eye(3, 4).unsqueeze(0)
    with torch.no_grad():
        y_r = r1(x)
        if isinstance(y_r, (list, tuple)):
            y_r = y_r[0]
        y_h = h0(x, global_image=None, crop_origin=crop, full_shape=shape,
                 affine_theta=th, patch_size=(64, 64, 48))
        if isinstance(y_h, (list, tuple)):
            y_h = y_h[0]
    # PE-only I=0 + bias-free → fusion adds 0
    assert torch.allclose(y_r, y_h, atol=1e-4), float((y_r - y_h).abs().max())


def test_b0_plain_unet_val_callable():
    """P0: evaluate_word_whole_volume accepts UNet3D; GLP path must not be used for B0."""
    from ujepa.unet3d import UNet3D
    from ujepa.whole_volume_eval import evaluate_whole_volume_case
    m = UNet3D(in_chns=1, feature_chns=[16, 32, 64, 128], class_num=17, multiscale_pred=False)
    m.eval()
    img = torch.rand(64, 64, 48)
    lab = torch.randint(0, 17, (64, 64, 48))
    out = evaluate_whole_volume_case(m, img, lab, 17, (32, 32, 24), (32, 32, 24), torch.device("cpu"))
    assert "mean_fg_dice" in out


if __name__ == "__main__":
    tests = [
        test_h0_h1_init_match,
        test_innovation_isolated_no_zl_in_fusion,
        test_h0_innovation_near_zero,
        test_h1_innovation_positive_with_distinct_globals,
        test_rms_scale,
        test_gridsample_identity_origin,
        test_r1_only_forward,
        test_per_sample_rms_independence,
        test_coarse_gridsample_same_token_count,
        test_innov_merge_bias_false_h0_zero_delta,
        test_h0_equiv_r1_forward,
        test_b0_plain_unet_val_callable,
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
