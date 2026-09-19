"""P0 contract tests for V3 GLP — especially global/local affine alignment."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.ct_augment import (
    local_token_to_global_voxel,
    mild_affine,
    token_coord_features,
)
from ujepa.global_local_predictive import GLPUNet
from ujepa.predictive_unet import target_grid_384


def test_identity_theta_alignment():
    """theta=None: token centers = crop_origin + geometric centers."""
    grid = (8, 8, 6)
    origin = (10, 20, 30)
    full = (200, 200, 160)
    patch = (128, 128, 96)
    g = local_token_to_global_voxel(grid, origin, full, patch, theta=None)
    # token (0,0,0) center in patch: (0.5*16, 0.5*16, 0.5*16)=(8,8,8)
    assert torch.allclose(g[0, 0, 0], torch.tensor([18.0, 28.0, 38.0])), g[0, 0, 0]
    # token (7,7,5): (7.5*16, 7.5*16, 5.5*16)=(120,120,88) + origin
    assert torch.allclose(g[7, 7, 5], torch.tensor([130.0, 140.0, 118.0])), g[7, 7, 5]


def test_identity_theta_tensor():
    th = torch.eye(3, 4).unsqueeze(0)
    g1 = local_token_to_global_voxel((4, 4, 4), (0, 0, 0), (64, 64, 48), (32, 32, 24), None)
    g2 = local_token_to_global_voxel((4, 4, 4), (0, 0, 0), (64, 64, 48), (32, 32, 24), th)
    assert torch.allclose(g1, g2, atol=1e-5), (g1[0, 0, 0], g2[0, 0, 0])


def test_affine_roundtrip_matches_grid_sample():
    """P0: after mild_affine, mapped global coords stay consistent with sampling.

    Contract: for identity-scale affine_grid, voxel mapping is exact.
    For non-trivial theta, verify our inverse-map matches F.affine_grid on
    the same patch for a probe point.
    """
    import torch.nn.functional as F

    pd, ph, pw = 32, 32, 24
    x = torch.randn(1, 1, pd, ph, pw)
    # force affine
    g = torch.Generator().manual_seed(0)
    x_t, theta = mild_affine(x, p=1.0, return_theta=True, generator=g)
    assert theta.shape == (1, 3, 4)
    # probe output voxel in patch
    v_out = torch.tensor([10.0, 12.0, 8.0])  # zyx
    u_out_xyz = torch.tensor([
        (v_out[2] + 0.5) / pw * 2 - 1,
        (v_out[1] + 0.5) / ph * 2 - 1,
        (v_out[0] + 0.5) / pd * 2 - 1,
    ])
    # affine_grid: input_norm = R @ out_norm + t  (xyz)
    R = theta[0, :3, :3]
    t = theta[0, :3, 3]
    u_in = R @ u_out_xyz + t
    # our mapper on single-token grid that lands on v_out
    # token grid 1x1x1 with patch = full → token center = patch center, not probe.
    # Instead verify with full token grid: find token whose geometric center ≈ v_out
    # Use grid equal to patch (1 voxel per token) for exactness
    grid = (pd, ph, pw)
    gvox = local_token_to_global_voxel(grid, (0, 0, 0), (pd, ph, pw), (pd, ph, pw), theta)
    z, y, x = int(v_out[0]), int(v_out[1]), int(v_out[2])
    # edge-based continuous center for output voxel-index (z,y,x)=(10,12,8)
    # voxel index i center is at i+0.5 in edge coords
    v_center = torch.tensor([10.0 + 0.5, 12.0 + 0.5, 8.0 + 0.5])  # zyx edge coords
    u_c = torch.tensor([
        v_center[2] / pw * 2 - 1,
        v_center[1] / ph * 2 - 1,
        v_center[0] / pd * 2 - 1,
    ])
    u_in_c = R @ u_c + t
    z_in = (u_in_c[2] + 1) * 0.5 * pd
    y_in = (u_in_c[1] + 1) * 0.5 * ph
    x_in = (u_in_c[0] + 1) * 0.5 * pw
    expected = torch.tensor([z_in, y_in, x_in])
    got = gvox[z, y, x]
    assert torch.allclose(got, expected, atol=1e-4), (got, expected)


def test_no_affine_dataset_coords():
    """When theta is identity, global coords ignore affine (P0 positive control)."""
    origin = (40, 50, 60)
    full = (300, 300, 200)
    patch = (128, 128, 96)
    th = torch.eye(3, 4)
    a = local_token_to_global_voxel((8, 8, 6), origin, full, patch, None)
    b = local_token_to_global_voxel((8, 8, 6), origin, full, patch, th)
    assert torch.allclose(a, b, atol=1e-5)
    assert a.min() >= -1 and a.max() <= max(full) + 64


def test_coord_features_range():
    g = local_token_to_global_voxel((8, 8, 6), (0, 0, 0), (256, 256, 192), (128, 128, 96), None)
    c = token_coord_features(g, (256, 256, 192), (128, 128, 96))
    assert c.shape[-1] == 6
    p = c[..., :3]
    assert p.min() >= -0.01 and p.max() <= 1.01


def test_alpha_g_init_zero_r1_equiv():
    """G1 starts with alpha_g=0 → F2* = F2_R1 (no global delta)."""
    torch.manual_seed(0)
    m = GLPUNet(arm="G1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    assert m.alpha_g_value == 0.0
    x = torch.randn(1, 1, 64, 64, 48)
    glob = torch.rand(1, 1, 64, 64, 64)
    origin = torch.tensor([[0, 0, 0]])
    shape = torch.tensor([[64, 64, 48]])
    th = torch.eye(3, 4).unsqueeze(0)
    m.eval()
    with torch.no_grad():
        y1 = m(x, global_image=glob, crop_origin=origin, full_shape=shape, affine_theta=th,
               patch_size=(64, 64, 48))
        if isinstance(y1, (list, tuple)):
            y1 = y1[0]
        # force nonzero merge weights but alpha=0
        if m.gl_merge is not None:
            m.gl_merge.weight.data.normal_(0, 0.1)
            m.gl_merge.bias.data.normal_(0, 0.1)
        y2 = m(x, global_image=glob, crop_origin=origin, full_shape=shape, affine_theta=th,
               patch_size=(64, 64, 48))
        if isinstance(y2, (list, tuple)):
            y2 = y2[0]
    assert torch.allclose(y1, y2, atol=1e-5), "alpha_g=0 must freeze global fusion"


def test_g0_g1_forward_shapes():
    for arm in ("G0", "G1", "G2"):
        m = GLPUNet(arm=arm, embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
        x = torch.randn(1, 1, 64, 64, 48)
        kw = dict(
            crop_origin=torch.tensor([[8, 8, 8]]),
            full_shape=torch.tensor([[128, 128, 96]]),
            affine_theta=torch.eye(3, 4).unsqueeze(0),
            patch_size=(64, 64, 48),
        )
        if arm != "G0":
            kw["global_image"] = torch.rand(1, 1, 64, 64, 64)
        y = m(x, **kw)
        if isinstance(y, (list, tuple)):
            y = y[0]
        assert y.shape[1] == 17, (arm, y.shape)


def test_g2_gl_loss_targets_original_f2():
    m = GLPUNet(arm="G2", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    m.train()
    x = torch.randn(1, 1, 64, 64, 48)
    y = m(
        x,
        global_image=torch.rand(1, 1, 64, 64, 64),
        crop_origin=torch.tensor([[0, 0, 0]]),
        full_shape=torch.tensor([[64, 64, 48]]),
        affine_theta=torch.eye(3, 4).unsqueeze(0),
        patch_size=(64, 64, 48),
    )
    gl = m.gl_loss
    assert torch.isfinite(gl)
    gl.backward()
    g = m.global_stem.proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "G2 L_GL must train global stem"


def test_target_grid_contract():
    assert target_grid_384((32, 32, 24)) == (8, 8, 6)


if __name__ == "__main__":
    tests = [
        test_identity_theta_alignment,
        test_identity_theta_tensor,
        test_affine_roundtrip_matches_grid_sample,
        test_no_affine_dataset_coords,
        test_coord_features_range,
        test_alpha_g_init_zero_r1_equiv,
        test_g0_g1_forward_shapes,
        test_g2_gl_loss_targets_original_f2,
        test_target_grid_contract,
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
    print("ALL_GLP_V3_TESTS_PASS")
