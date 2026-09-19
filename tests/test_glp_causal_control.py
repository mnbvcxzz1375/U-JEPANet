"""Final causal-control contracts for V3.1 G0 vs G1."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.global_local_predictive import GLPUNet


def _common_state(model):
    """Keys that must match between G0 and G1 under same seed."""
    sd = model.state_dict()
    return {k: v.clone() for k, v in sd.items() if not k.startswith("unused")}


def test_g0_g1_init_matching_same_seed():
    torch.manual_seed(42)
    g0 = GLPUNet(arm="G0", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    torch.manual_seed(42)
    g1 = GLPUNet(arm="G1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    s0, s1 = _common_state(g0), _common_state(g1)
    keys0, keys1 = set(s0), set(s1)
    assert keys0 == keys1, f"state key mismatch G0-only={keys0-keys1} G1-only={keys1-keys0}"
    # all shared modules must be bitwise-equal at init
    bad = []
    for k in sorted(keys0):
        if not torch.equal(s0[k], s1[k]):
            bad.append(k)
    assert not bad, f"G0/G1 init mismatch on {bad[:8]}..."
    # architecture parity: both have GlobalStem + full gl_pred stack
    assert g0.global_stem is not None and g1.global_stem is not None
    assert len(g0.gl_pred.blocks) == len(g1.gl_pred.blocks) == 2


def test_content_ablation_g1_equals_g0():
    """With alpha_g != 0 and same weights, G1 with Z_G content zeroed ≡ G0 PE-only."""
    torch.manual_seed(0)
    g0 = GLPUNet(arm="G0", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    torch.manual_seed(0)
    g1 = GLPUNet(arm="G1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    # force nonzero alpha so fusion path is active
    with torch.no_grad():
        g0.alpha_g.fill_(0.1)
        g1.alpha_g.fill_(0.1)
    g0.eval()
    g1.eval()
    B = 2
    x = torch.randn(B, 1, 64, 64, 48)
    crop = torch.tensor([[0, 0, 0], [40, 50, 30]], dtype=torch.long)
    shape = torch.tensor([[128, 128, 96]] * B, dtype=torch.long)
    th = torch.eye(3, 4).unsqueeze(0).expand(B, -1, -1).contiguous()
    kw = dict(crop_origin=crop, full_shape=shape, affine_theta=th, patch_size=(64, 64, 48))
    # G0 never reads image content
    with torch.no_grad():
        y0 = g0(x, global_image=torch.rand(B, 1, 64, 64, 64), **kw)
        if isinstance(y0, (list, tuple)):
            y0 = y0[0]
        # G1 with global_image that yields PE-only after zero-content:
        # monkey: call position_only path by passing None — G1 still uses PE-only when global_image is None
        y1_pe = g1(x, global_image=None, **kw)
        if isinstance(y1_pe, (list, tuple)):
            y1_pe = y1_pe[0]
        assert torch.allclose(y0, y1_pe, atol=1e-5), (
            "G0 PE-only must match G1(global_image=None) PE-only path",
            float((y0 - y1_pe).abs().max()),
        )
        # With real global content, G1 should differ (content is the only delta)
        y1_img = g1(x, global_image=torch.rand(B, 1, 64, 64, 64) + 2.0, **kw)
        if isinstance(y1_img, (list, tuple)):
            y1_img = y1_img[0]
        assert not torch.allclose(y0, y1_img, atol=1e-4), "G1 with content must differ from G0 PE-only"


def test_both_stacks_cross_attn():
    g0 = GLPUNet(arm="G0", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    g1 = GLPUNet(arm="G1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    assert len(g0.gl_pred.blocks) == len(g1.gl_pred.blocks)
    # G0 must not take the z_g=None shortcut that skips blocks
    # inspect: position_only produces non-None z_g → blocks always run
    z = g0.global_stem.position_only(2, torch.device("cpu"))
    assert z is not None and z.shape[1] == 8 * 8 * 8
    q = torch.randn(2, 384, 64)
    out = g0.gl_pred(q, z)
    assert out.shape == q.shape


if __name__ == "__main__":
    tests = [
        test_g0_g1_init_matching_same_seed,
        test_content_ablation_g1_equals_g0,
        test_both_stacks_cross_attn,
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
    print("ALL_GLP_CAUSAL_TESTS_PASS")
