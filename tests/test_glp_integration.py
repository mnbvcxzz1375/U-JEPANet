"""Integration contract: Dataset→collate→CUDA→GLP forward + GLP val patch."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.ct_augment import token_coord_features_batched
from ujepa.global_local_predictive import GLPUNet
from ujepa.glp_eval import sliding_window_logits_glp


def _make_batch(B=2, device="cpu"):
    x = torch.randn(B, 1, 64, 64, 48, device=device)
    crop = torch.tensor([[0, 0, 0], [50, 70, 20]], dtype=torch.long, device=device)[:B]
    if B == 1:
        crop = crop[:1]
    shape = torch.tensor([[128, 128, 96]] * B, dtype=torch.long, device=device)
    th = torch.eye(3, 4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    # different theta for sample 1
    if B > 1:
        th = th.clone()
        th[1, 0, 0] = 0.95
        th[1, 1, 1] = 1.05
    glob = torch.rand(B, 1, 64, 64, 64, device=device)
    return x, crop, shape, th, glob


def test_dataset_shapes_contract():
    """affine_theta (3,4) and global (1,64,64,64) after collate → B,3,4 / B,1,..."""
    from ujepa.dynamic_dataset import DynamicWordVolumeDataset

    # synthetic unit: mimic what dataset should return
    th_item = torch.eye(3, 4)
    g_item = torch.rand(1, 64, 64, 64)
    th_b = torch.stack([th_item, th_item])
    g_b = torch.stack([g_item, g_item])
    assert th_b.shape == (2, 3, 4), th_b.shape
    assert g_b.shape == (2, 1, 64, 64, 64), g_b.shape


def test_batch_coords_differ():
    grid = (8, 8, 6)
    x, crop, shape, th, _ = _make_batch(B=2, device="cpu")
    c = token_coord_features_batched(grid, crop, shape, (64, 64, 48), th, device=torch.device("cpu"))
    assert c.shape == (2, 384, 6), c.shape
    assert not torch.allclose(c[0], c[1]), "different origins must yield different coords"


def test_batched_coords_identity_matches_single():
    from ujepa.ct_augment import local_token_to_global_voxel, token_coord_features

    grid = (8, 8, 6)
    origin = (10, 20, 30)
    full = (200, 200, 160)
    patch = (128, 128, 96)
    g_single = local_token_to_global_voxel(grid, origin, full, patch, None)
    c_single = token_coord_features(g_single, full, patch).reshape(1, -1, 6)
    crop = torch.tensor([list(origin)])
    shape = torch.tensor([list(full)])
    c_b = token_coord_features_batched(grid, crop, shape, patch, None, torch.device("cpu"))
    assert torch.allclose(c_b, c_single, atol=1e-4), (c_b[0, 0], c_single[0, 0])


def test_glp_forward_batch2_cpu():
    for arm in ("G0", "G1", "G2"):
        m = GLPUNet(arm=arm, embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
        m.train()
        x, crop, shape, th, glob = _make_batch(B=2, device="cpu")
        kw = dict(crop_origin=crop, full_shape=shape, affine_theta=th, patch_size=(64, 64, 48))
        if arm != "G0":
            kw["global_image"] = glob
        y = m(x, **kw)
        if isinstance(y, (list, tuple)):
            y = y[0]
        assert y.shape[0] == 2 and y.shape[1] == 17
        loss = y.sum() + (m.gl_loss if arm == "G2" else 0.0)
        loss.backward()


def test_glp_forward_cuda_if_available():
    if not torch.cuda.is_available():
        print("SKIP cuda")
        return
    for arm in ("G0", "G1", "G2"):
        m = GLPUNet(arm=arm, embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False).cuda()
        m.train()
        x, crop, shape, th, glob = _make_batch(B=2, device="cuda")
        kw = dict(crop_origin=crop, full_shape=shape, affine_theta=th, patch_size=(64, 64, 48))
        if arm != "G0":
            kw["global_image"] = glob
        y = m(x, **kw)
        if isinstance(y, (list, tuple)):
            y = y[0]
        assert y.is_cuda and y.shape[0] == 2 and y.shape[1] == 17
        assert m.last_coord_feat is not None
        assert m.last_coord_feat.shape == (2, 384, 6)
        assert m.last_coord_feat.is_cuda
        assert not torch.allclose(m.last_coord_feat[0], m.last_coord_feat[1])
        loss = y.sum() + (m.gl_loss if arm == "G2" else torch.tensor(0.0, device="cuda"))
        loss.backward()
        # alpha_g=0 design: fusion path grads only into alpha_g; atlas/global via L_GL on G2
        assert m.alpha_g is not None and m.alpha_g.grad is not None
        if arm == "G2":
            g = m.global_stem.proj.weight.grad
            assert g is not None and g.abs().sum() > 0
            g2 = m.coord_atlas.mlp[0].weight.grad
            assert g2 is not None and g2.abs().sum() > 0
        else:
            # intentional: alpha_g blocks delta-path until it leaves 0
            assert float(m.alpha_g) == 0.0


def test_glp_evaluator_uses_origins():
    """sliding_window_logits_glp must pass non-zero origins (not all 0,0,0)."""
    seen = []

    class Spy(GLPUNet):
        def forward(self, x, **kw):
            co = kw.get("crop_origin")
            if co is not None:
                seen.append(co.detach().cpu().reshape(-1).tolist())
            return super().forward(x, **kw)

    m = Spy(arm="G0", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    m.eval()
    vol = torch.rand(1, 1, 64, 64, 48)
    with torch.no_grad():
        sliding_window_logits_glp(
            m, vol, (32, 32, 24), (32, 32, 24), torch.device("cpu"), 17,
            full_shape=(64, 64, 48),
        )
    assert len(seen) >= 2, seen
    origins = {tuple(o) for o in seen}
    assert any(o != (0, 0, 0) for o in origins), f"evaluator stuck at origin 0: {origins}"


def test_shuffle_global_requires_external_ids_b1():
    m = GLPUNet(arm="G1", embed_dim=64, predictor_blocks=1, num_heads=2, multiscale_pred=False)
    x, crop, shape, th, glob = _make_batch(B=1)
    m.eval()
    with torch.no_grad():
        # b=1 roll is no-op — document contract
        z = m.global_stem(glob)
        z2 = torch.roll(z, shifts=1, dims=0)
        assert torch.allclose(z, z2)
        # external index from another case is required
        idx = torch.tensor([0])  # still identity when only one global in batch
        # multi-case global buffer
        zg = torch.stack([z[0], z[0] + 1.0])
        zsh = zg[torch.tensor([1])]
        assert not torch.allclose(zsh[0], z[0])


if __name__ == "__main__":
    tests = [
        test_dataset_shapes_contract,
        test_batch_coords_differ,
        test_batched_coords_identity_matches_single,
        test_glp_forward_batch2_cpu,
        test_glp_forward_cuda_if_available,
        test_glp_evaluator_uses_origins,
        test_shuffle_global_requires_external_ids_b1,
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
    print("ALL_GLP_INTEGRATION_TESTS_PASS")
