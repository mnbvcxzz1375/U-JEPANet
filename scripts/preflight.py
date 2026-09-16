from __future__ import annotations

"""Lightweight interface / no-leakage checks for A0-A3. Run: python scripts/preflight.py"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.losses import dice_ce_loss, jepa_smooth_l1
from ujepa.masking import mask_hidden_fraction, sample_token_mask
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model, build_online_for_arm


def check_arm(arm: str) -> None:
    cfg = UJEPAConfig(
        arm=arm,
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
    model = build_model(cfg)
    x = torch.randn(2, 1, 32, 32, 32)
    y = torch.randint(0, 17, (2, 32, 32, 32))

    if arm == "A0":
        logits = model(x)
        loss = dice_ce_loss(logits, y, 17)
        loss.backward()
        print(f"[{arm}] seg-only OK params={sum(p.numel() for p in model.parameters()):,}")
        return

    assert isinstance(model, UJEPATrainer)
    logits = model.forward_seg(x)
    l_seg = dice_ce_loss(logits, y, 17)
    if arm == "A1":
        l_seg.backward()
        assert model.online.dualpath.alpha.grad is not None
        print(f"[{arm}] structure-only OK params={sum(p.numel() for p in model.parameters()):,}")
        return

    out = model.jepa_step(x)
    frac = 1.0 - out["visible"].float().mean().item()
    l_j = jepa_smooth_l1(out["pred_tokens"], out["tgt_tokens"], out["visible"])
    (l_seg + 0.3 * l_j).backward()
    print(
        f"[{arm}] joint OK l_seg={float(l_seg.detach()):.4f} l_jepa={float(l_j.detach()):.4f} "
        f"masked_token_frac={frac:.3f} N={out['pred_tokens'].shape[1]} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    assert 0.30 <= frac <= 0.55, f"masked token fraction {frac} not near mask_ratio"

    online = build_online_for_arm(cfg)
    online.load_state_dict(model.online.state_dict())
    with torch.no_grad():
        a = model.forward_seg(x)
        b = online(x)
    assert torch.allclose(a, b, atol=1e-5)
    print(f"[{arm}] inference online parity OK")

    # EMA in checkpoint
    keys = model.state_dict().keys()
    assert any(k.startswith("target.target.") for k in keys)
    print(f"[{arm}] EMA in state_dict OK")


def check_token_budget():
    cfg = UJEPAConfig(arm="A3", feature_chns=[8, 16, 32, 64], embed_dim=48, num_heads=2,
                      num_blocks=1, predictor_blocks=1, jepa_token_stride=16,
                      deep_stage=2, class_num=17, multiscale_pred=False)
    m = build_model(cfg)
    grid = m.online.jepa_token_grid((128, 128, 96))
    n = grid[0] * grid[1] * grid[2]
    print(f"[token-budget] 128x128x96 stride16 → {grid} N={n}")
    assert 256 <= n <= 1024


def check_mask_ratio_real_sizes():
    for grid in [(8, 8, 6), (32, 32, 24)]:
        g = torch.Generator().manual_seed(0)
        m = sample_token_mask(1, grid, mask_ratio=0.4, generator=g)
        frac = mask_hidden_fraction(m)
        print(f"[mask-ratio] grid={grid} hidden={frac:.3f}")
        assert abs(frac - 0.4) < 0.08


def main():
    for arm in ("A0", "A1", "A2", "A3"):
        check_arm(arm)
    check_token_budget()
    check_mask_ratio_real_sizes()
    print("PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
