from __future__ import annotations

"""Lightweight interface / no-leakage checks for A0-A3. Run: python scripts/preflight.py"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.losses import dice_ce_loss, jepa_smooth_l1
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model, build_online_for_arm
from ujepa.masking import sample_block_mask


def check_arm(arm: str) -> None:
    cfg = UJEPAConfig(
        arm=arm,
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
    x = torch.randn(2, 1, 32, 32, 32)
    y = torch.randint(0, 4, (2, 32, 32, 32))

    if arm == "A0":
        logits = model(x)
        loss = dice_ce_loss(logits, y, 4)
        loss.backward()
        print(f"[{arm}] seg-only OK params={sum(p.numel() for p in model.parameters()):,}")
        return

    assert isinstance(model, UJEPATrainer)
    logits = model.forward_seg(x)
    l_seg = dice_ce_loss(logits, y, 4)
    if arm == "A1":
        l_seg.backward()
        assert model.online.dualpath.alpha.grad is not None
        print(f"[{arm}] structure-only OK params={sum(p.numel() for p in model.parameters()):,}")
        return

    out = model.jepa_step(x)
    l_j = jepa_smooth_l1(out["pred_tokens"], out["tgt_tokens"], out["visible"])
    (l_seg + 0.3 * l_j).backward()
    assert model.online.dualpath is not None or model.online.jepa_head is not None
    print(
        f"[{arm}] joint OK l_seg={float(l_seg.detach()):.4f} l_jepa={float(l_j.detach()):.4f} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    # Inference path drops predictor
    online = build_online_for_arm(cfg)
    online.load_state_dict(model.online.state_dict())
    with torch.no_grad():
        a = model.forward_seg(x)
        b = online(x)
    assert torch.allclose(a, b, atol=1e-5)
    print(f"[{arm}] inference online parity OK")


def main():
    for arm in ("A0", "A1", "A2", "A3"):
        check_arm(arm)
    print("PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
