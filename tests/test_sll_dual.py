from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import build_sll_dual_loaders
from ujepa.model import UJEPAConfig, build_model
from ujepa.whole_volume_eval import sliding_window_logits


def test_dual_loader_synthetic_always_labeled_seg():
    cfg = UJEPAConfig(arm="A3", feature_chns=[8, 16, 32, 64], class_num=17, embed_dim=48,
                      num_heads=2, num_blocks=1, predictor_blocks=1, jepa_token_stride=8,
                      deep_stage=2, multiscale_pred=False)
    raw = {
        "data": {"synthetic": True, "patch_size": [32, 32, 32]},
        "train": {"batch_size": 2},
    }
    seg_loader, jepa_loader, _ = build_sll_dual_loaders(cfg, raw, seed=0)
    b = next(iter(seg_loader))
    assert bool(b["is_labeled"].all())
    jb = next(iter(jepa_loader))
    assert jb["image"].shape[0] == 2


def test_sliding_window_shape():
    cfg = UJEPAConfig(arm="A0", feature_chns=[8, 16, 32, 64], class_num=17, multiscale_pred=False)
    model = build_model(cfg)
    vol = torch.randn(1, 1, 40, 40, 32)
    logits = sliding_window_logits(model, vol, (32, 32, 32), (16, 16, 16), torch.device("cpu"), 17)
    assert logits.shape == (1, 17, 40, 40, 32)
