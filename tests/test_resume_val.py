from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import SyntheticWORDLike, collate_batch, load_checkpoint, train_loop
from ujepa.metrics import dice_per_class, run_validation
from ujepa.model import UJEPAConfig, build_model


def _tiny_cfg(arm="A3"):
    return UJEPAConfig(
        arm=arm,
        feature_chns=[8, 16, 32, 64],
        class_num=17,
        embed_dim=48,
        num_heads=2,
        num_blocks=1,
        predictor_blocks=1,
        jepa_token_stride=8,
        deep_stage=2,
        multiscale_pred=False,
        mask_ratio=0.4,
    )


def test_ema_stays_eval_after_model_train():
    cfg = _tiny_cfg("A3")
    model = build_model(cfg)
    model.train()
    assert model.target is not None
    assert model.target.training is False
    assert model.target.target.training is False
    model.update_ema(0.9)
    model.train()
    assert model.target.target.training is False


def test_dice_per_class():
    pred = torch.zeros(8, 8, 8, dtype=torch.long)
    tgt = torch.zeros(8, 8, 8, dtype=torch.long)
    pred[:4] = 1
    tgt[:4] = 1
    pred[4:, :4] = 2
    tgt[4:, :4] = 2
    pc = dice_per_class(pred, tgt, num_classes=17)
    assert abs(pc[1] - 1.0) < 1e-6
    assert abs(pc[2] - 1.0) < 1e-6


def test_resume_equivalence(tmp_path):
    """Same fixed batch every step: 20 continuous vs 10+save+resume+10."""
    # n=2, batch=2 → one batch/epoch, deterministic identical data each step.
    ds = SyntheticWORDLike(n=2, shape=(32, 32, 32), num_classes=17, seed=0, unlabeled_frac=0.0)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)
    device = torch.device("cpu")
    kwargs = dict(
        lr=1e-3,
        weight_decay=0.0,
        lambda_j_max=0.3,
        seg_only_steps=2,
        ramp_steps=2,
        grad_clip=1.0,
        log_every=100,
        save_every=0,
        val_loader=None,
        val_every=0,
        seed=0,
        out_dir=None,
    )

    torch.manual_seed(0)
    cfg = _tiny_cfg("A3")
    model_a = build_model(cfg)
    train_loop(model_a, loader, device, max_steps=20, **kwargs)
    w_a = {k: v.detach().float().clone() for k, v in model_a.online.state_dict().items()}

    torch.manual_seed(0)
    model_b = build_model(cfg)
    kw10 = dict(kwargs)
    kw10["out_dir"] = tmp_path
    kw10["save_every"] = 10
    train_loop(model_b, loader, device, max_steps=10, **kw10)
    ckpt_path = tmp_path / "last.pt"
    assert ckpt_path.exists()

    model_c = build_model(cfg)
    opt_c = torch.optim.AdamW(
        list(model_c.online.parameters()) + list(model_c.predictor.parameters()),
        lr=1e-3,
    )
    step = load_checkpoint(ckpt_path, model_c, opt_c, restore_rng=True)
    assert step == 10
    train_loop(model_c, loader, device, max_steps=20, resume_from=ckpt_path, **kwargs)

    max_diff = 0.0
    for k, va in w_a.items():
        max_diff = max(max_diff, float((va - model_c.online.state_dict()[k].float()).abs().max()))
    assert max_diff < 1e-4, f"resume mismatch max_diff={max_diff}"


def test_validation_runs():
    cfg = _tiny_cfg("A0")
    model = build_model(cfg)
    ds = SyntheticWORDLike(n=2, shape=(32, 32, 32), num_classes=17, seed=1)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_batch)
    m = run_validation(model, loader, torch.device("cpu"), 17)
    assert "mean_fg_dice" in m
    assert m["n_val_cases"] == 2
