#!/usr/bin/env python
"""B3: JEPA-only pretrain on 100 train images → segmentation fine-tune (no JEPA).

Stage 1: minimize L_J only (A2-style online+predictor+EMA).
Stage 2: load online encoder into A2 inference graph (or A0), train L_seg only
on 20 labeled; JEPA fully off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import build_sll_dual_loaders, save_checkpoint
from ujepa.losses import dice_ce_loss, jepa_smooth_l1, deep_supervision_loss
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model, build_online_for_arm
from ujepa.whole_volume_eval import evaluate_word_whole_volume


def seg_loss(logits, y, n_class):
    if isinstance(logits, (list, tuple)):
        return deep_supervision_loss(logits, y, n_class)
    return dice_ce_loss(logits, y, n_class)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/data/hyc/U-JEPANet/data/word_sll20_cache")
    p.add_argument("--split-dir", default="/data/hyc/U-JEPANet/data/splits_sll20")
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--out", required=True)
    p.add_argument("--pre-steps", type=int, default=8000)
    p.add_argument("--ft-steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = UJEPAConfig(
        arm="A2",
        feature_chns=[16, 32, 64, 128],
        class_num=17,
        embed_dim=256,
        num_heads=4,
        num_blocks=4,
        predictor_blocks=2,
        jepa_token_stride=16,
        deep_stage=2,
        multiscale_pred=True,
        mask_ratio=0.4,
    )
    model = build_model(cfg).to(device)
    assert isinstance(model, UJEPATrainer)
    raw = {
        "data": {
            "kind": "npy_sll",
            "root": args.cache,
            "val_ids_file": str(Path(args.split_dir) / "val_20.txt"),
            "word_root": args.word_root,
            "patch_size": [128, 128, 96],
        },
        "train": {"batch_size": args.batch},
    }
    seg_loader, jepa_loader, val_ids = build_sll_dual_loaders(cfg, raw, args.seed)
    val_ids = [ln.strip() for ln in (Path(args.split_dir) / "val_20.txt").read_text().splitlines() if ln.strip()]

    # ---- Stage 1: JEPA only ----
    params_j = list(model.online.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params_j, lr=args.lr, weight_decay=1e-4)
    it = iter(jepa_loader)
    print(f"=== STAGE1 JEPA pretrain {args.pre_steps} steps ===", flush=True)
    model.train()
    model.target.train(False)
    model.target.target.eval()
    t0 = time.time()
    for step in range(1, args.pre_steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(jepa_loader)
            b = next(it)
        x = b["image"].to(device)
        outj = model.jepa_step(x)
        loss = jepa_smooth_l1(outj["pred_tokens"], outj["tgt_tokens"], outj["visible"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_j, 1.0)
        opt.step()
        model.update_ema()
        model.target.train(False)
        model.target.target.eval()
        if step % 100 == 0 or step == 1:
            print(f"pre {step}/{args.pre_steps} l_j={float(loss.detach()):.4f}", flush=True)
    save_checkpoint(out / "pretrain_last.pt", model, opt, args.pre_steps, {"stage": "pretrain"}, cfg.to_dict())
    print(f"STAGE1_DONE {time.time()-t0:.1f}s", flush=True)

    # ---- Stage 2: seg fine-tune, JEPA off ----
    online = build_online_for_arm(cfg).to(device)
    online.load_state_dict(model.online.state_dict(), strict=True)
    del model
    torch.cuda.empty_cache()

    params_s = list(online.parameters())
    opt_s = torch.optim.AdamW(params_s, lr=args.lr, weight_decay=1e-4)
    it_s = iter(seg_loader)
    print(f"=== STAGE2 seg FT {args.ft_steps} steps (JEPA off) ===", flush=True)
    best = -1.0
    history = []
    for step in range(1, args.ft_steps + 1):
        try:
            b = next(it_s)
        except StopIteration:
            it_s = iter(seg_loader)
            b = next(it_s)
        x = b["image"].to(device)
        y = b["label"].to(device)
        online.train()
        logits = online(x)
        loss = seg_loss(logits, y, 17)
        opt_s.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_s, 1.0)
        opt_s.step()
        if step % 100 == 0 or step == 1:
            print(f"ft {step}/{args.ft_steps} l_seg={float(loss.detach()):.4f}", flush=True)
        if step % args.val_every == 0 or step == args.ft_steps:
            vm = evaluate_word_whole_volume(
                online, args.word_root, val_ids, 17, (128, 128, 96), (64, 64, 48), device
            )
            md = vm["mean_fg_dice"]
            history.append({"step": step, "mean_fg_dice": md})
            print(f"[val@{step}] mean_fg_dice={md:.4f}", flush=True)
            online.train()
            if md > best:
                best = md
                torch.save({"model": online.state_dict(), "config": cfg.to_dict(), "step": step}, out / "ft_best.pt")

    summary = {
        "protocol": "B3 JEPA-pretrain then seg-FT (JEPA off)",
        "pre_steps": args.pre_steps,
        "ft_steps": args.ft_steps,
        "best_val": best,
        "val_curve": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("B3_DONE", out, flush=True)


if __name__ == "__main__":
    main()
