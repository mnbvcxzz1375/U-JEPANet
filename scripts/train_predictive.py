#!/usr/bin/env python
"""Train Predictive U-Net arms (V1) against A0DA data pipeline.

Arms:
  P1 / B1  predictor in forward, lambda_p=0
  P2 / B2  + pred loss
  P3 / B3  pred loss, no residual merge
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

from train import _read_id_list
from ujepa.dynamic_dataset import DynamicWordVolumeDataset
from ujepa.losses import deep_supervision_loss, dice_ce_loss
from ujepa.predictive_unet import PredictiveUNet, pred_loss_from_aux
from ujepa.whole_volume_eval import evaluate_word_whole_volume


def collate(b):
    return {
        "image": torch.stack([i["image"] for i in b]),
        "label": torch.stack([i["label"] for i in b]),
        "is_labeled": torch.stack([i["is_labeled"] for i in b]),
    }


def seg_loss(logits, y, n):
    if isinstance(logits, (list, tuple)):
        return deep_supervision_loss(logits, y, n)
    return dice_ce_loss(logits, y, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["P1", "P2", "P3", "B1", "B2", "B3"])
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--split-dir", default="/data/hyc/U-JEPANet/data/splits_sll20")
    p.add_argument("--cache-dir", default="/tmp/ujepa_hu_p")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--lambda-p", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--intensity-mode", default="window", choices=["window", "minmax"],
                   help="dynamic-pipeline arms use window to match A0DA")
    args = p.parse_args()

    arm = args.arm.upper().replace("B", "P") if args.arm[0] in "Bb" else args.arm.upper()
    # P1 structure only; P2 + pred; P3 + pred no residual
    use_residual = arm != "P3"
    lambda_p = 0.0 if arm == "P1" else args.lambda_p

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = Path(args.split_dir)
    labeled = _read_id_list(split / "labeled_train_20.txt")
    val_ids = _read_id_list(split / "val_20.txt")

    # Same pipeline as A0DA: dynamic + CT-med, labeled-only seg
    ds = DynamicWordVolumeDataset(
        args.word_root, labeled, (128, 128, 96), labeled=True, seed=args.seed,
        mode="seg", strength=1.0, cache_dir=args.cache_dir, fg_crop_prob=0.7,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    model = PredictiveUNet(
        feature_chns=[16, 32, 64, 128],
        class_num=17,
        deep_stage=2,
        embed_dim=args.embed_dim,
        mask_ratio=args.mask_ratio,
        use_residual=use_residual,
        use_pred_in_forward=True,
        multiscale_pred=True,
        norm="instance",
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    it = iter(loader)
    best, history = -1.0, []
    t0 = time.time()
    print(f"arm={arm} lambda_p={lambda_p} residual={use_residual} n_l={len(labeled)}", flush=True)

    for step in range(1, args.steps + 1):
        try:
            bs = next(it)
        except StopIteration:
            it = iter(loader)
            bs = next(it)
        x = bs["image"].to(device)
        y = bs["label"].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, train_mode=True)
        l_s = seg_loss(logits, y, 17)
        l_p = pred_loss_from_aux(model.last_aux) if model.last_aux else torch.tensor(0.0, device=device)
        if not torch.is_tensor(l_p):
            l_p = torch.tensor(float(l_p), device=device)
        loss = l_s + lambda_p * l_p.to(device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"step={step} l_seg={float(l_s.detach()):.4f} l_p={float(l_p.detach()):.4f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            vm = evaluate_word_whole_volume(
                model, args.word_root, val_ids, 17, (128, 128, 96), (64, 64, 48), device,
                intensity_mode=args.intensity_mode,
            )
            md = vm["mean_fg_dice"]
            history.append({"step": step, "mean_fg_dice": md})
            print(f"[val@{step}] {md:.4f}", flush=True)
            model.train()
            if md > best:
                best = md
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arm": arm,
                        "lambda_p": lambda_p,
                        "use_residual": use_residual,
                        "mask_ratio": args.mask_ratio,
                        "embed_dim": args.embed_dim,
                        "intensity_mode": args.intensity_mode,
                        "step": step,
                    },
                    out / "best.pt",
                )

    summary = {
        "arm": arm, "best_val": best, "steps": args.steps,
        "lambda_p": lambda_p, "use_residual": use_residual,
        "elapsed_sec": time.time() - t0, "val_curve": history,
        "intensity_mode": args.intensity_mode, "baseline": "A0DA pipeline",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
