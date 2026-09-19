#!/usr/bin/env python
"""Train V3 GLP arms G0/G1/G2 (and R1 control). Selection on imagesVal only."""
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
from ujepa.global_local_predictive import GLPUNet
from ujepa.glp_eval import evaluate_word_whole_volume_glp
from ujepa.losses import deep_supervision_loss, dice_ce_loss


def collate(b):
    out = {
        "image": torch.stack([i["image"] for i in b]),
        "label": torch.stack([i["label"] for i in b]),
        "is_labeled": torch.stack([i["is_labeled"] for i in b]),
        "case_id": [i.get("case_id", "") for i in b],
    }
    if "crop_origin" in b[0]:
        out["crop_origin"] = torch.stack([i["crop_origin"] for i in b])
        out["full_shape"] = torch.stack([i["full_shape"] for i in b])
        out["affine_theta"] = torch.stack([i["affine_theta"] for i in b])
    if "global_image" in b[0]:
        out["global_image"] = torch.stack([i["global_image"] for i in b])
    return out


def seg_loss(logits, y, n):
    if isinstance(logits, (list, tuple)):
        return deep_supervision_loss(logits, y, n)
    return dice_ce_loss(logits, y, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["R1", "G0", "G1", "G2"])
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--split-dir", default="/data/hyc/U-JEPANet/data/splits_sll20")
    p.add_argument("--cache-dir", default="/tmp/ujepa_hu_glp")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--lambda-gl", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--intensity-mode", default="window")
    p.add_argument("--eval-selection", default="imagesVal", choices=["imagesVal"],
                   help="architecture selection locked to val; do not use test")
    args = p.parse_args()

    arm = args.arm.upper()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = Path(args.split_dir)
    labeled = _read_id_list(split / "labeled_train_20.txt")
    val_ids = _read_id_list(split / "val_20.txt")

    need_global = arm in ("G1", "G2")
    # unique cache per job/arm/seed to avoid multi-process npy races
    cache_dir = Path(args.cache_dir) / f"{arm}_s{args.seed}"
    ds = DynamicWordVolumeDataset(
        args.word_root, labeled, (128, 128, 96), labeled=True, seed=args.seed,
        mode="seg", strength=1.0, cache_dir=cache_dir, fg_crop_prob=0.7,
        return_crop_meta=True,
        return_global=need_global,
        global_size=(64, 64, 64),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        # P1: isolate shuffle RNG from model-init RNG
        generator=torch.Generator().manual_seed(args.seed + 10000),
    )

    model = GLPUNet(
        arm=arm,
        feature_chns=(16, 32, 64, 128),
        class_num=17,
        embed_dim=args.embed_dim,
        lambda_gl=args.lambda_gl,
        multiscale_pred=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    it = iter(loader)
    best, history = -1.0, []
    t0 = time.time()
    print(f"arm={arm} seed={args.seed} lambda_gl={args.lambda_gl} alpha_g0={model.alpha_g_value}", flush=True)

    for step in range(1, args.steps + 1):
        try:
            bs = next(it)
        except StopIteration:
            it = iter(loader)
            bs = next(it)
        x = bs["image"].to(device)
        y = bs["label"].to(device)
        kw = {}
        if "crop_origin" in bs:
            kw["crop_origin"] = bs["crop_origin"].to(device)
            kw["full_shape"] = bs["full_shape"].to(device)
            kw["affine_theta"] = bs["affine_theta"].to(device)
        if "global_image" in bs:
            kw["global_image"] = bs["global_image"].to(device)
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, **kw)
        l_s = seg_loss(logits, y, 17)
        l_g = model.gl_loss if arm == "G2" else torch.zeros((), device=device)
        loss = l_s + (args.lambda_gl * l_g if arm == "G2" else 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(
                f"step={step} l_seg={float(l_s.detach()):.4f} l_gl={float(l_g.detach()):.4f} "
                f"alpha_g={model.alpha_g_value:.4f}",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            # P0: GLP-aware val (real origins + whole-CT global when needed)
            vm = evaluate_word_whole_volume_glp(
                model, args.word_root, val_ids, 17, (128, 128, 96), (64, 64, 48), device,
                intensity_mode=args.intensity_mode,
                need_global=need_global,
                global_size=(64, 64, 64),
            )
            md = vm["mean_fg_dice"]
            history.append({"step": step, "mean_fg_dice": md, "alpha_g": model.alpha_g_value})
            print(f"[val@{step}] {md:.4f} alpha_g={model.alpha_g_value:.4f}", flush=True)
            if md > best:
                best = md
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arm": arm,
                        "code": "V3-GLP",
                        "lambda_gl": args.lambda_gl,
                        "embed_dim": args.embed_dim,
                        "alpha_g": model.alpha_g_value,
                        "intensity_mode": args.intensity_mode,
                        "selection": "imagesVal",
                        "step": step,
                    },
                    out / "best.pt",
                )

    summary = {
        "arm": arm,
        "code": "V3-GLP",
        "best_val": best,
        "steps": args.steps,
        "lambda_gl": args.lambda_gl if arm == "G2" else 0.0,
        "alpha_g_final": model.alpha_g_value,
        "val_curve": history,
        "intensity_mode": args.intensity_mode,
        "selection": "imagesVal only",
        "elapsed_sec": time.time() - t0,
        "note": "R1 is architecture baseline; primary gate G1-G0; then G2-G1",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
