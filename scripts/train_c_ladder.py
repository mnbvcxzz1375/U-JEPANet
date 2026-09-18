#!/usr/bin/env python
"""C0–C4 dynamic-crop / CT-aug ladder for A2 (JEPA) on WORD SLL 20L.

C0 is conceptual baseline = current fixed-crop (already run as A2).
C1 dynamic crop only
C2 dynamic + CT-med aug (seg)
C3 dynamic + same CT-med on JEPA
C4 dynamic + target weak / context strong
A0D = pure U-Net + dynamic crop (no JEPA)
A0DA = pure U-Net + dynamic crop + CT-med strength=1 (no JEPA) — missing 2x2 cell
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
from ujepa.losses import deep_supervision_loss, dice_ce_loss, jepa_smooth_l1, lambda_j_schedule
from ujepa.model import UNet3D, UJEPAConfig, UJEPATrainer, build_model
from ujepa.whole_volume_eval import evaluate_word_whole_volume


def collate(b):
    return {
        "image": torch.stack([i["image"] for i in b]),
        "label": torch.stack([i["label"] for i in b]),
        "is_labeled": torch.stack([i["is_labeled"] for i in b]),
        "case_id": [i.get("case_id", "") for i in b],
    }


def seg_loss(logits, y, n):
    if isinstance(logits, (list, tuple)):
        return deep_supervision_loss(logits, y, n)
    return dice_ce_loss(logits, y, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["C1", "C2", "C3", "C4", "A0D", "A0DA"])
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--split-dir", default="/data/hyc/U-JEPANet/data/splits_sll20")
    p.add_argument("--cache-dir", default="/data/hyc/U-JEPANet/data/word_hu_volumes")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--lambda-j", type=float, default=0.3)
    p.add_argument("--seg-only", type=int, default=3000)
    p.add_argument("--ramp", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--intensity-mode", default="window", choices=["window", "minmax"],
                   help="must match training crop path; dynamic arms use window")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = Path(args.split_dir)
    labeled = _read_id_list(split / "labeled_train_20.txt")
    unlabeled = _read_id_list(split / "unlabeled_train_80.txt")
    all_ids = labeled + unlabeled
    val_ids = _read_id_list(split / "val_20.txt")

    # Arm → (strength, jepa_mode, use_unet)
    if args.arm == "A0D":
        strength, jepa_mode, use_unet = 0.0, "none", True
    elif args.arm == "A0DA":
        strength, jepa_mode, use_unet = 1.0, "none", True  # CT-med on seg, no JEPA
    elif args.arm == "C1":
        strength, jepa_mode, use_unet = 0.0, "none", False
    elif args.arm == "C2":
        strength, jepa_mode, use_unet = 1.0, "none", False
    elif args.arm == "C3":
        # NOTE: historical C3 passed mode=jepa_context which uses jepa_strong_context,
        # NOT seg_augment_hu — not literally identical to seg aug. Retained as-is.
        strength, jepa_mode, use_unet = 1.0, "seg", False
    elif args.arm == "C4":
        # WARNING: historical C4 used unpaired context/target loaders. Invalid for
        # weak/strong conclusions. This path now pairs case+crop via paired mode.
        strength, jepa_mode, use_unet = 1.0, "ws", False
    else:
        raise ValueError(args.arm)

    seg_ds = DynamicWordVolumeDataset(
        args.word_root, labeled, (128, 128, 96), labeled=True, seed=args.seed,
        mode="seg", strength=strength, cache_dir=args.cache_dir, fg_crop_prob=0.7,
    )
    it_jt = None
    if jepa_mode == "ws":
        # Paired weak/strong: single dataset samples case+crop once, then two views.
        from ujepa.paired_views import PairedJEPADataset
        jepa_ds = PairedJEPADataset(
            args.word_root, all_ids, (128, 128, 96), seed=args.seed + 1,
            cache_dir=args.cache_dir,
        )
        jepa_tar_loader = None
        it_jt = None
    elif jepa_mode == "seg":
        jepa_ds = DynamicWordVolumeDataset(
            args.word_root, all_ids, (128, 128, 96), labeled=False, seed=args.seed + 1,
            mode="jepa_context", strength=strength, cache_dir=args.cache_dir,
        )
    else:
        jepa_ds = DynamicWordVolumeDataset(
            args.word_root, all_ids, (128, 128, 96), labeled=False, seed=args.seed + 1,
            mode="none", strength=0.0, cache_dir=args.cache_dir,
        )

    seg_loader = torch.utils.data.DataLoader(
        seg_ds, batch_size=args.batch, shuffle=True, collate_fn=collate
    )
    if jepa_mode == "ws":
        def collate_paired(batch):
            return {
                "context": torch.stack([b["context"] for b in batch]),
                "target": torch.stack([b["target"] for b in batch]),
                "case_id": [b["case_id"] for b in batch],
            }
        jepa_loader = torch.utils.data.DataLoader(
            jepa_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_paired
        )
    else:
        jepa_loader = torch.utils.data.DataLoader(
            jepa_ds, batch_size=args.batch, shuffle=True, collate_fn=collate
        )

    cfg = UJEPAConfig(
        arm="A0" if use_unet else "A2",
        feature_chns=[16, 32, 64, 128], class_num=17, embed_dim=256,
        num_heads=4, num_blocks=4, predictor_blocks=2, jepa_token_stride=16,
        deep_stage=2, multiscale_pred=True, mask_ratio=0.4,
    )
    model = build_model(cfg).to(device)
    use_jepa = not isinstance(model, UNet3D)
    if use_jepa:
        assert isinstance(model, UJEPATrainer)
        params = list(model.online.parameters()) + list(model.predictor.parameters())
    else:
        params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    it_s, it_j = iter(seg_loader), iter(jepa_loader)
    best, history = -1.0, []
    t0 = time.time()
    print(
        f"arm={args.arm} use_jepa={use_jepa} intensity={args.intensity_mode} "
        f"n_l={len(labeled)} strength={strength} jepa_mode={jepa_mode}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        try:
            bs = next(it_s)
        except StopIteration:
            it_s = iter(seg_loader)
            bs = next(it_s)

        x = bs["image"].to(device)
        y = bs["label"].to(device)
        opt.zero_grad(set_to_none=True)
        if use_jepa:
            l_s = seg_loss(model.forward_seg(x), y, 17)
        else:
            l_s = seg_loss(model(x), y, 17)
        l_s.backward()
        l_jv = 0.0
        lj = 0.0
        if use_jepa:
            try:
                bj = next(it_j)
            except StopIteration:
                it_j = iter(jepa_loader)
                bj = next(it_j)
            lj = lambda_j_schedule(step, args.seg_only, args.ramp, args.lambda_j)
            if lj > 0:
                if jepa_mode == "ws":
                    xj = bj["context"].to(device)
                    x_tar = bj["target"].to(device)
                    # assert paired: same batch size; content pairing guaranteed by dataset
                else:
                    xj = bj["image"].to(device)
                    x_tar = None
                outj = model.jepa_step(xj, x_target=x_tar)
                l_j = jepa_smooth_l1(outj["pred_tokens"], outj["tgt_tokens"], outj["visible"])
                (lj * l_j).backward()
                l_jv = float(l_j.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if use_jepa:
            model.update_ema()
            model.target.train(False)
            model.target.target.eval()
        if step % 200 == 0 or step == 1:
            print(f"step={step} l_seg={float(l_s.detach()):.4f} l_j={l_jv:.4f} lam={lj:.2f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            eval_m = model.online if use_jepa else model
            vm = evaluate_word_whole_volume(
                eval_m, args.word_root, val_ids, 17, (128, 128, 96), (64, 64, 48), device,
                intensity_mode=args.intensity_mode,
            )
            md = vm["mean_fg_dice"]
            history.append({"step": step, "mean_fg_dice": md})
            print(f"[val@{step}] {md:.4f}", flush=True)
            model.train()
            if use_jepa:
                model.target.train(False)
                model.target.target.eval()
            if md > best:
                best = md
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.to_dict(),
                        "step": step,
                        "intensity_mode": args.intensity_mode,
                        "arm": args.arm,
                    },
                    out / "best.pt",
                )

    summary = {
        "arm": args.arm, "best_val": best, "steps": args.steps,
        "elapsed_sec": time.time() - t0, "val_curve": history,
        "strength": strength, "jepa_mode": jepa_mode, "use_jepa": use_jepa,
        "intensity_mode": args.intensity_mode,
        "notes": (
            "C4 historical runs used unpaired loaders; do not use for weak/strong gates."
            if args.arm == "C4" else
            "C3 jepa path uses jepa_strong_context, not identical to seg_augment_hu."
            if args.arm == "C3" else ""
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
