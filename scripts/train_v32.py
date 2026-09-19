#!/usr/bin/env python
"""Train V3.2 / A0DA_GLP arms.

Arms:
  A0DA  plain U-Net + dyn+CT-med + isolated loader RNG  (R1 control)
  H0    V3.2 PE-only aligned innovation (same stack as H1)
  H1    V3.2 patient-global aligned innovation
  R1    local bottleneck only (already trained; optional re-run)
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
from ujepa.aligned_global_innovation import V32UNet
from ujepa.dynamic_dataset import DynamicWordVolumeDataset
from ujepa.glp_eval import evaluate_word_whole_volume_glp
from ujepa.losses import deep_supervision_loss, dice_ce_loss
from ujepa.unet3d import UNet3D
from ujepa.whole_volume_eval import evaluate_word_whole_volume


def collate(b):
    out = {
        "image": torch.stack([i["image"] for i in b]),
        "label": torch.stack([i["label"] for i in b]),
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
    p.add_argument("--arm", required=True, choices=["A0DA", "H0", "H1", "R1"])
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--split-dir", default="/data/hyc/U-JEPANet/data/splits_sll20")
    p.add_argument("--cache-dir", default="/tmp/ujepa_hu_v32")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha-init", type=float, default=0.04)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--intensity-mode", default="window")
    p.add_argument("--mech-every", type=int, default=200)
    args = p.parse_args()

    arm = args.arm.upper()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = Path(args.split_dir)
    labeled = _read_id_list(split / "labeled_train_20.txt")
    val_ids = _read_id_list(split / "val_20.txt")

    need_global = arm in ("H0", "H1")  # both run same sampler path
    need_meta = arm in ("H0", "H1", "R1")
    cache_dir = Path(args.cache_dir) / f"{arm}_s{args.seed}"
    ds = DynamicWordVolumeDataset(
        args.word_root, labeled, (128, 128, 96), labeled=True, seed=args.seed,
        mode="seg", strength=1.0, cache_dir=cache_dir, fg_crop_prob=0.7,
        return_crop_meta=need_meta,
        return_global=need_global,
        global_size=(64, 64, 64),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed + 10000),
    )

    if arm == "A0DA":
        model = UNet3D(
            in_chns=1, feature_chns=[16, 32, 64, 128], class_num=17,
            multiscale_pred=True, norm="instance",
        ).to(device)
        is_v32 = False
    else:
        model = V32UNet(
            arm=arm if arm != "A0DA" else "H1",
            feature_chns=(16, 32, 64, 128),
            class_num=17,
            embed_dim=args.embed_dim,
            alpha_init=args.alpha_init,
            use_image_global=(arm == "H1"),
            multiscale_pred=True,
        ).to(device)
        if arm == "H0":
            model.arm = "H0"
            model.use_image_global = False
        is_v32 = True

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    it = iter(loader)
    best, history, mech_log = -1.0, [], []
    t0 = time.time()
    print(f"arm={arm} seed={args.seed} alpha0={args.alpha_init if is_v32 else 0}", flush=True)

    for step in range(1, args.steps + 1):
        try:
            bs = next(it)
        except StopIteration:
            it = iter(loader)
            bs = next(it)
        x = bs["image"].to(device)
        y = bs["label"].to(device)
        kw = {}
        if is_v32 and "crop_origin" in bs:
            kw["crop_origin"] = bs["crop_origin"].to(device)
            kw["full_shape"] = bs["full_shape"].to(device)
            kw["affine_theta"] = bs["affine_theta"].to(device)
        if is_v32 and "global_image" in bs:
            kw["global_image"] = bs["global_image"].to(device)
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, **kw) if is_v32 else model(x)
        loss = seg_loss(logits, y, 17)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if is_v32 and (step % args.mech_every == 0 or step == 1):
            m = model.mech_dict
            mech_log.append({"step": step, **m})
            print(f"step={step} l={float(loss.detach()):.4f} mech={m}", flush=True)
        elif step % 200 == 0 or step == 1:
            print(f"step={step} l={float(loss.detach()):.4f}", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            if arm == "A0DA":
                # P0: plain UNet3D.forward(x) — do not call GLP kwargs
                vm = evaluate_word_whole_volume(
                    model, args.word_root, val_ids, 17, (128, 128, 96), (64, 64, 48), device,
                    intensity_mode=args.intensity_mode,
                )
            else:
                need_g = arm == "H1"
                vm = evaluate_word_whole_volume_glp(
                    model, args.word_root, val_ids, 17, (128, 128, 96), (64, 64, 48), device,
                    intensity_mode=args.intensity_mode,
                    need_global=need_g,
                )
            md = vm["mean_fg_dice"]
            history.append({"step": step, "mean_fg_dice": md,
                            "alpha": model.alpha_g_value if is_v32 else 0.0})
            print(f"[val@{step}] {md:.4f} alpha={model.alpha_g_value if is_v32 else 0:.4f}", flush=True)
            if md > best:
                best = md
                torch.save({
                    "model": model.state_dict(),
                    "arm": arm,
                    "code": "V3.2-aligned-innovation" if is_v32 else "A0DA_GLP",
                    "alpha_init": args.alpha_init if is_v32 else None,
                    "alpha_final": model.alpha_g_value if is_v32 else None,
                    "embed_dim": args.embed_dim,
                    "intensity_mode": args.intensity_mode,
                    "selection": "imagesVal",
                    "step": step,
                }, out / "best.pt")

    summary = {
        "arm": arm,
        "code": "V3.2-aligned-innovation" if is_v32 else "A0DA_GLP",
        "best_val": best,
        "steps": args.steps,
        "alpha_init": args.alpha_init if is_v32 else None,
        "alpha_final": model.alpha_g_value if is_v32 else None,
        "val_curve": history,
        "mech_log": mech_log,
        "intensity_mode": args.intensity_mode,
        "loader_generator": f"seed+{10000}",
        "elapsed_sec": time.time() - t0,
        "note": "H1-H0 = patient-specific aligned global content; A0DA_GLP controls R1 story",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "mech_log" and k != "val_curve"}, indent=2))


if __name__ == "__main__":
    main()
