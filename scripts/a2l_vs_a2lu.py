#!/usr/bin/env python
"""A2-L vs A2-LU: JEPA on 20 labeled only vs 20L+80U.

Same seg loader (20L); only the JEPA image pool differs.
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

from train import SyntheticWORDLike, _read_id_list, collate_batch, save_checkpoint
from ujepa.losses import dice_ce_loss, deep_supervision_loss, jepa_smooth_l1, lambda_j_schedule
from ujepa.metrics import dice_per_class
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model
from ujepa.whole_volume_eval import evaluate_word_whole_volume


class SplitNpy(torch.utils.data.Dataset):
    def __init__(self, root: Path, ids):
        self.root = root
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        import numpy as np

        img = np.load(self.root / "images" / f"{cid}.npy")
        image = torch.from_numpy(img.astype("float32"))
        if image.dim() == 3:
            image = image.unsqueeze(0)
        lab_p = self.root / "labels" / f"{cid}.npy"
        if lab_p.exists():
            label = torch.from_numpy(np.load(lab_p).astype("int64"))
            is_l = True
        else:
            label = torch.zeros(image.shape[-3:], dtype=torch.long)
            is_l = False
        return {
            "image": image,
            "label": label,
            "is_labeled": torch.tensor(is_l, dtype=torch.bool),
            "case_id": cid,
        }


def seg_loss(logits, y, n):
    if isinstance(logits, (list, tuple)):
        return deep_supervision_loss(logits, y, n)
    return dice_ce_loss(logits, y, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["L", "LU"])
    p.add_argument("--cache", default="/data/hyc/U-JEPANet/data/word_sll20_cache")
    p.add_argument("--split-dir", default="/data/hyc/U-JEPANet/data/splits_sll20")
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
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
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    split = Path(args.split_dir)

    labeled_ids = _read_id_list(cache / "splits" / "train_labeled.txt")
    if args.mode == "L":
        jepa_ids = labeled_ids
    else:
        jepa_ids = _read_id_list(cache / "splits" / "train_all.txt")
    val_ids = _read_id_list(split / "val_20.txt")

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

    ds_l = SplitNpy(cache, labeled_ids)
    ds_j = SplitNpy(cache, jepa_ids)
    seg_loader = torch.utils.data.DataLoader(ds_l, batch_size=args.batch, shuffle=True, collate_fn=collate_batch)
    jepa_loader = torch.utils.data.DataLoader(ds_j, batch_size=args.batch, shuffle=True, collate_fn=collate_batch)

    params = list(model.online.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    it_s = iter(seg_loader)
    it_j = iter(jepa_loader)
    best = -1.0
    history = []
    t0 = time.time()
    print(f"mode={args.mode} n_labeled={len(labeled_ids)} n_jepa={len(jepa_ids)}", flush=True)
    for step in range(1, args.steps + 1):
        try:
            bs = next(it_s)
        except StopIteration:
            it_s = iter(seg_loader)
            bs = next(it_s)
        try:
            bj = next(it_j)
        except StopIteration:
            it_j = iter(jepa_loader)
            bj = next(it_j)

        x = bs["image"].to(device)
        y = bs["label"].to(device)
        xj = bj["image"].to(device)
        lj = lambda_j_schedule(step, args.seg_only, args.ramp, args.lambda_j)
        opt.zero_grad(set_to_none=True)
        logits = model.forward_seg(x)
        l_s = seg_loss(logits, y, 17)
        l_s.backward()
        l_jv = 0.0
        if lj > 0:
            outj = model.jepa_step(xj)
            l_j = jepa_smooth_l1(outj["pred_tokens"], outj["tgt_tokens"], outj["visible"])
            (lj * l_j).backward()
            l_jv = float(l_j.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.update_ema()
        model.target.train(False)
        model.target.target.eval()
        if step % 200 == 0 or step == 1:
            print(f"step={step} l_seg={float(l_s.detach()):.4f} l_j={l_jv:.4f} lam={lj:.2f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            vm = evaluate_word_whole_volume(
                model.online if hasattr(model, "online") else model,
                args.word_root,
                val_ids,
                17,
                (128, 128, 96),
                (64, 64, 48),
                device,
            )
            md = vm["mean_fg_dice"]
            history.append({"step": step, "mean_fg_dice": md})
            print(f"[val@{step}] {md:.4f}", flush=True)
            model.train()
            model.target.train(False)
            model.target.target.eval()
            if md > best:
                best = md
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step}, out / "best.pt")

    summary = {
        "mode": args.mode,
        "n_jepa_images": len(jepa_ids),
        "steps": args.steps,
        "best_val": best,
        "elapsed_sec": time.time() - t0,
        "val_curve": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
