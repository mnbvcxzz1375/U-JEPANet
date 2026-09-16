#!/usr/bin/env python
"""Cache SLL patches for dual-loader 30k training.

- labeled: GT from labelsTr_All
- unlabeled: image-only (no label file written)
- val: optional centre crops for crop-mode val; whole-val uses live NIfTI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def read_nifti(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-root", required=True)
    p.add_argument("--split-dir", required=True, help="dir with labeled_train_20.txt / unlabeled_train_80.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--patch", default="128,128,96")
    p.add_argument("--crops-labeled", type=int, default=12)
    p.add_argument("--crops-unlabeled", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    patch = tuple(int(x) for x in args.patch.split(","))
    root = Path(args.word_root)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    labeled = [ln.strip() for ln in (Path(args.split_dir) / "labeled_train_20.txt").read_text().splitlines() if ln.strip()]
    unlabeled = [ln.strip() for ln in (Path(args.split_dir) / "unlabeled_train_80.txt").read_text().splitlines() if ln.strip()]
    g = torch.Generator().manual_seed(args.seed)

    def take(arr, z0, y0, x0):
        pd, ph, pw = patch
        c = arr[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
        if c.shape != patch:
            pad = (
                (0, max(0, pd - c.shape[0])),
                (0, max(0, ph - c.shape[1])),
                (0, max(0, pw - c.shape[2])),
            )
            c = np.pad(c, pad, mode="constant", constant_values=0)
        return c

    def random_origin(shape):
        d, h, w = shape
        pd, ph, pw = patch
        z0 = int(torch.randint(0, max(1, d - pd + 1), (1,), generator=g).item()) if d >= pd else 0
        y0 = int(torch.randint(0, max(1, h - ph + 1), (1,), generator=g).item()) if h >= ph else 0
        x0 = int(torch.randint(0, max(1, w - pw + 1), (1,), generator=g).item()) if w >= pw else 0
        return z0, y0, x0

    labeled_ids = []
    for cid in labeled:
        img = read_nifti(root / "imagesTr" / f"{cid}.nii.gz").astype(np.float32)
        lab = read_nifti(root / "labelsTr_All" / f"{cid}.nii.gz").astype(np.int64)
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / max(hi - lo, 1e-6)
        for k in range(args.crops_labeled):
            z0, y0, x0 = random_origin(img.shape)
            iid = f"{cid}_L{k}"
            np.save(out / "images" / f"{iid}.npy", take(img, z0, y0, x0))
            np.save(out / "labels" / f"{iid}.npy", take(lab, z0, y0, x0))
            labeled_ids.append(iid)
        print("labeled", cid, flush=True)

    all_ids = list(labeled_ids)
    for cid in unlabeled:
        img = read_nifti(root / "imagesTr" / f"{cid}.nii.gz").astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / max(hi - lo, 1e-6)
        for k in range(args.crops_unlabeled):
            z0, y0, x0 = random_origin(img.shape)
            iid = f"{cid}_U{k}"
            np.save(out / "images" / f"{iid}.npy", take(img, z0, y0, x0))
            all_ids.append(iid)
        print("unlabeled", cid, flush=True)

    (out / "splits" / "train_labeled.txt").write_text("\n".join(labeled_ids) + "\n", encoding="utf-8")
    (out / "splits" / "train_all.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    meta = {
        "patch": list(patch),
        "n_labeled_cases": len(labeled),
        "n_unlabeled_cases": len(unlabeled),
        "crops_labeled": args.crops_labeled,
        "crops_unlabeled": args.crops_unlabeled,
        "n_labeled_patches": len(labeled_ids),
        "n_all_patches": len(all_ids),
        "seed": args.seed,
        "no_imagesTs": True,
        "no_val_in_cache": True,
    }
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("SLL_CACHE_DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
