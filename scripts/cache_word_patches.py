#!/usr/bin/env python
"""Pre-cache fixed WORD patches to .npy for fast pilot training.

Uses imagesTr + labelsTr_All and imagesVal + labelsVal only.
No imagesTs access.
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


def crop_or_pad(arr: np.ndarray, patch) -> np.ndarray:
    pd, ph, pw = patch
    d, h, w = arr.shape
    # centre crop for val stability; jittered centre for train diversity saved as multiple
    z0 = max(0, (d - pd) // 2)
    y0 = max(0, (h - ph) // 2)
    x0 = max(0, (w - pw) // 2)
    c = arr[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
    if c.shape != tuple(patch):
        pad = (
            (0, max(0, pd - c.shape[0])),
            (0, max(0, ph - c.shape[1])),
            (0, max(0, pw - c.shape[2])),
        )
        c = np.pad(c, pad, mode="constant", constant_values=0)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--patch", default="96,96,64")
    p.add_argument("--train-cases", type=int, default=20)
    p.add_argument("--val-cases", type=int, default=8)
    p.add_argument("--crops-per-case", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    patch = tuple(int(x) for x in args.patch.split(","))
    root = Path(args.word_root)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(args.seed)

    def make_split(image_dir: str, label_dir: str, n_cases: int, split: str):
        files = sorted((root / image_dir).glob("word_*.nii.gz"))[:n_cases]
        ids = []
        for f in files:
            cid = f.name.replace(".nii.gz", "")
            img = read_nifti(f).astype(np.float32)
            lab = read_nifti(root / label_dir / f.name).astype(np.int64)
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / max(hi - lo, 1e-6)
            d, h, w = img.shape
            pd, ph, pw = patch
            for k in range(args.crops_per_case):
                if split == "val":
                    z0 = max(0, (d - pd) // 2)
                    y0 = max(0, (h - ph) // 2)
                    x0 = max(0, (w - pw) // 2)
                    if k > 0:
                        # one centre crop only for val per case
                        continue
                else:
                    z0 = int(torch.randint(0, max(1, d - pd + 1), (1,), generator=g).item()) if d >= pd else 0
                    y0 = int(torch.randint(0, max(1, h - ph + 1), (1,), generator=g).item()) if h >= ph else 0
                    x0 = int(torch.randint(0, max(1, w - pw + 1), (1,), generator=g).item()) if w >= pw else 0
                def take(a):
                    c = a[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                    if c.shape != patch:
                        pad = (
                            (0, max(0, pd - c.shape[0])),
                            (0, max(0, ph - c.shape[1])),
                            (0, max(0, pw - c.shape[2])),
                        )
                        c = np.pad(c, pad, mode="constant", constant_values=0)
                    return c
                iid = f"{cid}_c{k}" if split == "train" else f"{cid}_v{k}"
                np.save(out / "images" / f"{iid}.npy", take(img))
                np.save(out / "labels" / f"{iid}.npy", take(lab))
                ids.append(iid)
            print(f"{split} {cid}: done", flush=True)
        (out / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        return ids

    train_ids = make_split("imagesTr", "labelsTr_All", args.train_cases, "train")
    val_ids = make_split("imagesVal", "labelsVal", args.val_cases, "val")
    meta = {
        "patch": list(patch),
        "train_cases": args.train_cases,
        "val_cases": args.val_cases,
        "crops_per_case": args.crops_per_case,
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "word_root": str(root),
        "no_imagesTs": True,
    }
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("CACHE_DONE", meta, flush=True)


if __name__ == "__main__":
    main()
