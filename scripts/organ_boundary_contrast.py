#!/usr/bin/env python
"""Organ boundary contrast: inner band vs outer shells.

For each organ:
  inner_1: voxels in organ within 1 voxel of exterior (org & ~erode1)
  outer_k: voxels outside organ within k voxels (dilate_k & ~org), k=1,2,3
Metrics:
  mu_inner, mu_outer_k, signed contrast (inner-outer), |contrast|
  std_inner, std_outer, CNR = |mu_in-mu_out| / sqrt(var_in+var_out)
  |grad| on inner band (central differences on image)
  directional outer means along ±axis (6-neighborhood shells)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

WORD_ORGANS = {
    1: "Liver", 2: "Spleen", 3: "Kidney(L)", 4: "Kidney(R)", 5: "Stomach",
    6: "Gallbladder", 7: "Esophagus", 8: "Pancreas", 9: "Duodenum", 10: "Colon",
    11: "Intestine", 12: "Adrenal", 13: "Rectum", 14: "Bladder",
    15: "Femur(L)", 16: "Femur(R)",
}
AMOS_ORGANS = {
    1: "Spleen", 2: "Kidney(R)", 3: "Kidney(L)", 4: "Gallbladder", 5: "Esophagus",
    6: "Liver", 7: "Stomach", 8: "Aorta", 9: "Postcava", 10: "Pancreas",
    11: "Adrenal(R)", 12: "Adrenal(L)", 13: "Duodenum", 14: "Bladder",
    15: "Prostate/Uterus",
}
# FLARE2023 labeled 13 organs (nnU-Net style)
FLARE_ORGANS = {
    1: "Liver", 2: "Kidney(R)", 3: "Spleen", 4: "Pancreas", 5: "Aorta",
    6: "IVC", 7: "Adrenal(R)", 8: "Adrenal(L)", 9: "Gallbladder",
    10: "Esophagus", 11: "Stomach", 12: "Duodenum", 13: "Kidney(L)",
}


def read_nii(p: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))


def struct_ball(r: int) -> np.ndarray:
    d = 2 * r + 1
    zz, yy, xx = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (zz * zz + yy * yy + xx * xx) <= r * r + 1e-6


def boundary_metrics(img: np.ndarray, mask: np.ndarray, rings=(1, 2, 3)):
    """Return dict of metrics for one organ instance (one case).

    Operates on a padded bounding box around the organ for speed.
    """
    m_full = mask.astype(bool)
    if m_full.sum() < 50:
        return None
    coords = np.argwhere(m_full)
    pad = 6
    z0 = max(0, int(coords[:, 0].min()) - pad)
    z1 = min(m_full.shape[0], int(coords[:, 0].max()) + pad + 1)
    y0 = max(0, int(coords[:, 1].min()) - pad)
    y1 = min(m_full.shape[1], int(coords[:, 1].max()) + pad + 1)
    x0 = max(0, int(coords[:, 2].min()) - pad)
    x1 = min(m_full.shape[2], int(coords[:, 2].max()) + pad + 1)
    img = img[z0:z1, y0:y1, x0:x1]
    m = m_full[z0:z1, y0:y1, x0:x1]
    if m.sum() < 50:
        return None
    st1 = ndimage.generate_binary_structure(3, 1)
    er1 = ndimage.binary_erosion(m, structure=st1, iterations=1)
    inner1 = m & ~er1
    if inner1.sum() < 10:
        return None

    # image gradient magnitude (central diff) on crop only
    gy, gx, gz = np.gradient(img.astype(np.float64))
    gmag = np.sqrt(gy * gy + gx * gx + gz * gz)

    out = {
        "n_vox": int(m.sum()),
        "n_inner1": int(inner1.sum()),
        "mu_inner": float(img[inner1].mean()),
        "std_inner": float(img[inner1].std()),
        "grad_inner_mean": float(gmag[inner1].mean()),
        "grad_inner_p90": float(np.percentile(gmag[inner1], 90)),
    }

    dirs = {
        "out_z+": np.roll(m, -1, axis=0),
        "out_z-": np.roll(m, 1, axis=0),
        "out_y+": np.roll(m, -1, axis=1),
        "out_y-": np.roll(m, 1, axis=1),
        "out_x+": np.roll(m, -1, axis=2),
        "out_x-": np.roll(m, 1, axis=2),
    }
    dir_mu = {}
    for name, shifted in dirs.items():
        shell = shifted & ~m
        if shell.sum() >= 10:
            dir_mu[name] = float(img[shell].mean())
        else:
            dir_mu[name] = None
    out["dir_outer_mu"] = dir_mu
    valid_dirs = [v for v in dir_mu.values() if v is not None]
    if len(valid_dirs) >= 2:
        out["dir_outer_std_across_axes"] = float(np.std(valid_dirs))
        out["dir_outer_range"] = float(max(valid_dirs) - min(valid_dirs))
    else:
        out["dir_outer_std_across_axes"] = None
        out["dir_outer_range"] = None

    for k in rings:
        st = struct_ball(k)
        dil = ndimage.binary_dilation(m, structure=st, iterations=1)
        outer = dil & ~m
        n_out = int(outer.sum())
        key = f"r{k}"
        if n_out < 10:
            out[f"n_outer_{key}"] = n_out
            out[f"mu_outer_{key}"] = None
            out[f"std_outer_{key}"] = None
            out[f"contrast_{key}"] = None
            out[f"abs_contrast_{key}"] = None
            out[f"cnr_{key}"] = None
            continue
        mu_o = float(img[outer].mean())
        std_o = float(img[outer].std())
        mu_i = out["mu_inner"]
        std_i = out["std_inner"]
        contrast = mu_i - mu_o
        denom = np.sqrt(std_i ** 2 + std_o ** 2 + 1e-8)
        out[f"n_outer_{key}"] = n_out
        out[f"mu_outer_{key}"] = mu_o
        out[f"std_outer_{key}"] = std_o
        out[f"contrast_{key}"] = contrast
        out[f"abs_contrast_{key}"] = abs(contrast)
        out[f"cnr_{key}"] = abs(contrast) / denom
    return out


def aggregate(case_list):
    keys_mean = [
        "mu_inner", "std_inner", "grad_inner_mean", "grad_inner_p90",
        "dir_outer_std_across_axes", "dir_outer_range",
        "mu_outer_r1", "mu_outer_r2", "mu_outer_r3",
        "std_outer_r1", "std_outer_r2", "std_outer_r3",
        "contrast_r1", "contrast_r2", "contrast_r3",
        "abs_contrast_r1", "abs_contrast_r2", "abs_contrast_r3",
        "cnr_r1", "cnr_r2", "cnr_r3",
    ]
    dir_keys = ["out_z+", "out_z-", "out_y+", "out_y-", "out_x+", "out_x-"]
    agg = {"n_cases": len(case_list)}
    for k in keys_mean:
        vals = [c[k] for c in case_list if c.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
        agg[k + "_std"] = float(np.std(vals)) if vals else None
        agg[k + "_n"] = len(vals)
    # pooled directional means
    for dk in dir_keys:
        vals = [c["dir_outer_mu"].get(dk) for c in case_list if c.get("dir_outer_mu") and c["dir_outer_mu"].get(dk) is not None]
        agg[f"mu_{dk}"] = float(np.mean(vals)) if vals else None
    return agg


def run_dataset(name, root, img_dir, lab_dir, organs, pairs, max_cases, out_path):
    acc = defaultdict(list)
    files = sorted(Path(root, img_dir).glob("*.nii.gz"))
    if max_cases:
        files = files[:max_cases]
    print(f"[{name}] cases={len(files)} img={root}/{img_dir}", flush=True)
    for i, ip in enumerate(files):
        lp = Path(root, lab_dir) / ip.name
        if not lp.exists():
            continue
        img = read_nii(ip).astype(np.float32)
        lab = read_nii(lp).astype(np.int64)
        if img.shape != lab.shape:
            continue
        for c in organs:
            met = boundary_metrics(img, lab == c)
            if met:
                acc[c].append(met)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{name}] {i+1}/{len(files)}", flush=True)

    results = {}
    for c, oname in organs.items():
        if not acc[c]:
            results[oname] = {"label": c, "n_cases": 0}
            continue
        results[oname] = {"label": c, **aggregate(acc[c])}

    payload = {
        "dataset": name,
        "root": str(root),
        "n_files": len(files),
        "organs": organs,
        "metrics": results,
        "definition": {
            "inner_1": "organ voxels not in binary erosion(iter=1, 6-conn)",
            "outer_rk": "dilation(ball radius k) & ~organ",
            "contrast_rk": "mean(HU_inner1) - mean(HU_outer_rk)",
            "cnr_rk": "|contrast| / sqrt(var_inner + var_outer)",
            "grad_inner": "mean |nabla HU| on inner_1 (central differences)",
            "directions": "outer means via 6-neighborhood roll of organ mask",
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{name}] WROTE {out_path}")

    hdr = f"{'organ':16s} {'in':>7s} {'out1':>7s} {'c1':>7s} {'cnr1':>6s} {'out3':>7s} {'c3':>7s} {'cnr3':>6s} {'|g|':>7s} {'dir_rng':>7s}"
    print(f"[{name}] {hdr}")
    for c, oname in organs.items():
        r = results[oname]
        if r.get("n_cases", 0) == 0:
            print(f"[{name}] {oname:16s} empty")
            continue

        def f(k, w=7, d=2):
            v = r.get(k)
            return f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else " " * w

        print(
            f"[{name}] {oname:16s} "
            f"{f('mu_inner')} {f('mu_outer_r1')} {f('contrast_r1')} {f('cnr_r1',6,2)} "
            f"{f('mu_outer_r3')} {f('contrast_r3')} {f('cnr_r3',6,2)} "
            f"{f('grad_inner_mean')} {f('dir_outer_range')}"
        )
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["word", "amos", "flare"])
    p.add_argument("--root", required=True)
    p.add_argument("--img-dir", default=None)
    p.add_argument("--lab-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max-cases", type=int, default=0)
    args = p.parse_args()

    if args.dataset == "word":
        img_dir = args.img_dir or "imagesTr"
        lab_dir = args.lab_dir or "labelsTr_All"
        organs = WORD_ORGANS
        run_dataset(args.dataset, args.root, img_dir, lab_dir, organs, None, args.max_cases, args.out)
        return
    if args.dataset == "flare":
        run_flare(args)
        return

    # AMOS CT-only
    img_dir = args.img_dir or "imagesTr"
    lab_dir = args.lab_dir or "labelsTr"
    organs = AMOS_ORGANS
    files_probe = sorted(Path(args.root, img_dir).glob("*.nii.gz"))
    acc = []
    for f in files_probe:
        try:
            n = int(f.name.split("_")[1].split(".")[0])
        except Exception:
            continue
        if n < 500:
            acc.append(f)
    print(f"AMOS CT files {len(acc)}/{len(files_probe)}")
    if args.max_cases:
        acc = acc[: args.max_cases]
    _run_pairs("amos", args.root, acc, lab_dir, organs, args.out, lab_stem_strip=None)


def _run_pairs(name, root, img_files, lab_dir, organs, out_path, lab_stem_strip=None):
    acc = defaultdict(list)
    print(f"[{name}] cases={len(img_files)}", flush=True)
    for i, ip in enumerate(img_files):
        stem = ip.name
        if lab_stem_strip and stem.endswith(lab_stem_strip):
            stem = stem[: -len(lab_stem_strip)] + ".nii.gz"
        lp = Path(root, lab_dir) / stem
        if not lp.exists():
            continue
        img = read_nii(ip).astype(np.float32)
        lab = read_nii(lp).astype(np.int64)
        if img.shape != lab.shape:
            continue
        for c in organs:
            met = boundary_metrics(img, lab == c)
            if met:
                acc[c].append(met)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{name}] {i+1}/{len(img_files)}", flush=True)
    results = {}
    for c, oname in organs.items():
        results[oname] = {"label": c, **aggregate(acc[c])} if acc[c] else {"label": c, "n_cases": 0}
    payload = {
        "dataset": name, "root": str(root), "n_files": len(img_files),
        "organs": organs, "metrics": results,
        "definition": "inner1 vs outer r1..r3 shells; CNR; directional outer means",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{name}] WROTE {out_path}")
    hdr = f"{'organ':16s} {'in':>7s} {'out1':>7s} {'c1':>7s} {'cnr1':>6s} {'out3':>7s} {'c3':>7s} {'cnr3':>6s} {'|g|':>7s} {'dir_rng':>7s}"
    print(f"[{name}] {hdr}")
    for c, oname in organs.items():
        r = results[oname]
        if r.get("n_cases", 0) == 0:
            print(f"[{name}] {oname:16s} empty")
            continue

        def f(k, w=7, d=2):
            v = r.get(k)
            return f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else " " * w

        print(
            f"[{name}] {oname:16s} "
            f"{f('mu_inner')} {f('mu_outer_r1')} {f('contrast_r1')} {f('cnr_r1',6,2)} "
            f"{f('mu_outer_r3')} {f('contrast_r3')} {f('cnr_r3',6,2)} "
            f"{f('grad_inner_mean')} {f('dir_outer_range')}"
        )


def run_flare(args):
    root = Path(args.root)
    lab_dir = args.lab_dir or "labelsTr"
    imgs = []
    img_root = root / (args.img_dir or "imagesTr")
    for sub in sorted(img_root.iterdir()) if img_root.is_dir() else []:
        if sub.is_dir():
            imgs.extend(sorted(sub.glob("*.nii.gz")))
        elif sub.suffixes[-2:] == [".nii", ".gz"] or sub.name.endswith(".nii.gz"):
            imgs.append(sub)
    if not imgs:
        imgs = sorted(img_root.rglob("*_0000.nii.gz"))
    # match FLARE23_XXXX_0000.nii.gz -> FLARE23_XXXX.nii.gz
    _run_pairs("flare", str(root), imgs, lab_dir, FLARE_ORGANS, args.out, lab_stem_strip="_0000.nii.gz")


if __name__ == "__main__":
    main()
