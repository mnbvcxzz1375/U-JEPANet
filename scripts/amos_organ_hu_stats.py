#!/usr/bin/env python
"""AMOS 22 organ HU statistics: mean / boundary / interior."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# AMOS official labels (dataset.json)
ORGAN_NAMES = {
    1: "Spleen",
    2: "Kidney(R)",
    3: "Kidney(L)",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Postcava",
    10: "Pancreas",
    11: "Adrenal(R)",
    12: "Adrenal(L)",
    13: "Duodenum",
    14: "Bladder",
    15: "Prostate/Uterus",
}


def read_nifti(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def edge_interior(mask: np.ndarray, iterations: int = 1):
    m = mask.astype(bool)
    if not m.any():
        return m, m
    st = ndimage.generate_binary_structure(3, 1)
    er = ndimage.binary_erosion(m, structure=st, iterations=iterations)
    return m & ~er, er


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="extracted amos22 dir with imagesTr/labelsTr")
    p.add_argument("--out", required=True)
    p.add_argument("--ct-only", action="store_true", help="keep amos_id < 500 (CT)")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--erosion-iter", type=int, default=1)
    args = p.parse_args()

    root = Path(args.root)
    img_dir = root / "imagesTr"
    lab_dir = root / "labelsTr"
    files = sorted(img_dir.glob("amos_*.nii.gz"))
    if args.ct_only:
        def ct_id(p: Path):
            try:
                return int(p.name.split("_")[1].split(".")[0])
            except Exception:
                return 10**9
        files = [f for f in files if ct_id(f) < 500]
    if args.max_cases:
        files = files[: args.max_cases]

    acc = defaultdict(lambda: {
        "sum_all": 0.0, "n_all": 0,
        "sum_edge": 0.0, "n_edge": 0,
        "sum_int": 0.0, "n_int": 0,
        "cases_with_organ": 0,
        "case_means": [], "case_edge_means": [], "case_int_means": [],
    })

    print(f"cases={len(files)} root={root} ct_only={args.ct_only}", flush=True)
    n_ok = 0
    for i, ip in enumerate(files):
        lp = lab_dir / ip.name
        if not lp.exists():
            continue
        img = read_nifti(ip).astype(np.float32)
        lab = read_nifti(lp).astype(np.int64)
        if img.shape != lab.shape:
            print(f"skip {ip.name}: shape {img.shape} vs {lab.shape}", flush=True)
            continue
        n_ok += 1
        for c in range(1, 16):
            m = lab == c
            if not m.any():
                continue
            a = acc[c]
            hu = img[m]
            a["sum_all"] += float(hu.sum())
            a["n_all"] += int(m.sum())
            a["cases_with_organ"] += 1
            a["case_means"].append(float(hu.mean()))
            edge, interior = edge_interior(m, args.erosion_iter)
            if edge.any():
                ehu = img[edge]
                a["sum_edge"] += float(ehu.sum())
                a["n_edge"] += int(edge.sum())
                a["case_edge_means"].append(float(ehu.mean()))
            if interior.any():
                ihu = img[interior]
                a["sum_int"] += float(ihu.sum())
                a["n_int"] += int(interior.sum())
                a["case_int_means"].append(float(ihu.mean()))
        if (i + 1) % 20 == 0 or i == 0:
            print(f"processed {i+1}/{len(files)}", flush=True)

    results = {}
    for c in range(1, 16):
        a = acc[c]
        name = ORGAN_NAMES[c]
        if a["n_all"] == 0:
            results[name] = {"label": c, "n_voxels": 0}
            continue
        mean = a["sum_all"] / a["n_all"]
        em = a["sum_edge"] / a["n_edge"] if a["n_edge"] else None
        im = a["sum_int"] / a["n_int"] if a["n_int"] else None
        results[name] = {
            "label": c,
            "n_voxels": a["n_all"],
            "n_edge_voxels": a["n_edge"],
            "n_interior_voxels": a["n_int"],
            "cases_with_organ": a["cases_with_organ"],
            "voxel_mean_hu": mean,
            "edge_mean_hu": em,
            "interior_mean_hu": im,
            "case_mean_hu_avg": float(np.mean(a["case_means"])) if a["case_means"] else None,
            "case_edge_hu_avg": float(np.mean(a["case_edge_means"])) if a["case_edge_means"] else None,
            "case_interior_hu_avg": float(np.mean(a["case_int_means"])) if a["case_int_means"] else None,
            "edge_minus_interior": (em - im) if (em is not None and im is not None) else None,
        }

    out = {
        "dataset": "AMOS22",
        "root": str(root),
        "n_cases": n_ok,
        "ct_only": bool(args.ct_only),
        "erosion_iterations": args.erosion_iter,
        "organ_names": ORGAN_NAMES,
        "organs": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", args.out)
    print(f"{'organ':16s} {'mean_HU':>9s} {'edge_HU':>9s} {'int_HU':>9s} {'e-i':>8s} {'n_vox':>12s} {'n_case':>6s}")
    for c in range(1, 16):
        r = results[ORGAN_NAMES[c]]
        if r.get("n_voxels", 0) == 0:
            print(f"{ORGAN_NAMES[c]:16s} empty")
            continue
        em = r["edge_mean_hu"]
        im = r["interior_mean_hu"]
        ei = r["edge_minus_interior"]
        print(
            f"{ORGAN_NAMES[c]:16s} {r['voxel_mean_hu']:9.2f} "
            f"{(em if em is not None else float('nan')):9.2f} "
            f"{(im if im is not None else float('nan')):9.2f} "
            f"{(ei if ei is not None else float('nan')):8.2f} "
            f"{r['n_voxels']:12d} {r['cases_with_organ']:6d}"
        )


if __name__ == "__main__":
    main()
