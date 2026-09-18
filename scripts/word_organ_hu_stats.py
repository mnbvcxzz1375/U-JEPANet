#!/usr/bin/env python
"""WORD organ HU statistics: mean / boundary / interior (non-edge).

Uses imagesTr + labelsTr_All only. CPU. PLS4MIS label order.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

ORGAN_NAMES = {
    1: "Liver",
    2: "Spleen",
    3: "Kidney(L)",
    4: "Kidney(R)",
    5: "Stomach",
    6: "Gallbladder",
    7: "Esophagus",
    8: "Pancreas",
    9: "Duodenum",
    10: "Colon",
    11: "Intestine",
    12: "Adrenal",
    13: "Rectum",
    14: "Bladder",
    15: "Femur(L)",
    16: "Femur(R)",
}


def read_nifti(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def organ_boundary_mask(mask: np.ndarray, iterations: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (edge_mask, interior_mask) for a binary organ mask."""
    if mask.sum() == 0:
        return mask.astype(bool), mask.astype(bool)
    # 6-connectivity structure
    st = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=st, iterations=iterations)
    edge = mask.astype(bool) & ~eroded
    interior = eroded
    return edge, interior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--out", default="/data/hyc/U-JEPANet/runs/word_organ_hu_stats.json")
    p.add_argument("--max-cases", type=int, default=0, help="0 = all imagesTr")
    p.add_argument("--erosion-iter", type=int, default=1)
    args = p.parse_args()

    root = Path(args.word_root)
    img_dir = root / "imagesTr"
    lab_dir = root / "labelsTr_All"
    files = sorted(img_dir.glob("word_*.nii.gz"))
    if args.max_cases:
        files = files[: args.max_cases]

    # accumulators: organ -> list of (mean, edge_mean, interior_mean, n_vox, n_edge, n_int)
    acc = defaultdict(lambda: {
        "sum_all": 0.0, "n_all": 0,
        "sum_edge": 0.0, "n_edge": 0,
        "sum_int": 0.0, "n_int": 0,
        "case_means": [], "case_edge_means": [], "case_int_means": [],
        "cases_with_organ": 0,
    })

    print(f"cases={len(files)} root={root}", flush=True)
    for i, ip in enumerate(files):
        cid = ip.name.replace(".nii.gz", "")
        lp = lab_dir / ip.name
        if not lp.exists():
            print(f"skip {cid}: no label", flush=True)
            continue
        img = read_nifti(ip).astype(np.float32)
        lab = read_nifti(lp).astype(np.int64)
        if img.shape != lab.shape:
            print(f"skip {cid}: shape mismatch {img.shape} vs {lab.shape}", flush=True)
            continue
        for c in range(1, 17):
            m = lab == c
            n = int(m.sum())
            if n == 0:
                continue
            hu = img[m]
            edge, interior = organ_boundary_mask(m, args.erosion_iter)
            a = acc[c]
            a["sum_all"] += float(hu.sum())
            a["n_all"] += n
            a["case_means"].append(float(hu.mean()))
            a["cases_with_organ"] += 1
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
    for c in range(1, 17):
        a = acc[c]
        name = ORGAN_NAMES[c]
        if a["n_all"] == 0:
            results[name] = {"label": c, "n_voxels": 0}
            continue
        results[name] = {
            "label": c,
            "n_voxels": a["n_all"],
            "n_edge_voxels": a["n_edge"],
            "n_interior_voxels": a["n_int"],
            "cases_with_organ": a["cases_with_organ"],
            "voxel_mean_hu": a["sum_all"] / a["n_all"],
            "edge_mean_hu": (a["sum_edge"] / a["n_edge"]) if a["n_edge"] else None,
            "interior_mean_hu": (a["sum_int"] / a["n_int"]) if a["n_int"] else None,
            "case_mean_hu_avg": float(np.mean(a["case_means"])) if a["case_means"] else None,
            "case_edge_hu_avg": float(np.mean(a["case_edge_means"])) if a["case_edge_means"] else None,
            "case_interior_hu_avg": float(np.mean(a["case_int_means"])) if a["case_int_means"] else None,
            "edge_minus_interior": (
                (a["sum_edge"] / a["n_edge"]) - (a["sum_int"] / a["n_int"])
                if a["n_edge"] and a["n_int"] else None
            ),
        }

    out = {
        "word_root": str(root),
        "n_cases": len(files),
        "label_view": "labelsTr_All",
        "erosion_iterations": args.erosion_iter,
        "organ_names_pls4mis": ORGAN_NAMES,
        "organs": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", args.out)
    print(f"{'organ':14s} {'mean_HU':>9s} {'edge_HU':>9s} {'int_HU':>9s} {'e-i':>8s} {'n_vox':>12s}")
    for c in range(1, 17):
        r = results[ORGAN_NAMES[c]]
        if r.get("n_voxels", 0) == 0:
            print(f"{ORGAN_NAMES[c]:14s}  empty")
            continue
        em = r["edge_mean_hu"]
        im = r["interior_mean_hu"]
        ei = r["edge_minus_interior"]
        print(
            f"{ORGAN_NAMES[c]:14s} {r['voxel_mean_hu']:9.2f} "
            f"{(em if em is not None else float('nan')):9.2f} "
            f"{(im if im is not None else float('nan')):9.2f} "
            f"{(ei if ei is not None else float('nan')):8.2f} "
            f"{r['n_voxels']:12d}"
        )


if __name__ == "__main__":
    main()
