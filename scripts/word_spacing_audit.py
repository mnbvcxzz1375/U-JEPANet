#!/usr/bin/env python
"""WORD CT spacing / direction / FOV audit (train imagesTr)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/hyc/PLS4MIS/code/datasets/WORD")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "word_spacing_audit.json")
    rows = []
    for split in ("imagesTr", "imagesVal", "imagesTs"):
        d = root / split
        if not d.exists():
            continue
        for p in sorted(d.glob("word_*.nii.gz")):
            img = sitk.ReadImage(str(p))
            sp = img.GetSpacing()  # (x,y,z) mm
            origin = img.GetOrigin()
            direction = img.GetDirection()
            arr = sitk.GetArrayFromImage(img)  # z,y,x
            rows.append({
                "case": p.stem.replace(".nii", ""),
                "split": split,
                "shape_zyx": list(arr.shape),
                "spacing_xyz": list(sp),
                "origin": list(origin),
                "direction": list(direction),
                "fov_mm": [arr.shape[2] * sp[0], arr.shape[1] * sp[1], arr.shape[0] * sp[2]],
                "ident_direction": bool(direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)),
            })
    arr_sp = np.array([r["spacing_xyz"] for r in rows], dtype=np.float64)
    arr_fov = np.array([r["fov_mm"] for r in rows], dtype=np.float64)
    arr_sh = np.array([r["shape_zyx"] for r in rows], dtype=np.float64)
    summary = {
        "n": len(rows),
        "spacing_xyz_mean": arr_sp.mean(0).tolist(),
        "spacing_xyz_std": arr_sp.std(0).tolist(),
        "spacing_xyz_min": arr_sp.min(0).tolist(),
        "spacing_xyz_max": arr_sp.max(0).tolist(),
        "fov_mm_mean": arr_fov.mean(0).tolist(),
        "fov_mm_std": arr_fov.std(0).tolist(),
        "fov_mm_min": arr_fov.min(0).tolist(),
        "fov_mm_max": arr_fov.max(0).tolist(),
        "shape_mean": arr_sh.mean(0).tolist(),
        "shape_std": arr_sh.std(0).tolist(),
        "n_ident_direction": sum(1 for r in rows if r["ident_direction"]),
        "n_total": len(rows),
        "note": "If spacing/FOV vary a lot, 64^3 squeeze is voxel-normalized not physical-anatomy-normalized.",
    }
    out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
