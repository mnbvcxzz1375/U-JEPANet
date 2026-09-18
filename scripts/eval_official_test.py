#!/usr/bin/env python
"""Official WORD imagesTs/labelsTs whole-volume evaluation for SLL20 arms.

User-authorized 2026-09-17. Screening arms only; no re-selection on test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.metrics import dice_per_class
from ujepa.model import UJEPAConfig, build_online_for_arm
from ujepa.whole_volume_eval import sliding_window_logits

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

DEFAULT_ARMS = [
    ("A0", "/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/a0/best.pt"),
    ("A1", "/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/a1/best.pt"),
    ("A2", "/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/a2/best.pt"),
    ("A3", "/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/a3/best.pt"),
    ("A2-L", "/data/hyc/U-JEPANet/runs/a2l_lu/L/best.pt"),
    ("A2-LU", "/data/hyc/U-JEPANet/runs/a2l_lu/LU/best.pt"),
]


def parse_arm_overrides(spec: str) -> dict[str, str]:
    """Parse 'A1=/path.pt,A3=/path.pt' into a mapping."""
    out: dict[str, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, path = part.split("=", 1)
        out[name.strip().upper()] = path.strip()
    return out


def load_arm_model(arm: str, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config") or {}
    cfg = UJEPAConfig(**{k: v for k, v in cfg_dict.items() if k in UJEPAConfig.__dataclass_fields__})
    # Inference arm for A2-L / A2-LU is A2 (JEPA head, no dualpath residual in online seg path)
    infer_arm = "A2" if arm.startswith("A2") else arm.upper()
    cfg.arm = infer_arm
    model = build_online_for_arm(cfg)
    state = ckpt["model"]
    if any(k.startswith("online.") for k in state):
        state = {k[len("online."):]: v for k, v in state.items() if k.startswith("online.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Filter expected non-online keys already stripped; report real issues
    missing = [k for k in missing if not k.startswith(("predictor.", "target."))]
    unexpected = [k for k in unexpected if not k.startswith(("predictor.", "target."))]
    if missing or unexpected:
        print(f"  load warn missing={missing[:5]} unexpected={unexpected[:5]}", flush=True)
    model = model.to(device).eval()
    return model, cfg


@torch.no_grad()
def eval_arm_test(
    model,
    word_root: Path,
    test_ids: list[str],
    num_classes: int,
    patch,
    stride,
    device: torch.device,
):
    acc = {c: [] for c in range(1, num_classes)}
    per_case = []
    for i, cid in enumerate(test_ids):
        t0 = time.time()
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / "imagesTs" / f"{cid}.nii.gz"))).astype(np.float32)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / "labelsTs" / f"{cid}.nii.gz"))).astype(np.int64)
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / max(hi - lo, 1e-6)
        vol = torch.from_numpy(img)[None, None]
        logits = sliding_window_logits(model, vol, patch, stride, device, num_classes)
        pred = torch.argmax(logits, dim=1)[0].cpu()
        pc = dice_per_class(pred, torch.from_numpy(lab), num_classes)
        finite = {c: v for c, v in pc.items() if v == v}
        mean = float(np.mean(list(finite.values()))) if finite else float("nan")
        for c, v in finite.items():
            acc[c].append(v)
        per_case.append({"case": cid, "mean_fg_dice": mean, "per_class": {str(k): (None if v != v else float(v)) for k, v in pc.items()}})
        print(f"  [{i+1}/{len(test_ids)}] {cid} mean={mean:.4f} t={time.time()-t0:.1f}s", flush=True)
    per = {c: float(np.mean(vs)) if vs else float("nan") for c, vs in acc.items()}
    vals = [v for v in per.values() if v == v]
    per["ALL"] = float(np.mean(vals)) if vals else float("nan")
    return per, per_case


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--test-ids", default="/data/hyc/U-JEPANet/data/splits_sll20/test_30.txt")
    p.add_argument("--out", default="/data/hyc/U-JEPANet/runs/official_test_20260917/test_per_organ.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--arms", default="", help="comma list; empty = all")
    p.add_argument("--arm-ckpt", default="", help="override e.g. A1=/path/best.pt,A3=/path/best.pt")
    p.add_argument("--intensity-mode", default="minmax", choices=["minmax", "window"],
                   help="minmax=legacy PL-Seg style; window=canonical L40/W400 to match dynamic-crop training")
    args = p.parse_args()

    device = torch.device(args.device)
    word_root = Path(args.word_root)
    test_ids = [ln.strip() for ln in Path(args.test_ids).read_text().splitlines() if ln.strip()]
    print(f"n_test={len(test_ids)} device={device} intensity={args.intensity_mode}", flush=True)

    overrides = parse_arm_overrides(args.arm_ckpt)
    arms = [(n, overrides.get(n.upper(), pth)) for n, pth in DEFAULT_ARMS]
    selected = arms
    if args.arms:
        want = {x.strip().upper() for x in args.arms.split(",") if x.strip()}
        selected = [(n, pth) for n, pth in arms if n.upper() in want]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    patch, stride = (128, 128, 96), (64, 64, 48)

    for arm, ckpt in selected:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"SKIP {arm}: missing {ckpt}", flush=True)
            continue
        print(f"=== {arm} {ckpt} ===", flush=True)
        t0 = time.time()
        model, cfg = load_arm_model(arm, ckpt_path, device)
        # wrap eval to pass intensity_mode
        from ujepa.whole_volume_eval import sliding_window_logits
        import numpy as np
        import SimpleITK as sitk

        if args.intensity_mode == "window":
            from ujepa.ct_augment import canonical_window
        acc = {c: [] for c in range(1, cfg.class_num)}
        per_case = []
        for i, cid in enumerate(test_ids):
            tc0 = time.time()
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / "imagesTs" / f"{cid}.nii.gz"))).astype(np.float32)
            lab = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / "labelsTs" / f"{cid}.nii.gz"))).astype(np.int64)
            if args.intensity_mode == "window":
                vol = canonical_window(torch.from_numpy(img)[None, None])
            else:
                lo, hi = float(img.min()), float(img.max())
                vol = torch.from_numpy(img)[None, None]
                vol = (vol - lo) / max(hi - lo, 1e-6)
            logits = sliding_window_logits(model, vol, patch, stride, device, cfg.class_num)
            pred = torch.argmax(logits, dim=1)[0].cpu()
            pc = dice_per_class(pred, torch.from_numpy(lab), cfg.class_num)
            finite = {c: v for c, v in pc.items() if v == v}
            mean = float(np.mean(list(finite.values()))) if finite else float("nan")
            for c, v in finite.items():
                acc[c].append(v)
            per_case.append({"case": cid, "mean_fg_dice": mean,
                             "per_class": {str(k): (None if v != v else float(v)) for k, v in pc.items()}})
            print(f"  [{i+1}/{len(test_ids)}] {cid} mean={mean:.4f} t={time.time()-tc0:.1f}s", flush=True)
        per = {c: float(np.mean(vs)) if vs else float("nan") for c, vs in acc.items()}
        vals = [v for v in per.values() if v == v]
        per["ALL"] = float(np.mean(vals)) if vals else float("nan")
        results[arm] = {
            "per_organ": per,
            "n_test": len(test_ids),
            "elapsed_sec": time.time() - t0,
            "ckpt": ckpt,
            "per_case": per_case,
            "intensity_mode": args.intensity_mode,
        }
        print(f"{arm} ALL={per['ALL']:.4f} elapsed={time.time()-t0:.1f}s", flush=True)
        payload = {
            "protocol": f"WORD official imagesTs whole-volume 128x128x96 stride 64x64x48 intensity={args.intensity_mode}",
            "intensity_mode": args.intensity_mode,
            "authorized": "user 2026-09-17",
            "test_ids": test_ids,
            "organ_names": ORGAN_NAMES,
            "arms": {k: {"per_organ": v["per_organ"], "n_test": v["n_test"], "ckpt": v["ckpt"],
                         "elapsed_sec": v["elapsed_sec"], "intensity_mode": v.get("intensity_mode")} for k, v in results.items()},
            "per_case": {k: v["per_case"] for k, v in results.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    names = list(results.keys())
    print("\norgan | " + " | ".join(names))
    for c in range(1, 17):
        row = [f"{results[a]['per_organ'].get(c, float('nan')):.4f}" for a in names]
        print(f"{c:02d} {ORGAN_NAMES[c]:14s} | " + " | ".join(row))
    row = [f"{results[a]['per_organ']['ALL']:.4f}" for a in names]
    print("ALL              | " + " | ".join(row))
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
