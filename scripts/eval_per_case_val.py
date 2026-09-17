#!/usr/bin/env python
"""Per-case whole-volume eval on official imagesVal (20 cases) for locked best.pt arms."""
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
    1: "Liver", 2: "Spleen", 3: "Kidney(L)", 4: "Kidney(R)", 5: "Stomach",
    6: "Gallbladder", 7: "Esophagus", 8: "Pancreas", 9: "Duodenum", 10: "Colon",
    11: "Intestine", 12: "Adrenal", 13: "Rectum", 14: "Bladder", 15: "Femur(L)", 16: "Femur(R)",
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
    infer_arm = "A2" if arm.upper().startswith("A2") else arm.upper()
    cfg.arm = infer_arm
    model = build_online_for_arm(cfg)
    state = ckpt["model"]
    if any(k.startswith("online.") for k in state):
        state = {k[len("online."):]: v for k, v in state.items() if k.startswith("online.")}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), cfg


@torch.no_grad()
def eval_split(model, word_root: Path, ids, image_dir: str, label_dir: str,
               num_classes: int, patch, stride, device):
    acc = {c: [] for c in range(1, num_classes)}
    per_case = []
    for i, cid in enumerate(ids):
        t0 = time.time()
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / image_dir / f"{cid}.nii.gz"))).astype(np.float32)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / label_dir / f"{cid}.nii.gz"))).astype(np.int64)
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
        per_case.append({
            "case": cid,
            "mean_fg_dice": mean,
            "per_class": {str(k): (None if v != v else float(v)) for k, v in pc.items()},
        })
        print(f"  [{i+1}/{len(ids)}] {cid} mean={mean:.4f} t={time.time()-t0:.1f}s", flush=True)
    per = {c: float(np.mean(vs)) if vs else float("nan") for c, vs in acc.items()}
    vals = [v for v in per.values() if v == v]
    per["ALL"] = float(np.mean(vals)) if vals else float("nan")
    return per, per_case


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--ids", required=True)
    p.add_argument("--image-dir", default="imagesVal")
    p.add_argument("--label-dir", default="labelsVal")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="")
    p.add_argument("--arm-ckpt", default="")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    word_root = Path(args.word_root)
    ids = [ln.strip() for ln in Path(args.ids).read_text().splitlines() if ln.strip()]
    print(f"n={len(ids)} {args.image_dir} device={device}", flush=True)

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
        model, cfg = load_arm_model(arm, ckpt_path, device)
        per, per_case = eval_split(
            model, word_root, ids, args.image_dir, args.label_dir,
            cfg.class_num, patch, stride, device,
        )
        results[arm] = {
            "per_organ": per,
            "per_case": per_case,
            "n": len(ids),
            "ckpt": ckpt,
        }
        print(f"{arm} ALL={per['ALL']:.4f}", flush=True)
        payload = {
            "protocol": f"whole-volume {args.image_dir} 128x128x96 stride 64x64x48",
            "ids": ids,
            "organ_names": ORGAN_NAMES,
            "arms": {k: {"per_organ": v["per_organ"], "n": v["n"], "ckpt": v["ckpt"]} for k, v in results.items()},
            "per_case": {k: v["per_case"] for k, v in results.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
