#!/usr/bin/env python
"""Per-organ whole-volume Dice for SLL20 A0-A3 best checkpoints.

Uses official 20 imagesVal + labelsVal only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.metrics import dice_per_class
from ujepa.model import UJEPAConfig, build_model, build_online_for_arm
from ujepa.whole_volume_eval import sliding_window_logits

# WORD 16 organ names (1-indexed labels)
ORGAN_NAMES = {
    1: "Liver",
    2: "Right Kidney",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "Inferior Vena Cava",
    7: "Right Adrenal Gland",
    8: "Left Adrenal Gland",
    9: "Gallbladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Duodenum",
    13: "Left Kidney",
    14: "Bladder",
    15: "Rectum",
    16: "Left Femoral Head",
}


def load_arm_model(arm: str, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config") or {}
    cfg = UJEPAConfig(**{k: v for k, v in cfg_dict.items() if k in UJEPAConfig.__dataclass_fields__})
    cfg.arm = arm.upper()
    model = build_online_for_arm(cfg)
    # strip trainer prefixes if present
    state = ckpt["model"]
    if any(k.startswith("online.") for k in state):
        state = {k[len("online.") :]: v for k, v in state.items() if k.startswith("online.")}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, cfg


@torch.no_grad()
def eval_arm(
    model,
    word_root: Path,
    val_ids: list[str],
    num_classes: int,
    patch,
    stride,
    device: torch.device,
    max_cases: int | None = None,
):
    acc = {c: [] for c in range(1, num_classes)}
    ids = val_ids[:max_cases] if max_cases else val_ids
    for cid in ids:
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / "imagesVal" / f"{cid}.nii.gz"))).astype(np.float32)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(word_root / "labelsVal" / f"{cid}.nii.gz"))).astype(np.int64)
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / max(hi - lo, 1e-6)
        vol = torch.from_numpy(img)[None, None]
        logits = sliding_window_logits(model, vol, patch, stride, device, num_classes)
        pred = torch.argmax(logits, dim=1)[0].cpu()
        pc = dice_per_class(pred, torch.from_numpy(lab), num_classes)
        for c, v in pc.items():
            if v == v:
                acc[c].append(v)
        print(f"  {cid} mean={np.nanmean([v for v in pc.values() if v==v]):.4f}", flush=True)
    per = {c: float(np.mean(vs)) if vs else float("nan") for c, vs in acc.items()}
    vals = [v for v in per.values() if v == v]
    per["ALL"] = float(np.mean(vals)) if vals else float("nan")
    return per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-root", default="/data/hyc/PLS4MIS/code/datasets/WORD")
    p.add_argument("--run-root", default="/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923")
    p.add_argument("--val-ids", default="/data/hyc/U-JEPANet/data/splits_sll20/val_20.txt")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-cases", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    word_root = Path(args.word_root)
    run_root = Path(args.run_root)
    val_ids = [ln.strip() for ln in Path(args.val_ids).read_text().splitlines() if ln.strip()]
    if args.max_cases:
        val_ids = val_ids[: args.max_cases]

    arms = ["a0", "a1", "a2", "a3"]
    results = {}
    for arm in arms:
        ckpt = run_root / arm / "best.pt"
        if not ckpt.exists():
            print(f"skip {arm}: no best.pt")
            continue
        print(f"=== {arm} {ckpt} ===", flush=True)
        model, cfg = load_arm_model(arm, ckpt, device)
        patch = (128, 128, 96)
        stride = (64, 64, 48)
        per = eval_arm(model, word_root, val_ids, cfg.class_num, patch, stride, device)
        results[arm] = per
        print(f"{arm} ALL={per['ALL']:.4f}", flush=True)

    # deltas vs A0
    if "a0" in results:
        for arm in arms:
            if arm == "a0" or arm not in results:
                continue
            results[arm]["delta_vs_a0"] = {
                c: results[arm][c] - results["a0"][c]
                for c in list(range(1, 17)) + ["ALL"]
                if c in results[arm] and c in results["a0"]
            }

    out = Path(args.out) if args.out else run_root / "per_organ_dice.json"
    payload = {
        "protocol": "SLL20-30k best.pt whole-volume imagesVal",
        "val_cases": val_ids,
        "organ_names": ORGAN_NAMES,
        "arms": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", out)

    # human table
    print("\norgan | " + " | ".join(arms) + " | A3-A0")
    for c in range(1, 17):
        row = [f"{results[a].get(c, float('nan')):.4f}" if a in results else "—" for a in arms]
        d = results.get("a3", {}).get("delta_vs_a0", {}).get(c, float("nan"))
        name = ORGAN_NAMES.get(c, str(c))
        print(f"{c:02d} {name:22s} | " + " | ".join(row) + f" | {d:+.4f}")
    print("ALL | " + " | ".join(f"{results[a]['ALL']:.4f}" if a in results else "—" for a in arms))


if __name__ == "__main__":
    main()
