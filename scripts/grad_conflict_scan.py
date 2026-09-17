#!/usr/bin/env python
"""Run gradient-conflict diagnostic on A2/A3 checkpoints (or fresh)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import build_sll_dual_loaders
from ujepa.grad_conflict import run_grad_conflict_scan
from ujepa.model import UJEPAConfig, build_model


def load_model(arm: str, ckpt: Path | None, device):
    if ckpt and ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg_d = payload.get("config") or {}
        cfg = UJEPAConfig(**{k: v for k, v in cfg_d.items() if k in UJEPAConfig.__dataclass_fields__})
        cfg.arm = arm.upper()
        model = build_model(cfg)
        model.load_state_dict(payload["model"], strict=True)
    else:
        cfg = UJEPAConfig(
            arm=arm.upper(),
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
        model = build_model(cfg)
    return model.to(device), cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="a2", choices=["a2", "a3"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--cache", default="/data/hyc/U-JEPANet/data/word_sll20_cache")
    p.add_argument("--split-dir", default="/data/hyc/U-JEPANet/data/splits_sll20")
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = Path(args.ckpt) if args.ckpt else None
    model, cfg = load_model(args.arm, ckpt, device)
    raw = {
        "data": {
            "kind": "npy_sll",
            "root": args.cache,
            "val_ids_file": str(Path(args.split_dir) / "val_20.txt"),
            "patch_size": [128, 128, 96],
        },
        "train": {"batch_size": 1},
    }
    seg_loader, jepa_loader, _ = build_sll_dual_loaders(cfg, raw, seed=42)
    stats = run_grad_conflict_scan(
        model, seg_loader, jepa_loader, device, cfg.class_num, n_batches=args.batches
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "rows"}, indent=2))
    out = Path(args.out) if args.out else Path(f"/data/hyc/U-JEPANet/runs/grad_conflict_{args.arm}.json")
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("WROTE", out)


if __name__ == "__main__":
    main()
