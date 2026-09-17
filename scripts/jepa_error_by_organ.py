#!/usr/bin/env python
"""Per-organ JEPA prediction error on labeled val crops (image-level, train-only cache).

For each labeled train patch with GT, compute token-level JEPA error on masked
positions overlapping each organ. Tests whether high-unpredictability organs
(e.g. Rectum) have higher L_J than stable ones (Femur L).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.model import UJEPAConfig, UJEPATrainer, build_model
from ujepa.masking import sample_token_mask, token_mask_to_volume_mask

ORGAN_NAMES = {
    1: "Liver", 2: "Spleen", 3: "Kidney(L)", 4: "Kidney(R)", 5: "Stomach",
    6: "Gallbladder", 7: "Esophagus", 8: "Pancreas", 9: "Duodenum", 10: "Colon",
    11: "Intestine", 12: "Adrenal", 13: "Rectum", 14: "Bladder", 15: "Femur(L)", 16: "Femur(R)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/a2/best.pt")
    p.add_argument("--cache", default="/data/hyc/U-JEPANet/data/word_sll20_cache")
    p.add_argument("--n-patches", type=int, default=80)
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/data/hyc/U-JEPANet/runs/jepa_error_by_organ.json")
    args = p.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_d = payload.get("config") or {}
    cfg = UJEPAConfig(**{k: v for k, v in cfg_d.items() if k in UJEPAConfig.__dataclass_fields__})
    cfg.arm = "A2"
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    cache = Path(args.cache)
    labeled_ids = [ln.strip() for ln in (cache / "splits" / "train_labeled.txt").read_text().splitlines() if ln.strip()]
    labeled_ids = labeled_ids[: args.n_patches]

    org_err = defaultdict(list)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for cid in labeled_ids:
            img = np.load(cache / "images" / f"{cid}.npy").astype("float32")
            lab = np.load(cache / "labels" / f"{cid}.npy").astype(np.int64)
            x = torch.from_numpy(img)[None, None].to(device)
            grid = model.online.jepa_token_grid(tuple(x.shape[-3:]))
            tmask = sample_token_mask(1, grid, mask_ratio=args.mask_ratio, generator=g, device=device)
            out = model.jepa_step(x, mask=tmask)
            # token-level error
            pred, tgt, vis = out["pred_tokens"], out["tgt_tokens"], out["visible"]
            tgt_n = F.layer_norm(tgt, (tgt.shape[-1],))
            err = (pred - tgt_n).pow(2).mean(-1)[0]  # (N,)
            masked = (~vis[0]).float()  # (N,)
            # organ occupancy on token grid
            lab_t = torch.from_numpy(lab)[None, None].float().to(device)
            lab_ds = F.interpolate(lab_t, size=grid, mode="nearest")[0, 0].long()  # (D,H,W)
            lab_flat = lab_ds.reshape(-1)
            for c in range(1, 17):
                m_org = (lab_flat == c) & (masked > 0.5)
                if m_org.any():
                    org_err[c].append(float(err[m_org].mean().item()))

    summary = {}
    for c in range(1, 17):
        vals = org_err.get(c, [])
        summary[ORGAN_NAMES[c]] = {
            "n_patches": len(vals),
            "mean_jepa_mse": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
        }
    # rank by error
    ranked = sorted(
        [(k, v["mean_jepa_mse"]) for k, v in summary.items() if v["mean_jepa_mse"] is not None],
        key=lambda kv: -kv[1],
    )
    out = {"ckpt": args.ckpt, "n_patches": len(labeled_ids), "by_organ": summary, "ranked_high_error": ranked}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"ranked_high_error": ranked}, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
