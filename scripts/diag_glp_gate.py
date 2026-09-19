#!/usr/bin/env python
"""P0 diagnostics on existing GLP G1/G0 checkpoints (no training).

S_pre  — relative Z_hat change under shuffled global (before alpha)
S_fuse — ||alpha * Delta|| / ||F2_R1||
S_out  — Dice normal - Dice shuffled (optional, needs val cases)
Also: per-step branch grad norms on a synthetic batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ujepa.global_local_predictive import GLPUNet


def load(arm: str, ckpt: Path):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = GLPUNet(
        arm=arm,
        feature_chns=(16, 32, 64, 128),
        class_num=17,
        embed_dim=int(ck.get("embed_dim", 256)),
        multiscale_pred=True,
    )
    m.load_state_dict(ck["model"])
    return m, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="G1", choices=["G0", "G1"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()
    device = torch.device(args.device)
    model, ck = load(args.arm, Path(args.ckpt))
    model = model.to(device)
    model.eval()

    B = args.batch
    torch.manual_seed(0)
    x = torch.randn(B, 1, 64, 64, 48, device=device)
    crop = torch.tensor([[0, 0, 0], [40, 50, 20], [80, 70, 40], [20, 90, 30]],
                        dtype=torch.long, device=device)[:B]
    shape = torch.tensor([[128, 128, 96]] * B, dtype=torch.long, device=device)
    th = torch.eye(3, 4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    # distinct global CTs
    glob = torch.stack([torch.rand(1, 64, 64, 64, device=device) + i * 0.3 for i in range(B)])
    kw = dict(crop_origin=crop, full_shape=shape, affine_theta=th, patch_size=(64, 64, 48))

    # --- S_pre: Z_hat sensitivity to shuffled global (alpha-independent) ---
    with torch.no_grad():
        # hook to capture last_zhat-like tensors via forward internals
        feats = model.backbone.encode(x) if False else None
        # manual path
        x0 = model.backbone.in_conv(x)
        x1 = model.backbone.down1(x0)
        f2 = model.backbone.down2(x1)
        z_l, grid, spatial = model._local_tokens(f2)
        cfeat = __import__("ujepa.ct_augment", fromlist=["token_coord_features_batched"]).token_coord_features_batched(
            grid, crop, shape, (64, 64, 48), th, device=device
        )
        q = model.coord_atlas(cfeat)
        zg_i = model.global_stem(glob)
        zhat_i = model.gl_pred(q, zg_i)
        zg_j = torch.roll(zg_i, shifts=1, dims=0)
        zhat_j = model.gl_pred(q, zg_j)
        num = (zhat_i - zhat_j).norm(dim=-1).mean()
        den = zhat_i.norm(dim=-1).mean() + 1e-6
        s_pre = float(num / den)
        # fusion ratio at current alpha
        alpha = model.alpha_g_value
        zhat_v = zhat_i.transpose(1, 2).reshape(B, model.embed_dim, *grid)
        r_v = (z_l - zhat_i).transpose(1, 2).reshape(B, model.embed_dim, *grid)
        d, h, w = spatial
        zhat_v = F.interpolate(zhat_v, size=(d, h, w), mode="trilinear", align_corners=False)
        r_v = F.interpolate(r_v, size=(d, h, w), mode="trilinear", align_corners=False)
        delta = model.gl_merge(torch.cat([zhat_v, r_v], dim=1))
        f2_r1, _ = model.local_bn.full_context_forward(f2)
        s_fuse = float((abs(alpha) * delta.norm()) / (f2_r1.norm() + 1e-6))

    # --- branch grads on synthetic seg loss ---
    model.train()
    for p_ in model.parameters():
        p_.grad = None
    y = model(x, global_image=glob, **kw)
    if isinstance(y, (list, tuple)):
        y = y[0]
    loss = y.sum()
    loss.backward()
    def gsum(mod):
        if mod is None:
            return None
        tot = 0.0
        n = 0
        for p_ in mod.parameters():
            if p_.grad is not None:
                tot += float(p_.grad.abs().sum())
                n += 1
        return {"sum": tot, "n_params_with_grad": n}

    report = {
        "arm": args.arm,
        "ckpt": str(args.ckpt),
        "best_val": ck.get("best_val"),
        "alpha_g": alpha,
        "S_pre": s_pre,
        "S_fuse_at_current_alpha": s_fuse,
        "grad_global_stem": gsum(model.global_stem),
        "grad_coord_atlas": gsum(model.coord_atlas),
        "grad_gl_pred": gsum(model.gl_pred),
        "grad_alpha_g": None if model.alpha_g.grad is None else float(model.alpha_g.grad),
        "interpretation": {
            "S_pre~0": "predictor ignores global content",
            "S_pre>0 & S_fuse~0": "global content differs but gate kills fusion",
            "S_pre>0 & S_fuse>0": "content reaches F2*; test harm may be real",
        },
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("WROTE", args.out)


if __name__ == "__main__":
    main()
