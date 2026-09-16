from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from ujepa.losses import (
    deep_supervision_loss,
    dice_ce_loss,
    jepa_smooth_l1,
    lambda_j_schedule,
)
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model
from ujepa.unet3d import UNet3D


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_from_dict(d: Dict[str, Any]) -> UJEPAConfig:
    known = {k: v for k, v in d.items() if k in UJEPAConfig.__dataclass_fields__}
    return UJEPAConfig(**known)


class SyntheticWORDLike(torch.utils.data.Dataset):
    """CPU/GPU synthetic 3D volumes for smoke / interface tests only."""

    def __init__(self, n: int = 8, shape=(64, 64, 64), num_classes: int = 16, seed: int = 0):
        self.n = n
        self.shape = shape
        self.num_classes = num_classes
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 1, *shape, generator=g)
        # A few random ellipsoid-ish organ blobs.
        labels = torch.zeros(n, *shape, dtype=torch.long)
        for i in range(n):
            for c in range(1, min(6, num_classes)):
                cz = torch.randint(8, shape[0] - 8, (1,), generator=g).item()
                cy = torch.randint(8, shape[1] - 8, (1,), generator=g).item()
                cx = torch.randint(8, shape[2] - 8, (1,), generator=g).item()
                rz = torch.randint(3, 10, (1,), generator=g).item()
                zz, yy, xx = torch.meshgrid(
                    torch.arange(shape[0]),
                    torch.arange(shape[1]),
                    torch.arange(shape[2]),
                    indexing="ij",
                )
                mask = ((zz - cz) / rz) ** 2 + ((yy - cy) / rz) ** 2 + ((xx - cx) / rz) ** 2 <= 1.0
                labels[i][mask] = c
        self.labels = labels

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def train_loop(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    max_steps: int,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    lambda_j_max: float = 0.3,
    seg_only_steps: int = 50,
    ramp_steps: int = 50,
    grad_clip: float = 1.0,
    log_every: int = 10,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    model = model.to(device)
    is_jepa_trainer = isinstance(model, UJEPATrainer)
    if is_jepa_trainer:
        params = list(model.online.parameters())
        if model.predictor is not None:
            params = params + list(model.predictor.parameters())
        if model.target is not None:
            # Device-safe EMA target (online already moved).
            model.target.target = model.target.target.to(device)
    else:
        params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    history = []
    step = 0
    t0 = time.time()
    model.train()
    data_iter = iter(loader)

    while step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x = x.to(device)
        y = y.to(device)

        lj = lambda_j_schedule(step, seg_only_steps, ramp_steps, lambda_j_max)

        # Segmentation forward (full image; dual-path uses raw x).
        if is_jepa_trainer:
            logits = model.forward_seg(x)
        else:
            logits = model(x)

        if isinstance(logits, (list, tuple)):
            n_class = logits[0].shape[1]
            l_seg = deep_supervision_loss(logits, y, n_class)
        else:
            n_class = logits.shape[1]
            l_seg = dice_ce_loss(logits, y, n_class)

        l_j = torch.zeros((), device=device)
        if is_jepa_trainer and lj > 0:
            out = model.jepa_step(x)
            l_j = jepa_smooth_l1(out["pred_tokens"], out["tgt_tokens"], out["visible"])

        loss = l_seg + lj * l_j
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        if is_jepa_trainer:
            model.update_ema()

        step += 1
        rec = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "l_seg": float(l_seg.detach().cpu()),
            "l_jepa": float(l_j.detach().cpu()),
            "lambda_j": float(lj),
        }
        if step % log_every == 0 or step == 1:
            history.append(rec)
            print(
                f"step={step:05d} loss={rec['loss']:.4f} seg={rec['l_seg']:.4f} "
                f"jepa={rec['l_jepa']:.4f} lambda={lj:.3f}"
            )

    elapsed = time.time() - t0
    result = {"steps": step, "elapsed_sec": elapsed, "history": history}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model": model.state_dict(),
            "config": getattr(model, "cfg", None) and model.cfg.to_dict(),
            "history": history,
        }
        torch.save(ckpt, out_dir / "last.pt")
        with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description="U-JEPANet A0-A3 training")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--smoke", action="store_true", help="Synthetic data, few steps")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    raw = load_config(args.config)
    train_cfg = raw.get("train", {})
    model_cfg_raw = raw.get("model", raw)
    cfg = cfg_from_dict(model_cfg_raw)

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arm={cfg.arm} params={n_params:,} device={args.device}")

    if args.smoke or raw.get("data", {}).get("synthetic", False):
        ds = SyntheticWORDLike(
            n=8,
            shape=tuple(raw.get("data", {}).get("patch_size", [64, 64, 64])),
            num_classes=cfg.class_num,
            seed=args.seed,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    else:
        raise SystemExit(
            "Real WORD data path is not wired yet. Use --smoke or set data.synthetic=true. "
            "Wire dataset adapters in a follow-up task after remote layout is confirmed."
        )

    out_dir = Path(args.out or f"runs/{cfg.arm}_{int(time.time())}")
    max_steps = args.max_steps or int(train_cfg.get("max_steps", 50 if args.smoke else 30000))

    train_loop(
        model,
        loader,
        torch.device(args.device),
        max_steps=max_steps,
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        lambda_j_max=float(train_cfg.get("lambda_j_max", 0.3)),
        seg_only_steps=int(train_cfg.get("seg_only_steps", 50 if args.smoke else 3000)),
        ramp_steps=int(train_cfg.get("ramp_steps", 50 if args.smoke else 3000)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        log_every=int(train_cfg.get("log_every", 10)),
        out_dir=out_dir,
    )
    print(f"done. artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
