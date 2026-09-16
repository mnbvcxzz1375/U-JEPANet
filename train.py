from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from ujepa.data import NiftiWordDataset, NpyVolumeDataset, collate_batch
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

    def __init__(
        self,
        n: int = 8,
        shape=(64, 64, 64),
        num_classes: int = 17,
        seed: int = 0,
        unlabeled_frac: float = 0.0,
    ):
        self.n = n
        self.shape = shape
        self.num_classes = num_classes
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 1, *shape, generator=g)
        labels = torch.zeros(n, *shape, dtype=torch.long)
        zz, yy, xx = torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing="ij",
        )
        for i in range(n):
            for c in range(1, min(6, num_classes)):
                cz = int(torch.randint(8, shape[0] - 8, (1,), generator=g).item())
                cy = int(torch.randint(8, shape[1] - 8, (1,), generator=g).item())
                cx = int(torch.randint(8, shape[2] - 8, (1,), generator=g).item())
                rz = int(torch.randint(3, 10, (1,), generator=g).item())
                mask = ((zz - cz) / rz) ** 2 + ((yy - cy) / rz) ** 2 + ((xx - cx) / rz) ** 2 <= 1.0
                labels[i][mask] = c
        self.labels = labels
        n_u = int(n * unlabeled_frac)
        self.is_labeled = torch.ones(n, dtype=torch.bool)
        if n_u > 0:
            self.is_labeled[-n_u:] = False

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "is_labeled": self.is_labeled[idx],
            "case_id": f"syn_{idx:03d}",
        }


def _seg_loss(logits, y, n_class: int) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        return deep_supervision_loss(logits, y, n_class)
    return dice_ce_loss(logits, y, n_class)


def train_loop(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    max_steps: int,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    lambda_j_max: float = 0.3,
    seg_only_steps: int = 3000,
    ramp_steps: int = 3000,
    grad_clip: float = 1.0,
    log_every: int = 10,
    out_dir: Optional[Path] = None,
    save_every: int = 0,
) -> Dict[str, Any]:
    model = model.to(device)
    is_jepa_trainer = isinstance(model, UJEPATrainer)
    if is_jepa_trainer:
        params = list(model.online.parameters())
        if model.predictor is not None:
            params = params + list(model.predictor.parameters())
        if model.target is not None:
            model.target.target = model.target.target.to(device)
    else:
        params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    history = []
    step = 0
    t0 = time.time()
    model.train()
    data_iter = iter(loader)
    n_class = getattr(model, "cfg", None)
    n_class = n_class.class_num if n_class is not None else 17

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        if isinstance(batch, (list, tuple)):
            # Fallback for raw (x, y) loaders in older tests.
            x, y = batch
            is_labeled = torch.ones(x.shape[0], dtype=torch.bool, device=device)
            batch = {"image": x, "label": y, "is_labeled": is_labeled}

        x = batch["image"].to(device)
        y = batch["label"].to(device)
        labeled = batch["is_labeled"].to(device).bool()
        if labeled.dim() == 0:
            labeled = labeled.unsqueeze(0)

        lj = lambda_j_schedule(step, seg_only_steps, ramp_steps, lambda_j_max)
        opt.zero_grad(set_to_none=True)

        # --- Segmentation: only labeled rows ---
        has_labeled = bool(labeled.any().item())
        l_seg_val = 0.0
        if has_labeled:
            x_l = x[labeled]
            y_l = y[labeled]
            if is_jepa_trainer:
                logits = model.forward_seg(x_l)
            else:
                logits = model(x_l)
            l_seg = _seg_loss(logits, y_l, n_class)
            l_seg.backward()
            l_seg_val = float(l_seg.detach().cpu())
        else:
            l_seg = torch.zeros((), device=device)

        # --- JEPA: all rows in this batch (image-only) ---
        l_j_val = 0.0
        if is_jepa_trainer and lj > 0:
            out = model.jepa_step(x)
            l_j = jepa_smooth_l1(out["pred_tokens"], out["tgt_tokens"], out["visible"])
            (lj * l_j).backward()
            l_j_val = float(l_j.detach().cpu())
        else:
            l_j = torch.zeros((), device=device)

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        if is_jepa_trainer:
            model.update_ema()

        step += 1
        rec = {
            "step": step,
            "l_seg": l_seg_val,
            "l_jepa": l_j_val,
            "lambda_j": float(lj),
            "n_labeled": int(labeled.sum().item()),
        }
        if step % log_every == 0 or step == 1:
            history.append(rec)
            print(
                f"step={step:05d} seg={rec['l_seg']:.4f} jepa={rec['l_jepa']:.4f} "
                f"lambda={lj:.3f} labeled={rec['n_labeled']}/{len(labeled)}"
            )

        if out_dir is not None and save_every > 0 and step % save_every == 0:
            _save_ckpt(out_dir / f"step_{step:06d}.pt", model, opt, step, rec)

    elapsed = time.time() - t0
    result = {"steps": step, "elapsed_sec": elapsed, "history": history}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_ckpt(out_dir / "last.pt", model, opt, step, history[-1] if history else {})
        with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def _save_ckpt(path: Path, model, opt, step: int, meta: dict) -> None:
    """Persist online+EMA+predictor via state_dict, plus optimizer/step."""
    payload = {
        "step": step,
        "meta": meta,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        payload["config"] = cfg.to_dict()
    torch.save(payload, path)


def build_loader(cfg: UJEPAConfig, raw: Dict[str, Any], seed: int, smoke: bool):
    data_cfg = raw.get("data", {})
    train_cfg = raw.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    if smoke or data_cfg.get("synthetic", False):
        ds = SyntheticWORDLike(
            n=8,
            shape=tuple(data_cfg.get("patch_size", [64, 64, 64])),
            num_classes=cfg.class_num,
            seed=seed,
            unlabeled_frac=float(data_cfg.get("unlabeled_frac", 0.0)),
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
        )

    kind = data_cfg.get("kind", "nifti_word")
    unlabeled_ids = data_cfg.get("unlabeled_ids") or []
    patch = tuple(data_cfg.get("patch_size", [128, 128, 96]))
    if kind == "nifti_word":
        root = data_cfg.get("word_root")
        if not root:
            raise SystemExit("data.word_root is required for non-synthetic training")
        ds = NiftiWordDataset(
            word_root=root,
            split=data_cfg.get("split", "imagesTr"),
            label_view=data_cfg.get("label_view", "labelsTr_All"),
            patch_size=patch,
            unlabeled_ids=unlabeled_ids,
            max_cases=data_cfg.get("max_cases"),
            seed=seed,
        )
    elif kind == "npy":
        root = data_cfg.get("root")
        if not root:
            raise SystemExit("data.root is required for npy datasets")
        ds = NpyVolumeDataset(
            root=root,
            split=data_cfg.get("split", "train"),
            patch_size=patch,
            unlabeled_ids=unlabeled_ids,
        )
    else:
        raise SystemExit(f"unknown data.kind={kind}")
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0
    )


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
    print(f"arm={cfg.arm} params={n_params:,} class_num={cfg.class_num} device={args.device}")

    loader = build_loader(cfg, raw, args.seed, args.smoke)
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
        save_every=int(train_cfg.get("save_every", 0)),
    )
    print(f"done. artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
