from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from ujepa.data import NiftiWordDataset, NpyVolumeDataset, collate_batch
from ujepa.losses import (
    deep_supervision_loss,
    dice_ce_loss,
    jepa_smooth_l1,
    lambda_j_schedule,
)
from ujepa.metrics import run_validation
from ujepa.model import UJEPAConfig, UJEPATrainer, build_model
from ujepa.unet3d import UNet3D


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_from_dict(d: Dict[str, Any]) -> UJEPAConfig:
    known = {k: v for k, v in d.items() if k in UJEPAConfig.__dataclass_fields__}
    return UJEPAConfig(**known)


def git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class SyntheticWORDLike(torch.utils.data.Dataset):
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    step: int,
    meta: dict,
    config: Optional[dict] = None,
) -> None:
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
    if config is not None:
        payload["config"] = config
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> int:
    """Load model/EMA/optimizer/step/RNG. Returns restored step."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if restore_rng and "rng" in ckpt:
        try:
            torch.set_rng_state(ckpt["rng"]["torch"].cpu())
            if ckpt["rng"].get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [s.cpu() for s in ckpt["rng"]["cuda"]]
                )
        except Exception as e:
            print(f"[resume] RNG restore skipped: {e}")
    # Re-assert EMA eval invariant after load.
    tgt = getattr(model, "target", None)
    if tgt is not None and hasattr(tgt, "target"):
        tgt.target.eval()
        tgt.train(False)
    return int(ckpt.get("step", 0))


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
    val_loader=None,
    val_every: int = 0,
    resume_from: Optional[Path] = None,
    seed: int = 42,
    jepa_loader=None,
    val_mode: str = "crop",
    word_root: Optional[str] = None,
    val_case_ids: Optional[list] = None,
    val_patch: tuple = (128, 128, 96),
    val_stride: tuple = (64, 64, 48),
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

    step = 0
    if resume_from is not None and Path(resume_from).exists():
        step = load_checkpoint(Path(resume_from), model, opt, map_location=device)
        print(f"[resume] loaded {resume_from} at step={step}")

    model.train()
    if is_jepa_trainer and model.target is not None:
        # Critical invariant: EMA teacher stays eval.
        model.target.train(False)
        model.target.target.eval()

    history = []
    val_history = []
    best_dice = float("-inf")
    best_path = None
    t0 = time.time()
    data_iter = iter(loader)
    jepa_iter = iter(jepa_loader) if jepa_loader is not None else None
    n_class = getattr(model, "cfg", None)
    n_class = n_class.class_num if n_class is not None else 17
    start_step = step
    commit = git_commit_sha()

    def _next(it, dl):
        try:
            return next(it), it
        except StopIteration:
            it2 = iter(dl)
            return next(it2), it2

    while step < max_steps:
        # --- Segmentation batch: always from labeled loader when dual-mode ---
        batch, data_iter = _next(data_iter, loader)
        if isinstance(batch, (list, tuple)):
            x, y = batch
            is_labeled = torch.ones(x.shape[0], dtype=torch.bool, device=device)
            batch = {"image": x, "label": y, "is_labeled": is_labeled}

        x = batch["image"].to(device)
        y = batch["label"].to(device)
        labeled = batch["is_labeled"].to(device).bool()
        if labeled.dim() == 0:
            labeled = labeled.unsqueeze(0)

        # Dual-mode: optional JEPA batch from labeled∪unlabeled train pool
        x_j = None
        if jepa_loader is not None:
            jbatch, jepa_iter = _next(jepa_iter, jepa_loader)
            if isinstance(jbatch, (list, tuple)):
                x_j = jbatch[0].to(device)
            else:
                x_j = jbatch["image"].to(device)

        lj = lambda_j_schedule(step, seg_only_steps, ramp_steps, lambda_j_max)
        opt.zero_grad(set_to_none=True)

        has_labeled = bool(labeled.any().item())
        l_seg_val = 0.0
        if has_labeled:
            x_l = x[labeled]
            y_l = y[labeled]
            logits = model.forward_seg(x_l) if is_jepa_trainer else model(x_l)
            l_seg = _seg_loss(logits, y_l, n_class)
            l_seg.backward()
            l_seg_val = float(l_seg.detach().cpu())

        l_j_val = 0.0
        if is_jepa_trainer and lj > 0:
            jepa_x = x_j if x_j is not None else x
            out = model.jepa_step(jepa_x)
            l_j = jepa_smooth_l1(out["pred_tokens"], out["tgt_tokens"], out["visible"])
            (lj * l_j).backward()
            l_j_val = float(l_j.detach().cpu())

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        if is_jepa_trainer:
            model.update_ema()
            if model.target is not None:
                model.target.train(False)
                model.target.target.eval()

        step += 1
        rec = {
            "step": step,
            "l_seg": l_seg_val,
            "l_jepa": l_j_val,
            "lambda_j": float(lj),
            "n_labeled": int(labeled.sum().item()),
            "commit": commit,
        }
        if step % log_every == 0 or step == 1 or step == max_steps:
            history.append(rec)
            alpha = None
            if is_jepa_trainer and getattr(model.online, "dualpath", None) is not None:
                alpha = float(model.online.dualpath.alpha.detach().cpu())
            extra = f" alpha={alpha:.4f}" if alpha is not None else ""
            print(
                f"step={step:05d} seg={rec['l_seg']:.4f} jepa={rec['l_jepa']:.4f} "
                f"lambda={lj:.3f} labeled={rec['n_labeled']}/{len(labeled)}{extra}"
            )

        if out_dir is not None and save_every > 0 and step % save_every == 0:
            save_checkpoint(
                out_dir / f"step_{step:06d}.pt",
                model,
                opt,
                step,
                rec,
                config=getattr(model, "cfg", None) and model.cfg.to_dict(),
            )

        if (
            (val_loader is not None or (val_mode == "whole" and word_root and val_case_ids))
            and val_every > 0
            and (step % val_every == 0 or step == max_steps)
        ):
            if val_mode == "whole" and word_root and val_case_ids:
                from ujepa.whole_volume_eval import evaluate_word_whole_volume

                vm = evaluate_word_whole_volume(
                    model,
                    word_root,
                    val_case_ids,
                    n_class,
                    val_patch,
                    val_stride,
                    device,
                )
            else:
                vm = run_validation(model, val_loader, device, n_class)
            vm["step"] = step
            vm["commit"] = commit
            vm["val_mode"] = val_mode
            val_history.append(vm)
            md = vm.get("mean_fg_dice", float("nan"))
            print(
                f"[val@{step:05d}][{val_mode}] mean_fg_dice={md:.4f} "
                f"n_val={vm.get('n_val_cases')}"
            )
            if out_dir is not None and md == md and md > best_dice:
                best_dice = md
                best_path = out_dir / "best.pt"
                save_checkpoint(
                    best_path,
                    model,
                    opt,
                    step,
                    {**rec, "val_mean_fg_dice": md},
                    config=getattr(model, "cfg", None) and model.cfg.to_dict(),
                )
            model.train()
            if is_jepa_trainer and model.target is not None:
                model.target.train(False)
                model.target.target.eval()

    elapsed = time.time() - t0
    result = {
        "steps": step,
        "start_step": start_step,
        "elapsed_sec": elapsed,
        "history": history,
        "val_history": val_history,
        "best_mean_fg_dice": best_dice if best_dice > float("-inf") else None,
        "commit": commit,
        "device": str(device),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out_dir / "last.pt",
            model,
            opt,
            step,
            history[-1] if history else {},
            config=getattr(model, "cfg", None) and model.cfg.to_dict(),
        )
        with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def build_loader(cfg: UJEPAConfig, raw: Dict[str, Any], seed: int, smoke: bool, split_role: str = "train"):
    data_cfg = raw.get("data", {})
    train_cfg = raw.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    if smoke or data_cfg.get("synthetic", False):
        n = 8 if split_role == "train" else 4
        ds = SyntheticWORDLike(
            n=n,
            shape=tuple(data_cfg.get("patch_size", [64, 64, 64])),
            num_classes=cfg.class_num,
            seed=seed if split_role == "train" else seed + 1,
            unlabeled_frac=float(data_cfg.get("unlabeled_frac", 0.0)) if split_role == "train" else 0.0,
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(split_role == "train"), collate_fn=collate_batch
        )

    kind = data_cfg.get("kind", "nifti_word")
    unlabeled_ids = data_cfg.get("unlabeled_ids") or []
    patch = tuple(data_cfg.get("patch_size", [128, 128, 96]))
    if kind == "nifti_word":
        root = data_cfg.get("word_root")
        if not root:
            raise SystemExit("data.word_root is required for non-synthetic training")
        if split_role == "val":
            ds = NiftiWordDataset(
                word_root=root,
                split=data_cfg.get("val_split", "imagesVal"),
                label_view=data_cfg.get("val_label_view", "labelsVal"),
                patch_size=patch,
                unlabeled_ids=[],
                max_cases=data_cfg.get("val_max_cases"),
                seed=seed + 1,
            )
        else:
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
        split = data_cfg.get("val_split", "val") if split_role == "val" else data_cfg.get("split", "train")
        ds = NpyVolumeDataset(
            root=root,
            split=split,
            patch_size=patch,
            unlabeled_ids=[] if split_role == "val" else unlabeled_ids,
        )
    else:
        raise SystemExit(f"unknown data.kind={kind}")
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split_role == "train"),
        collate_fn=collate_batch,
        num_workers=0,
    )


def _read_id_list(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_sll_dual_loaders(cfg: UJEPAConfig, raw: Dict[str, Any], seed: int):
    """Seg loader from 20 labeled only; JEPA loader from 100 train (L+U).

    Uses npy patch caches produced by cache_sll_patches.py, or synthetic fallback.
    """
    data_cfg = raw.get("data", {})
    train_cfg = raw.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    kind = data_cfg.get("kind", "npy_sll")
    patch = tuple(data_cfg.get("patch_size", [128, 128, 96]))

    if data_cfg.get("synthetic", False):
        ds_l = SyntheticWORDLike(n=8, shape=patch, num_classes=cfg.class_num, seed=seed, unlabeled_frac=0.0)
        ds_j = SyntheticWORDLike(n=16, shape=patch, num_classes=cfg.class_num, seed=seed + 1, unlabeled_frac=0.5)
        seg_loader = torch.utils.data.DataLoader(ds_l, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        jepa_loader = torch.utils.data.DataLoader(ds_j, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        return seg_loader, jepa_loader, []

    if kind == "npy_sll":
        root = Path(data_cfg["root"])
        labeled_ids = _read_id_list(root / "splits" / "train_labeled.txt")
        jepa_ids = _read_id_list(root / "splits" / "train_all.txt")
        val_ids = []
        val_list = data_cfg.get("val_ids_file")
        if val_list:
            val_ids = _read_id_list(val_list)

        class SplitNpy(torch.utils.data.Dataset):
            def __init__(self, ids, require_label: bool):
                self.ids = ids
                self.require_label = require_label

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, idx):
                cid = self.ids[idx]
                img = np.load(root / "images" / f"{cid}.npy")
                image = torch.from_numpy(img.astype("float32"))
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                lab_path = root / "labels" / f"{cid}.npy"
                if lab_path.exists() and self.require_label:
                    label = torch.from_numpy(np.load(lab_path).astype("int64"))
                    is_l = True
                else:
                    label = torch.zeros(image.shape[-3:], dtype=torch.long)
                    is_l = not self.require_label or lab_path.exists()
                return {
                    "image": image,
                    "label": label,
                    "is_labeled": torch.tensor(bool(is_l and self.require_label), dtype=torch.bool),
                    "case_id": cid,
                }

        ds_l = SplitNpy(labeled_ids, require_label=True)
        ds_j = SplitNpy(jepa_ids, require_label=False)
        seg_loader = torch.utils.data.DataLoader(ds_l, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        jepa_loader = torch.utils.data.DataLoader(ds_j, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        return seg_loader, jepa_loader, val_ids

    raise SystemExit(f"unsupported data.kind for dual loaders: {kind}")


def main():
    p = argparse.ArgumentParser(description="U-JEPANet A0-A3 training")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--smoke", action="store_true", help="Synthetic data, few steps")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    raw = load_config(args.config)
    train_cfg = raw.get("train", {})
    model_cfg_raw = raw.get("model", raw)
    cfg = cfg_from_dict(model_cfg_raw)

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"arm={cfg.arm} params={n_params:,} class_num={cfg.class_num} "
        f"device={args.device} commit={git_commit_sha()}"
    )

    loader = build_loader(cfg, raw, args.seed, args.smoke, split_role="train")
    val_loader = None
    val_every = int(train_cfg.get("val_every", 0))
    jepa_loader = None
    val_mode = train_cfg.get("val_mode", "crop")
    word_root = raw.get("data", {}).get("word_root")
    val_case_ids = []
    val_patch = tuple(raw.get("data", {}).get("patch_size", [128, 128, 96]))
    val_stride = tuple(train_cfg.get("val_stride", [64, 64, 48]))

    if train_cfg.get("dual_loader", False):
        seg_loader, jepa_loader, val_case_ids = build_sll_dual_loaders(cfg, raw, args.seed)
        loader = seg_loader
        print(f"[protocol] dual_loader labeled={len(seg_loader.dataset)} jepa={len(jepa_loader.dataset)}")
        val_mode = train_cfg.get("val_mode", "whole")
    elif val_every > 0 or train_cfg.get("do_val", False):
        val_loader = build_loader(cfg, raw, args.seed, args.smoke, split_role="val")
        if val_every <= 0:
            val_every = max(1, int(train_cfg.get("max_steps", 100) // 4))

    if val_mode == "whole":
        if not val_case_ids:
            vf = raw.get("data", {}).get("val_ids_file")
            if vf:
                val_case_ids = _read_id_list(vf)
        if not word_root or not val_case_ids:
            raise SystemExit("whole val requires data.word_root and val_case_ids / val_ids_file")
        print(f"[protocol] whole-volume val on {len(val_case_ids)} cases")

    out_dir = Path(args.out or f"runs/{cfg.arm}_{int(time.time())}")
    max_steps = args.max_steps or int(train_cfg.get("max_steps", 50 if args.smoke else 30000))

    result = train_loop(
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
        val_loader=val_loader,
        val_every=val_every if val_mode != "whole" or val_every else int(train_cfg.get("val_every", 2000)),
        resume_from=Path(args.resume) if args.resume else None,
        seed=args.seed,
        jepa_loader=jepa_loader,
        val_mode=val_mode,
        word_root=word_root,
        val_case_ids=val_case_ids,
        val_patch=val_patch,
        val_stride=val_stride,
    )
    print(
        f"done. steps={result['steps']} best_val={result.get('best_mean_fg_dice')} "
        f"commit={result.get('commit')} artifacts -> {out_dir}"
    )


if __name__ == "__main__":
    main()
