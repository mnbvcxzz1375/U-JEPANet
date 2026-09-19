"""GLP whole-volume validation — P0: real crop origins + whole-CT global."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import dice_per_class


def _gaussian_importance(patch_size, sigma_scale=0.125) -> torch.Tensor:
    axes = []
    for s in patch_size:
        coords = torch.arange(s, dtype=torch.float32)
        coords = coords - (s - 1) / 2.0
        axes.append(torch.exp(-0.5 * (coords / (sigma_scale * s)) ** 2))
    g = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return g / g.max()


@torch.no_grad()
def sliding_window_logits_glp(
    model,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device,
    num_classes: int,
    global_image: Optional[torch.Tensor] = None,
    full_shape: Optional[Tuple[int, int, int]] = None,
):
    """volume: (1,1,D,H,W) windowed. global_image: (1,1,gD,gH,gW) or None.

    Each patch passes its true crop_origin in whole-volume voxel coords.
    Eval has no aug → affine_theta = I.
    """
    model.eval()
    vol = volume.to(device)
    _, _, d, h, w = vol.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    sd = max(1, min(sd, pd))
    sh = max(1, min(sh, ph))
    sw = max(1, min(sw, pw))
    if full_shape is None:
        full_shape = (d, h, w)

    zs = list(range(0, max(1, d - pd + 1), sd))
    ys = list(range(0, max(1, h - ph + 1), sh))
    xs = list(range(0, max(1, w - pw + 1), sw))
    if zs[-1] != max(0, d - pd):
        zs.append(max(0, d - pd))
    if ys[-1] != max(0, h - ph):
        ys.append(max(0, h - ph))
    if xs[-1] != max(0, w - pw):
        xs.append(max(0, w - pw))

    acc = torch.zeros((1, num_classes, d, h, w), device=device)
    weight = torch.zeros((1, 1, d, h, w), device=device)
    importance = _gaussian_importance((pd, ph, pw)).to(device)[None, None]
    theta = torch.eye(3, 4, device=device).unsqueeze(0)  # (1,3,4)
    fs = torch.tensor([[full_shape[0], full_shape[1], full_shape[2]]], dtype=torch.long, device=device)
    gi = global_image.to(device) if global_image is not None else None

    for z in zs:
        for y in ys:
            for x in xs:
                patch = vol[:, :, z : z + pd, y : y + ph, x : x + pw]
                if patch.shape[-3:] != (pd, ph, pw):
                    pad_d = max(0, pd - patch.shape[-3])
                    pad_h = max(0, ph - patch.shape[-2])
                    pad_w = max(0, pw - patch.shape[-1])
                    patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))
                origin = torch.tensor([[z, y, x]], dtype=torch.long, device=device)
                logits = model(
                    patch,
                    global_image=gi,
                    crop_origin=origin,
                    full_shape=fs,
                    affine_theta=theta,
                    patch_size=(pd, ph, pw),
                )
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                acc[:, :, z : z + pd, y : y + ph, x : x + pw] += logits * importance
                weight[:, :, z : z + pd, y : y + ph, x : x + pw] += importance
    weight = torch.clamp(weight, min=1e-8)
    return acc / weight


@torch.no_grad()
def evaluate_word_whole_volume_glp(
    model,
    word_root: str,
    val_ids: Sequence[str],
    num_classes: int,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device,
    label_view: str = "labelsVal",
    image_dir: str = "imagesVal",
    intensity_mode: str = "window",
    need_global: bool = False,
    global_size: Tuple[int, int, int] = (64, 64, 64),
    max_cases: Optional[int] = None,
) -> Dict[str, float]:
    """GLP-aware val: whole CT global once per case + real patch origins."""
    from pathlib import Path

    import SimpleITK as sitk

    from .ct_augment import canonical_window

    was_training = model.training
    model.eval()
    root = Path(word_root)
    ids = list(val_ids)
    if max_cases is not None:
        ids = ids[: int(max_cases)]

    per_class_acc: Dict[int, List[float]] = {c: [] for c in range(1, num_classes)}
    case_means: List[float] = []
    for cid in ids:
        ip = root / image_dir / f"{cid}.nii.gz"
        lp = root / label_view / f"{cid}.nii.gz"
        if not ip.exists() or not lp.exists():
            raise FileNotFoundError(f"missing case {cid}")
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(ip))).astype(np.float32)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(lp))).astype(np.int64)
        image_t = torch.from_numpy(img)[None]  # (1,D,H,W)
        if intensity_mode == "window":
            vol = canonical_window(image_t)[None]  # (1,1,D,H,W)
        else:
            lo, hi = float(img.min()), float(img.max())
            vol = ((image_t - lo) / max(hi - lo, 1e-6))[None]
        glob = None
        if need_global:
            g = canonical_window(torch.from_numpy(img).float())
            glob = F.interpolate(g[None, None], size=global_size, mode="trilinear", align_corners=False)
        logits = sliding_window_logits_glp(
            model, vol, patch_size, stride, device, num_classes,
            global_image=glob, full_shape=tuple(img.shape),
        )
        pred = torch.argmax(logits, dim=1)[0].cpu()
        pc = dice_per_class(pred, torch.from_numpy(lab), num_classes)
        for c, v in pc.items():
            if v == v:
                per_class_acc[c].append(v)
        vals = [v for v in pc.values() if v == v]
        case_means.append(float(np.mean(vals)) if vals else float("nan"))
        print(f"  [glp-val] {cid} mean={case_means[-1]:.4f}", flush=True)

    if was_training:
        model.train()
    metrics: Dict[str, float] = {}
    for c, vs in per_class_acc.items():
        metrics[f"dice_c{c:02d}"] = float(sum(vs) / len(vs)) if vs else float("nan")
    scored = [metrics[f"dice_c{c:02d}"] for c in per_class_acc if metrics[f"dice_c{c:02d}"] == metrics[f"dice_c{c:02d}"]]
    metrics["mean_fg_dice"] = float(sum(scored) / len(scored)) if scored else float("nan")
    metrics["n_val_cases"] = float(len(ids))
    return metrics
