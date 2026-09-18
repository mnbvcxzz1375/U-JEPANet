from __future__ import annotations

"""Whole-volume sliding-window validation for locked WORD imagesVal."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import dice_per_class


def _gaussian_importance(patch_size: Sequence[int], sigma_scale: float = 0.125) -> torch.Tensor:
    axes = []
    for s in patch_size:
        coords = torch.arange(s, dtype=torch.float32) - (s - 1) / 2.0
        axes.append(torch.exp(-0.5 * (coords / (sigma_scale * s)) ** 2))
    g = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return g / g.max()


@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    """volume: (1,1,D,H,W) already normalized. Returns (1,C,D,H,W) logits."""
    model.eval()
    vol = volume.to(device)
    _, _, d, h, w = vol.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    sd = max(1, min(sd, pd))
    sh = max(1, min(sh, ph))
    sw = max(1, min(sw, pw))

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

    for z in zs:
        for y in ys:
            for x in xs:
                patch = vol[:, :, z : z + pd, y : y + ph, x : x + pw]
                # pad if volume smaller than patch
                if patch.shape[-3:] != (pd, ph, pw):
                    pad_d = max(0, pd - patch.shape[-3])
                    pad_h = max(0, ph - patch.shape[-2])
                    pad_w = max(0, pw - patch.shape[-1])
                    patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))
                logits = model(patch)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                acc[:, :, z : z + pd, y : y + ph, x : x + pw] += logits * importance
                weight[:, :, z : z + pd, y : y + ph, x : x + pw] += importance
    weight = torch.clamp(weight, min=1e-8)
    return acc / weight


@torch.no_grad()
def evaluate_whole_volume_case(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: torch.device,
    intensity_mode: str = "minmax",
) -> Dict[str, float]:
    """image (1,D,H,W) or (1,1,D,H,W); label (D,H,W).

    intensity_mode:
      - minmax: (x-min)/(max-min)  [legacy fixed-crop / official PL-Seg style]
      - window: fixed CT window L=40 W=400  [matches DynamicWordVolumeDataset training]
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() == 4:
        image = image.unsqueeze(0) if image.shape[0] != 1 else image.unsqueeze(1)
    if intensity_mode == "window":
        from .ct_augment import canonical_window

        # image is (1,1,D,H,W); canonical_window expects HU volume
        image = canonical_window(image)
    else:
        lo = float(image.min())
        hi = float(image.max())
        image = (image - lo) / max(hi - lo, 1e-6)
    logits = sliding_window_logits(model, image, patch_size, stride, device, num_classes)
    pred = torch.argmax(logits, dim=1)[0].cpu()
    pc = dice_per_class(pred, label.cpu(), num_classes)
    out = {f"dice_c{c:02d}": v for c, v in pc.items()}
    vals = [v for v in pc.values() if v == v]
    out["mean_fg_dice"] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


@torch.no_grad()
def evaluate_word_whole_volume(
    model: torch.nn.Module,
    word_root: str,
    val_ids: Sequence[str],
    num_classes: int,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    device: torch.device,
    label_view: str = "labelsVal",
    max_cases: Optional[int] = None,
    intensity_mode: str = "minmax",
    image_dir: str = "imagesVal",
) -> Dict[str, float]:
    """Official-style whole-volume val on listed case ids.

    intensity_mode must match how the checkpoint was trained:
      - minmax for fixed-crop cache arms (A0/A2/A3 historical)
      - window for DynamicWordVolumeDataset arms (A0D/C*)
    """
    from pathlib import Path

    import SimpleITK as sitk

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
            raise FileNotFoundError(f"missing val case {cid}")
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(ip))).astype(np.float32)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(lp))).astype(np.int64)
        image_t = torch.from_numpy(img)[None]  # (1,D,H,W)
        label_t = torch.from_numpy(lab)
        m = evaluate_whole_volume_case(
            model, image_t, label_t, num_classes, patch_size, stride, device,
            intensity_mode=intensity_mode,
        )
        for c in range(1, num_classes):
            v = m.get(f"dice_c{c:02d}", float("nan"))
            if v == v:
                per_class_acc[c].append(v)
        case_means.append(m["mean_fg_dice"])

    if was_training:
        model.train()
        tgt = getattr(model, "target", None)
        if tgt is not None and getattr(tgt, "target", None) is not None:
            tgt.train(False)
            tgt.target.eval()

    metrics: Dict[str, float] = {}
    for c, vs in per_class_acc.items():
        metrics[f"dice_c{c:02d}"] = float(sum(vs) / len(vs)) if vs else float("nan")
    scored = [metrics[f"dice_c{c:02d}"] for c in per_class_acc if metrics[f"dice_c{c:02d}"] == metrics[f"dice_c{c:02d}"]]
    metrics["mean_fg_dice"] = float(sum(scored) / len(scored)) if scored else float("nan")
    metrics["n_val_cases"] = float(len(ids))
    return metrics
