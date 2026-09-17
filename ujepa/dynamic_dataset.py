from __future__ import annotations

"""Dynamic-crop WORD dataset from full HU volumes (not fixed patch cache)."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .ct_augment import (
    canonical_window,
    jepa_strong_context,
    jepa_weak_target,
    mild_affine,
    random_window,
    seg_augment_hu,
)


def _read_nifti(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


class DynamicWordVolumeDataset(Dataset):
    """Load full volumes once; crop dynamically every __getitem__.

    Images stored/used as raw HU. Windowing happens at batch time.
    Labeled: optional foreground-biased crop using GT.
    Unlabeled: body-band + z-stratified crop (no GT).
    """

    def __init__(
        self,
        word_root: str | Path,
        case_ids: Sequence[str],
        patch_size: Sequence[int] = (128, 128, 96),
        labeled: bool = True,
        label_view: str = "labelsTr_All",
        image_dir: str = "imagesTr",
        fg_crop_prob: float = 0.7,
        z_strata: int = 3,
        seed: int = 0,
        mode: str = "seg",  # seg | jepa_target | jepa_context | none
        strength: float = 1.0,
        cache_dir: Optional[str | Path] = None,
        preload: bool = False,
    ):
        self.root = Path(word_root)
        self.ids = list(case_ids)
        self.patch = tuple(int(v) for v in patch_size)
        self.labeled = labeled
        self.label_view = label_view
        self.image_dir = image_dir
        self.fg_crop_prob = fg_crop_prob
        self.z_strata = z_strata
        self.mode = mode
        self.strength = strength
        self._g = torch.Generator().manual_seed(seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Lazy volume store: do not preload all CTs into RAM.
        self._img_cache: Dict[str, np.ndarray] = {}
        self._lab_cache: Dict[str, np.ndarray] = {}
        self._preload = bool(preload)
        if self._preload:
            for cid in self.ids:
                self._img_cache[cid] = self._load_img(cid)
                if labeled:
                    self._lab_cache[cid] = self._load_lab(cid)

    def _load_img(self, cid: str) -> np.ndarray:
        if self.cache_dir and (self.cache_dir / f"{cid}_img.npy").exists():
            return np.load(self.cache_dir / f"{cid}_img.npy")
        arr = _read_nifti(self.root / self.image_dir / f"{cid}.nii.gz")
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{cid}_img.npy", arr)
        return arr

    def _load_lab(self, cid: str) -> np.ndarray:
        if self.cache_dir and (self.cache_dir / f"{cid}_lab.npy").exists():
            return np.load(self.cache_dir / f"{cid}_lab.npy")
        arr = _read_nifti(self.root / self.label_view / f"{cid}.nii.gz").astype(np.int64)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{cid}_lab.npy", arr)
        return arr

    def __len__(self) -> int:
        return len(self.ids)

    def _sample_origin(self, img: np.ndarray, lab: Optional[np.ndarray]) -> Tuple[int, int, int]:
        pd, ph, pw = self.patch
        d, h, w = img.shape
        z0 = y0 = x0 = 0
        use_fg = (
            self.labeled
            and lab is not None
            and torch.rand(1, generator=self._g).item() < self.fg_crop_prob
            and (lab > 0).any()
        )
        if use_fg:
            fg = np.argwhere(lab > 0)
            i = int(torch.randint(0, len(fg), (1,), generator=self._g).item())
            cz, cy, cx = fg[i]
            z0 = int(np.clip(cz - pd // 2, 0, max(0, d - pd)))
            y0 = int(np.clip(cy - ph // 2, 0, max(0, h - ph)))
            x0 = int(np.clip(cx - pw // 2, 0, max(0, w - pw)))
        else:
            # body-ish: skip pure air using intensity percentile on a coarse grid
            # z-stratified for unlabeled pelvic coverage
            s = self.z_strata
            band = int(torch.randint(0, max(1, s), (1,), generator=self._g).item())
            z_lo = int(band * max(0, d - pd) / max(1, s - 1)) if s > 1 else 0
            z_hi = int((band + 1) * max(0, d - pd) / max(1, s - 1)) if s > 1 else max(0, d - pd)
            if z_hi <= z_lo:
                z_lo, z_hi = 0, max(0, d - pd)
            z0 = int(torch.randint(z_lo, z_hi + 1, (1,), generator=self._g).item()) if d > pd else 0
            y0 = int(torch.randint(0, max(1, h - ph + 1), (1,), generator=self._g).item()) if h >= ph else 0
            x0 = int(torch.randint(0, max(1, w - pw + 1), (1,), generator=self._g).item()) if w >= pw else 0
        return z0, y0, x0

    def _crop(self, arr: np.ndarray, z0: int, y0: int, x0: int) -> np.ndarray:
        pd, ph, pw = self.patch
        c = arr[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
        if c.shape != self.patch:
            pad = (
                (0, max(0, pd - c.shape[0])),
                (0, max(0, ph - c.shape[1])),
                (0, max(0, pw - c.shape[2])),
            )
            c = np.pad(c, pad, mode="constant", constant_values=-1000 if arr.dtype != np.int64 else 0)
        return c

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | bool]:
        cid = self.ids[idx]
        if cid in self._img_cache:
            img = self._img_cache[cid]
        else:
            img = self._load_img(cid)
            if len(self._img_cache) < 8:  # tiny LRU-ish cap
                self._img_cache[cid] = img
        if self.labeled:
            if cid in self._lab_cache:
                lab = self._lab_cache[cid]
            else:
                lab = self._load_lab(cid)
                if len(self._lab_cache) < 8:
                    self._lab_cache[cid] = lab
        else:
            lab = None
        z0, y0, x0 = self._sample_origin(img, lab)
        crop_hu = self._crop(img, z0, y0, x0)
        x_hu = torch.from_numpy(np.ascontiguousarray(crop_hu)).float().unsqueeze(0)  # (1,D,H,W)

        if self.mode == "jepa_target":
            x = jepa_weak_target(x_hu)
        elif self.mode == "jepa_context":
            x = jepa_strong_context(x_hu, generator=self._g)
        elif self.mode == "seg":
            x = seg_augment_hu(x_hu, strength=self.strength, generator=self._g)
        else:
            x = canonical_window(x_hu)

        if lab is None:
            label = torch.zeros(self.patch, dtype=torch.long)
            is_l = False
        else:
            label = torch.from_numpy(np.ascontiguousarray(self._crop(lab, z0, y0, x0))).long()
            is_l = True
            # shared mild affine for seg only
            if self.mode == "seg" and self.strength > 0:
                # x is (1,D,H,W); mild_affine expects (B,C,D,H,W)
                lab_t = label.unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
                x5 = x.unsqueeze(0)  # (1,1,D,H,W)
                x5, lab_t = mild_affine(x5, lab_t, p=0.5)
                if lab_t.dim() == 5:
                    label = lab_t[0, 0].long()
                elif lab_t.dim() == 4:
                    label = lab_t[0].long()
                else:
                    label = lab_t.long()
                x = x5[0] if x5.dim() == 5 else x5

        return {
            "image": x,
            "label": label,
            "is_labeled": torch.tensor(is_l, dtype=torch.bool),
            "case_id": cid,
        }


def build_dynamic_loaders(
    word_root: str,
    labeled_ids: Sequence[str],
    all_ids: Sequence[str],
    patch=(128, 128, 96),
    batch_size=2,
    seed=42,
    strength=1.0,
    jepa_aug: str = "none",  # none | same | weak_strong
    cache_dir: Optional[str] = None,
):
    """Return (seg_loader, jepa_loader_or_None)."""
    seg_ds = DynamicWordVolumeDataset(
        word_root, labeled_ids, patch, labeled=True, seed=seed, mode="seg", strength=strength, cache_dir=cache_dir
    )
    seg_loader = torch.utils.data.DataLoader(
        seg_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: {
            "image": torch.stack([i["image"] for i in b]),
            "label": torch.stack([i["label"] for i in b]),
            "is_labeled": torch.stack([i["is_labeled"] for i in b]),
            "case_id": [i["case_id"] for i in b],
        },
    )
    if jepa_aug == "weak_strong":
        # context path strong; target uses canonical — handled in trainer by two datasets
        jepa_ctx = DynamicWordVolumeDataset(
            word_root, all_ids, patch, labeled=False, seed=seed + 1, mode="jepa_context", cache_dir=cache_dir
        )
        jepa_tar = DynamicWordVolumeDataset(
            word_root, all_ids, patch, labeled=False, seed=seed + 2, mode="jepa_target", cache_dir=cache_dir
        )
        return seg_loader, (jepa_ctx, jepa_tar)
    if jepa_aug == "same":
        jepa_ds = DynamicWordVolumeDataset(
            word_root, all_ids, patch, labeled=False, seed=seed + 1, mode="seg", strength=strength, cache_dir=cache_dir
        )
    else:
        jepa_ds = DynamicWordVolumeDataset(
            word_root, all_ids, patch, labeled=False, seed=seed + 1, mode="none", cache_dir=cache_dir
        )
    jepa_loader = torch.utils.data.DataLoader(
        jepa_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: {
            "image": torch.stack([i["image"] for i in b]),
            "label": torch.stack([i["label"] for i in b]),
            "is_labeled": torch.stack([i["is_labeled"] for i in b]),
            "case_id": [i["case_id"] for i in b],
        },
    )
    return seg_loader, jepa_loader
