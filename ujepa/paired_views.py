from __future__ import annotations

"""Paired weak/strong JEPA views: same case + same crop, two appearance views."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .ct_augment import canonical_window, jepa_strong_context, jepa_weak_target
from .dynamic_dataset import DynamicWordVolumeDataset


class PairedJEPADataset(Dataset):
    """Sample case+crop once; emit (context_strong, target_weak) of the same content.

    Geometry is shared by construction: one origin, one crop, then two intensity maps.
    """

    def __init__(
        self,
        word_root: str | Path,
        case_ids: Sequence[str],
        patch_size: Sequence[int] = (128, 128, 96),
        seed: int = 0,
        cache_dir: Optional[str | Path] = None,
    ):
        self.inner = DynamicWordVolumeDataset(
            word_root,
            case_ids,
            patch_size=patch_size,
            labeled=False,
            seed=seed,
            mode="none",  # raw window, no random appearance
            strength=0.0,
            cache_dir=cache_dir,
        )
        self._g = torch.Generator().manual_seed(seed + 997)
        self.patch = tuple(int(v) for v in patch_size)
        self.ids = list(case_ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        # Re-sample crop with DynamicWordVolumeDataset; then two views from same HU crop.
        # Access private helpers to keep one origin.
        cid = self.ids[idx]
        img = self.inner._load_img(cid)
        if len(self.inner._img_cache) < 8:
            self.inner._img_cache[cid] = img
        z0, y0, x0 = self.inner._sample_origin(img, None)
        crop_hu = self.inner._crop(img, z0, y0, x0)
        x_hu = torch.from_numpy(np.ascontiguousarray(crop_hu)).float().unsqueeze(0)
        context = jepa_strong_context(x_hu, generator=self._g)
        target = jepa_weak_target(x_hu)
        return {
            "context": context,
            "target": target,
            "case_id": cid,
            "origin": torch.tensor([z0, y0, x0], dtype=torch.long),
        }
