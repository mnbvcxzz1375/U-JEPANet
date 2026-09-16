from __future__ import annotations

"""WORD-style dataset adapters.

Real remote layouts differ (labelsTr vs labelsTr_2/All, h5 caches, LPS).
Wire the concrete root after confirming the host path. Until then, only the
synthetic path in train.py is enabled by default.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class NpyVolumeDataset(Dataset):
    """Read preprocessed volumes from .npy files.

    Expected layout:
        root/
          images/{case_id}.npy   # float32 (D, H, W) or (C, D, H, W)
          labels/{case_id}.npy   # int64 (D, H, W)  [optional for unlabeled]
          splits/{split}.txt     # one case_id per line
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: Sequence[int] = (128, 128, 96),
        with_labels: bool = True,
        unlabeled_ids: Optional[Sequence[str]] = None,
    ):
        self.root = Path(root)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.with_labels = with_labels
        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        self.ids: List[str] = [
            line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        unlabeled = set(unlabeled_ids or [])
        self.labeled_ids = [i for i in self.ids if i not in unlabeled]
        self.unlabeled_ids = [i for i in self.ids if i in unlabeled]
        self.all_ids = list(self.ids)

    def _load_image(self, case_id: str) -> torch.Tensor:
        import numpy as np

        arr = np.load(self.root / "images" / f"{case_id}.npy")
        t = torch.from_numpy(arr).float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t

    def _load_label(self, case_id: str) -> Optional[torch.Tensor]:
        import numpy as np

        path = self.root / "labels" / f"{case_id}.npy"
        if not path.exists():
            return None
        return torch.from_numpy(np.load(path)).long()

    def _random_crop(self, image: torch.Tensor, label: Optional[torch.Tensor]):
        d, h, w = image.shape[-3:]
        pd, ph, pw = self.patch_size
        sd = max(0, d - pd)
        sh = max(0, h - ph)
        sw = max(0, w - pw)
        z = torch.randint(0, sd + 1, (1,)).item() if sd > 0 else 0
        y = torch.randint(0, sh + 1, (1,)).item() if sh > 0 else 0
        x = torch.randint(0, sw + 1, (1,)).item() if sw > 0 else 0
        img = image[..., z : z + pd, y : y + ph, x : x + pw]
        if img.shape[-3:] != self.patch_size:
            img = torch.nn.functional.pad(
                img,
                (
                    0,
                    max(0, pw - img.shape[-1]),
                    0,
                    max(0, ph - img.shape[-2]),
                    0,
                    max(0, pd - img.shape[-3]),
                ),
            )
        if label is None:
            return img, None
        lab = label[z : z + pd, y : y + ph, x : x + pw]
        if lab.shape[-3:] != self.patch_size:
            lab = torch.nn.functional.pad(
                lab,
                (
                    0,
                    max(0, pw - lab.shape[-1]),
                    0,
                    max(0, ph - lab.shape[-2]),
                    0,
                    max(0, pd - lab.shape[-3]),
                ),
            )
        return img, lab

    def __len__(self) -> int:
        return len(self.all_ids)

    def __getitem__(self, idx: int):
        case_id = self.all_ids[idx]
        image = self._load_image(case_id)
        label = self._load_label(case_id) if self.with_labels else None
        image, label = self._random_crop(image, label)
        if label is None:
            # Return a dummy label for collate; trainer should skip seg loss
            # for unlabeled rows via a flag dataset if needed.
            label = torch.zeros(image.shape[-3:], dtype=torch.long)
        return image, label
