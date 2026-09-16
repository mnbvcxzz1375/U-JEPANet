from __future__ import annotations

"""WORD-style dataset adapters with fail-closed unlabeled routing.

Unlabeled cases never load GT even if the label file exists on disk.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class NpyVolumeDataset(Dataset):
    """Preprocessed .npy volumes.

    Layout:
        root/
          images/{case_id}.npy
          labels/{case_id}.npy          # optional
          splits/{split}.txt
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: Sequence[int] = (128, 128, 96),
        unlabeled_ids: Optional[Sequence[str]] = None,
    ):
        self.root = Path(root)
        self.patch_size = tuple(int(v) for v in patch_size)
        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        self.ids: List[str] = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.unlabeled = frozenset(unlabeled_ids or [])

    def is_labeled(self, case_id: str) -> bool:
        return case_id not in self.unlabeled

    def _load_image(self, case_id: str) -> torch.Tensor:
        import numpy as np

        arr = np.load(self.root / "images" / f"{case_id}.npy")
        t = torch.from_numpy(arr).float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t

    def _load_label(self, case_id: str) -> torch.Tensor:
        import numpy as np

        path = self.root / "labels" / f"{case_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"labeled case missing label file: {path}")
        return torch.from_numpy(np.load(path)).long()

    def _random_crop(self, image: torch.Tensor, label: Optional[torch.Tensor]):
        d, h, w = image.shape[-3:]
        pd, ph, pw = self.patch_size
        sd = max(0, d - pd)
        sh = max(0, h - ph)
        sw = max(0, w - pw)
        z = int(torch.randint(0, sd + 1, (1,)).item()) if sd > 0 else 0
        y = int(torch.randint(0, sh + 1, (1,)).item()) if sh > 0 else 0
        x = int(torch.randint(0, sw + 1, (1,)).item()) if sw > 0 else 0
        img = image[..., z : z + pd, y : y + ph, x : x + pw]
        if img.shape[-3:] != self.patch_size:
            img = torch.nn.functional.pad(
                img,
                (
                    0, max(0, pw - img.shape[-1]),
                    0, max(0, ph - img.shape[-2]),
                    0, max(0, pd - img.shape[-3]),
                ),
            )
        if label is None:
            return img, None
        lab = label[z : z + pd, y : y + ph, x : x + pw]
        if lab.shape[-3:] != self.patch_size:
            lab = torch.nn.functional.pad(
                lab,
                (
                    0, max(0, pw - lab.shape[-1]),
                    0, max(0, ph - lab.shape[-2]),
                    0, max(0, pd - lab.shape[-3]),
                ),
            )
        return img, lab

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | bool]:
        case_id = self.ids[idx]
        labeled = self.is_labeled(case_id)
        image = self._load_image(case_id)
        label = self._load_label(case_id) if labeled else None
        image, label = self._random_crop(image, label)
        if label is None:
            # Dummy label for collate only; is_labeled=False must gate L_seg.
            label = torch.zeros(image.shape[-3:], dtype=torch.long)
        return {
            "image": image,
            "label": label,
            "is_labeled": torch.tensor(labeled, dtype=torch.bool),
            "case_id": case_id,
        }


def collate_batch(items: List[Dict]):
    images = torch.stack([it["image"] for it in items], dim=0)
    labels = torch.stack([it["label"] for it in items], dim=0)
    is_labeled = torch.stack([it["is_labeled"] for it in items], dim=0)
    case_ids = [it["case_id"] for it in items]
    return {
        "image": images,
        "label": labels,
        "is_labeled": is_labeled,
        "case_id": case_ids,
    }


class NiftiWordDataset(Dataset):
    """Direct NIfTI WORD reader with fail-closed unlabeled routing.

    Never opens labels for cases in ``unlabeled_ids``, even if files exist.
    """

    def __init__(
        self,
        word_root: str | Path,
        split: str = "imagesTr",
        label_view: str = "labelsTr_All",
        patch_size: Sequence[int] = (128, 128, 96),
        unlabeled_ids: Optional[Sequence[str]] = None,
        max_cases: Optional[int] = None,
        seed: int = 0,
    ):
        self.root = Path(word_root)
        self.split = split
        self.label_view = label_view
        self.patch_size = tuple(int(v) for v in patch_size)
        self.unlabeled = frozenset(unlabeled_ids or [])
        img_dir = self.root / split
        if not img_dir.exists():
            raise FileNotFoundError(img_dir)
        files = sorted(img_dir.glob("word_*.nii.gz"))
        if max_cases is not None:
            files = files[: int(max_cases)]
        self.files = files
        self._rng = torch.Generator().manual_seed(seed)

    def is_labeled(self, case_id: str) -> bool:
        return case_id not in self.unlabeled

    def __len__(self) -> int:
        return len(self.files)

    def _read_nifti(self, path: Path):
        import numpy as np

        try:
            import SimpleITK as sitk

            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        except ImportError:
            import nibabel as nib

            arr = np.asanyarray(nib.load(str(path)).dataobj).transpose(2, 1, 0)
        return arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | bool]:
        import numpy as np

        path = self.files[idx]
        case_id = path.name.replace(".nii.gz", "")
        labeled = self.is_labeled(case_id)
        img = self._read_nifti(path).astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / max(hi - lo, 1e-6)

        lab = None
        if labeled:
            lab_path = self.root / self.label_view / path.name
            if not lab_path.exists():
                raise FileNotFoundError(f"labeled case missing label: {lab_path}")
            lab = self._read_nifti(lab_path).astype(np.int64)

        pd, ph, pw = self.patch_size
        d, h, w = img.shape
        z0 = int(torch.randint(0, max(1, d - pd + 1), (1,), generator=self._rng).item()) if d >= pd else 0
        y0 = int(torch.randint(0, max(1, h - ph + 1), (1,), generator=self._rng).item()) if h >= ph else 0
        x0 = int(torch.randint(0, max(1, w - pw + 1), (1,), generator=self._rng).item()) if w >= pw else 0

        def crop_pad(arr, is_label=False):
            c = arr[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
            if c.shape != self.patch_size:
                pad = (
                    (0, max(0, pd - c.shape[0])),
                    (0, max(0, ph - c.shape[1])),
                    (0, max(0, pw - c.shape[2])),
                )
                c = np.pad(c, pad, mode="constant", constant_values=0)
            return c

        img_c = crop_pad(img)
        image = torch.from_numpy(img_c).float().unsqueeze(0)
        if lab is None:
            label = torch.zeros(self.patch_size, dtype=torch.long)
        else:
            label = torch.from_numpy(crop_pad(lab)).long()
        return {
            "image": image,
            "label": label,
            "is_labeled": torch.tensor(labeled, dtype=torch.bool),
            "case_id": case_id,
        }
