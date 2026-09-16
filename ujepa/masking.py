from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _block_size_for_grid(grid: Tuple[int, int, int], mask_ratio: float) -> Tuple[int, int, int]:
    """Pick token-grid block sizes that can actually reach ``mask_ratio``."""
    d, h, w = grid
    # Aim for a few large blocks; scale with grid so 8^3 and 32^24 both work.
    base = max(1, int(round(min(d, h, w) * 0.35)))
    # Wider grids need anisotropic blocks to place enough volume.
    bd = max(1, min(d, max(1, int(d * 0.35))))
    bh = max(1, min(h, max(1, int(h * 0.35))))
    bw = max(1, min(w, max(1, int(w * 0.35))))
    if mask_ratio >= 0.5:
        bd, bh, bw = max(bd, d // 2), max(bh, h // 2), max(bw, w // 2)
    return (min(bd, d), min(bh, h), min(bw, w))


def sample_token_mask(
    batch: int,
    grid: Tuple[int, int, int],
    mask_ratio: float = 0.4,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    max_attempts: int = 64,
) -> torch.Tensor:
    """Sample visibility mask **on the JEPA token grid**.

    Returns M with shape (B, 1, D, H, W), M=1 visible, M=0 hidden.
    Guarantees hidden fraction >= min(mask_ratio, 0.95) when possible
    (falls back to random token fill if blocks cannot cover enough).
    """
    d, h, w = grid
    device = device or torch.device("cpu")
    n = d * h * w
    target_hidden = int(round(n * float(mask_ratio)))
    target_hidden = max(0, min(n, target_hidden))
    mask = torch.ones(batch, 1, d, h, w, device=device)
    if target_hidden <= 0:
        return mask

    bd, bh, bw = _block_size_for_grid(grid, mask_ratio)
    for b in range(batch):
        hidden = 0
        attempts = 0
        while hidden < target_hidden and attempts < max_attempts:
            attempts += 1
            # Shrink blocks near the end so we don't wildly overshoot forever.
            remain = target_hidden - hidden
            if remain < bd * bh * bw // 4:
                bd_i = max(1, min(bd, remain))
                bh_i = max(1, min(bh, int(remain ** (1 / 3)) + 1))
                bw_i = max(1, min(bw, int(remain ** (1 / 3)) + 1))
            else:
                bd_i, bh_i, bw_i = bd, bh, bw
            y0 = int(torch.randint(0, max(1, d - bd_i + 1), (1,), generator=generator).item()) if d > bd_i else 0
            x0 = int(torch.randint(0, max(1, h - bh_i + 1), (1,), generator=generator).item()) if h > bh_i else 0
            z0 = int(torch.randint(0, max(1, w - bw_i + 1), (1,), generator=generator).item()) if w > bw_i else 0
            sl = mask[b, 0, y0 : y0 + bd_i, x0 : x0 + bh_i, z0 : z0 + bw_i]
            newly = int((sl > 0.5).sum().item())
            sl.zero_()
            hidden += newly

        if hidden < target_hidden:
            # Deterministic fallback: randomly hide remaining tokens without replacement.
            flat = mask[b, 0].reshape(-1)
            visible_idx = torch.nonzero(flat > 0.5, as_tuple=False).squeeze(-1)
            need = target_hidden - hidden
            if visible_idx.numel() > 0:
                perm = torch.randperm(visible_idx.numel(), generator=generator, device=device)[:need]
                flat[visible_idx[perm]] = 0.0
    return mask


def token_mask_to_volume_mask(
    token_mask: torch.Tensor,
    volume_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Nearest-upsample a token-grid mask (B,1,D',H',W') to (B,1,D,H,W)."""
    if token_mask.shape[-3:] == volume_size:
        return token_mask
    return F.interpolate(token_mask.float(), size=volume_size, mode="nearest")


def sample_block_mask(
    batch: int,
    grid: Tuple[int, int, int],
    mask_ratio: float = 0.4,
    min_block: Tuple[int, int, int] = (2, 2, 2),
    max_block: Tuple[int, int, int] = (6, 6, 6),
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Deprecated voxel-grid masker kept for tests; prefer sample_token_mask.

    Note: with tiny max_block this cannot reach high mask ratios on large grids.
    """
    return sample_token_mask(
        batch, grid, mask_ratio=mask_ratio, generator=generator, device=device
    )


def apply_mask(
    x: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0
) -> torch.Tensor:
    """Apply visibility mask (token or volume grid) to a volume/feature map."""
    if mask.shape[-3:] != x.shape[-3:]:
        mask = token_mask_to_volume_mask(mask, tuple(x.shape[-3:]))
    fill = torch.full_like(x, fill_value)
    return x * mask + fill * (1.0 - mask)


def mask_hidden_fraction(mask: torch.Tensor) -> float:
    return float((mask < 0.5).float().mean().item())


def mask_to_token_ids(
    mask: torch.Tensor, grid: Optional[Tuple[int, int, int]] = None
) -> torch.Tensor:
    b, _, d, h, w = mask.shape
    return mask.reshape(b, d * h * w) > 0.5
