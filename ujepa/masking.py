from __future__ import annotations

from typing import Optional, Tuple

import torch


def sample_block_mask(
    batch: int,
    grid: Tuple[int, int, int],
    mask_ratio: float = 0.4,
    min_block: Tuple[int, int, int] = (2, 2, 2),
    max_block: Tuple[int, int, int] = (6, 6, 6),
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample a binary visibility mask M with shape (B, 1, D, H, W).

    M = 1 means visible. A single (or few) axis-aligned 3D block is filled as
    hidden (0) until the hidden voxel ratio approximates ``mask_ratio``.
    Uses continuous 3D blocks; never uses organ GT.
    """
    d, h, w = grid
    device = device or torch.device("cpu")
    mask = torch.ones(batch, 1, d, h, w, device=device)
    total = d * h * w
    target_hidden = int(round(total * mask_ratio))
    if target_hidden <= 0:
        return mask

    min_d, min_h, min_w = (max(1, min(x, s)) for x, s in zip(min_block, (d, h, w)))
    max_d = max(min_d, min(max_block[0], d))
    max_h = max(min_h, min(max_block[1], h))
    max_w = max(min_w, min(max_block[2], w))

    for b in range(batch):
        hidden = 0
        attempts = 0
        while hidden < target_hidden and attempts < 16:
            attempts += 1
            bd = int(torch.randint(min_d, max_d + 1, (1,), generator=generator).item())
            bh = int(torch.randint(min_h, max_h + 1, (1,), generator=generator).item())
            bw = int(torch.randint(min_w, max_w + 1, (1,), generator=generator).item())
            if d > bd:
                y0 = int(torch.randint(0, d - bd + 1, (1,), generator=generator).item())
            else:
                y0 = 0
            if h > bh:
                x0 = int(torch.randint(0, h - bh + 1, (1,), generator=generator).item())
            else:
                x0 = 0
            if w > bw:
                z0 = int(torch.randint(0, w - bw + 1, (1,), generator=generator).item())
            else:
                z0 = 0
            block = mask[b, 0, y0 : y0 + bd, x0 : x0 + bh, z0 : z0 + bw]
            newly = int((block == 1).sum().item())
            block.zero_()
            hidden += newly
    return mask


def apply_mask(
    x: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0
) -> torch.Tensor:
    """Apply visibility mask to a volume or dense feature map.

    mask: (B, 1, D, H, W) with 1=visible. Spatial size of mask may be smaller
    than x; it is then upsampled with nearest interpolation to x's spatial size.
    """
    if mask.shape[-3:] != x.shape[-3:]:
        mask = torch.nn.functional.interpolate(
            mask.float(), size=x.shape[-3:], mode="nearest"
        )
    fill = torch.full_like(x, fill_value)
    return x * mask + fill * (1.0 - mask)


def mask_to_token_ids(
    mask: torch.Tensor, grid: Optional[Tuple[int, int, int]] = None
) -> torch.Tensor:
    """Convert (B, 1, D, H, W) visibility to (B, N) token visibility."""
    b, _, d, h, w = mask.shape
    return mask.reshape(b, d * h * w) > 0.5
