from __future__ import annotations

"""CT-specific intensity augmentations. No LR flip (anatomy-preserving)."""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def ct_window(
    x_hu: torch.Tensor,
    level: float,
    width: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map HU to [0,1] via window: clip((hu - (L-W/2)) / W, 0, 1)."""
    lo = level - width / 2.0
    return ((x_hu - lo) / max(width, eps)).clamp(0.0, 1.0)


def random_window(
    x_hu: torch.Tensor,
    base_level: float = 40.0,
    base_width: float = 400.0,
    d_level: float = 30.0,
    width_scale: Tuple[float, float] = (0.8, 1.2),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    dl = (torch.rand(1, generator=generator).item() * 2 - 1) * d_level
    r = width_scale[0] + torch.rand(1, generator=generator).item() * (width_scale[1] - width_scale[0])
    return ct_window(x_hu, base_level + dl, base_width * r)


def canonical_window(
    x_hu: torch.Tensor,
    base_level: float = 40.0,
    base_width: float = 400.0,
) -> torch.Tensor:
    return ct_window(x_hu, base_level, base_width)


def gaussian_noise(
    x: torch.Tensor,
    p: float = 0.15,
    std: float = 0.03,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if torch.rand(1, generator=generator).item() > p:
        return x
    return (x + torch.randn_like(x) * std).clamp(0, 1)


def gaussian_blur(
    x: torch.Tensor,
    p: float = 0.1,
    sigma: float = 0.8,
) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return x
    squeeze = False
    if x.dim() == 4:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.dim() != 5:
        raise ValueError(f"gaussian_blur expects 4D/5D, got {tuple(x.shape)}")
    radius = max(1, int(round(3 * sigma)))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-0.5 * (k / sigma) ** 2)
    g = g / g.sum()
    b, ch, d, h, w = x.shape
    g1 = g.view(1, 1, -1, 1, 1).expand(ch, 1, -1, 1, 1)
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, radius, radius), mode="replicate"), g1, groups=ch)
    g2 = g.view(1, 1, 1, -1, 1).expand(ch, 1, 1, -1, 1)
    x = F.conv3d(F.pad(x, (0, 0, radius, radius, 0, 0), mode="replicate"), g2, groups=ch)
    g3 = g.view(1, 1, 1, 1, -1).expand(ch, 1, 1, 1, -1)
    x = F.conv3d(F.pad(x, (radius, radius, 0, 0, 0, 0), mode="replicate"), g3, groups=ch)
    x = x.clamp(0, 1)
    return x.squeeze(0) if squeeze else x


def low_resolution(
    x: torch.Tensor,
    p: float = 0.15,
    scale_range: Tuple[float, float] = (0.5, 1.0),
) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return x
    s = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
    if s >= 0.99:
        return x
    squeeze = False
    if x.dim() == 4:
        x = x.unsqueeze(0)
        squeeze = True
    size = [max(1, int(round(dim * s))) for dim in x.shape[-3:]]
    y = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    y = F.interpolate(y, size=x.shape[-3:], mode="trilinear", align_corners=False).clamp(0, 1)
    return y.squeeze(0) if squeeze else y


def mild_affine(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    max_rotate_deg: float = 12.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    p: float = 0.5,
):
    """Shared affine for image (+ optional label). No LR flip.

    Image: bilinear; label: nearest.
    """
    if torch.rand(1).item() > p:
        return (x, y) if y is not None else x
    if x.dim() == 4:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    b, c, d, h, w = x.shape
    # sample angles around D/H/W axes (small)
    import math

    def ang():
        return math.radians((torch.rand(1).item() * 2 - 1) * max_rotate_deg)

    a, bb, cc = ang(), ang(), ang()
    s = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(bb), math.sin(bb)
    cx, sx = math.cos(cc), math.sin(cc)
    # R = Rz @ Ry @ Rx (simplified composition), scaled
    R = torch.tensor(
        [
            [cb * cx, -cb * sx, sb],
            [sa * sb * cx + ca * sx, -sa * sb * sx + ca * cx, -sa * cb],
            [-ca * sb * cx + sa * sx, ca * sb * sx + sa * cx, ca * cb],
        ],
        dtype=x.dtype,
        device=x.device,
    ) * s
    theta = torch.eye(3, 4, dtype=x.dtype, device=x.device)
    theta[:3, :3] = R
    theta = theta.unsqueeze(0).expand(b, -1, -1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x_t = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if y is not None:
        if y.dim() == 3:
            y = y.unsqueeze(0).unsqueeze(0)
        y_t = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="border", align_corners=False)
        y_t = y_t.squeeze(0).squeeze(0).long() if squeeze else y_t.squeeze(1).long()
        out_y = y_t
    else:
        out_y = None
    x_out = x_t.squeeze(0) if squeeze else x_t
    return (x_out, out_y) if out_y is not None else x_out


def seg_augment_hu(
    x_hu: torch.Tensor,
    strength: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Full-volume or patch in HU → windowed [0,1] + mild intensity noise."""
    x = random_window(
        x_hu,
        d_level=30.0 * strength,
        width_scale=(1.0 - 0.2 * strength, 1.0 + 0.2 * strength),
        generator=generator,
    )
    x = gaussian_noise(x, p=0.15 * strength, std=0.03 * strength, generator=generator)
    x = gaussian_blur(x, p=0.1 * strength, sigma=0.8)
    if strength > 0.5:
        x = low_resolution(x, p=0.15 * strength)
    return x


def jepa_weak_target(x_hu: torch.Tensor) -> torch.Tensor:
    return canonical_window(x_hu)


def jepa_strong_context(x_hu: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    x = random_window(x_hu, generator=generator)
    x = gaussian_noise(x, p=0.25, std=0.04, generator=generator)
    x = gaussian_blur(x, p=0.2, sigma=1.0)
    return x
