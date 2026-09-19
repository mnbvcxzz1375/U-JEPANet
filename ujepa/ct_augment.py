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
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if torch.rand(1, generator=generator).item() > p:
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
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if torch.rand(1, generator=generator).item() > p:
        return x
    s = scale_range[0] + torch.rand(1, generator=generator).item() * (scale_range[1] - scale_range[0])
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


def _identity_theta(batch: int = 1, device=None, dtype=torch.float32) -> torch.Tensor:
    th = torch.eye(3, 4, dtype=dtype, device=device)
    return th.unsqueeze(0).expand(batch, -1, -1).contiguous()


def mild_affine(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    max_rotate_deg: float = 12.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    p: float = 0.5,
    return_theta: bool = False,
    generator: Optional[torch.Generator] = None,
):
    """Shared affine for image (+ optional label). No LR flip.

    return_theta: also return (B,3,4) affine_grid theta (output→input norm coords).
    Identity when no affine is applied.
    """

    def _rand() -> float:
        if generator is None:
            return torch.rand(1).item()
        return torch.rand(1, generator=generator).item()

    if _rand() > p:
        b = x.shape[0] if x.dim() == 5 else 1
        th0 = _identity_theta(b, x.device, x.dtype)
        if y is not None:
            return (x, y, th0) if return_theta else (x, y)
        return (x, th0) if return_theta else x

    if x.dim() == 4:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    b, c, d, h, w = x.shape
    import math

    def ang():
        return math.radians((_rand() * 2 - 1) * max_rotate_deg)

    a, bb, cc = ang(), ang(), ang()
    s = scale_range[0] + _rand() * (scale_range[1] - scale_range[0])
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(bb), math.sin(bb)
    cx, sx = math.cos(cc), math.sin(cc)
    R = torch.tensor(
        [
            [cb * cx, -cb * sx, sb],
            [sa * sb * cx + ca * sx, -sa * sb * sx + ca * cx, -ca * cb],
            [-ca * sb * cx + sa * sx, ca * sb * sx + ca * cb, ca * cb],
        ],
        dtype=x.dtype,
        device=x.device,
    ) * s
    theta = torch.eye(3, 4, dtype=x.dtype, device=x.device)
    theta[:3, :3] = R
    theta = theta.unsqueeze(0).expand(b, -1, -1).contiguous()
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
    if return_theta:
        if y is not None:
            return x_out, out_y, theta
        return x_out, theta
    return (x_out, out_y) if out_y is not None else x_out


def local_token_to_global_voxel(
    token_grid_zyx: Tuple[int, int, int],
    crop_origin: Sequence[int],
    full_shape: Sequence[int],
    patch_size: Sequence[int],
    theta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Local token centers → whole-volume voxel (z,y,x).

    If theta is None/identity: crop_origin + center offset in patch (no affine).
    Otherwise apply affine_grid theta (output→input on patch, xyz order, align_corners=False)
    then add crop_origin.
    """
    gd, gh, gw = [int(v) for v in token_grid_zyx]
    pd, ph, pw = [int(v) for v in patch_size]
    z0, y0, x0 = [int(v) for v in crop_origin]
    zs = (torch.arange(gd, dtype=torch.float32) + 0.5) * (pd / gd)
    ys = (torch.arange(gh, dtype=torch.float32) + 0.5) * (ph / gh)
    xs = (torch.arange(gw, dtype=torch.float32) + 0.5) * (pw / gw)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    local_vox = torch.stack([zz, yy, xx], dim=-1)  # zyx

    if theta is None:
        return local_vox + torch.tensor([z0, y0, x0], dtype=torch.float32)

    th = theta.detach().float()
    if th.dim() == 3:
        th = th[0]
    # local_vox is edge-based continuous center: voxel-index i has center i+0.5.
    # align_corners=False: u = 2*c/S - 1, c = (u+1)*0.5*S.
    u_z = local_vox[..., 0] / pd * 2 - 1
    u_y = local_vox[..., 1] / ph * 2 - 1
    u_x = local_vox[..., 2] / pw * 2 - 1
    xyz_out = torch.stack([u_x, u_y, u_z], dim=-1)
    R = th[:3, :3]
    t = th[:3, 3]
    xyz_in = torch.einsum("ij,...j->...i", R, xyz_out) + t
    z_in = (xyz_in[..., 2] + 1) * 0.5 * pd
    y_in = (xyz_in[..., 1] + 1) * 0.5 * ph
    x_in = (xyz_in[..., 0] + 1) * 0.5 * pw
    local_orig = torch.stack([z_in, y_in, x_in], dim=-1)
    return local_orig + torch.tensor([z0, y0, x0], dtype=torch.float32)


def token_coord_features(
    global_vox_zyx: torch.Tensor,
    full_shape: Sequence[int],
    patch_size: Sequence[int],
) -> torch.Tensor:
    """c_i = [p_i, s] with p in [0,1]^3 (z,y,x), s=patch/full."""
    D, H, W = [float(v) for v in full_shape]
    pd, ph, pw = [float(v) for v in patch_size]
    p = global_vox_zyx.clone()
    p[..., 0] = p[..., 0] / max(D, 1)
    p[..., 1] = p[..., 1] / max(H, 1)
    p[..., 2] = p[..., 2] / max(W, 1)
    s = torch.tensor([pd / max(D, 1), ph / max(H, 1), pw / max(W, 1)], dtype=p.dtype)
    s = s.view(*([1] * (p.dim() - 1)), 3).expand_as(p)
    return torch.cat([p, s], dim=-1)


def token_coord_features_batched(
    token_grid_zyx: Tuple[int, int, int],
    crop_origin: torch.Tensor,
    full_shape: torch.Tensor,
    patch_size: Sequence[int],
    theta: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Batched (B,N,6) coord features on `device`. P0: no first-sample expand.

    crop_origin: (B,3) long/float zyx
    full_shape: (B,3) long
    theta: (B,3,4) or None
    """
    gd, gh, gw = [int(v) for v in token_grid_zyx]
    pd, ph, pw = [int(v) for v in patch_size]
    if device is None:
        device = crop_origin.device
    B = crop_origin.shape[0]
    zs = (torch.arange(gd, device=device, dtype=torch.float32) + 0.5) * (pd / gd)
    ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * (ph / gh)
    xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * (pw / gw)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    local = torch.stack([zz, yy, xx], dim=-1).reshape(1, -1, 3)  # (1,N,3) zyx edge centers
    local = local.expand(B, -1, -1)

    # affine: u = 2c/S - 1, xyz order for affine_grid
    u_z = local[..., 0] / pd * 2 - 1
    u_y = local[..., 1] / ph * 2 - 1
    u_x = local[..., 2] / pw * 2 - 1
    xyz_out = torch.stack([u_x, u_y, u_z], dim=-1)  # (B,N,3)

    if theta is None:
        local_orig = local
    else:
        th = theta.to(device=device, dtype=torch.float32)
        if th.dim() == 2:
            th = th.unsqueeze(0)
        if th.dim() == 4:
            th = th[:, 0]
        R = th[:, :3, :3]  # (B,3,3)
        t = th[:, :3, 3]  # (B,3)
        # xyz_in[b,n,:] = R[b] @ xyz_out[b,n] + t[b]
        xyz_in = torch.einsum("bij,bnj->bni", R, xyz_out) + t.unsqueeze(1)
        z_in = (xyz_in[..., 2] + 1) * 0.5 * pd
        y_in = (xyz_in[..., 1] + 1) * 0.5 * ph
        x_in = (xyz_in[..., 0] + 1) * 0.5 * pw
        local_orig = torch.stack([z_in, y_in, x_in], dim=-1)

    origin = crop_origin.to(device=device, dtype=torch.float32).unsqueeze(1)  # (B,1,3)
    gvox = local_orig + origin  # (B,N,3)

    fs = full_shape.to(device=device, dtype=torch.float32)  # (B,3)
    p = gvox / fs.unsqueeze(1).clamp(min=1.0)
    s = torch.tensor([pd, ph, pw], device=device, dtype=torch.float32) / fs.clamp(min=1.0)
    s = s.unsqueeze(1).expand(B, gvox.shape[1], 3)
    return torch.cat([p, s], dim=-1)


def seg_augment_hu(
    x_hu: torch.Tensor,
    strength: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    x = random_window(
        x_hu,
        d_level=30.0 * strength,
        width_scale=(1.0 - 0.2 * strength, 1.0 + 0.2 * strength),
        generator=generator,
    )
    x = gaussian_noise(x, p=0.15 * strength, std=0.03 * strength, generator=generator)
    x = gaussian_blur(x, p=0.1 * strength, sigma=0.8, generator=generator)
    if strength > 0.5:
        x = low_resolution(x, p=0.15 * strength, generator=generator)
    return x


def jepa_weak_target(x_hu: torch.Tensor) -> torch.Tensor:
    return canonical_window(x_hu)


def jepa_strong_context(x_hu: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    x = random_window(x_hu, generator=generator)
    x = gaussian_noise(x, p=0.25, std=0.04, generator=generator)
    x = gaussian_blur(x, p=0.2, sigma=1.0, generator=generator)
    return x
