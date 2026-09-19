"""V3.2 Aligned Global Innovation U-Net.

Hard-aligned global sampling (GridSample) + patient innovation + RMS gate.

I = Z_hat^G - Z_hat^A   # patient global minus PE-only atlas at same p_i
F2* = F2_R1 + α * RMS-norm(A(I))
No Z_L residual in fusion (cannot degrade into local adapter).

H0: Z_G = PE only
H1: Z_G = E_G(X_G) + PE
Both: same sampler / predictor / RMS gate / α0.

Mechanism logs (train): alpha, S_pre=||ZhatG-ZhatA||/||ZhatG||,
S_fuse=||α Δ~||/||F2_R1||, S_shuffle on real case swap when provided.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ct_augment import token_coord_features_batched
from .predictive_unet import PredictiveBottleneck, target_grid_384
from .unet3d import UNet3D


class GlobalFeatureMap(nn.Module):
    """Dense global CNN → feature map G (B,C,Dg,Hg,Wg), not bag-of-tokens."""

    def __init__(self, embed_dim: int = 256, map_size: Tuple[int, int, int] = (32, 32, 32)):
        super().__init__()
        self.map_size = map_size
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj = nn.Conv3d(64, embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_g: torch.Tensor) -> torch.Tensor:
        if x_g.dim() == 4:
            x_g = x_g.unsqueeze(1)
        h = self.proj(self.stem(x_g))
        h = F.interpolate(h, size=self.map_size, mode="trilinear", align_corners=False)
        return h  # (B,C,Dg,Hg,Wg)


def gridsample_global(
    G: torch.Tensor,
    grid_zyx: Tuple[int, int, int],
    crop_origin: torch.Tensor,
    full_shape: torch.Tensor,
    patch_size: Sequence[int],
    theta: Optional[torch.Tensor] = None,
    align_corners: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample G at local-token whole-volume positions p_i (batched).

    Returns:
      g_fine (B,N,C) sampled at p_i
      norm_coords (B, D',H',W', 3) in grid_sample order (x,y,z) ∈ [-1,1]
    """
    B = crop_origin.shape[0]
    device = crop_origin.device
    gd, gh, gw = grid_zyx
    pd, ph, pw = [int(v) for v in patch_size]
    # token centers in patch edge coords
    zs = (torch.arange(gd, device=device, dtype=torch.float32) + 0.5) * (pd / gd)
    ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * (ph / gh)
    xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * (pw / gw)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    local = torch.stack([zz, yy, xx], dim=-1)  # (gd,gh,gw,3) zyx
    local = local.view(1, -1, 3).expand(B, -1, -1)

    if theta is None:
        local_orig = local
    else:
        th = theta.to(device=device, dtype=torch.float32)
        if th.dim() == 2:
            th = th.unsqueeze(0)
        if th.dim() == 4:
            th = th[:, 0]
        u_z = local[..., 0] / pd * 2 - 1
        u_y = local[..., 1] / ph * 2 - 1
        u_x = local[..., 2] / pw * 2 - 1
        xyz_out = torch.stack([u_x, u_y, u_z], dim=-1)
        R = th[:, :3, :3]
        t = th[:, :3, 3]
        xyz_in = torch.einsum("bij,bnj->bni", R, xyz_out) + t.unsqueeze(1)
        local_orig = torch.stack([
            (xyz_in[..., 2] + 1) * 0.5 * pd,
            (xyz_in[..., 1] + 1) * 0.5 * ph,
            (xyz_in[..., 0] + 1) * 0.5 * pw,
        ], dim=-1)

    origin = crop_origin.to(device=device, dtype=torch.float32).unsqueeze(1)
    gvox = local_orig + origin  # (B,N,3) zyx whole-volume voxel
    fs = full_shape.to(device=device, dtype=torch.float32).unsqueeze(1)  # (B,1,3)
    # normalized to [-1,1] for the **patient's own volume frame** (edge-centered)
    # p = 2*c/size - 1
    p_zyx = gvox / fs.clamp(min=1.0) * 2.0 - 1.0
    # grid_sample 5D grid: (B, Dout, Hout, Wout, 3) with last dim (x,y,z)
    grid = torch.stack([p_zyx[..., 2], p_zyx[..., 1], p_zyx[..., 0]], dim=-1)
    grid = grid.view(B, gd, gh, gw, 3)
    g_fine = F.grid_sample(G, grid, mode="bilinear", padding_mode="border", align_corners=align_corners)
    # (B,C,gd,gh,gw) → (B,N,C)
    C = g_fine.shape[1]
    g_tok = g_fine.reshape(B, C, -1).transpose(1, 2)
    return g_tok, grid


class PEAtlas(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, cfeat: torch.Tensor) -> torch.Tensor:
        return self.mlp(cfeat)


class ExpectationHead(nn.Module):
    """Z_hat = MLP([q, g_fine, g_coarse, g_pooled]) — used for both A and G arms."""

    def __init__(self, embed_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, q, g_fine, g_coarse, g_pool):
        if g_fine is None:
            # atlas path: no image global — use zeros for global slots
            z = torch.zeros_like(q)
            g_fine = g_coarse = g_pool = z
        g_pool = g_pool.unsqueeze(1).expand_as(q) if g_pool.dim() == 2 else g_pool
        return self.mlp(torch.cat([q, g_fine, g_coarse, g_pool], dim=-1))


def rms_norm_delta(delta: torch.Tensor, ref: torch.Tensor, stop_grad_scale: bool = True) -> torch.Tensor:
    """Per-sample RMS: tildeΔ_b = Δ_b * RMS(ref_b) / (RMS(Δ_b)+eps).

    P0: scale must not mix patients in the batch; α=0.04 means ~4% RMS **per case**.
    """
    # delta, ref: (B,C,D,H,W)
    dims = tuple(range(1, delta.dim()))
    rms_d = delta.pow(2).mean(dim=dims).sqrt().view(-1, *([1] * (delta.dim() - 1)))
    rms_r = ref.pow(2).mean(dim=dims).sqrt().view(-1, *([1] * (ref.dim() - 1)))
    if stop_grad_scale:
        scale = (rms_r / (rms_d + 1e-6)).detach()
    else:
        scale = rms_r / (rms_d + 1e-6)
    return delta * scale


class V32UNet(nn.Module):
    """arm: H0 PE-only global innovation | H1 patient global innovation.

    R1 local bottleneck always on. Fusion only sees I = ZhatG - ZhatA.
    """

    def __init__(
        self,
        arm: str = "H1",
        feature_chns: Sequence[int] = (16, 32, 64, 128),
        class_num: int = 17,
        embed_dim: int = 256,
        num_heads: int = 4,
        predictor_blocks: int = 2,
        mask_ratio: float = 0.4,
        target_tokens: int = 384,
        alpha_init: float = 0.04,
        use_image_global: bool = True,
        multiscale_pred: bool = True,
        norm: str = "instance",
        global_map_size: Tuple[int, int, int] = (32, 32, 32),
    ):
        super().__init__()
        self.arm = arm.upper()
        if self.arm not in ("H0", "H1", "B1", "R1"):
            raise ValueError(arm)
        self.embed_dim = embed_dim
        self.target_tokens = target_tokens
        self.use_image_global = bool(use_image_global) and self.arm == "H1"
        self.backbone = UNet3D(
            in_chns=1,
            feature_chns=list(feature_chns),
            class_num=class_num,
            multiscale_pred=multiscale_pred,
            norm=norm,
        )
        ch = list(feature_chns)[2]
        self.local_bn = PredictiveBottleneck(
            in_channels=ch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            predictor_blocks=predictor_blocks,
            mask_ratio=mask_ratio,
            use_residual=True,
            target_tokens=target_tokens,
        )
        self.is_r1_only = self.arm in ("R1", "B1")
        if not self.is_r1_only:
            self.global_map = GlobalFeatureMap(embed_dim, global_map_size)
            self.atlas = PEAtlas(embed_dim)
            # Single shared P for causal I = P(q,Z_G) - P(q,Z_PE)
            self.exp_head = ExpectationHead(embed_dim)
            # P1: bias=False so I=0 ⇒ Δ=0 ⇒ H0 ≡ R1 (when weights match)
            self.innov_merge = nn.Conv3d(embed_dim, ch, kernel_size=1, bias=False)
            self.alpha_g = nn.Parameter(torch.tensor(float(alpha_init)))
            self.local_tok_proj = nn.Conv3d(ch, embed_dim, 1)
            self.local_tok_norm = nn.LayerNorm(embed_dim)

        self.last_aux: Dict[str, torch.Tensor] = {}
        self.mech: Dict[str, float] = {}

    def _local_tokens(self, f2: torch.Tensor):
        b, c, d, h, w = f2.shape
        grid = target_grid_384((d, h, w))
        vol = self.local_tok_proj(f2)
        if tuple(vol.shape[-3:]) != tuple(grid):
            vol = F.adaptive_avg_pool3d(vol, grid)
        tok = self.local_tok_norm(self._tokens(vol))
        return tok, grid, (d, h, w)

    @staticmethod
    def _tokens(vol: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = vol.shape
        return vol.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)

    def forward(
        self,
        x: torch.Tensor,
        global_image: Optional[torch.Tensor] = None,
        crop_origin: Optional[torch.Tensor] = None,
        full_shape: Optional[torch.Tensor] = None,
        affine_theta: Optional[torch.Tensor] = None,
        patch_size: Tuple[int, int, int] = (128, 128, 96),
        shuffle_global: bool = False,
        shuffle_global_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bb = self.backbone
        x0 = bb.in_conv(x)
        x1 = bb.down1(x0)
        f2 = bb.down2(x1)
        f2_r1, aux = self.local_bn.full_context_forward(f2)
        self.last_aux = aux
        self.mech = {"alpha": 0.0, "S_pre": 0.0, "S_fuse": 0.0, "S_shuffle": 0.0}
        if self.is_r1_only:
            f3 = bb.down3(f2_r1)
            feats = [x0, x1, f2_r1, f3]
            if len(bb.ft_chns) == 5:
                feats.append(bb.down4(f3))
            return bb.decode(feats)

        b = f2.shape[0]
        device = f2.device
        z_l, grid, spatial = self._local_tokens(f2)
        if crop_origin is None:
            crop_origin = torch.zeros(b, 3, dtype=torch.long, device=device)
        if full_shape is None:
            full_shape = torch.tensor([[*patch_size]] * b, dtype=torch.long, device=device)
        if affine_theta is None:
            affine_theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(b, -1, -1)
        cfeat = token_coord_features_batched(
            grid, crop_origin, full_shape, tuple(patch_size), affine_theta, device=device
        )
        q = self.atlas(cfeat)
        # PE-only global slots (Z_PE): zeros in image feature channels
        z_pe = torch.zeros(b, self.embed_dim, *self.global_map.map_size, device=device)
        g_pe, _ = gridsample_global(z_pe, grid, crop_origin, full_shape, tuple(patch_size), affine_theta)
        # P1: coarse branch also GridSample at same p_i (not flatten+1D interp)
        z_pe_coarse = F.adaptive_avg_pool3d(
            z_pe, (max(1, self.global_map.map_size[0] // 2),
                   max(1, self.global_map.map_size[1] // 2),
                   max(1, self.global_map.map_size[2] // 2))
        )
        g_pe_c, _ = gridsample_global(z_pe_coarse, grid, crop_origin, full_shape, tuple(patch_size), affine_theta)
        g_pe_pool = z_pe.mean(dim=(2, 3, 4))
        # Patient image global (H1) or same PE slots (H0)
        if self.use_image_global and global_image is not None:
            G = self.global_map(global_image)
            if shuffle_global:
                if shuffle_global_images is not None:
                    G = self.global_map(shuffle_global_images)
                elif b > 1:
                    G = torch.roll(G, shifts=1, dims=0)
        else:
            G = z_pe
        g_fine, _ = gridsample_global(G, grid, crop_origin, full_shape, tuple(patch_size), affine_theta)
        G_coarse = F.adaptive_avg_pool3d(
            G, (max(1, self.global_map.map_size[0] // 2),
                max(1, self.global_map.map_size[1] // 2),
                max(1, self.global_map.map_size[2] // 2))
        )
        g_c, _ = gridsample_global(G_coarse, grid, crop_origin, full_shape, tuple(patch_size), affine_theta)
        g_pool = G.mean(dim=(2, 3, 4))

        # Shared P: I = P(q, Z_G) - P(q, Z_PE)
        zhat_a = self.exp_head(q, g_pe, g_pe_c, g_pe_pool)
        zhat_g = self.exp_head(q, g_fine, g_c, g_pool)
        I = zhat_g - zhat_a
        self.mech["S_pre"] = float((zhat_g - zhat_a).norm() / (zhat_g.norm() + 1e-6))

        d, h, w = spatial
        gd, gh, gw = grid
        I_v = I.transpose(1, 2).reshape(b, self.embed_dim, gd, gh, gw)
        I_v = F.interpolate(I_v, size=(d, h, w), mode="trilinear", align_corners=False)
        delta = self.innov_merge(I_v)
        delta_t = rms_norm_delta(delta, f2_r1.detach())
        alpha = self.alpha_g
        f2_star = f2_r1 + alpha * delta_t
        self.mech["alpha"] = float(alpha.detach())
        self.mech["S_fuse"] = float((alpha * delta_t).norm() / (f2_r1.norm() + 1e-6))

        f3 = bb.down3(f2_star)
        feats = [x0, x1, f2_star, f3]
        if len(bb.ft_chns) == 5:
            feats.append(bb.down4(f3))
        return bb.decode(feats)

    @property
    def alpha_g_value(self) -> float:
        return float(self.alpha_g.detach()) if hasattr(self, "alpha_g") and self.alpha_g is not None else 0.0

    @property
    def mech_dict(self) -> Dict[str, float]:
        return dict(self.mech)
