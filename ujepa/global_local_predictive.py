"""Global-to-Local Anatomy Predictive U-Net (V3.1 GLP) — integration-fixed.

P0 fixes after 2829c2a review:
- batched per-sample coords (no crop_origin[0] expand)
- CUDA-safe coordinate tensors
- dataset affine_theta → (3,4); global_image → (1,D,H,W)
- GLP whole-volume evaluator with real crop origins + whole CT
- global tokens get whole-volume positional encoding
- isolated DataLoader generator
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ct_augment import token_coord_features_batched
from .dualpath_jepa import tokens_from_volume
from .predictive_unet import PredictiveBottleneck, target_grid_384
from .unet3d import UNet3D


class GlobalStem(nn.Module):
    """Global tokens = optional image features + whole-volume PE.

    G0: position_only() → PE-only tokens (no patient image content)
    G1/G2: forward(X_G) → E_G(X_G) + PE
    Same module always instantiated so G0/G1 init RNG order matches.
    """

    def __init__(self, embed_dim: int = 256, grid: Tuple[int, int, int] = (8, 8, 8)):
        super().__init__()
        self.grid = grid
        self.embed_dim = embed_dim
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
        self.pe_mlp = nn.Sequential(nn.Linear(3, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        gd, gh, gw = grid
        zs = (torch.arange(gd, dtype=torch.float32) + 0.5) / gd
        ys = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
        xs = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        pe = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        self.register_buffer("pe_grid", pe, persistent=False)

    def _pe(self, batch: int, device, dtype) -> torch.Tensor:
        pe = self.pe_mlp(self.pe_grid.to(device=device, dtype=dtype))
        return pe.unsqueeze(0).expand(batch, -1, -1)

    def position_only(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        """G0 global state: PE tokens only (atlas-like spatial code, no CT content)."""
        return self._pe(batch, device, dtype)

    def forward(self, x_g: torch.Tensor) -> torch.Tensor:
        if x_g.dim() == 4:
            x_g = x_g.unsqueeze(1)
        h = self.stem(x_g)
        h = F.adaptive_avg_pool3d(h, self.grid)
        h = self.proj(h)
        tok = tokens_from_volume(h)
        pe = self._pe(tok.shape[0], tok.device, tok.dtype)
        return self.norm(tok + pe)


class CoordAtlas(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, coord_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(coord_feat)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(q), kv, kv, need_weights=False)
        q = q + h
        q = q + self.ff(self.norm2(q))
        return q


class GlobalToLocalPredictor(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttnBlock(embed_dim, num_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q: torch.Tensor, z_g: Optional[torch.Tensor]) -> torch.Tensor:
        if z_g is None:
            return self.norm(q)
        x = q
        for blk in self.blocks:
            x = blk(x, z_g)
        return self.norm(x)


class GLPUNet(nn.Module):
    def __init__(
        self,
        arm: str = "R1",
        feature_chns: Sequence[int] = (16, 32, 64, 128),
        class_num: int = 17,
        embed_dim: int = 256,
        num_heads: int = 4,
        predictor_blocks: int = 2,
        mask_ratio: float = 0.4,
        use_residual: bool = True,
        target_tokens: int = 384,
        global_grid: Tuple[int, int, int] = (8, 8, 8),
        lambda_gl: float = 0.3,
        multiscale_pred: bool = True,
        norm: str = "instance",
    ):
        super().__init__()
        self.arm = arm.upper()
        if self.arm not in ("R1", "G0", "G1", "G2"):
            raise ValueError(arm)
        self.embed_dim = embed_dim
        self.lambda_gl = lambda_gl
        self.target_tokens = target_tokens
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
            use_residual=use_residual,
            target_tokens=target_tokens,
        )
        self.use_coord = self.arm != "R1"
        # Causal control: G0/G1/G2 ALL instantiate GlobalStem (same RNG/init order).
        # Only difference: whether Z_G includes patient image content.
        self.use_global_image = self.arm in ("G1", "G2")  # content path
        self.use_gl_loss = self.arm == "G2"
        self.coord_atlas = CoordAtlas(embed_dim) if self.use_coord else None
        self.global_stem = GlobalStem(embed_dim, global_grid) if self.use_coord else None
        self.gl_pred = GlobalToLocalPredictor(embed_dim, num_heads, n_blocks=2) if self.use_coord else None
        merge_in = embed_dim * 2 if self.use_coord else 0
        self.gl_merge = nn.Conv3d(merge_in, ch, kernel_size=1) if self.use_coord else None
        self.alpha_g = nn.Parameter(torch.zeros(())) if self.use_coord else None
        self.local_tok_proj = nn.Conv3d(ch, embed_dim, 1) if self.use_coord else None
        self.local_tok_norm = nn.LayerNorm(embed_dim) if self.use_coord else None

        self.last_aux: Dict[str, torch.Tensor] = {}
        self._last_gl_loss: Optional[torch.Tensor] = None
        self.last_coord_feat: Optional[torch.Tensor] = None
        self.last_global_shuffled = False

    def _local_tokens(self, f2: torch.Tensor):
        b, c, d, h, w = f2.shape
        grid = target_grid_384((d, h, w))
        vol = self.local_tok_proj(f2)
        if tuple(vol.shape[-3:]) != tuple(grid):
            vol = F.adaptive_avg_pool3d(vol, grid)
        tok = self.local_tok_norm(tokens_from_volume(vol))
        return tok, grid, (d, h, w)

    def forward(
        self,
        x: torch.Tensor,
        global_image: Optional[torch.Tensor] = None,
        crop_origin: Optional[torch.Tensor] = None,
        full_shape: Optional[torch.Tensor] = None,
        affine_theta: Optional[torch.Tensor] = None,
        patch_size: Tuple[int, int, int] = (128, 128, 96),
        shuffle_global: bool = False,
        shuffle_global_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bb = self.backbone
        x0 = bb.in_conv(x)
        x1 = bb.down1(x0)
        f2 = bb.down2(x1)
        f2_r1, aux_local = self.local_bn.full_context_forward(f2)

        self._last_gl_loss = None
        self.last_global_shuffled = bool(shuffle_global)
        f2_star = f2_r1

        if self.use_coord:
            b = f2.shape[0]
            device = f2.device
            z_l, grid, spatial = self._local_tokens(f2)
            if crop_origin is None:
                crop_origin = torch.zeros(b, 3, dtype=torch.long, device=device)
            if full_shape is None:
                full_shape = torch.tensor([[*patch_size]] * b, dtype=torch.long, device=device)
            if affine_theta is None:
                affine_theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(b, -1, -1)
            # P0: batched coords, CUDA-safe
            cfeat = token_coord_features_batched(
                token_grid_zyx=grid,
                crop_origin=crop_origin,
                full_shape=full_shape,
                patch_size=tuple(patch_size),
                theta=affine_theta,
                device=device,
            )  # (B,N,6)
            self.last_coord_feat = cfeat
            q = self.coord_atlas(cfeat)

            z_g = None
            if self.use_coord and self.global_stem is not None:
                # Causal G0 vs G1: same predictor path; only Z_G content differs.
                if self.use_global_image and global_image is not None:
                    z_g = self.global_stem(global_image)
                    if shuffle_global:
                        if shuffle_global_ids is not None:
                            z_g = z_g[shuffle_global_ids]
                        elif b > 1:
                            z_g = torch.roll(z_g, shifts=1, dims=0)
                else:
                    # G0 (or G1/G2 without image at train-step edge): PE-only tokens
                    z_g = self.global_stem.position_only(b, device, dtype=cfeat.dtype)
            zhat = self.gl_pred(q, z_g)
            r_gl = z_l - zhat
            d, h, w = spatial
            gd, gh, gw = grid
            zhat_v = zhat.transpose(1, 2).reshape(b, self.embed_dim, gd, gh, gw)
            r_v = r_gl.transpose(1, 2).reshape(b, self.embed_dim, gd, gh, gw)
            zhat_v = F.interpolate(zhat_v, size=(d, h, w), mode="trilinear", align_corners=False)
            r_v = F.interpolate(r_v, size=(d, h, w), mode="trilinear", align_corners=False)
            delta = self.gl_merge(torch.cat([zhat_v, r_v], dim=1))
            f2_star = f2_r1 + self.alpha_g * delta

            if self.training and self.use_gl_loss:
                self._last_gl_loss = F.smooth_l1_loss(zhat, z_l.detach(), reduction="none").mean(-1).mean()

        f3 = bb.down3(f2_star)
        feats = [x0, x1, f2_star, f3]
        if len(bb.ft_chns) == 5:
            feats.append(bb.down4(f3))
        self.last_aux = aux_local
        return bb.decode(feats)

    @property
    def gl_loss(self) -> torch.Tensor:
        if self._last_gl_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._last_gl_loss

    @property
    def alpha_g_value(self) -> float:
        return float(self.alpha_g.detach()) if self.alpha_g is not None else 0.0
