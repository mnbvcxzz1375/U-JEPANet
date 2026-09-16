from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    CrossAttentionBlock,
    DenseAdapter,
    FeatureProj,
    ImageEmbed3D,
    SinCosPositionalEncoding3D,
    TransformerBlock,
    tokens_to_volume,
)


class DualPathDeepModule(nn.Module):
    """Dual-path conditioned deep encoder at a **low-res JEPA token grid**.

    Feature tokens (from F_{s-1}) attend to image tokens (strided patch embed
    of X). Output residual is adapted back to the CNN stage spatial size.

    JEPA token count is controlled by ``jepa_token_stride`` so attention stays
    in the ~256-1024 range for WORD-scale patches.
    """

    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        out_channels: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 4,
        jepa_token_stride: int = 16,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.jepa_token_stride = jepa_token_stride
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        # Target grid is computed at runtime from image spatial size / stride.
        self.feat_proj = FeatureProj(feature_channels, embed_dim, target_size=None)
        self.img_embed = ImageEmbed3D(image_channels, embed_dim, target_stride=jepa_token_stride)
        self.pos_enc = SinCosPositionalEncoding3D(embed_dim)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.self_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(max(1, num_blocks // 2))]
        )
        self.adapter = DenseAdapter(embed_dim, out_channels)
        # 1x1 upsample of residual volume to CNN stage size is done by caller
        # or adapter via interpolate — keep adapter channel-only.

    def expected_token_grid(self, image_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        s = self.jepa_token_stride
        return tuple(max(1, int(dim // s)) for dim in image_size)  # type: ignore[return-value]

    def forward(
        self,
        feat: torch.Tensor,
        image: torch.Tensor,
        return_tokens: bool = False,
    ):
        grid_x = self.expected_token_grid(tuple(image.shape[-3:]))
        # Rebuild feature proj target each call so stride stays consistent.
        t_x, grid_x = self.img_embed(image)
        # Project F_{s-1} onto the same JEPA grid.
        vol_f = self.feat_proj.proj(feat)
        if tuple(vol_f.shape[-3:]) != grid_x:
            vol_f = F.interpolate(vol_f, size=grid_x, mode="trilinear", align_corners=False)
        t_f = self.feat_proj.norm(tokens_from_volume(vol_f))

        pe = self.pos_enc(grid_x, t_f.device)
        q = t_f + pe
        kv = t_x + pe
        for blk in self.blocks:
            q = blk(q, kv)
        for blk in self.self_blocks:
            q = blk(q)

        residual = self.adapter(q, grid_x)
        if return_tokens:
            return residual, q, grid_x
        return residual


def tokens_from_volume(x: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)


class JEPAFeatureHead(nn.Module):
    """Project dense deep features to JEPA tokens on a low-res grid (A2)."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        jepa_token_stride: int = 16,
    ):
        super().__init__()
        self.jepa_token_stride = jepa_token_stride
        # Strided conv: cheap downsample + project without full-res embed map.
        k = max(1, jepa_token_stride // 4)  # if feat already at 1/4, further /4 → 1/16
        # Caller passes the spatial downsample factor from input → this feature.
        self.extra_stride = None  # resolved at forward from target grid
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, target_grid: Tuple[int, int, int]):
        feat = self.proj(x)
        if tuple(feat.shape[-3:]) != tuple(target_grid):
            feat = F.interpolate(feat, size=target_grid, mode="trilinear", align_corners=False)
        tokens = tokens_from_volume(feat)
        return self.norm(tokens), tuple(target_grid)


class JEPAPredictor(nn.Module):
    """Predict masked-position latents. **Always adds 3D target positional
    encoding** so masked queries are not permutation-symmetric.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, num_blocks: int = 2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_enc = SinCosPositionalEncoding3D(embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        visible: torch.Tensor,
        grid: Tuple[int, int, int],
    ) -> torch.Tensor:
        """context_tokens: (B, N, C) online tokens on the JEPA grid.

        visible: (B, N) bool. grid: (D', H', W') matching N.
        """
        b, n, c = context_tokens.shape
        pe = self.pos_enc(grid, context_tokens.device)  # (1, N, C)
        ctx = context_tokens + pe
        mask_q = self.mask_token.expand(b, n, c) + pe
        x = torch.where(visible.unsqueeze(-1), ctx, mask_q)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
