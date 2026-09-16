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
    volume_to_tokens,
    tokens_to_volume,
)


class DualPathDeepModule(nn.Module):
    """Dual-path conditioned deep encoder module.

    Inputs:
      - feature F_{s-1}: (B, C_f, D_f, H_f, W_f)
      - image X: (B, C_x, D, H, W) at full input resolution

    Outputs:
      - Z_J tokens (B, N, dim) and dense adapted map A(Z_J) matching F_s spatial
        size *after* the CNN stage that consumes F_{s-1}. The CNN stage itself
        is applied outside; this module produces a residual that is added to
        F_s_CNN.

    Interaction: feature tokens as Query, image tokens as Key/Value.
    """

    def __init__(
        self,
        feature_channels: int,
        image_channels: int,
        out_channels: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 4,
        image_stride: int = 4,
        feature_downsample: int = 1,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.feat_proj = FeatureProj(feature_channels, embed_dim, downsample=feature_downsample)
        self.img_embed = ImageEmbed3D(image_channels, embed_dim, target_stride=image_stride)
        self.pos_enc = SinCosPositionalEncoding3D(embed_dim)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        # Extra self-attn on fused tokens for residual context mixing.
        self.self_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(max(1, num_blocks // 2))]
        )
        self.adapter = DenseAdapter(embed_dim, out_channels)

    def forward(
        self,
        feat: torch.Tensor,
        image: torch.Tensor,
        return_tokens: bool = False,
    ):
        t_f, grid_f = self.feat_proj(feat)
        t_x, grid_x = self.img_embed(image)
        if grid_f != grid_x:
            # Align feature tokens spatially to the image-token grid.
            vol_f = tokens_to_volume(t_f, grid_f, self.embed_dim)
            vol_f = F.interpolate(vol_f, size=grid_x, mode="trilinear", align_corners=False)
            t_f, grid_f = volume_to_tokens(vol_f)

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


class JEPAFeatureHead(nn.Module):
    """Project dense deep features to JEPA target tokens (A2 uses this on CNN F_s)."""

    def __init__(self, in_channels: int, embed_dim: int = 256, downsample: int = 1):
        super().__init__()
        self.downsample = downsample
        layers = []
        if downsample > 1:
            layers.append(nn.AvgPool3d(downsample, downsample))
        layers.append(nn.Conv3d(in_channels, embed_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        feat = self.net(x)
        tokens, grid = volume_to_tokens(feat)
        return self.norm(tokens), grid


class JEPAPredictor(nn.Module):
    """Predict masked-position latent tokens from visible context tokens."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, num_blocks: int = 2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        visible: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """context_tokens: (B, N, C) online tokens (full grid, masked positions may be garbage).

        visible: (B, N) bool, True where context is visible.
        Returns predicted tokens for all positions; loss applies only on ~visible.
        """
        b, n, c = context_tokens.shape
        x = context_tokens.clone()
        if pos_embed is not None:
            x = x + pos_embed
        x = torch.where(visible.unsqueeze(-1), x, self.mask_token.expand(b, n, c))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
