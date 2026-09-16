from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3D convs. InstanceNorm by default: batch is typically 1-2 for 3D CT,
    and the same online encoder sees clean CT (seg) and masked CT (JEPA).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        norm: str = "instance",
    ):
        super().__init__()

        def make_norm(c: int) -> nn.Module:
            if norm == "batch":
                return nn.BatchNorm3d(c)
            if norm == "group":
                groups = 8 if c % 8 == 0 else 1
                return nn.GroupNorm(groups, c)
            return nn.InstanceNorm3d(c, affine=True)

        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            make_norm(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            make_norm(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_conv(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        norm: str = "instance",
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels, dropout_p, norm=norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels1: int,
        in_channels2: int,
        out_channels: int,
        dropout_p: float = 0.0,
        trilinear: bool = True,
        norm: str = "instance",
    ):
        super().__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def volume_to_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """(B, C, D, H, W) -> (B, N, C), (D, H, W)."""
    b, c, d, h, w = x.shape
    tokens = x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
    return tokens, (d, h, w)


def tokens_to_volume(
    tokens: torch.Tensor, grid: Tuple[int, int, int], channels: int
) -> torch.Tensor:
    """(B, N, C) -> (B, C, D, H, W)."""
    d, h, w = grid
    b = tokens.shape[0]
    return tokens.reshape(b, d, h, w, channels).permute(0, 4, 1, 2, 3).contiguous()


class SinCosPositionalEncoding3D(nn.Module):
    """Fixed sin-cos encoding for (D, H, W) token grids.

    Works for any embed_dim >= 6: uses 6 * (embed_dim // 6) channels and
    zero-pads any remainder so the output is always (1, N, embed_dim).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        if embed_dim < 6:
            raise ValueError(f"embed_dim must be >= 6, got {embed_dim}")
        self.embed_dim = embed_dim
        self.core_dim = (embed_dim // 6) * 6

    @staticmethod
    def _pe_1d(length: int, dim_t: int, device: torch.device) -> torch.Tensor:
        if dim_t <= 0:
            return torch.zeros(length, 0, device=device)
        omega = torch.arange(dim_t, dtype=torch.float32, device=device)
        omega = 1.0 / (10000 ** (omega / float(dim_t)))
        pos = torch.arange(length, dtype=torch.float32, device=device)
        out = pos[:, None] * omega[None, :]  # (length, dim_t)
        return torch.cat([out.sin(), out.cos()], dim=-1)  # (length, 2*dim_t)

    def forward(self, grid: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        d, h, w = grid
        dim_t = self.core_dim // 6
        pe_d = self._pe_1d(d, dim_t, device)[:, None, None, :].expand(d, h, w, 2 * dim_t)
        pe_h = self._pe_1d(h, dim_t, device)[None, :, None, :].expand(d, h, w, 2 * dim_t)
        pe_w = self._pe_1d(w, dim_t, device)[None, None, :, :].expand(d, h, w, 2 * dim_t)
        pe = torch.cat([pe_d, pe_h, pe_w], dim=-1)
        if pe.shape[-1] < self.embed_dim:
            pad = torch.zeros(d, h, w, self.embed_dim - pe.shape[-1], device=device)
            pe = torch.cat([pe, pad], dim=-1)
        return pe.reshape(1, d * h * w, self.embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        if key is None:
            attn_out, _ = self.attn(h, h, h, need_weights=False)
        else:
            k = self.norm1(key) if key is not x else h
            v = value if value is not None else k
            if v is not key and value is None:
                v = k
            attn_out, _ = self.attn(h, k, v, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Query from feature tokens, key/value from image tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm_mlp(q))
        return q


class ImageEmbed3D(nn.Module):
    """Strided patch embedding — projects and downsamples in one conv.

    Avoids materializing a full-resolution embed_dim feature map before pooling
    (which OOMs on 128x128x96 with embed_dim=256).
    """

    def __init__(self, in_channels: int, embed_dim: int, target_stride: int):
        super().__init__()
        if target_stride < 1:
            raise ValueError("target_stride must be >= 1")
        self.target_stride = target_stride
        # kernel=stride gives non-overlapping patches (ViT-style 3D).
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=target_stride,
            stride=target_stride,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        feat = self.proj(x)
        tokens, grid = volume_to_tokens(feat)
        return self.norm(tokens), grid


class FeatureProj(nn.Module):
    """Project encoder feature to token dim and resample to a JEPA token grid.

    ``target_size`` is the JEPA grid (D', H', W'), independent of U-Net stage.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        target_size: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.target_size = target_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        feat = self.proj(x)
        if self.target_size is not None and tuple(feat.shape[-3:]) != tuple(self.target_size):
            feat = F.interpolate(feat, size=self.target_size, mode="trilinear", align_corners=False)
        tokens, grid = volume_to_tokens(feat)
        return self.norm(tokens), grid


class DenseAdapter(nn.Module):
    """Map token sequence back to dense volume channels for residual fusion."""

    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        vol = tokens_to_volume(self.norm(tokens), grid, self.embed_dim)
        return self.proj(vol)

    @property
    def embed_dim(self) -> int:
        return self.proj.in_channels
