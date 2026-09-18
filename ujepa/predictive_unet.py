from __future__ import annotations

"""V1.2 Corrected Predictive Bottleneck (post user review of 4b7971f).

P0-1: F2* must feed E3 (deep path), not only decoder skip.
P0-2: L_P on original F2; context NOT detached; target stop-grad → encoder
      receives predictive gradient.

Graph (locked):
  X → E0 → E1 → F2
       ├─ B_full(F2) → F2* → E3 → F3* → Decoder(F0,F1,F2*,F3*)
       └─ B_mask(F2) → L_P   (context grad, target sg)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dualpath_jepa import JEPAPredictor, tokens_from_volume
from .unet3d import UNet3D


def target_grid_384(spatial: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Explicit grid for WORD-style 128x128x96 stage-2 → 8x8x6=384 tokens.

    For other sizes: pick the /div grid closest to 384 tokens without exceeding 3*384.
    """
    d, h, w = spatial
    if (d, h, w) == (32, 32, 24):
        return (8, 8, 6)
    best = (max(1, d // 4), max(1, h // 4), max(1, w // 4))
    best_n = best[0] * best[1] * best[2]
    for div in (2, 4, 8, 16):
        g = (max(1, d // div), max(1, h // div), max(1, w // div))
        n = g[0] * g[1] * g[2]
        if n <= 384 * 3 and abs(n - 384) < abs(best_n - 384):
            best, best_n = g, n
    return best


class PredictiveBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        predictor_blocks: int = 2,
        mask_ratio: float = 0.4,
        use_residual: bool = True,
        target_tokens: int = 384,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.use_residual = use_residual
        self.target_tokens = int(target_tokens)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.predictor = JEPAPredictor(embed_dim=embed_dim, num_heads=num_heads, num_blocks=predictor_blocks)
        merge_in = embed_dim * (3 if use_residual else 2)
        self.merge = nn.Conv3d(merge_in, in_channels, kernel_size=1)

    def _pool(self, vol: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        return vol if tuple(vol.shape[-3:]) == tuple(grid) else F.adaptive_avg_pool3d(vol, grid)

    def _up(self, vol: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        return vol if tuple(vol.shape[-3:]) == tuple(size) else F.interpolate(
            vol, size=size, mode="trilinear", align_corners=False
        )

    def encode_tokens(self, f2: torch.Tensor, grid: Optional[Tuple[int, int, int]] = None):
        b, c, d, h, w = f2.shape
        if grid is None:
            grid = target_grid_384((d, h, w))
        vol = self.proj(f2)
        z_vol = self._pool(vol, grid)
        tokens = self.norm(tokens_from_volume(z_vol))
        return tokens, grid, (d, h, w)

    def full_context_forward(self, f2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Seg path: always P(Z, 1). Deterministic under eval()."""
        tokens, grid, spatial = self.encode_tokens(f2)
        b = tokens.shape[0]
        n = tokens.shape[1]
        vis_all = torch.ones(b, n, dtype=torch.bool, device=f2.device)
        pred = self.predictor(tokens, vis_all, grid)

        def to_vol(tok):
            gd, gh, gw = grid
            return tok.transpose(1, 2).reshape(b, self.embed_dim, gd, gh, gw)

        d, h, w = spatial
        z_b = self._up(to_vol(tokens), (d, h, w))
        zhat_b = self._up(to_vol(pred), (d, h, w))
        if self.use_residual:
            cat = torch.cat([z_b, zhat_b, z_b - zhat_b], dim=1)
        else:
            cat = torch.cat([z_b, zhat_b], dim=1)
        f2_star = f2 + self.merge(cat)
        aux = {"z": tokens, "pred_full": pred, "grid": grid}
        return f2_star, aux

    def masked_pred_loss(self, f2_base: torch.Tensor) -> torch.Tensor:
        """P0-2: run on ORIGINAL F2; context keeps grad; target stop-grad.

        ∇θ_E L_P ≠ 0 via visible context tokens → predictor.
        ∇θ_proj L_P ≠ 0 via context path.
        ∇θ_P L_P ≠ 0 via predictor.
        """
        if self.mask_ratio <= 0:
            return f2_base.new_zeros(())
        # context tokens: NO detach → encoder/proj receive gradient
        tokens, grid, _ = self.encode_tokens(f2_base)
        z_target = tokens.detach()
        b, n, _ = tokens.shape
        gd, gh, gw = grid
        n_mask = max(1, int(round(self.mask_ratio * n)))
        visible = torch.ones(b, n, dtype=torch.bool, device=f2_base.device)
        for i in range(b):
            idx = torch.randperm(n, device=f2_base.device)[:n_mask]
            visible[i, idx] = False
        pred = self.predictor(tokens, visible, grid)  # tokens = live context
        masked = ~visible
        if not masked.any():
            return f2_base.new_zeros(())
        diff = F.smooth_l1_loss(pred, z_target, reduction="none").mean(-1)
        return diff[masked].mean()


class PredictiveUNet(nn.Module):
    """P0-1: bottleneck sits ON the deep path — E3 consumes F2*."""

    def __init__(
        self,
        feature_chns: Tuple[int, ...] = (16, 32, 64, 128),
        class_num: int = 17,
        deep_stage: int = 2,
        embed_dim: int = 256,
        num_heads: int = 4,
        predictor_blocks: int = 2,
        mask_ratio: float = 0.4,
        use_residual: bool = True,
        target_tokens: int = 384,
        multiscale_pred: bool = True,
        norm: str = "instance",
        dropout: Optional[List[float]] = None,
    ):
        super().__init__()
        if deep_stage != 2:
            raise ValueError("V1.2 locks deep_stage=2 (E2 → bottleneck → E3)")
        self.cfg_deep_stage = deep_stage
        self.backbone = UNet3D(
            in_chns=1,
            feature_chns=list(feature_chns),
            dropout=dropout,
            class_num=class_num,
            multiscale_pred=multiscale_pred,
            norm=norm,
        )
        ch = list(feature_chns)[deep_stage]
        self.bottleneck = PredictiveBottleneck(
            in_channels=ch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            predictor_blocks=predictor_blocks,
            mask_ratio=mask_ratio,
            use_residual=use_residual,
            target_tokens=target_tokens,
        )
        self.last_aux: Dict[str, torch.Tensor] = {}
        self._last_pred_loss: Optional[torch.Tensor] = None
        self._last_f2_base: Optional[torch.Tensor] = None
        self._last_f2_star: Optional[torch.Tensor] = None
        self._last_f3: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        # Stage-by-stage encode so E3 sees F2*
        x0 = bb.in_conv(x)
        x1 = bb.down1(x0)
        f2 = bb.down2(x1)
        self._last_f2_base = f2
        f2_star, aux = self.bottleneck.full_context_forward(f2)
        self._last_f2_star = f2_star
        f3 = bb.down3(f2_star)  # P0-1: deep path uses predictive F2*
        self._last_f3 = f3
        feats = [x0, x1, f2_star, f3]
        if len(bb.ft_chns) == 5:
            f4 = bb.down4(f3)
            feats.append(f4)
        self.last_aux = aux
        if self.training:
            # P0-2: aux branch from ORIGINAL f2, encoder gets pred grad
            self._last_pred_loss = self.bottleneck.masked_pred_loss(f2)
        else:
            self._last_pred_loss = None
        return bb.decode(feats)

    @property
    def pred_loss(self) -> torch.Tensor:
        if self._last_pred_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._last_pred_loss
