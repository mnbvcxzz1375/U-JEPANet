from __future__ import annotations

"""V1.1 Corrected Predictive-Residual bottleneck.

Fixes vs V1:
- Segmentation forward is ALWAYS full-context (train == inference semantics).
- `model.eval()` automatically disables stochastic masking (no train_mode arg needed).
- Predictive loss uses a separate masked aux branch, loss only on masked tokens.
- Tokens are pooled to a low-res grid (default 8x8x6) before attention.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dualpath_jepa import JEPAPredictor, tokens_from_volume
from .unet3d import UNet3D


def _target_grid(spatial: Tuple[int, int, int], target_tokens: int = 384) -> Tuple[int, int, int]:
    """Pick a D,H,W grid with ~target_tokens cells, preferring /4 then /8."""
    d, h, w = spatial
    for div in (2, 4, 8, 16):
        gd, gh, gw = max(1, d // div), max(1, h // div), max(1, w // div)
        if gd * gh * gw <= max(target_tokens * 2, 64) and gd * gh * gw >= max(32, target_tokens // 8):
            # refine toward target_tokens
            pass
        if gd * gh * gw <= target_tokens * 3:
            return gd, gh, gw
    return max(1, d // 8), max(1, h // 8), max(1, w // 8)


class PredictiveBottleneck(nn.Module):
    """Pool F_s -> Z (low-res tokens) -> full-context predict -> merge -> F_s*."""

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
        self.last_aux: Dict[str, torch.Tensor] = {}

    def _pool_to_grid(self, vol: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        if tuple(vol.shape[-3:]) == tuple(grid):
            return vol
        return F.adaptive_avg_pool3d(vol, grid)

    def _upsample_to(self, vol: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        if tuple(vol.shape[-3:]) == tuple(size):
            return vol
        return F.interpolate(vol, size=size, mode="trilinear", align_corners=False)

    def _sample_visible(self, b: int, grid: Tuple[int, int, int], device) -> torch.Tensor:
        d, h, w = grid
        n = d * h * w
        n_mask = max(1, int(round(self.mask_ratio * n)))
        visible = torch.ones(b, n, dtype=torch.bool, device=device)
        for i in range(b):
            idx = torch.randperm(n, device=device)[:n_mask]
            visible[i, idx] = False
        return visible.view(b, d, h, w)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Always full-context for the seg path. Deterministic under eval()."""
        b, c, d, h, w = feat.shape
        grid = _target_grid((d, h, w), self.target_tokens)
        vol = self.proj(feat)
        z_vol = self._pool_to_grid(vol, grid)  # (B,E,gd,gh,gw)
        tokens = self.norm(tokens_from_volume(z_vol))  # (B,N,E)
        n = tokens.shape[1]
        # Full context: all tokens visible — same path for train and infer
        vis_all = torch.ones(b, n, dtype=torch.bool, device=feat.device)
        pred_tokens = self.predictor(tokens, vis_all, grid)

        def to_vol(tok: torch.Tensor) -> torch.Tensor:
            gd, gh, gw = grid
            return tok.transpose(1, 2).reshape(b, self.embed_dim, gd, gh, gw)

        z_back = self._upsample_to(to_vol(tokens), (d, h, w))
        zhat_back = self._upsample_to(to_vol(pred_tokens), (d, h, w))
        if self.use_residual:
            r_back = z_back - zhat_back
            cat = torch.cat([z_back, zhat_back, r_back], dim=1)
        else:
            cat = torch.cat([z_back, zhat_back], dim=1)
        f_star = self.merge(cat)
        aux = {
            "z_tokens": tokens.detach(),
            "pred_tokens": pred_tokens,
            "grid": torch.tensor(grid, device=feat.device),
        }
        self.last_aux = aux
        return feat + f_star, aux

    def masked_pred_loss(self, feat: torch.Tensor) -> torch.Tensor:
        """Auxiliary masked prediction; loss only on masked positions.

        Does NOT affect the segmentation feature path (separate forward).
        """
        if not self.training or self.mask_ratio <= 0:
            return feat.new_zeros(())
        b, c, d, h, w = feat.shape
        grid = _target_grid((d, h, w), self.target_tokens)
        with torch.no_grad():
            vol = self.proj(feat)
            z_vol = self._pool_to_grid(vol, grid)
            z_tok = self.norm(tokens_from_volume(z_vol)).detach()
        # recompute proj for grad into predictor only (proj can stay frozen for this aux)
        vis2d = self._sample_visible(b, grid, feat.device)
        visible = vis2d.reshape(b, -1)
        # predictor input: visible tokens keep z, masked use mask_token via visible flag
        pred = self.predictor(z_tok, visible, grid)
        masked = ~visible  # (B,N)
        if not masked.any():
            return feat.new_zeros(())
        # Huber / SmoothL1 only on masked
        diff = F.smooth_l1_loss(pred, z_tok, reduction="none").mean(-1)  # (B,N)
        return diff[masked].mean()


class PredictiveUNet(nn.Module):
    """U-Net + corrected full-context predictive bottleneck at deep_stage."""

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
        dropout: Optional[list] = None,
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encode(x)
        s = self.cfg_deep_stage
        f_s, aux = self.bottleneck(feats[s])
        feats[s] = f_s
        self.last_aux = aux
        if self.training:
            self._last_pred_loss = self.bottleneck.masked_pred_loss(feats[s].detach())
        else:
            self._last_pred_loss = None
        return self.backbone.decode(feats)

    @property
    def pred_loss(self) -> torch.Tensor:
        if self._last_pred_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._last_pred_loss
