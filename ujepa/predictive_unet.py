from __future__ import annotations

"""Predictive-Residual U-Net (V1): predictor stays in the inference graph."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dualpath_jepa import JEPAPredictor, tokens_from_volume
from .unet3d import UNet3D


class PredictiveBottleneck(nn.Module):
    """F_s -> tokens -> (predict) -> merge(Z, Z_hat, R) -> F_s*."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        predictor_blocks: int = 2,
        mask_ratio: float = 0.4,
        use_residual: bool = True,
        token_stride_hint: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.use_residual = use_residual
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.predictor = JEPAPredictor(embed_dim=embed_dim, num_heads=num_heads, num_blocks=predictor_blocks)
        merge_in = embed_dim * (3 if use_residual else 2)
        self.merge = nn.Conv3d(merge_in, in_channels, kernel_size=1)
        self.last_z: Optional[torch.Tensor] = None
        self.last_zhat: Optional[torch.Tensor] = None

    def _sample_mask(self, n: int, grid: Tuple[int, int, int], device) -> torch.Tensor:
        # token grid mask (1,D',H',W') bool True=visible
        d, h, w = grid
        n_tok = d * h * w
        n_mask = max(1, int(round(self.mask_ratio * n_tok)))
        idx = torch.randperm(n_tok, device=device)[: n_tok - n_mask]
        vis = torch.zeros(n_tok, dtype=torch.bool, device=device)
        vis[idx] = True
        return vis.view(1, d, h, w)

    def forward(
        self,
        feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        train_mode: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """feat: (B,C,D,H,W) at deep stage. Returns F* same shape + aux."""
        b, c, d, h, w = feat.shape
        vol = self.proj(feat)  # (B,E,D,H,W)
        grid = (d, h, w)
        tokens = tokens_from_volume(vol)  # (B,N,E)
        tokens = self.norm(tokens)
        n = tokens.shape[1]
        if mask is None:
            # visible mask on this native grid
            vis2d = torch.stack([self._sample_mask(n, grid, feat.device)[0] for _ in range(b)], 0)  # (B,D,H,W)
        else:
            vis2d = mask
        visible = vis2d.reshape(b, -1) > 0.5  # (B,N)

        # Always run predictor on (partially) masked tokens for structure; at infer use all-visible
        # so prediction is conditioned on full anatomy context then refined.
        if train_mode and self.mask_ratio > 0:
            # hide masked positions from predictor input by zeroing visible flag usage in predictor
            zhat_tok = self.predictor(tokens, visible, grid)
            # For merge we want a full-grid prediction: for visible use input, masked use pred
            z_merge_tok = torch.where(visible.unsqueeze(-1), tokens, zhat_tok)
        else:
            # inference: predict every token from all tokens (context = full)
            vis_all = torch.ones_like(visible)
            zhat_tok = self.predictor(tokens, vis_all, grid)
            z_merge_tok = zhat_tok

        def to_vol(tok: torch.Tensor) -> torch.Tensor:
            return tok.transpose(1, 2).reshape(b, self.embed_dim, d, h, w)

        z_vol = to_vol(tokens)
        zhat_vol = to_vol(z_merge_tok)
        if self.use_residual:
            r_vol = z_vol - zhat_vol
            cat = torch.cat([z_vol, zhat_vol, r_vol], dim=1)
        else:
            cat = torch.cat([z_vol, zhat_vol], dim=1)
        f_star = self.merge(cat)
        aux = {
            "z": tokens,
            "z_hat": z_merge_tok,
            "visible": visible,
            "grid": torch.tensor(grid, device=feat.device),
            "pred_tokens": zhat_tok,
        }
        self.last_z, self.last_zhat = tokens.detach(), z_merge_tok.detach()
        return feat + f_star, aux


class PredictiveUNet(nn.Module):
    """U-Net with predictive-residual bottleneck at deep_stage (default 2)."""

    def __init__(
        self,
        feature_chns: Sequence[int] = (16, 32, 64, 128),
        class_num: int = 17,
        deep_stage: int = 2,
        embed_dim: int = 256,
        num_heads: int = 4,
        predictor_blocks: int = 2,
        mask_ratio: float = 0.4,
        use_residual: bool = True,
        use_pred_in_forward: bool = True,
        multiscale_pred: bool = True,
        norm: str = "instance",
    ):
        super().__init__()
        self.cfg_deep_stage = deep_stage
        self.use_pred_in_forward = use_pred_in_forward
        self.backbone = UNet3D(
            in_chns=1,
            feature_chns=list(feature_chns),
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
        )
        self.last_aux: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, train_mode: bool = True):
        """Returns decoder output; may be a list when multiscale_pred=True (deep supervision)."""
        feats = self.backbone.encode(x)
        s = self.cfg_deep_stage
        if self.use_pred_in_forward:
            f_s, aux = self.bottleneck(feats[s], train_mode=train_mode)
            feats[s] = f_s
            self.last_aux = aux
        else:
            self.last_aux = {}
        return self.backbone.decode(feats)


def pred_loss_from_aux(aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    """SmoothL1 on predicted tokens vs stop-grad Z (no EMA required for V1)."""
    if not aux:
        return torch.tensor(0.0)
    z = aux["z"]
    zhat = aux["pred_tokens"]
    vis = aux["visible"]
    # loss on all tokens (dense-ish): pred should reconstruct Z from context
    diff = F.smooth_l1_loss(zhat, z.detach(), reduction="none").mean(-1)  # (B,N)
    return diff.mean()
