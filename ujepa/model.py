from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .dualpath_jepa import DualPathDeepModule, JEPAFeatureHead, JEPAPredictor
from .ema import EMATarget
from .masking import apply_mask, sample_block_mask
from .unet3d import UNet3D


@dataclass
class UJEPAConfig:
    # Experiment arm: A0 | A1 | A2 | A3
    arm: str = "A0"
    in_chns: int = 1
    feature_chns: Sequence[int] = field(default_factory=lambda: [16, 32, 64, 128])
    dropout: Optional[Sequence[float]] = None
    class_num: int = 16
    trilinear: bool = True
    multiscale_pred: bool = True

    # Deep stage to enhance / apply JEPA (index into feature_chns / encoder feats).
    # feats[0] is stem (stride 1), feats[1] stride 2, ...
    deep_stage: int = 2

    # Dual-path module
    embed_dim: int = 256
    num_heads: int = 4
    num_blocks: int = 4
    predictor_blocks: int = 2
    image_stride: int = 4
    feature_downsample: int = 1
    alpha_init: float = 0.1
    use_cnn_branch: bool = True  # False => full replacement (A6, not first-round default)

    # JEPA
    mask_ratio: float = 0.4
    ema_decay: float = 0.996
    fill_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feature_chns"] = list(self.feature_chns)
        if self.dropout is not None:
            d["dropout"] = list(self.dropout)
        return d

    @property
    def uses_dualpath(self) -> bool:
        return self.arm in ("A1", "A3") or not self.use_cnn_branch

    @property
    def uses_jepa(self) -> bool:
        return self.arm in ("A2", "A3")


class UJEPAOnline(nn.Module):
    """Online network for segmentation (+ optional dual-path deep module).

    JEPA predictor and EMA target are handled by the training wrapper so that
    inference can drop them cleanly.
    """

    def __init__(self, cfg: UJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = UNet3D(
            in_chns=cfg.in_chns,
            feature_chns=cfg.feature_chns,
            dropout=cfg.dropout,
            class_num=cfg.class_num,
            trilinear=cfg.trilinear,
            multiscale_pred=cfg.multiscale_pred,
        )
        self.dualpath: Optional[DualPathDeepModule] = None
        self.jepa_head: Optional[JEPAFeatureHead] = None

        if cfg.uses_dualpath:
            prev_ch = cfg.feature_chns[cfg.deep_stage - 1]
            deep_ch = cfg.feature_chns[cfg.deep_stage]
            self.dualpath = DualPathDeepModule(
                feature_channels=prev_ch,
                image_channels=cfg.in_chns,
                out_channels=deep_ch,
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                num_blocks=cfg.num_blocks,
                image_stride=cfg.image_stride,
                feature_downsample=cfg.feature_downsample,
                alpha_init=cfg.alpha_init,
            )
        if cfg.uses_jepa and not cfg.uses_dualpath:
            # A2: project existing CNN deep features to JEPA tokens.
            self.jepa_head = JEPAFeatureHead(
                in_channels=cfg.feature_chns[cfg.deep_stage],
                embed_dim=cfg.embed_dim,
                downsample=max(1, cfg.image_stride // (2**cfg.deep_stage)),
            )

    def _encode_with_optional_dualpath(
        self, x: torch.Tensor, return_aux: bool = False
    ):
        """Manual encode so we can inject the dual-path residual at stage s."""
        b = self.backbone
        s = self.cfg.deep_stage
        feats: List[torch.Tensor] = []
        x0 = b.in_conv(x)
        feats.append(x0)
        downs = [b.down1, b.down2, b.down3]
        if len(b.ft_chns) == 5:
            downs.append(b.down4)
        cur = x0
        dual_tokens = None
        dual_grid = None
        for i, down in enumerate(downs, start=1):
            cur = down(cur)
            if i == s and self.dualpath is not None:
                residual = self.dualpath(feats[i - 1], x, return_tokens=return_aux)
                if return_aux:
                    residual, dual_tokens, dual_grid = residual
                if self.cfg.use_cnn_branch:
                    alpha = self.dualpath.alpha
                    cur = cur + alpha * residual
                else:
                    # Full replacement of this stage's CNN output.
                    if residual.shape[-3:] != cur.shape[-3:]:
                        residual = torch.nn.functional.interpolate(
                            residual, size=cur.shape[-3:], mode="trilinear", align_corners=False
                        )
                    cur = residual
            feats.append(cur)

        aux = {}
        if return_aux:
            if self.dualpath is not None:
                aux["dual_tokens"] = dual_tokens
                aux["dual_grid"] = dual_grid
            elif self.jepa_head is not None:
                aux["dual_tokens"], aux["dual_grid"] = self.jepa_head(feats[s])
        if return_aux:
            return feats, aux
        return feats

    def forward(self, x: torch.Tensor):
        feats = self._encode_with_optional_dualpath(x, return_aux=False)
        return self.backbone.decode(feats)

    def forward_with_aux(self, x: torch.Tensor):
        feats, aux = self._encode_with_optional_dualpath(x, return_aux=True)
        logits = self.backbone.decode(feats)
        aux["deep_feat"] = feats[self.cfg.deep_stage]
        return logits, aux


class UJEPATrainer(nn.Module):
    """Holds online network, JEPA predictor, and EMA target for A2/A3."""

    def __init__(self, cfg: UJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.online = UJEPAOnline(cfg)
        self.predictor: Optional[JEPAPredictor] = None
        self.target: Optional[EMATarget] = None
        if cfg.uses_jepa:
            self.predictor = JEPAPredictor(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                num_blocks=cfg.predictor_blocks,
            )
            self.target = EMATarget(self.online, decay=cfg.ema_decay)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.target is not None:
            # Keep EMA target on the same device as the online module.
            device = next(self.online.parameters()).device
            self.target.target = self.target.target.to(device)
        return self

    @torch.no_grad()
    def sync_target(self) -> None:
        if self.target is not None:
            self.target.load_online(self.online)

    @torch.no_grad()
    def update_ema(self, decay: Optional[float] = None) -> None:
        if self.target is not None:
            self.target.update(self.online, decay=decay)

    def forward_seg(self, x: torch.Tensor):
        return self.online.forward(x)

    def jepa_step(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """One JEPA forward. Online sees masked X on BOTH paths; target sees clean X.

        Critical no-leakage contract: dual-path / image embed both consume X_mask.
        """
        if not self.cfg.uses_jepa or self.target is None or self.predictor is None:
            raise RuntimeError("jepa_step called on a non-JEPA arm")

        b, _, d, h, w = x.shape
        if mask is None:
            # Mask at the JEPA token grid if available; else input grid.
            # Using input grid is always safe; interpolate happens inside apply_mask.
            mask = sample_block_mask(
                b, (d, h, w), mask_ratio=self.cfg.mask_ratio, generator=generator, device=x.device
            )
        x_mask = apply_mask(x, mask, fill_value=self.cfg.fill_value)

        # Online path with masked input (both branches see X_mask by construction).
        _, aux_online = self.online.forward_with_aux(x_mask)
        ctx_tokens = aux_online["dual_tokens"]
        grid = aux_online["dual_grid"]
        if grid is None:
            raise RuntimeError("JEPA online did not return a token grid")

        # Build visible token flags from the volume mask at the token grid.
        vis_vol = torch.nn.functional.interpolate(
            mask.float(), size=grid, mode="nearest"
        )  # (B,1,D',H',W')
        visible = vis_vol.reshape(b, -1) > 0.5  # (B, N)

        # Target tokens from clean image through EMA online.
        with torch.no_grad():
            _, aux_tgt = self.target.target.forward_with_aux(x)
            tgt_tokens = aux_tgt["dual_tokens"].detach()

        pred_tokens = self.predictor(ctx_tokens, visible)
        return {
            "pred_tokens": pred_tokens,
            "tgt_tokens": tgt_tokens,
            "visible": visible,
            "mask": mask,
        }


def build_model(cfg: UJEPAConfig) -> nn.Module:
    arm = cfg.arm.upper()
    if arm not in ("A0", "A1", "A2", "A3"):
        raise ValueError(f"Unknown arm {cfg.arm}")
    # Normalize cfg.arm
    cfg.arm = arm
    if arm == "A0":
        # A0: plain U-Net only
        return UNet3D(
            in_chns=cfg.in_chns,
            feature_chns=cfg.feature_chns,
            dropout=cfg.dropout,
            class_num=cfg.class_num,
            trilinear=cfg.trilinear,
            multiscale_pred=cfg.multiscale_pred,
        )
    return UJEPATrainer(cfg)


def build_online_for_arm(cfg: UJEPAConfig) -> nn.Module:
    """Return the inference-time module (no predictor / EMA)."""
    arm = cfg.arm.upper()
    cfg.arm = arm
    if arm == "A0":
        return UNet3D(
            in_chns=cfg.in_chns,
            feature_chns=cfg.feature_chns,
            dropout=cfg.dropout,
            class_num=cfg.class_num,
            trilinear=cfg.trilinear,
            multiscale_pred=cfg.multiscale_pred,
        )
    return UJEPAOnline(cfg)
