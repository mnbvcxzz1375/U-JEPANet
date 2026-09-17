from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dualpath_jepa import DualPathDeepModule, JEPAFeatureHead, JEPAPredictor
from .ema import EMATarget
from .masking import sample_token_mask, token_mask_to_volume_mask
from .unet3d import UNet3D


@dataclass
class UJEPAConfig:
    # Experiment arm: A0 | A1 | A2 | A3
    arm: str = "A0"
    in_chns: int = 1
    feature_chns: Sequence[int] = field(default_factory=lambda: [16, 32, 64, 128])
    dropout: Optional[Sequence[float]] = None
    # WORD labelsTr_All is 0..16 → 17 classes (bg + 16 organs).
    class_num: int = 17
    trilinear: bool = True
    multiscale_pred: bool = True
    norm: str = "instance"  # instance | batch | group

    # Deep stage whose CNN output receives the residual (index into encoder feats).
    deep_stage: int = 2

    # Dual-path / JEPA
    embed_dim: int = 256
    num_heads: int = 4
    num_blocks: int = 4
    predictor_blocks: int = 2
    # JEPA works on a low-res token grid independent of deep_stage.
    # For 128x128x96 and stride=16 → 8x8x6 = 384 tokens.
    jepa_token_stride: int = 16
    alpha_init: float = 0.1
    use_cnn_branch: bool = True

    # JEPA objective
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
        return self.arm in ("A1", "A3") or (self.arm not in ("A0", "A2") and not self.use_cnn_branch)

    @property
    def uses_jepa(self) -> bool:
        return self.arm in ("A2", "A3")


class UJEPAOnline(nn.Module):
    """Online network for segmentation (+ optional dual-path deep module)."""

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
            norm=cfg.norm,
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
                jepa_token_stride=cfg.jepa_token_stride,
                alpha_init=cfg.alpha_init,
            )
        if cfg.uses_jepa and not cfg.uses_dualpath:
            self.jepa_head = JEPAFeatureHead(
                in_channels=cfg.feature_chns[cfg.deep_stage],
                embed_dim=cfg.embed_dim,
                jepa_token_stride=cfg.jepa_token_stride,
            )

    def jepa_token_grid(self, image_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        s = max(1, int(self.cfg.jepa_token_stride))
        return tuple(max(1, int(dim // s)) for dim in image_size)  # type: ignore[return-value]

    def encode(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> List[torch.Tensor] | Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Encoder forward with optional dual-path residual. **No decoder.**"""
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
                out = self.dualpath(feats[i - 1], x, return_tokens=return_aux)
                if return_aux:
                    residual, dual_tokens, dual_grid = out
                else:
                    residual = out
                # Always align residual to CNN stage size (decouple grids).
                if residual.shape[-3:] != cur.shape[-3:]:
                    residual = F.interpolate(
                        residual, size=cur.shape[-3:], mode="trilinear", align_corners=False
                    )
                if self.cfg.use_cnn_branch:
                    cur = cur + self.dualpath.alpha * residual
                else:
                    cur = residual
            feats.append(cur)

        aux: Dict[str, Any] = {}
        if return_aux:
            grid = self.jepa_token_grid(tuple(x.shape[-3:]))
            if self.dualpath is not None:
                aux["dual_tokens"] = dual_tokens
                aux["dual_grid"] = dual_grid if dual_grid is not None else grid
            elif self.jepa_head is not None:
                aux["dual_tokens"], aux["dual_grid"] = self.jepa_head(feats[s], grid)
            aux["deep_feat"] = feats[s]
            return feats, aux
        return feats

    def forward(self, x: torch.Tensor):
        feats = self.encode(x, return_aux=False)
        assert isinstance(feats, list)
        return self.backbone.decode(feats)

    def forward_seg(self, x: torch.Tensor):
        return self.forward(x)

    def forward_with_aux(self, x: torch.Tensor):
        feats, aux = self.encode(x, return_aux=True)
        logits = self.backbone.decode(feats)
        return logits, aux

    def encode_jepa_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """JEPA-only path: encoder features, **no decoder**."""
        _, aux = self.encode(x, return_aux=True)
        tokens = aux["dual_tokens"]
        grid = aux["dual_grid"]
        if tokens is None or grid is None:
            raise RuntimeError("encode_jepa_tokens requires dualpath or jepa_head")
        return tokens, grid


class UJEPATrainer(nn.Module):
    """Holds online network, JEPA predictor, and EMA target (registered for ckpt)."""

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
            # EMATarget is nn.Module so EMA weights appear in state_dict().
            self.target = EMATarget(self.online, decay=cfg.ema_decay)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.target is not None:
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

    def forward(self, x: torch.Tensor):
        return self.forward_seg(x)

    def jepa_step(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        x_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """One JEPA forward. Online sees masked X on BOTH paths; target sees clean X.

        ``x_target`` optional weak/canonical view for the EMA target (C4).
        When None, target uses the same ``x`` (minus the online mask path).
        """
        if not self.cfg.uses_jepa or self.target is None or self.predictor is None:
            raise RuntimeError("jepa_step called on a non-JEPA arm")

        b = x.shape[0]
        grid = self.online.jepa_token_grid(tuple(x.shape[-3:]))
        if mask is None:
            token_mask = sample_token_mask(
                b,
                grid,
                mask_ratio=self.cfg.mask_ratio,
                generator=generator,
                device=x.device,
            )
        elif mask.shape[-3:] == tuple(grid):
            token_mask = mask.to(x.device)
        else:
            token_mask = F.interpolate(
                mask.float(), size=grid, mode="nearest"
            ).to(x.device)

        vol_mask = token_mask_to_volume_mask(token_mask, tuple(x.shape[-3:]))
        x_mask = x * vol_mask + self.cfg.fill_value * (1.0 - vol_mask)

        # Online: encoder only (no decoder).
        ctx_tokens, ctx_grid = self.online.encode_jepa_tokens(x_mask)
        if tuple(ctx_grid) != tuple(grid):
            grid = ctx_grid

        visible = token_mask.reshape(b, -1) > 0.5  # (B, N) on token grid

        tar_x = x if x_target is None else x_target
        with torch.no_grad():
            tgt_tokens, tgt_grid = self.target.target.encode_jepa_tokens(tar_x)
            tgt_tokens = tgt_tokens.detach()
            if tuple(tgt_grid) != tuple(grid):
                raise RuntimeError(f"target grid {tgt_grid} != online grid {grid}")

        pred_tokens = self.predictor(ctx_tokens, visible, grid)
        return {
            "pred_tokens": pred_tokens,
            "tgt_tokens": tgt_tokens,
            "visible": visible,
            "token_mask": token_mask,
            "vol_mask": vol_mask,
            "grid": grid,
        }


def build_model(cfg: UJEPAConfig) -> nn.Module:
    arm = str(cfg.arm).upper()
    if arm not in ("A0", "A1", "A2", "A3"):
        raise ValueError(f"Unknown arm {cfg.arm}")
    cfg.arm = arm
    if arm == "A0":
        return UNet3D(
            in_chns=cfg.in_chns,
            feature_chns=cfg.feature_chns,
            dropout=cfg.dropout,
            class_num=cfg.class_num,
            trilinear=cfg.trilinear,
            multiscale_pred=cfg.multiscale_pred,
            norm=cfg.norm,
        )
    return UJEPATrainer(cfg)


def build_online_for_arm(cfg: UJEPAConfig) -> nn.Module:
    arm = str(cfg.arm).upper()
    cfg.arm = arm
    if arm == "A0":
        return UNet3D(
            in_chns=cfg.in_chns,
            feature_chns=cfg.feature_chns,
            dropout=cfg.dropout,
            class_num=cfg.class_num,
            trilinear=cfg.trilinear,
            multiscale_pred=cfg.multiscale_pred,
            norm=cfg.norm,
        )
    return UJEPAOnline(cfg)
