from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .blocks import ConvBlock, DownBlock, UpBlock


class UNet3D(nn.Module):
    """Baseline 3D U-Net compatible with the PL-Seg feature channel convention.

    feature_chns length 4 or 5. Multiscale deep supervision optional.
    """

    def __init__(
        self,
        in_chns: int = 1,
        feature_chns: Sequence[int] = (16, 32, 64, 128),
        dropout: Optional[Sequence[float]] = None,
        class_num: int = 16,
        trilinear: bool = True,
        multiscale_pred: bool = False,
    ):
        super().__init__()
        self.ft_chns = list(feature_chns)
        if dropout is None:
            dropout = [0.0] * len(self.ft_chns)
        self.dropout = list(dropout)
        self.in_chns = in_chns
        self.n_class = class_num
        self.trilinear = trilinear
        self.mul_pred = multiscale_pred
        assert len(self.ft_chns) in (4, 5)
        assert len(self.dropout) == len(self.ft_chns)

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if len(self.ft_chns) == 5:
            self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
            self.up1 = UpBlock(
                self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], self.dropout[3], trilinear
            )
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], self.dropout[2], trilinear)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], self.dropout[1], trilinear)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], self.dropout[0], trilinear)
        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=1)
        if self.mul_pred:
            self.out_conv1 = nn.Conv3d(self.ft_chns[1], self.n_class, kernel_size=1)
            self.out_conv2 = nn.Conv3d(self.ft_chns[2], self.n_class, kernel_size=1)
            self.out_conv3 = nn.Conv3d(self.ft_chns[3], self.n_class, kernel_size=1)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        feats = [x0, x1, x2, x3]
        if len(self.ft_chns) == 5:
            x4 = self.down4(x3)
            feats.append(x4)
        return feats

    def decode(self, feats: List[torch.Tensor]):
        if len(self.ft_chns) == 5:
            x0, x1, x2, x3, x4 = feats
            x_d3 = self.up1(x4, x3)
        else:
            x0, x1, x2, x3 = feats
            x_d3 = x3
        x_d2 = self.up2(x_d3, x2)
        x_d1 = self.up3(x_d2, x1)
        x_d0 = self.up4(x_d1, x0)
        output = self.out_conv(x_d0)
        if self.mul_pred:
            output = [
                output,
                self.out_conv1(x_d1),
                self.out_conv2(x_d2),
                self.out_conv3(x_d3),
            ]
        return output

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))
