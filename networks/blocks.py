from typing import Optional
from typing import Tuple
from typing import Any

import torch
import torch.nn as nn


RELU_INPLACE = True
NORM_TYPE = 'groupnorm'
ACT_TYPE = 'leaky_relu'


def conv5x5(in_channels, out_channels, stride=1, padding=2, bias=True):
    return nn.Conv2d(in_channels, out_channels, 5, stride, padding, bias=bias)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=bias)


def conv1x1(in_channels, out_channels, stride=1, padding=0, bias=True):
    return nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=bias)


class Activation(nn.Module):
    def __init__(self,
                 type: str,
                 ) -> None:
        super().__init__()
        assert type in {'relu', 'leaky_relu'}

        if type == 'relu':
            self.act = nn.ReLU(inplace=RELU_INPLACE)
        elif type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=RELU_INPLACE)
        else:
            raise NotImplementedError

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.act(x)


class Normalize(nn.Module):
    def __init__(self,
                 type: str,
                 num_features: Optional[int],
                 ) -> None:
        super().__init__()
        assert type in {'none', 'batchnorm', 'instancenorm', 'groupnorm'}

        if type == 'none':
            self.norm = lambda x: x
        elif type == 'batchnorm':
            self.norm = nn.BatchNorm2d(num_features)
        elif type == 'instancenorm':
            self.norm = nn.InstanceNorm2d(num_features)
        elif type == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=8, num_channels=num_features)
        else:
            raise NotImplementedError

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        out = self.norm(x)
        return out


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()

        self.conv = nn.Sequential(Normalize(NORM_TYPE, in_channels),
                                  Activation(ACT_TYPE),
                                  nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
                                  Normalize(NORM_TYPE, out_channels),
                                  Activation(ACT_TYPE),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True))

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
            )
        else:
            self.downsample = lambda x: x

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.downsample(x) + self.conv(x)


class DownConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()

        self.conv = nn.Sequential(Normalize(NORM_TYPE, in_channels),
                                  Activation(ACT_TYPE),
                                  nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True))

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()

        self.up = nn.Sequential(Normalize(NORM_TYPE, in_channels),
                                Activation(ACT_TYPE),
                                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        x = self.up(x)
        return x


class SPADE(nn.Module):
    # original: https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
    def __init__(self,
                 num_features: int,
                 hidden_channels: int,
                 label_channels: int,
                 ) -> None:
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(num_features, affine=False)
        self.mlp_shared = nn.Sequential(
            conv3x3(label_channels, hidden_channels),
            nn.ReLU(inplace=RELU_INPLACE),
        )
        self.mlp_gamma = conv3x3(hidden_channels, num_features)
        self.mlp_beta = conv3x3(hidden_channels, num_features)

    def forward(self, x: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class SPADEConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 label_channels: int,
                 ) -> None:
        super().__init__()

        self.activation = Activation(ACT_TYPE)
        self.spade1 = SPADE(in_channels, in_channels // 2, label_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.spade2 = SPADE(out_channels, out_channels // 2, label_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
            )
        else:
            self.downsample = lambda x: x

    def forward(self,
                x: torch.Tensor,
                feat: torch.Tensor,
                ) -> torch.Tensor:

        out = self.spade1(x, feat)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.spade2(out, feat)
        out = self.activation(out)
        out = self.conv2(out)
        out += self.downsample(x)

        return out
