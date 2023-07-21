import torch
import torch.nn as nn
from torch.nn import functional as F

from .blocks import ConvBlock
from .blocks import UpConv
from .blocks import Normalize
from .blocks import SPADEConvBlock


class SegDecoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        self.initial_conv = nn.Conv2d(emb_dim, filters[0], 3, 1, 1)
        self.dec_block_0 = ConvBlock(filters[0], filters[0])

        self.len_filters = len(filters)

        for i in range(len(filters) - 1):
            in_channels = filters[i]
            out_channels = filters[i + 1]

            self.add_module('dec_up_{}'.format(str(i + 1)), UpConv(in_channels, out_channels))
            self.add_module('dec_block_{}'.format(str(i + 1)), ConvBlock(out_channels, out_channels))

        self.dec_end = nn.Conv2d(filters[-1], output_dim, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.dec_block_0(x)

        for i in range(self.len_filters - 1):
            x = getattr(self, 'dec_up_{}'.format(str(i + 1)))(x)
            x = getattr(self, 'dec_block_{}'.format(str(i + 1)))(x)

        x = self.dec_end(x)

        return x


class ImgDecoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 img_output_act: str,
                 emb_dim: int,
                 label_channels: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        assert img_output_act in {'none', 'tanh'}

        self.n_blocks = len(filters)

        self.initial_conv = nn.Conv2d(emb_dim, filters[0], 3, 1, 1)
        self.dec_block_0 = SPADEConvBlock(filters[0], filters[0], label_channels)

        self.len_filters = len(filters)
        # module_list = []
        for i in range(len(filters) - 1):
            in_channels = filters[i]
            out_channels = filters[i + 1]

            self.add_module('dec_up_{}'.format(str(i + 1)), UpConv(in_channels, out_channels))
            self.add_module('dec_block_{}'.format(str(i + 1)), SPADEConvBlock(out_channels, out_channels, label_channels))

        self.dec_end = nn.Conv2d(filters[-1], output_dim, 1, 1, 0, bias=True)

        if img_output_act == 'none':
            self.final = lambda x: x
        elif img_output_act == 'tanh':
            self.final = nn.Tanh()

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        feat_list = [feat]
        for _ in range(self.n_blocks-1):
            feat = F.avg_pool2d(feat, kernel_size=2)
            feat_list.append(feat)

        x = self.initial_conv(x)
        x = self.dec_block_0(x, feat_list.pop())

        for i in range(self.len_filters - 1):
            x = getattr(self, 'dec_up_{}'.format(str(i + 1)))(x)
            x = getattr(self, 'dec_block_{}'.format(str(i + 1)))(x, feat_list.pop())

        x = self.dec_end(x)

        return self.final(x)
