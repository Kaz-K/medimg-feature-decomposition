import torch
import torch.nn as nn
import math


class DownResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 4, 2, 1),
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, 2, 0)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.double_conv(x) + self.downsample(x))


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ) -> None:
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.double_conv(x) + x)


class LatentDiscriminator(nn.Module):
    baseline = 8

    def __init__(self,
                 emb_dim: int,
                 latent_size: int,
                 ) -> None:
        super().__init__()

        self.basemodel = []
        if latent_size > 4:
            num_downconv = int(math.log2(latent_size // self.baseline))

            for i in range(num_downconv):
                self.basemodel.extend([
                    nn.Conv2d(emb_dim, emb_dim, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                ])

            self.basemodel.extend([
                nn.Conv2d(emb_dim, emb_dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(emb_dim, emb_dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(emb_dim, emb_dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
            ])

        else:
            self.basemodel.extend([
                nn.Conv2d(emb_dim, emb_dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(emb_dim, emb_dim, 4, 2, 1),
                nn.LeakyReLU(0.2),
            ])

        self.basemodel = nn.Sequential(*self.basemodel)

        self.discriminator = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(emb_dim, 1, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.basemodel(x)
        return self.discriminator(x).view(-1)


class LatentDiscriminatorV2(nn.Module):
    baseline = 8

    def __init__(self,
                 emb_dim: int,
                 latent_size: int,
                 ) -> None:
        super().__init__()

        num_downconv = int(math.log2(latent_size // self.baseline))

        self.basemodel = []
        for i in range(num_downconv):
            self.basemodel.extend([
                DownResBlock(emb_dim, emb_dim),
            ])

        self.basemodel.extend([
            DownResBlock(emb_dim, emb_dim),  # 4x4
            DownResBlock(emb_dim, emb_dim),  # 2x2
            DownResBlock(emb_dim, emb_dim),  # 1x1
        ])

        self.basemodel = nn.Sequential(*self.basemodel)

        self.discriminator = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(emb_dim, 1, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.basemodel(x)
        return self.discriminator(x).view(-1)


class ImageDiscriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, filters[0], 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters[0], filters[1], 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters[1], filters[2], 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters[2], filters[3], 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters[3], filters[4], 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters[4], filters[5], 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * filters[5], 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x.view(-1)
