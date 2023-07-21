import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_conv: int,
                 ) -> None:
        super().__init__()

        module_list = []
        for i in range(n_conv):
            if i == 0:
                input_dim = in_channels
                output_dim = out_channels
            else:
                input_dim = out_channels
                output_dim = out_channels

            module_list.extend([
                nn.Conv2d(input_dim, output_dim, 3, 1, 1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ])

        self.conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.conv(x)


class NormalityClassifier(nn.Module):

    def __init__(self,
                 input_dim: int,
                 filters: list = [32, 64, 128, 256, 512, 512],
                 ) -> None:

        super().__init__()

        self.conv = nn.Sequential(
            DoubleConv(input_dim, filters[0], n_conv=2),
            nn.MaxPool2d(2),
            DoubleConv(filters[0], filters[1], n_conv=2),
            nn.MaxPool2d(2),
            DoubleConv(filters[1], filters[2], n_conv=3),
            nn.MaxPool2d(2),
            DoubleConv(filters[2], filters[3], n_conv=3),
            nn.MaxPool2d(2),
            DoubleConv(filters[3], filters[4], n_conv=3),
            nn.MaxPool2d(2),
            DoubleConv(filters[4], filters[5], n_conv=3),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * filters[5], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x.view(-1)
