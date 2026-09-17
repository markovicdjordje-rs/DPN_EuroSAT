# DPN model je zasnovan na radu Chen i saradnika
# Implementacija od nule u PyTorchu

import torch
import torch.nn as nn

class DualPathBlock(nn.Module):

    def __init__(self, residual_channels, dense_channels, growth_rate):

        super().__init__()

        input_channels = residual_channels + dense_channels

        self.residual_channels = residual_channels

        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(

            input_channels,
            residual_channels,
            kernel_size=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(residual_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(

            residual_channels,
            residual_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(residual_channels)

        self.conv3 = nn.Conv2d(

            residual_channels,
            residual_channels + growth_rate,
            kernel_size=1,
            bias=False
        )


        self.bn3 = nn.BatchNorm2d(residual_channels + growth_rate)

    def forward(self, residual, dense):

        x = torch.cat([residual, dense], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        new_residual = x[:, :self.residual_channels]
        new_dense = x[:, self.residual_channels:]

        residual = residual + new_residual
        residual = self.relu(residual)

        dense = torch.cat([dense, new_dense], dim=1)

        return residual, dense
    
class DPNTransition(nn.Module):

    def __init__(self, in_channels, residual_channels, dense_channels):

        super().__init__()

        self.residual_channels = residual_channels

        out_channels = residual_channels + dense_channels

        self.conv = nn.Conv2d(

            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, residual, dense):

        x = torch.cat([residual, dense], dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        residual = x[:, :self.residual_channels]
        dense = x[:, self.residual_channels:]

        return residual, dense

class DPN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                80,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(80),
            nn.ReLU()
        )

        self.block1 = DualPathBlock(
            residual_channels=64,
            dense_channels=16,
            growth_rate=16
        )

        self.block2 = DualPathBlock(
            residual_channels=64,
            dense_channels=32,
            growth_rate=16
        )

        self.transition1 = DPNTransition(
            in_channels=112,
            residual_channels=128,
            dense_channels=16
        )

        self.block3 = DualPathBlock(
            residual_channels=128,
            dense_channels=16,
            growth_rate=16
        )

        self.block4 = DualPathBlock(
            residual_channels=128,
            dense_channels=32,
            growth_rate=16
        )

        self.transition2 = DPNTransition(
            in_channels=176,
            residual_channels=256,
            dense_channels=16
        )

        self.block5 = DualPathBlock(
            residual_channels=256,
            dense_channels=16,
            growth_rate=16
        )

        self.block6 = DualPathBlock(
            residual_channels=256,
            dense_channels=32,
            growth_rate=16
        )

        self.transition3 = DPNTransition(
            in_channels=304,
            residual_channels=512,
            dense_channels=16
        )

        self.block7 = DualPathBlock(
            residual_channels=512,
            dense_channels=16,
            growth_rate=16
        )

        self.block8 = DualPathBlock(
            residual_channels=512,
            dense_channels=32,
            growth_rate=16
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(560, num_classes)
        
    def forward(self, x):

        x = self.stem(x)

        residual = x[:, :64]
        dense = x[:, 64:]

        residual, dense = self.block1(residual, dense)
        residual, dense = self.block2(residual, dense)
        residual, dense = self.transition1(residual, dense)

        residual, dense = self.block3(residual, dense)
        residual, dense = self.block4(residual, dense)
        residual, dense = self.transition2(residual, dense)

        residual, dense = self.block5(residual, dense)
        residual, dense = self.block6(residual, dense)
        residual, dense = self.transition3(residual, dense)

        residual, dense = self.block7(residual, dense)
        residual, dense = self.block8(residual, dense)

        x = torch.cat([residual, dense], dim=1)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x