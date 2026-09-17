# Densenet model je zasnovan na radu Huang i saradnika
# Implementacija od nule u PyTorchu

import torch
import torch.nn as nn

class DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate):

        super().__init__()

        hidden_channels = 4 * growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(
            hidden_channels,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = torch.cat([x, out], dim=1)

        return out

class DenseBlock(nn.Module):

    def __init__(self, num_layers, in_channels, growth_rate):

        super().__init__()
        layers = []
        current_channels = in_channels

        for _ in range(num_layers):

            layers.append(
                DenseLayer(current_channels, growth_rate)
            )
            current_channels = current_channels + growth_rate

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)

class TransitionLayer(nn.Module):

    def __init__(self, in_channels):

        super().__init__()
        out_channels = in_channels // 2
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)

        return x

class DenseNet121(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.block1 = DenseBlock(6, 64, 32)
        self.transition1 = TransitionLayer(256)

        self.block2 = DenseBlock(12, 128, 32)
        self.transition2 = TransitionLayer(512)

        self.block3 = DenseBlock(24, 256, 32)
        self.transition3 = TransitionLayer(1024)

        self.block4 = DenseBlock(16, 512, 32)

        self.bn_final = nn.BatchNorm2d(1024)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.transition1(x)

        x = self.block2(x)
        x = self.transition2(x)

        x = self.block3(x)
        x = self.transition3(x)

        x = self.block4(x)

        x = self.bn_final(x)
        x = self.relu(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x