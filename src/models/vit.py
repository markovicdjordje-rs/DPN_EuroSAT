# Vision Transformer zasnovan na radu Dosovitsky i saradnika
# Slika je vredna 16x16 reci 
# kao i za resnet, densenet i dpn - implementacija od nule u PyTorchu

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(self, patch_size=8, embed_dim=128):

        super().__init__()

        self.projection = nn.Conv2d(

            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x
    
class VisionTransformer(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        embed_dim = 128
        num_patches = 64

        self.patch_embedding = PatchEmbedding(
            patch_size=8,
            embed_dim=embed_dim
        )

        self.class_token = nn.Parameter(

            torch.zeros(1, 1, embed_dim)
        ) 
        # dodaje se jedan token za kasnije klasifikaciju

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,
            nhead=4,
            dim_feedforward=256,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.patch_embedding(x)

        batch_size = x.shape[0]

        class_token = self.class_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat([class_token, x], dim=1)
        x = x + self.position_embedding

        x = self.encoder(x)

        x = x[:, 0]
        x = self.norm(x)
        x = self.fc(x)

        return x