# src/models/vision_transformer.py
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, emb_size=1024, image_size=64):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = (image_size // patch_size) ** 2
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positions
        return x

class VisionTransformer(nn.Module):
    def __init__(self, emb_size=1024, depth=6, heads=16, mlp_dim=2048, **kwargs):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.patch_embed = PatchEmbedding(emb_size=emb_size, **kwargs)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return x[:, 0]