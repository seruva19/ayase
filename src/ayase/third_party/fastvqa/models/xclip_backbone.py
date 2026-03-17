# This source code is licensed under the S-Lab License 1.0 found in the
# LICENSE file in the current directory's parent directory.
"""
The code has been adopted from FAST-VQA-and-FasterVQA
(https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/blob/dev/fastvqa/models/xclip_backbone.py)
"""

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=width,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size),
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, multi=False, layer=-1):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        x = x.reshape(B, T, -1).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return x


def build_x_clip_model(state_dict: dict = None, **kwargs):
    vit = "ViT-B/16"
    if vit == "ViT-B/16":
        vision_width = 768
        vision_layers = 12
        vision_patch_size = 16
        grid_size = 14
        embed_dim = 512
        vision_heads = vision_width // 64
    elif vit == "ViT-L/14":
        vision_width = 1024
        vision_layers = 24
        vision_patch_size = 14
        grid_size = 16
        embed_dim = 768
        vision_heads = vision_width // 64

    model = VisualTransformer(
        input_resolution=224,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
    )

    if state_dict is not None:
        new_state_dict = {}
        for key, value in state_dict.items():
            if "visual." in key:
                new_key = key.replace("visual.", "")
                if "conv1.weight" in new_key:
                    value = value.unsqueeze(2) / 2
                    value = torch.cat([value, value], dim=2)
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)

    return model
