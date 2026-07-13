"""StableVQA — Video Stability Quality Assessment (ACM MM 2023).

Faithful re-implementation of the published StableVQA predictor of Kou et al.,
"StableVQA: A Deep No-Reference Quality Assessment Model for Video Stability"
(ACM MM 2023).  Official code: https://github.com/QMME/StableVQA

The network (``Stablev2Evaluator`` upstream) fuses three feature branches and
regresses a single stability score:

  * **Semantic branch** — a 2-D Swin-Transformer (SwinV1-Tiny, ``resize_backbone``)
    extracts a 768-d feature per sampled frame (32 frames → 24576-d).
  * **Motion / camera branch** — dense optical flow between adjacent frames is
    computed with RAFT (``flow_model``); the stacked flow field is passed
    through a 3-D ResNet-18 (``motion_analyzer``, 2-channel input) → 512-d.
  * **Blur branch** — the Stripformer deblurring encoder (``deblur_net``)
    produces a 320-channel map for 8 sampled frames → 2560-d.

The three feature vectors are concatenated (512 + 768*32 + 320*8 = 27648) and
passed through a 2-layer MLP (``quality``) to give ``stablevqa_score``
(higher = more stable).

The trained checkpoint (all four sub-networks, 861 tensors) is mirrored on
HuggingFace and loaded verbatim; the whole model definition is vendored below
so no upstream package is required.  If the weights or torch are unavailable the
metric is reported as unavailable and ``stablevqa_score`` is left unset — no
proxy or heuristic score is substituted (no-heuristic policy).
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# HuggingFace mirror of the official StableVQA checkpoint.
_HF_REPO = "AkaneTendo25/ayase-models"
_HF_FILENAME = "stablevqa/stablevqa_checkpoint.pth"

# Frame preprocessing constants (upstream FusionDataset).
_MEAN = [123.675, 116.28, 103.53]
_STD = [58.395, 57.12, 57.375]
_CLIP_LEN = 32       # number of frames fed to the model (fixes img_feat = 768*32)
_FRAME_SIZE = 224    # Swin / RAFT input resolution
_BLUR_FRAMES = 8     # frames sampled for the blur branch (fixes blur_feat = 320*8)

# Cache for the lazily-built (torch-dependent) model class.
_EVALUATOR_CLASS = None


# ---------------------------------------------------------------------------
# Vendored model definition (built lazily so importing this module never
# requires torch).  Reconstructs the exact upstream ``Stablev2Evaluator`` so the
# published 861-tensor state_dict loads strictly.
# ---------------------------------------------------------------------------
def _load_model_definitions():
    """Build and return the vendored ``Stablev2Evaluator`` class (cached)."""
    global _EVALUATOR_CLASS
    if _EVALUATOR_CLASS is not None:
        return _EVALUATOR_CLASS

    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from timm.layers import DropPath, to_2tuple, trunc_normal_

    # ================= Swin-Transformer (semantic branch) =================
    class _SwinMlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None,
                     act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    def _window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def _window_reverse(windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    class _WindowAttention(nn.Module):
        def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                     attn_drop=0., proj_drop=0.):
            super().__init__()
            self.dim = dim
            self.window_size = window_size
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            trunc_normal_(self.relative_position_bias_table, std=.02)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, mask=None):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class _SwinBlock(nn.Module):
        def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            self.mlp_ratio = mlp_ratio
            if min(self.input_resolution) <= self.window_size:
                self.shift_size = 0
                self.window_size = min(self.input_resolution)
            assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

            self.norm1 = norm_layer(dim)
            self.attn = _WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = _SwinMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                                act_layer=act_layer, drop=drop)

            if self.shift_size > 0:
                H, W = self.input_resolution
                img_mask = torch.zeros((1, H, W, 1))
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                mask_windows = _window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

        def forward(self, x):
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = _window_partition(shifted_x, self.window_size)
            else:
                shifted_x = x
                x_windows = _window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            if self.shift_size > 0:
                shifted_x = _window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                shifted_x = _window_reverse(attn_windows, self.window_size, H, W)
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    class _PatchMerging(nn.Module):
        def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

        def forward(self, x):
            H, W = self.input_resolution
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
            x = x.view(B, -1, 4 * C)
            x = self.norm(x)
            x = self.reduction(x)
            return x

    class _SwinBasicLayer(nn.Module):
        def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.depth = depth
            self.blocks = nn.ModuleList([
                _SwinBlock(dim=dim, input_resolution=input_resolution,
                           num_heads=num_heads, window_size=window_size,
                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop, attn_drop=attn_drop,
                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                           norm_layer=norm_layer)
                for i in range(depth)])
            if downsample is not None:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            else:
                self.downsample = None

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
            return x

    class _PatchEmbed(nn.Module):
        def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
            super().__init__()
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
            self.img_size = img_size
            self.patch_size = patch_size
            self.patches_resolution = patches_resolution
            self.num_patches = patches_resolution[0] * patches_resolution[1]
            self.in_chans = in_chans
            self.embed_dim = embed_dim
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            if norm_layer is not None:
                self.norm = norm_layer(embed_dim)
            else:
                self.norm = None

        def forward(self, x):
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x).flatten(2).transpose(1, 2)
            if self.norm is not None:
                x = self.norm(x)
            return x

    class _SwinTransformer(nn.Module):
        def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                     embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):
            super().__init__()
            self.num_classes = num_classes
            self.num_layers = len(depths)
            self.embed_dim = embed_dim
            self.ape = ape
            self.patch_norm = patch_norm
            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
            self.mlp_ratio = mlp_ratio
            self.patch_embed = _PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            num_patches = self.patch_embed.num_patches
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution
            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
                trunc_normal_(self.absolute_pos_embed, std=.02)
            self.pos_drop = nn.Dropout(p=drop_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = _SwinBasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                      patches_resolution[1] // (2 ** i_layer)),
                    depth=depths[i_layer], num_heads=num_heads[i_layer],
                    window_size=window_size, mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=_PatchMerging if (i_layer < self.num_layers - 1) else None)
                self.layers.append(layer)
            self.norm = norm_layer(self.num_features)
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = torch.flatten(x, 1)
            return x

    # ================= 3-D ResNet-18 (motion / camera branch) =================
    def _conv3x3x3(in_planes, out_planes, stride=1):
        return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def _conv1x1x1(in_planes, out_planes, stride=1):
        return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    class _BasicBlock3D(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = _conv3x3x3(in_planes, planes, stride)
            self.bn1 = nn.BatchNorm3d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = _conv3x3x3(planes, planes)
            self.bn2 = nn.BatchNorm3d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    class _ResNet3D(nn.Module):
        def __init__(self, block, layers, block_inplanes, n_input_channels=3,
                     conv1_t_size=7, conv1_t_stride=1, no_max_pool=False,
                     shortcut_type='B', widen_factor=1.0, n_classes=400):
            super().__init__()
            block_inplanes = [int(x * widen_factor) for x in block_inplanes]
            self.in_planes = block_inplanes[0]
            self.no_max_pool = no_max_pool
            self.conv1 = nn.Conv3d(n_input_channels, self.in_planes,
                                   kernel_size=(conv1_t_size, 7, 7),
                                   stride=(conv1_t_stride, 2, 2),
                                   padding=(conv1_t_size // 2, 3, 3), bias=False)
            self.bn1 = nn.BatchNorm3d(self.in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
            self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
            self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
            self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
            downsample = None
            if stride != 1 or self.in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    _conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))
            layers = [block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample)]
            self.in_planes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if not self.no_max_pool:
                x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return x

    # ================= Stripformer (blur branch) =================
    class _StripEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = nn.LeakyReLU(0.2, True)
            self.en_layer1_1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), self.activation)
            self.en_layer1_2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), self.activation, nn.Conv2d(64, 64, 3, padding=1))
            self.en_layer1_3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), self.activation, nn.Conv2d(64, 64, 3, padding=1))
            self.en_layer1_4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), self.activation, nn.Conv2d(64, 64, 3, padding=1))
            self.en_layer2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), self.activation)
            self.en_layer2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), self.activation, nn.Conv2d(128, 128, 3, padding=1))
            self.en_layer2_3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), self.activation, nn.Conv2d(128, 128, 3, padding=1))
            self.en_layer2_4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), self.activation, nn.Conv2d(128, 128, 3, padding=1))
            self.en_layer3_1 = nn.Sequential(nn.Conv2d(128, 320, 3, stride=2, padding=1), self.activation)

        def forward(self, x):
            hx = self.en_layer1_1(x)
            hx = self.activation(self.en_layer1_2(hx) + hx)
            hx = self.activation(self.en_layer1_3(hx) + hx)
            hx = self.activation(self.en_layer1_4(hx) + hx)
            hx = self.en_layer2_1(hx)
            hx = self.activation(self.en_layer2_2(hx) + hx)
            hx = self.activation(self.en_layer2_3(hx) + hx)
            hx = self.activation(self.en_layer2_4(hx) + hx)
            hx = self.en_layer3_1(hx)
            return hx

    class _StripAttention(nn.Module):
        def __init__(self, head_num):
            super().__init__()
            self.num_attention_heads = head_num
            self.softmax = nn.Softmax(dim=-1)

        def transpose_for_scores(self, x):
            B, N, C = x.size()
            attention_head_size = int(C / self.num_attention_heads)
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3).contiguous()

        def forward(self, query_layer, key_layer, value_layer):
            B, N, C = query_layer.size()
            query_layer = self.transpose_for_scores(query_layer)
            key_layer = self.transpose_for_scores(key_layer)
            value_layer = self.transpose_for_scores(value_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            _, _, _, d = query_layer.size()
            attention_scores = attention_scores / math.sqrt(d)
            attention_probs = self.softmax(attention_scores)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (C,)
            return context_layer.view(*new_context_layer_shape)

    class _StripMlp(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
            self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
            self.act_fn = torch.nn.functional.gelu

        def forward(self, x):
            x = self.fc1(x)
            x = self.act_fn(x)
            x = self.fc2(x)
            return x

    class _PEG(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.PEG = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size)

        def forward(self, x):
            return self.PEG(x) + x

    class _IntraSA(nn.Module):
        def __init__(self, dim, head_num):
            super().__init__()
            self.hidden_size = dim // 2
            self.head_num = head_num
            self.attention_norm = nn.LayerNorm(dim)
            self.conv_input = nn.Conv2d(dim, dim, 1, padding=0)
            self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)
            self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)
            self.fuse_out = nn.Conv2d(dim, dim, 1, padding=0)
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = _StripMlp(dim)
            self.attn = _StripAttention(head_num=self.head_num)
            self.PEG = _PEG(dim)

        def forward(self, x):
            h = x
            B, C, H, W = x.size()
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            x = self.attention_norm(x).permute(0, 2, 1).contiguous()
            x = x.view(B, C, H, W)
            x_input = torch.chunk(self.conv_input(x), 2, dim=1)
            feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous().view(B * H, W, C // 2)
            feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous().view(B * W, H, C // 2)
            qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
            qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
            q_h, k_h, v_h = qkv_h
            q_v, k_v, v_v = qkv_v
            if H == W:
                query = torch.cat((q_h, q_v), dim=0)
                key = torch.cat((k_h, k_v), dim=0)
                value = torch.cat((v_h, v_v), dim=0)
                attention_output = torch.chunk(self.attn(query, key, value), 2, dim=0)
                attention_output_h = attention_output[0].view(B, H, W, C // 2).permute(0, 3, 1, 2).contiguous()
                attention_output_v = attention_output[1].view(B, W, H, C // 2).permute(0, 3, 2, 1).contiguous()
            else:
                attention_output_h = self.attn(q_h, k_h, v_h).view(B, H, W, C // 2).permute(0, 3, 1, 2).contiguous()
                attention_output_v = self.attn(q_v, k_v, v_v).view(B, W, H, C // 2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
            x = attn_out + h
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
            return self.PEG(x)

    class _InterSA(nn.Module):
        def __init__(self, dim, head_num):
            super().__init__()
            self.hidden_size = dim
            self.head_num = head_num
            self.attention_norm = nn.LayerNorm(self.hidden_size)
            self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, 1, padding=0)
            self.conv_h = nn.Conv2d(self.hidden_size // 2, 3 * (self.hidden_size // 2), 1, padding=0)
            self.conv_v = nn.Conv2d(self.hidden_size // 2, 3 * (self.hidden_size // 2), 1, padding=0)
            self.ffn_norm = nn.LayerNorm(self.hidden_size)
            self.ffn = _StripMlp(self.hidden_size)
            self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, 1, padding=0)
            self.attn = _StripAttention(head_num=self.head_num)
            self.PEG = _PEG(dim)

        def forward(self, x):
            h = x
            B, C, H, W = x.size()
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            x = self.attention_norm(x).permute(0, 2, 1).contiguous()
            x = x.view(B, C, H, W)
            x_input = torch.chunk(self.conv_input(x), 2, dim=1)
            feature_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
            feature_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
            query_h, key_h, value_h = feature_h
            query_v, key_v, value_v = feature_v
            horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0).permute(0, 2, 1, 3).contiguous().view(3 * B, H, -1)
            horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
            query_h, key_h, value_h = horizontal_groups
            vertical_groups = torch.cat((query_v, key_v, value_v), dim=0).permute(0, 3, 1, 2).contiguous().view(3 * B, W, -1)
            vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
            query_v, key_v, value_v = vertical_groups
            if H == W:
                query = torch.cat((query_h, query_v), dim=0)
                key = torch.cat((key_h, key_v), dim=0)
                value = torch.cat((value_h, value_v), dim=0)
                attention_output = torch.chunk(self.attn(query, key, value), 2, dim=0)
                attention_output_h = attention_output[0].view(B, H, C // 2, W).permute(0, 2, 1, 3).contiguous()
                attention_output_v = attention_output[1].view(B, W, C // 2, H).permute(0, 2, 3, 1).contiguous()
            else:
                attention_output_h = self.attn(query_h, key_h, value_h).view(B, H, C // 2, W).permute(0, 2, 1, 3).contiguous()
                attention_output_v = self.attn(query_v, key_v, value_v).view(B, W, C // 2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
            x = attn_out + h
            x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
            return self.PEG(x)

    class _Stripformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _StripEmbeddings()
            head_num = 5
            dim = 320
            self.Trans_block_1 = _IntraSA(dim, head_num)
            self.Trans_block_2 = _InterSA(dim, head_num)
            self.Trans_block_3 = _IntraSA(dim, head_num)
            self.Trans_block_4 = _InterSA(dim, head_num)
            self.Trans_block_5 = _IntraSA(dim, head_num)
            self.Trans_block_6 = _InterSA(dim, head_num)
            self.Trans_block_7 = _IntraSA(dim, head_num)
            self.Trans_block_8 = _InterSA(dim, head_num)
            self.Trans_block_9 = _IntraSA(dim, head_num)
            self.Trans_block_10 = _InterSA(dim, head_num)
            self.Trans_block_11 = _IntraSA(dim, head_num)
            self.Trans_block_12 = _InterSA(dim, head_num)
            self.decoder = _StripEmbeddingsOutput()

        def forward(self, x):
            hx = self.encoder(x)
            hx = self.Trans_block_1(hx)
            hx = self.Trans_block_2(hx)
            hx = self.Trans_block_3(hx)
            hx = self.Trans_block_4(hx)
            hx = self.Trans_block_5(hx)
            hx = self.Trans_block_6(hx)
            hx = self.Trans_block_7(hx)
            hx = self.Trans_block_8(hx)
            hx = self.Trans_block_9(hx)
            hx = self.Trans_block_10(hx)
            hx = self.Trans_block_11(hx)
            hx = self.Trans_block_12(hx)
            return hx  # decoder is intentionally unused (upstream), 320-channel map

    class _StripEmbeddingsOutput(nn.Module):
        # Present only to preserve the ``decoder.*`` parameter keys in the
        # checkpoint (upstream never calls it during feature extraction).
        def __init__(self):
            super().__init__()
            self.activation = nn.LeakyReLU(0.2, True)
            self.de_layer3_1 = nn.Sequential(nn.ConvTranspose2d(320, 192, 4, stride=2, padding=1), self.activation)
            self.de_layer2_2 = nn.Sequential(nn.Conv2d(192 + 128, 192, 1, padding=0), self.activation)
            self.de_block_1 = _IntraSA(192, 3)
            self.de_block_2 = _InterSA(192, 3)
            self.de_block_3 = _IntraSA(192, 3)
            self.de_block_4 = _InterSA(192, 3)
            self.de_block_5 = _IntraSA(192, 3)
            self.de_block_6 = _InterSA(192, 3)
            self.de_layer2_1 = nn.Sequential(nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1), self.activation)
            self.de_layer1_3 = nn.Sequential(nn.Conv2d(128, 64, 1, padding=0), self.activation, nn.Conv2d(64, 64, 3, padding=1))
            self.de_layer1_2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), self.activation, nn.Conv2d(64, 64, 3, padding=1))
            self.de_layer1_1 = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1), self.activation)

    # ================= RAFT (optical-flow estimator) =================
    def _coords_grid(batch, ht, wd):
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def _bilinear_sampler(img, coords, mode='bilinear'):
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1)
        return F.grid_sample(img, grid, align_corners=True)

    def _upflow8(flow, mode='bilinear'):
        new_size = (8 * flow.shape[2], 8 * flow.shape[3])
        return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    class _CorrBlock:
        def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
            self.num_levels = num_levels
            self.radius = radius
            self.corr_pyramid = []
            corr = _CorrBlock.corr(fmap1, fmap2)
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            self.corr_pyramid.append(corr)
            for _ in range(self.num_levels - 1):
                corr = F.avg_pool2d(corr, 2, stride=2)
                self.corr_pyramid.append(corr)

        def __call__(self, coords):
            r = self.radius
            coords = coords.permute(0, 2, 3, 1)
            batch, h1, w1, _ = coords.shape
            out_pyramid = []
            for i in range(self.num_levels):
                corr = self.corr_pyramid[i]
                dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
                dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
                delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)
                centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
                delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
                coords_lvl = centroid_lvl + delta_lvl
                corr = _bilinear_sampler(corr, coords_lvl)
                corr = corr.view(batch, h1, w1, -1)
                out_pyramid.append(corr)
            out = torch.cat(out_pyramid, dim=-1)
            return out.permute(0, 3, 1, 2).contiguous().float()

        @staticmethod
        def corr(fmap1, fmap2):
            batch, dim, ht, wd = fmap1.shape
            fmap1 = fmap1.view(batch, dim, ht * wd)
            fmap2 = fmap2.view(batch, dim, ht * wd)
            corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
            corr = corr.view(batch, ht, wd, 1, ht, wd)
            return corr / torch.sqrt(torch.tensor(dim).float())

    class _RaftResidualBlock(nn.Module):
        def __init__(self, in_planes, planes, norm_fn='group', stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride)
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            num_groups = planes // 8
            if norm_fn == 'group':
                self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
                self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
                if stride != 1:
                    self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            elif norm_fn == 'batch':
                self.norm1 = nn.BatchNorm2d(planes)
                self.norm2 = nn.BatchNorm2d(planes)
                if stride != 1:
                    self.norm3 = nn.BatchNorm2d(planes)
            elif norm_fn == 'instance':
                self.norm1 = nn.InstanceNorm2d(planes)
                self.norm2 = nn.InstanceNorm2d(planes)
                if stride != 1:
                    self.norm3 = nn.InstanceNorm2d(planes)
            elif norm_fn == 'none':
                self.norm1 = nn.Sequential()
                self.norm2 = nn.Sequential()
                if stride != 1:
                    self.norm3 = nn.Sequential()
            if stride == 1:
                self.downsample = None
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride), self.norm3)

        def forward(self, x):
            y = x
            y = self.relu(self.norm1(self.conv1(y)))
            y = self.relu(self.norm2(self.conv2(y)))
            if self.downsample is not None:
                x = self.downsample(x)
            return self.relu(x + y)

    class _BasicEncoder(nn.Module):
        def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
            super().__init__()
            self.norm_fn = norm_fn
            if self.norm_fn == 'group':
                self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            elif self.norm_fn == 'batch':
                self.norm1 = nn.BatchNorm2d(64)
            elif self.norm_fn == 'instance':
                self.norm1 = nn.InstanceNorm2d(64)
            elif self.norm_fn == 'none':
                self.norm1 = nn.Sequential()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.relu1 = nn.ReLU(inplace=True)
            self.in_planes = 64
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128, output_dim, 1)
            self.dropout = None
            if dropout > 0:
                self.dropout = nn.Dropout2d(p=dropout)

        def _make_layer(self, dim, stride=1):
            layer1 = _RaftResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
            layer2 = _RaftResidualBlock(dim, dim, self.norm_fn, stride=1)
            self.in_planes = dim
            return nn.Sequential(layer1, layer2)

        def forward(self, x):
            is_list = isinstance(x, (tuple, list))
            if is_list:
                batch_dim = x[0].shape[0]
                x = torch.cat(x, dim=0)
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.conv2(x)
            if self.training and self.dropout is not None:
                x = self.dropout(x)
            if is_list:
                x = torch.split(x, [batch_dim, batch_dim], dim=0)
            return x

    class _FlowHead(nn.Module):
        def __init__(self, input_dim=128, hidden_dim=256):
            super().__init__()
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.conv2(self.relu(self.conv1(x)))

    class _SepConvGRU(nn.Module):
        def __init__(self, hidden_dim=128, input_dim=192 + 128):
            super().__init__()
            self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
            self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
            self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
            self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
            self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
            self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

        def forward(self, h, x):
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(self.convz1(hx))
            r = torch.sigmoid(self.convr1(hx))
            q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
            h = (1 - z) * h + z * q
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(self.convz2(hx))
            r = torch.sigmoid(self.convr2(hx))
            q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
            h = (1 - z) * h + z * q
            return h

    class _BasicMotionEncoder(nn.Module):
        def __init__(self, corr_levels, corr_radius):
            super().__init__()
            cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
            self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
            self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
            self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
            self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
            self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

        def forward(self, flow, corr):
            cor = F.relu(self.convc1(corr))
            cor = F.relu(self.convc2(cor))
            flo = F.relu(self.convf1(flow))
            flo = F.relu(self.convf2(flo))
            cor_flo = torch.cat([cor, flo], dim=1)
            out = F.relu(self.conv(cor_flo))
            return torch.cat([out, flow], dim=1)

    class _BasicUpdateBlock(nn.Module):
        def __init__(self, corr_levels, corr_radius, hidden_dim=128, input_dim=128):
            super().__init__()
            self.encoder = _BasicMotionEncoder(corr_levels, corr_radius)
            self.gru = _SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
            self.flow_head = _FlowHead(hidden_dim, hidden_dim=256)
            self.mask = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64 * 9, 1, padding=0))

        def forward(self, net, inp, corr, flow, upsample=True):
            motion_features = self.encoder(flow, corr)
            inp = torch.cat([inp, motion_features], dim=1)
            net = self.gru(net, inp)
            delta_flow = self.flow_head(net)
            mask = .25 * self.mask(net)
            return net, mask, delta_flow

    class _RAFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.corr_levels = 4
            self.corr_radius = 4
            self.fnet = _BasicEncoder(output_dim=256, norm_fn='instance', dropout=0)
            self.cnet = _BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=0)
            self.update_block = _BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)

        def initialize_flow(self, img):
            N, C, H, W = img.shape
            coords0 = _coords_grid(N, H // 8, W // 8).to(img.device)
            coords1 = _coords_grid(N, H // 8, W // 8).to(img.device)
            return coords0, coords1

        def upsample_flow(self, flow, mask):
            N, _, H, W = flow.shape
            mask = mask.view(N, 1, 9, 8, 8, H, W)
            mask = torch.softmax(mask, dim=2)
            up_flow = F.unfold(8 * flow, [3, 3], padding=1)
            up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
            up_flow = torch.sum(mask * up_flow, dim=2)
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
            return up_flow.reshape(N, 2, 8 * H, 8 * W)

        def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0
            image1 = image1.contiguous()
            image2 = image2.contiguous()
            hdim = self.hidden_dim
            cdim = self.context_dim
            fmap1, fmap2 = self.fnet([image1, image2])
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
            corr_fn = _CorrBlock(fmap1, fmap2, radius=self.corr_radius)
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            coords0, coords1 = self.initialize_flow(image1)
            if flow_init is not None:
                coords1 = coords1 + flow_init
            flow_predictions = []
            for _ in range(iters):
                coords1 = coords1.detach()
                corr = corr_fn(coords1)
                flow = coords1 - coords0
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
                coords1 = coords1 + delta_flow
                if up_mask is None:
                    flow_up = _upflow8(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                flow_predictions.append(flow_up)
            if test_mode:
                return coords1 - coords0, flow_up
            return flow_predictions

    # ================= Fusion evaluator =================
    class Stablev2Evaluator(nn.Module):
        """Vendored upstream ``Stablev2Evaluator`` (inference-only)."""

        def __init__(self):
            super().__init__()
            self.blur = _BLUR_FRAMES
            # Semantic branch: SwinV1-Tiny (default config)
            self.resize_backbone = _SwinTransformer()
            # Blur branch: Stripformer wrapped in DataParallel to match key prefix
            self.deblur_net = nn.DataParallel(_Stripformer())
            # Motion branch: 3-D ResNet-18, 2-channel flow input
            self.motion_analyzer = _ResNet3D(_BasicBlock3D, [2, 2, 2, 2],
                                             [64, 128, 256, 512],
                                             n_input_channels=2, n_classes=256)
            self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
            # Optical-flow estimator: RAFT
            self.flow_model = _RAFT()
            # Fusion + regression head
            self.quality = nn.Sequential(
                nn.Linear(512 + 768 * 32 + 320 * self.blur, 128),
                nn.Linear(128, 1),
            )

        def get_blur_vec(self, frames, num):
            # frames: (n, d, c, h, w)
            _, d, c, h, w = frames.shape
            img_tensor = frames[:, 0:d:int(d / num), :, :, :]
            img_tensor = img_tensor.reshape(-1, c, h, w)
            factor = 8
            padh = (factor - h % factor) % factor
            padw = (factor - w % factor) % factor
            img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
            # bypass DataParallel scatter (safe on both CPU and single GPU)
            return self.deblur_net.module(img_tensor)

        def forward(self, vclips):
            scores = []
            for key in vclips:
                n, c, d, h, w = vclips[key].shape
                tmp = vclips[key].permute(0, 2, 1, 3, 4)  # n d c h w
                x = vclips[key].reshape(-1, c, h, w)
                optical_flows = []
                for i in range(d):
                    if i + 1 < d:
                        flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i + 1, :, :])
                    else:
                        flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                    optical_flows.append(flow_up[0])
                img_f = getattr(self, key.split("_")[0] + "_backbone")(x)
                img_feat = img_f.reshape(n, d * img_f.size(1))
                optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
                blur_feats = self.get_blur_vec(tmp, self.blur)
                blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                total_feat = torch.cat([blur_feats, img_feat, optical_feat], 1)
                scores.append(self.quality(total_feat))
            return scores

    _EVALUATOR_CLASS = Stablev2Evaluator
    return _EVALUATOR_CLASS


class StableVQAModule(PipelineModule):
    name = "stablevqa"
    description = "StableVQA video stability quality assessment (ACM MM 2023)"
    default_config = {
        "device": "auto",
        "clip_len": _CLIP_LEN,
        "frame_size": _FRAME_SIZE,
    }
    metric_groups = {
        "stablevqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            evaluator_cls = _load_model_definitions()
            model = evaluator_cls()
            model.eval()

            ckpt_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                # Refuse to emit a score from a partially-initialised model.
                logger.warning(
                    "StableVQA: checkpoint did not load cleanly "
                    "(%d missing, %d unexpected keys); metric unavailable.",
                    len(missing), len(unexpected),
                )
                self._backend = "unavailable"
                return

            model.to(self._device)
            model.eval()
            self._model = model
            self._ml_available = True
            self._backend = "real"
            logger.info("StableVQA initialised on %s (real backend, 861 tensors loaded).",
                        self._device)

        except ImportError as e:
            self._backend = "unavailable"
            logger.warning("StableVQA requires torch/timm/huggingface_hub: %s", e)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("StableVQA unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._model is None:
            # No trained backend -> leave stablevqa_score unset (no heuristic).
            return sample
        try:
            score = self._compute(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.stablevqa_score = float(score)
                logger.debug("StableVQA for %s: %.4f", sample.path.name, score)
        except Exception as e:
            logger.warning("StableVQA failed for %s: %s", sample.path, e)
        return sample

    def _compute(self, sample: Sample) -> Optional[float]:
        import numpy as np
        import torch

        if not getattr(sample, "is_video", False):
            return None

        clip = self._load_clip(sample)
        if clip is None:
            return None

        clip = clip.to(self._device)
        with torch.no_grad():
            scores = self._model({"resize": clip})
        if not scores:
            return None
        return float(np.mean(scores[0].detach().cpu().numpy()))

    def _load_clip(self, sample: Sample):
        """Sample ``clip_len`` frames, resize to 224 and mean/std normalise.

        Returns a tensor of shape (1, 3, clip_len, 224, 224) or None.
        """
        import cv2
        import numpy as np
        import torch

        clip_len = int(self.config.get("clip_len", _CLIP_LEN))
        size = int(self.config.get("frame_size", _FRAME_SIZE))

        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        indices = np.linspace(0, total - 1, clip_len).astype(int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1])
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)
        cap.release()

        if len(frames) < clip_len:
            if not frames:
                return None
            while len(frames) < clip_len:
                frames.append(frames[-1])
        frames = frames[:clip_len]

        arr = np.stack(frames).astype(np.float32)  # (T, H, W, 3) RGB
        t = torch.from_numpy(arr)
        mean = torch.tensor(_MEAN)
        std = torch.tensor(_STD)
        t = (t - mean) / std              # normalise on RGB channel (last dim)
        t = t.permute(3, 0, 1, 2)         # (3, T, H, W)
        return t.unsqueeze(0).contiguous()  # (1, 3, T, H, W)
