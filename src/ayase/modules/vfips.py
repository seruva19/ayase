"""VFIPS — Video Frame Interpolation Perceptual Similarity (ECCV 2022).

Full-reference perceptual video metric designed for frame-interpolation
evaluation (Hou, Qi et al., github.com/hqqxyy/VFIPS). It compares a distorted /
interpolated video against a reference video using a trained multi-scale
spatiotemporal network: a 2-D convolutional feature extractor is applied
per-frame, features across a 12-frame temporal window are stacked and fused by
per-scale Swin-transformer "merge" layers, and a linear head aggregates the
per-scale perceptual differences into a single distance.

The real architecture (LPIPS_3D_Diff / MultiScaleV33 / SwinDiffTiny) is vendored
below verbatim from the upstream ``networks/`` package, and the 101-key trained
state-dict is loaded strictly from the mirrored weights. There is no proxy: if
torch / timm / the weights are unavailable, or no reference video is provided,
``vfips_score`` is left unset.

vfips_score — lower = better (perceptual distance), populated only with the real
trained backend and a reference video.
"""

import logging
from pathlib import Path
from typing import List, Optional

from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)

# Temporal window (frames per clip). The trained "merge" Swin layers have their
# input channel count baked in as 12 * chn * 3, so this is fixed at 12.
_WINDOW = 12

# Minimum frame side so the 5x-downsampled feature stays >= the Swin window (4).
_MIN_SIDE = 128

_HF_REPO = "AkaneTendo25/ayase-models"
_HF_WEIGHTS = "vfips/model.pytorch"


# ---------------------------------------------------------------------------
# Vendored model definition (upstream networks/common.py, multi_scale.py,
# swinir.py, lpips_3d.py). Attribute names are preserved exactly so the trained
# 101-key state-dict loads strictly.
# ---------------------------------------------------------------------------
def _build_arch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    try:
        from timm.layers import DropPath, to_2tuple, trunc_normal_
    except Exception:  # pragma: no cover - older timm
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_

    # ----- common.py -----
    class ScalingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
            self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

        def forward(self, inp):
            return (inp - self.shift) / self.scale

    class NetLinLayer(nn.Module):
        """A single linear layer that does a 1x1 conv."""

        def __init__(self, chn_in, chn_out=1, use_dropout=False):
            super().__init__()
            layers = [nn.Dropout()] if use_dropout else []
            layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    def spatial_average(in_tens, keepdim=True):
        return in_tens.mean([2, 3], keepdim=keepdim)

    def normalize_tensor(in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)

    # ----- swinir.py -----
    class Mlp(nn.Module):
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

    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    class WindowAttention(nn.Module):
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
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
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

    class SwinTransformerBlock(nn.Module):
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

            self.norm1 = norm_layer(dim) if norm_layer is not None else None
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim) if norm_layer is not None else None
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

            if self.shift_size > 0:
                attn_mask = self.calculate_mask(self.input_resolution)
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

        def calculate_mask(self, x_size):
            H, W = x_size
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
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            return attn_mask

        def forward(self, x, x_size):
            H, W = x_size
            B, L, C = x.shape

            shortcut = x
            if self.norm1 is not None:
                x = self.norm1(x)
            x = x.view(B, H, W, C)

            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

            if self.input_resolution == x_size:
                attn_windows = self.attn(x_windows, mask=self.attn_mask)
            else:
                attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)

            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)

            x = shortcut + self.drop_path(x)
            if self.norm2 is None:
                x = x + self.drop_path(self.mlp(x))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    class BasicLayer(nn.Module):
        def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.depth = depth
            self.use_checkpoint = use_checkpoint

            self.blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])

            if downsample is not None:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            else:
                self.downsample = None

        def forward(self, x, x_size):
            for blk in self.blocks:
                x = blk(x, x_size)
            if self.downsample is not None:
                x = self.downsample(x)
            return x

    class PatchEmbed(nn.Module):
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
            self.norm = norm_layer(embed_dim) if norm_layer is not None else None

        def forward(self, x):
            x = x.flatten(2).transpose(1, 2)
            if self.norm is not None:
                x = self.norm(x)
            return x

    class PatchUnEmbed(nn.Module):
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

        def forward(self, x, x_size):
            B, HW, C = x.shape
            x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
            return x

    class RSTBTiny(nn.Module):
        def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                     img_size=224, patch_size=4, resi_connection='1conv'):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.residual_group = BasicLayer(dim=dim, input_resolution=input_resolution,
                                             depth=depth, num_heads=num_heads, window_size=window_size,
                                             mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                             norm_layer=norm_layer, downsample=downsample,
                                             use_checkpoint=use_checkpoint)

            if resi_connection == '1conv':
                self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
            elif resi_connection == '3conv':
                self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1),
                                          nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                          nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                          nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                          nn.Conv2d(dim // 4, dim, 3, 1, 1))
            else:
                self.conv = nn.Sequential()

            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=0, embed_dim=dim, norm_layer=None)
            self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                              in_chans=0, embed_dim=dim, norm_layer=None)

        def forward(self, x, x_size):
            return self.residual_group(x, x_size)

    class SwinDiffTiny(nn.Module):
        def __init__(self, in_chans=3, out_chans=64, embed_dim=180, depths=(3,),
                     num_heads=(3,), window_size=8, mlp_ratio=2.):
            super().__init__()
            qkv_bias = True
            qk_scale = None
            drop_rate = 0
            attn_drop_rate = 0
            drop_path_rate = 0.1
            norm_layer = None
            ape = False
            patch_norm = True
            use_checkpoint = False

            num_in_ch = in_chans
            num_out_ch = out_chans
            self.window_size = window_size

            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

            self.num_layers = len(depths)
            self.embed_dim = embed_dim
            self.ape = ape
            self.patch_norm = patch_norm
            self.num_features = embed_dim
            self.mlp_ratio = mlp_ratio

            img_size = (64, 64)

            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=1, in_chans=embed_dim,
                                          embed_dim=embed_dim,
                                          norm_layer=norm_layer if self.patch_norm else None)
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution

            self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=1, in_chans=embed_dim,
                                              embed_dim=embed_dim,
                                              norm_layer=norm_layer if self.patch_norm else None)

            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
                trunc_normal_(self.absolute_pos_embed, std=.02)

            self.pos_drop = nn.Dropout(p=drop_rate)

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = RSTBTiny(dim=embed_dim,
                                 input_resolution=(patches_resolution[0], patches_resolution[1]),
                                 depth=depths[i_layer], num_heads=num_heads[i_layer],
                                 window_size=window_size, mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                 norm_layer=norm_layer, downsample=None,
                                 use_checkpoint=use_checkpoint, img_size=img_size,
                                 patch_size=1, resi_connection=None)
                self.layers.append(layer)

            self.norm = norm_layer(self.num_features) if norm_layer is not None else None
            self.conv_after_body = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        def check_image_size(self, x):
            _, _, h, w = x.size()
            mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
            mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            return x

        def forward_features(self, x):
            x_size = (x.shape[2], x.shape[3])
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)
            for layer in self.layers:
                x = layer(x, x_size)
            if self.norm is not None:
                x = self.norm(x)
            x = self.patch_unembed(x, x_size)
            return x

        def forward(self, x):
            H, W = x.shape[2:]
            x = self.check_image_size(x)
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x))
            return x[:, :, :H, :W]

    # ----- multi_scale.py -----
    class Extractor(nn.Module):
        def __init__(self):
            super().__init__()

            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(cout, cout, 3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

            self.moduleFirst = block(3, 16)
            self.moduleSecond = block(16, 32)
            self.moduleThird = block(32, 64)
            self.moduleFourth = block(64, 96)
            self.moduleFifth = block(96, 128)
            self.moduleSixth = block(128, 192)

        def forward(self, tensorInput):
            t1 = self.moduleFirst(tensorInput)
            t2 = self.moduleSecond(t1)
            t3 = self.moduleThird(t2)
            t4 = self.moduleFourth(t3)
            t5 = self.moduleFifth(t4)
            t6 = self.moduleSixth(t5)
            return [t1, t2, t3, t4, t5, t6]

    class MultiScaleV33(nn.Module):
        def __init__(self):
            super().__init__()
            self.moduleExtractor = Extractor()
            chns = [16, 32, 64, 96, 128]
            out_chn = 32
            merge_layers = []
            for chn in chns:
                merge_layers.append(
                    SwinDiffTiny(in_chans=_WINDOW * chn * 3, out_chans=out_chn, embed_dim=32,
                                 depths=[1], num_heads=[2], window_size=4, mlp_ratio=2.))
            self.merge_layers = nn.ModuleList(merge_layers)

        def forward(self, inputFirst, inputSecond):
            B, V, C, H, W = inputFirst.size()
            inputFirst = inputFirst.view(B * V, C, H, W)
            inputSecond = inputSecond.view(B * V, C, H, W)

            feasFirst = self.moduleExtractor(inputFirst)[:-1]
            feasSecond = self.moduleExtractor(inputSecond)[:-1]

            tensorFeas = []
            for fa, fb in zip(feasFirst, feasSecond):
                fa = normalize_tensor(fa)
                fb = normalize_tensor(fb)
                fd = torch.abs(fa - fb)
                tensorFeas.append(torch.cat([fa, fb, fd], dim=1))

            tensorFeas = [p.view(B, -1, p.size(2), p.size(3)) for p in tensorFeas]
            outs = []
            for merge_layer, fea in zip(self.merge_layers, tensorFeas):
                outs.append(merge_layer(fea))
            return outs

    # ----- lpips_3d.py -----
    class LPIPS_3D_Diff(nn.Module):
        def __init__(self):
            super().__init__()
            self.scaling_layer = ScalingLayer()
            self.net = MultiScaleV33()
            self.chns = [32, 32, 32, 32, 32]
            self.L = len(self.chns)
            self.lins = nn.ModuleList([NetLinLayer(chn, use_dropout=True) for chn in self.chns])

        def forward(self, in0, in1, normalize=False):
            if normalize:
                in0 = 2 * in0 - 1
                in1 = 2 * in1 - 1
            in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
            diffs = self.net.forward(in0_input, in1_input)
            res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
            val = res[0]
            for layer_index in range(1, self.L):
                val = val + res[layer_index]
            return val

    return LPIPS_3D_Diff


class VFIPSModule(ReferenceBasedModule):
    name = "vfips"
    description = "VFIPS frame interpolation perceptual similarity (ECCV 2022, FR)"
    metric_field = "vfips_score"
    default_config = {"max_clips": 8, "device": "auto"}
    metric_groups = {
        "vfips_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self._device = "cpu"
        self.max_clips = int(self.config.get("max_clips", 8))

    def setup(self) -> None:
        self._backend = "unavailable"
        self._ml_available = False
        try:
            import torch
            from huggingface_hub import hf_hub_download

            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            arch = _build_arch()
            model = arch()
            ckpt = hf_hub_download(repo_id=_HF_REPO, filename=_HF_WEIGHTS)
            state_dict = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            model.to(self._device).eval()

            self._model = model
            self._backend = "real"
            self._ml_available = True
            logger.info("VFIPS initialised (trained multiscale_v33 network) on %s", self._device)
        except ImportError as e:
            logger.warning("VFIPS unavailable: requires torch, timm and huggingface_hub (%s)", e)
        except Exception as e:
            logger.warning("VFIPS unavailable: backend load failed (%s)", e)

    def _preprocess_frame(self, rgb):
        """RGB uint8 HxWx3 -> float tensor CxHxW in [-1, 1] (ToTensor + Normalize(0.5))."""
        import numpy as np
        import torch

        arr = np.ascontiguousarray(rgb).astype(np.float32) / 255.0  # [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
        t = (t - 0.5) / 0.5  # -> [-1,1]
        return t

    def _read_windows(self, ref_path: Path, dist_path: Path):
        """Yield aligned (ref_clip, dist_clip) tensors [1, _WINDOW, 3, H, W]."""
        import cv2
        import torch

        cap_ref = cv2.VideoCapture(str(ref_path))
        cap_dist = cv2.VideoCapture(str(dist_path))
        clips: List = []
        try:
            if not cap_ref.isOpened() or not cap_dist.isOpened():
                return clips
            while len(clips) < self.max_clips:
                ref_frames = []
                dist_frames = []
                ok = True
                target_hw = None
                for _ in range(_WINDOW):
                    r1, f1 = cap_ref.read()
                    r2, f2 = cap_dist.read()
                    if not (r1 and r2):
                        ok = False
                        break
                    rgb_ref = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                    rgb_dist = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                    if target_hw is None:
                        h0, w0 = rgb_ref.shape[0], rgb_ref.shape[1]
                        # The extractor downsamples 5x (/32); the deepest merge
                        # Swin layer (window_size 4) needs that feature >= 4, i.e.
                        # each side >= _MIN_SIDE. Upscale (aspect-preserving) only
                        # when a frame is smaller than this; native otherwise.
                        if min(h0, w0) < _MIN_SIDE:
                            s = _MIN_SIDE / float(min(h0, w0))
                            w0 = int(round(w0 * s))
                            h0 = int(round(h0 * s))
                        target_hw = (w0, h0)  # (W, H)
                    if (rgb_ref.shape[1], rgb_ref.shape[0]) != target_hw:
                        rgb_ref = cv2.resize(rgb_ref, target_hw)
                    if (rgb_dist.shape[1], rgb_dist.shape[0]) != target_hw:
                        rgb_dist = cv2.resize(rgb_dist, target_hw)
                    ref_frames.append(self._preprocess_frame(rgb_ref))
                    dist_frames.append(self._preprocess_frame(rgb_dist))
                if not ok or len(ref_frames) < _WINDOW:
                    break
                clip_ref = torch.stack(ref_frames, dim=0).unsqueeze(0)
                clip_dist = torch.stack(dist_frames, dim=0).unsqueeze(0)
                clips.append((clip_ref, clip_dist))
        finally:
            cap_ref.release()
            cap_dist.release()
        return clips

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        if self._backend != "real" or self._model is None:
            return None
        try:
            import torch

            clips = self._read_windows(reference_path, sample_path)
            if not clips:
                return None

            scores: List[float] = []
            with torch.no_grad():
                for clip_ref, clip_dist in clips:
                    ref_t = clip_ref.to(self._device)
                    dist_t = clip_dist.to(self._device)
                    val = self._model(ref_t, dist_t)  # forward(in0=ref, in1=distorted)
                    v = float(val.detach().cpu().numpy().flatten().mean())
                    if v == v and abs(v) != float("inf"):  # skip nan/inf
                        scores.append(v)
            if not scores:
                return None
            return float(sum(scores) / len(scores))
        except Exception as e:
            logger.warning("VFIPS computation failed: %s", e)
            return None
