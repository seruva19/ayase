"""Zoom-VQA -- Patches, Frames and Clips Integration for VQA.

Zhao, Yuan, Sun, Wen. CVPRW 2023 (arXiv:2304.06440). Runner-up in the
NTIRE-2023 Quality Assessment of Video Enhancement Challenge.

GitHub: https://github.com/k-zha14/Zoom-VQA

Dual-branch late-fusion blind video-quality model:

* **IQA branch** -- ``CPNet`` ("ConvNet + Patch-level head"): a timm
  ``convnext_tiny`` feature extractor (``features_only=True``, multi-scale
  hyper-column of 96+192+384+768 = 1440 channels) followed by two
  patch-level MLP heads (score head + weight head) whose weighted average
  yields a frame quality score. Runs per frame (2 fps in the paper).
* **VQA branch** -- a Video Swin Transformer ("swin_tiny_grpb", i.e. Swin-3D
  Tiny with Gated Relative Position Bias / fragment position bias) operating
  on FAST-VQA style spatial *fragments* + temporal *clips*, with a Conv3D
  regression head.

The two branch scores are combined by late fusion. The published pipeline
z-score-normalises each branch across the whole evaluation set and applies a
sigmoid rescale before averaging (0.5 / 0.5); that dataset-relative rescale is
a monotonic re-ranking and is *undefined for single-sample inference*, so this
module reports the direct fused output of the two real trained networks
(``0.5 * iqa + 0.5 * vqa``). Both branch scores come from the real released
checkpoints -- no proxy or heuristic model is substituted.

Real backend requires ``torch``, ``timm``, ``einops``, ``decord`` and the two
released checkpoints (mirrored on HF ``AkaneTendo25/ayase-models``). When any
are missing the metric is left ``None``.

zoomvqa_score -- higher = better quality
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

IQA_WEIGHT = "weights/zoomvqa/iqa_best_29epoch_checkpoint.pth.tar"
VQA_WEIGHT = "weights/zoomvqa/vqa_best_29e_val-vqpve_s.pth"
HF_REPO = "AkaneTendo25/ayase-models"

# ---------------------------------------------------------------------------
# Vendored minimal model definitions (from github.com/k-zha14/Zoom-VQA).
# Heavy imports are guarded so the module still registers when torch/timm/
# einops are unavailable; setup() then reports the metric as unavailable.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
    try:
        from timm.layers import DropPath, trunc_normal_
    except Exception:  # older timm
        from timm.models.layers import DropPath, trunc_normal_
    from functools import reduce, lru_cache
    from operator import mul

    _TORCH_OK = True
except Exception:  # pragma: no cover - exercised only without deps
    _TORCH_OK = False


if _TORCH_OK:

    # ----- VQA branch: Video Swin Transformer (swin_tiny_grpb) -------------
    # Source: VQA/fastvqa/models/swin_backbone.py + head.py

    class _Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None,
                     act_layer=nn.GELU, drop=0.0):
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
        B, D, H, W, C = x.shape
        x = x.view(B, D // window_size[0], window_size[0],
                   H // window_size[1], window_size[1],
                   W // window_size[2], window_size[2], C)
        windows = (x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
                   .view(-1, reduce(mul, window_size), C))
        return windows

    def _window_reverse(windows, window_size, B, D, H, W):
        x = windows.view(B, D // window_size[0], H // window_size[1],
                         W // window_size[2], window_size[0], window_size[1],
                         window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
        return x

    def _get_window_size(x_size, window_size, shift_size=None):
        use_window_size = list(window_size)
        if shift_size is not None:
            use_shift_size = list(shift_size)
        for i in range(len(x_size)):
            if x_size[i] <= window_size[i]:
                use_window_size[i] = x_size[i]
                if shift_size is not None:
                    use_shift_size[i] = 0
        if shift_size is None:
            return tuple(use_window_size)
        return tuple(use_window_size), tuple(use_shift_size)

    @lru_cache
    def _get_adaptive_window_size(base_window_size, input_x_size, base_x_size):
        tw, hw, ww = base_window_size
        tx_, hx_, wx_ = input_x_size
        tx, hx, wx = base_x_size
        return (tw * tx_) // tx, (hw * hx_) // hx, (ww * wx_) // wx

    @lru_cache
    def _global_position_index(D, H, W, fragments=(1, 7, 7),
                               window_size=(8, 7, 7), shift_size=(0, 0, 0),
                               device="cpu"):
        frags_d = torch.arange(fragments[0])
        frags_h = torch.arange(fragments[1])
        frags_w = torch.arange(fragments[2])
        frags = torch.stack(
            torch.meshgrid(frags_d, frags_h, frags_w, indexing="ij")
        ).float()
        coords = (torch.nn.functional.interpolate(frags[None].to(device),
                                                  size=(D, H, W))
                  .long().permute(0, 2, 3, 4, 1))
        coords = torch.roll(
            coords, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
            dims=(1, 2, 3))
        window_coords = _window_partition(coords, window_size)
        relative_coords = window_coords[:, None, :] - window_coords[:, :, None]
        return relative_coords

    @lru_cache()
    def _compute_mask(D, H, W, window_size, shift_size, device):
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        cnt = 0
        for d in (slice(-window_size[0]),
                  slice(-window_size[0], -shift_size[0]),
                  slice(-shift_size[0], None)):
            for h in (slice(-window_size[1]),
                      slice(-window_size[1], -shift_size[1]),
                      slice(-shift_size[1], None)):
                for w in (slice(-window_size[2]),
                          slice(-window_size[2], -shift_size[2]),
                          slice(-shift_size[2], None)):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = _window_partition(img_mask, window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = (attn_mask.masked_fill(attn_mask != 0, float(-100.0))
                     .masked_fill(attn_mask == 0, float(0.0)))
        return attn_mask

    class _WindowAttention3D(nn.Module):
        def __init__(self, dim, window_size, num_heads, qkv_bias=False,
                     qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                     frag_bias=False):
            super().__init__()
            self.dim = dim
            self.window_size = window_size
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5

            self.relative_position_bias_table = nn.Parameter(torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1), num_heads))
            if frag_bias:
                self.fragment_position_bias_table = nn.Parameter(torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
                    * (2 * window_size[2] - 1), num_heads))

            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            coords = torch.stack(
                torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = (coords_flatten[:, :, None]
                               - coords_flatten[:, None, :])
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= ((2 * self.window_size[1] - 1)
                                         * (2 * self.window_size[2] - 1))
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index",
                                 relative_position_index)

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            trunc_normal_(self.relative_position_bias_table, std=0.02)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, mask=None, fmask=None, resized_window_size=None):
            B_, N, C = x.shape
            qkv = (self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                       C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            if resized_window_size is None:
                rpi = self.relative_position_index[:N, :N]
            else:
                relative_position_index = self.relative_position_index.reshape(
                    *self.window_size, *self.window_size)
                d, h, w = resized_window_size
                rpi = relative_position_index[:d, :h, :w, :d, :h, :w]
            relative_position_bias = self.relative_position_bias_table[
                rpi.reshape(-1)].reshape(N, N, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()
            if hasattr(self, "fragment_position_bias_table"):
                fragment_position_bias = self.fragment_position_bias_table[
                    rpi.reshape(-1)].reshape(N, N, -1)
                fragment_position_bias = fragment_position_bias.permute(
                    2, 0, 1).contiguous()

            if fmask is not None:
                fgate = fmask.abs().sum(-1)
                nW = fmask.shape[0]
                relative_position_bias = relative_position_bias.unsqueeze(0)
                fgate = fgate.unsqueeze(1)
                if hasattr(self, "fragment_position_bias_table"):
                    relative_position_bias = (
                        relative_position_bias * fgate
                        + fragment_position_bias * (1 - fgate))
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                    + relative_position_bias.unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            else:
                attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                    + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class _SwinTransformerBlock3D(nn.Module):
        def __init__(self, dim, num_heads, window_size=(2, 7, 7),
                     shift_size=(0, 0, 0), mlp_ratio=4.0, qkv_bias=True,
                     qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                     act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                     jump_attention=False, frag_bias=False):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            self.mlp_ratio = mlp_ratio
            self.jump_attention = jump_attention
            self.norm1 = norm_layer(dim)
            self.attn = _WindowAttention3D(
                dim, window_size=self.window_size, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, frag_bias=frag_bias)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 \
                else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                            act_layer=act_layer, drop=drop)

        def forward_part1(self, x, mask_matrix, resized_window_size=None):
            B, D, H, W, C = x.shape
            window_size, shift_size = _get_window_size(
                (D, H, W),
                self.window_size if resized_window_size is None
                else resized_window_size,
                self.shift_size)
            x = self.norm1(x)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, Dp, Hp, Wp, _ = x.shape
            if any(i > 0 for i in shift_size):
                shifted_x = torch.roll(
                    x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                    dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None
            x_windows = _window_partition(shifted_x, window_size)
            gpi = _global_position_index(
                Dp, Hp, Wp, fragments=(1,) + window_size[1:],
                window_size=window_size, shift_size=shift_size, device=x.device)
            attn_windows = self.attn(
                x_windows, mask=attn_mask, fmask=gpi,
                resized_window_size=window_size
                if resized_window_size is not None else None)
            attn_windows = attn_windows.view(-1, *(window_size + (C,)))
            shifted_x = _window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
            if any(i > 0 for i in shift_size):
                x = torch.roll(
                    shifted_x,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3))
            else:
                x = shifted_x
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :D, :H, :W, :].contiguous()
            return x

        def forward_part2(self, x):
            return self.drop_path(self.mlp(self.norm2(x)))

        def forward(self, x, mask_matrix, resized_window_size=None):
            shortcut = x
            if not self.jump_attention:
                x = self.forward_part1(x, mask_matrix, resized_window_size)
                x = shortcut + self.drop_path(x)
            x = x + self.forward_part2(x)
            return x

    class _PatchMerging(nn.Module):
        def __init__(self, dim, norm_layer=nn.LayerNorm):
            super().__init__()
            self.dim = dim
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

        def forward(self, x):
            B, D, H, W, C = x.shape
            pad_input = (H % 2 == 1) or (W % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            x0 = x[:, :, 0::2, 0::2, :]
            x1 = x[:, :, 1::2, 0::2, :]
            x2 = x[:, :, 0::2, 1::2, :]
            x3 = x[:, :, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
            x = self.norm(x)
            x = self.reduction(x)
            return x

    class _BasicLayer(nn.Module):
        def __init__(self, dim, depth, num_heads, window_size=(1, 7, 7),
                     mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0,
                     attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                     downsample=None, jump_attention=False, frag_bias=False):
            super().__init__()
            self.window_size = window_size
            self.shift_size = tuple(i // 2 for i in window_size)
            self.depth = depth
            self.blocks = nn.ModuleList([
                _SwinTransformerBlock3D(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer, jump_attention=jump_attention,
                    frag_bias=frag_bias)
                for i in range(depth)])
            self.downsample = downsample
            if self.downsample is not None:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer)

        def forward(self, x, resized_window_size=None):
            B, C, D, H, W = x.shape
            window_size, shift_size = _get_window_size(
                (D, H, W),
                self.window_size if resized_window_size is None
                else resized_window_size,
                self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            Dp = int(np.ceil(D / window_size[0])) * window_size[0]
            Hp = int(np.ceil(H / window_size[1])) * window_size[1]
            Wp = int(np.ceil(W / window_size[2])) * window_size[2]
            attn_mask = _compute_mask(Dp, Hp, Wp, window_size, shift_size,
                                      x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, resized_window_size=resized_window_size)
            x = x.view(B, D, H, W, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")
            return x

    class _PatchEmbed3D(nn.Module):
        def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96,
                     norm_layer=None):
            super().__init__()
            self.patch_size = patch_size
            self.in_chans = in_chans
            self.embed_dim = embed_dim
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=patch_size)
            self.norm = norm_layer(embed_dim) if norm_layer is not None else None

        def forward(self, x):
            _, _, D, H, W = x.size()
            if W % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
            if H % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0,
                              self.patch_size[1] - H % self.patch_size[1]))
            if D % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0,
                              self.patch_size[0] - D % self.patch_size[0]))
            x = self.proj(x)
            if self.norm is not None:
                D, Wh, Ww = x.size(2), x.size(3), x.size(4)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
            return x

    class _SwinTransformer3D(nn.Module):
        """Trimmed SwinTransformer3D (inference-only; no pretrained loaders)."""

        def __init__(self, patch_size=(2, 6, 6), in_chans=3, embed_dim=96,
                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                     window_size=(8, 7, 7), mlp_ratio=4.0, qkv_bias=True,
                     qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                     drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                     patch_norm=True,
                     frag_biases=(True, True, True, False)):
            super().__init__()
            self.num_layers = len(depths)
            self.embed_dim = embed_dim
            self.patch_norm = patch_norm
            self.window_size = window_size
            self.patch_size = patch_size
            self.base_x_size = (32, 56 * patch_size[-1], 56 * patch_size[-1])

            self.patch_embed = _PatchEmbed3D(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            self.pos_drop = nn.Dropout(p=drop_rate)
            dpr = [x.item() for x in
                   torch.linspace(0, drop_path_rate, sum(depths))]
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = _BasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer], num_heads=num_heads[i_layer],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):
                                  sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=_PatchMerging
                    if i_layer < self.num_layers - 1 else None,
                    frag_bias=frag_biases[i_layer])
                self.layers.append(layer)
            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
            self.norm = norm_layer(self.num_features)

        def forward(self, x, multi=False, layer=-1, adaptive_window_size=True):
            if adaptive_window_size:
                resized_window_size = _get_adaptive_window_size(
                    self.window_size, tuple(x.shape[2:]), self.base_x_size)
            else:
                resized_window_size = None
            x = self.patch_embed(x)
            x = self.pos_drop(x)
            for mlayer in self.layers:
                x = mlayer(x.contiguous(), resized_window_size)
            x = rearrange(x, "n c d h w -> n d h w c")
            x = self.norm(x)
            x = rearrange(x, "n d h w c -> n c d h w")
            return x

    class _VQAHead(nn.Module):
        def __init__(self, in_channels=768, hidden_channels=64,
                     dropout_ratio=0.5):
            super().__init__()
            self.dropout_ratio = dropout_ratio
            self.dropout = nn.Dropout(p=dropout_ratio) \
                if dropout_ratio != 0 else None
            self.fc_hid = nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))
            self.fc_last = nn.Conv3d(hidden_channels, 1, (1, 1, 1))
            self.gelu = nn.GELU()
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        def forward(self, x):
            x = self.dropout(x)
            return self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))

    class _VQAEvaluator(nn.Module):
        """Matches checkpoint prefixes ``fragments_backbone.*`` + ``vqa_head.*``
        (DiViDeAddEvaluator, backbone_preserve_keys='fragments')."""

        def __init__(self, patch_size=(2, 6, 6)):
            super().__init__()
            self.fragments_backbone = _SwinTransformer3D(
                patch_size=patch_size, window_size=(8, 7, 7),
                drop_path_rate=0.0, frag_biases=(True, True, True, False))
            self.vqa_head = _VQAHead(in_channels=768, hidden_channels=64,
                                     dropout_ratio=0.5)

        def forward(self, x):
            feat = self.fragments_backbone(x)
            return self.vqa_head(feat)

    # ----- IQA branch: CPNet (convnext_tiny + patch heads) ----------------
    # Source: IQA/models/CPNetMulti.py

    class _CPNet(nn.Module):
        def __init__(self, csize=320, drop_path=0.0):
            super().__init__()
            import timm
            from einops.layers.torch import Rearrange

            self.backbone = timm.create_model(
                "convnext_tiny", pretrained=False, drop_path_rate=drop_path,
                features_only=True)
            self.rerange_layer = Rearrange("b c h w -> b (h w) c")
            self.avg_pool = nn.AdaptiveAvgPool2d(csize // 32)
            embed_dim = 1440
            self.head_score = nn.Sequential(
                nn.Linear(embed_dim, 384), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(384, 1), nn.ReLU())
            self.head_weight = nn.Sequential(
                nn.Linear(embed_dim, 384), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(384, 1), nn.Sigmoid())

        def forward(self, x):
            feats = self.backbone(x)
            feats = [self.avg_pool(feat) for feat in feats]
            feats = torch.cat(feats, dim=1)
            feats = self.rerange_layer(feats)
            scores = self.head_score(feats)
            weights = self.head_weight(feats)
            y = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)
            return y

    # ----- Preprocessing (FAST-VQA fragment / clip sampling) --------------
    # Source: VQA/fastvqa/datasets/fusion_datasets.py

    class _FragmentSampleFrames:
        def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1):
            self.fragments_t = fragments_t
            self.fsize_t = fsize_t
            self.frame_interval = frame_interval
            self.num_clips = num_clips

        def _get_frame_indices(self, num_frames):
            tlength = num_frames // self.fragments_t
            tgrids = np.array([tlength * i for i in range(self.fragments_t)],
                              dtype=np.int32)
            if tlength > self.fsize_t * self.frame_interval:
                rnd_t = np.array(
                    [(tlength - self.fsize_t * self.frame_interval) // 2]
                    * len(tgrids))
            else:
                rnd_t = np.zeros(len(tgrids), dtype=np.int32)
            ranges_t = (np.arange(self.fsize_t)[None, :] * self.frame_interval
                        + rnd_t[:, None] + tgrids[:, None])
            return np.concatenate(ranges_t)

        def __call__(self, total_frames, start_index=0):
            frame_inds = [self._get_frame_indices(total_frames)
                          for _ in range(self.num_clips)]
            frame_inds = np.concatenate(frame_inds)
            frame_inds = np.mod(frame_inds + start_index, total_frames)
            return frame_inds.astype(np.int32)

    def _get_spatial_fragments(video, fragments_h=7, fragments_w=7,
                               fsize_h=32, fsize_w=32, aligned=32):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w
        if video.shape[1] == 1:
            aligned = 1
        dur_t, res_h, res_w = video.shape[-3:]
        ratio = min(res_h / size_h, res_w / size_w)
        if ratio < 1:
            ovideo = video
            video = torch.nn.functional.interpolate(
                video / 255.0, scale_factor=1 / ratio, mode="bilinear")
            video = (video * 255.0).type_as(ovideo)
        assert dur_t % aligned == 0
        size = size_h, size_w
        hlength, wlength = res_h // fragments_h, res_w // fragments_w
        hgrids = torch.LongTensor(
            [min(hlength * i, res_h - fsize_h) for i in range(fragments_h)])
        wgrids = torch.LongTensor(
            [min(wlength * i, res_w - fsize_w) for i in range(fragments_w)])
        if hlength > fsize_h:
            rnd_h = (hlength - fsize_h) // 2 * torch.ones(
                (len(hgrids), len(wgrids), dur_t // aligned)).int()
        else:
            rnd_h = torch.zeros(
                (len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = (wlength - fsize_w) // 2 * torch.ones(
                (len(hgrids), len(wgrids), dur_t // aligned)).int()
        else:
            rnd_w = torch.zeros(
                (len(hgrids), len(wgrids), dur_t // aligned)).int()
        target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                for t in range(dur_t // aligned):
                    t_s, t_e = t * aligned, (t + 1) * aligned
                    h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                    w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                    h_so = hs + rnd_h[i][j][t]
                    w_so = ws + rnd_w[i][j][t]
                    target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                        :, t_s:t_e, h_so:h_so + fsize_h, w_so:w_so + fsize_w]
        return target_video


class ZoomVQAModule(PipelineModule):
    name = "zoomvqa"
    description = (
        "Zoom-VQA dual-branch IQA+VQA late-fusion blind VQA (CVPRW 2023)"
    )
    default_config = {
        "subsample": 16,          # IQA frames sampled from a video
        "iqa_rsize": 512,
        "iqa_csize": 320,
        "vqa_rsize": 480,         # min-edge resize before spatial fragments
        "vqa_patch_size": 6,      # -> fragment fsize = patch_size * 8 = 48
        "vqa_clip_len": 32,       # frames per temporal fragment
        "vqa_num_clips": 4,       # temporal fragments / clips
        "vqa_frame_interval": 2,
        "fusion_iqa_weight": 0.5,
        "device": "auto",
    }
    metric_groups = {
        "zoomvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._iqa = None
        self._vqa = None
        self._device = None
        self._iqa_tf = None

    def setup(self) -> None:
        if self.test_mode:
            return

        if not _TORCH_OK:
            self._backend = "unavailable"
            logger.warning(
                "Zoom-VQA unavailable: torch/timm/einops not importable")
            return

        try:
            import timm  # noqa: F401
            import decord  # noqa: F401
            from torchvision import transforms
            from huggingface_hub import hf_hub_download
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("Zoom-VQA unavailable: missing dependency (%s)", e)
            return

        try:
            from ayase.runtime import resolve_torch_device
            self._device = resolve_torch_device(self.config.get("device", "auto"))
        except Exception:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            iqa_path = hf_hub_download(repo_id=HF_REPO, filename=IQA_WEIGHT)
            vqa_path = hf_hub_download(repo_id=HF_REPO, filename=VQA_WEIGHT)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("Zoom-VQA unavailable: checkpoints not found (%s)", e)
            return

        try:
            # IQA branch
            iqa = _CPNet(csize=self.config.get("iqa_csize", 320))
            isd = torch.load(iqa_path, map_location="cpu",
                             weights_only=False)["state_dict"]
            isd = {k.replace("module.", ""): v for k, v in isd.items()}
            iqa.load_state_dict(isd, strict=True)
            iqa.eval().to(self._device)

            # VQA branch
            ps = int(self.config.get("vqa_patch_size", 6))
            vqa = _VQAEvaluator(patch_size=(2, ps, ps))
            vsd = torch.load(vqa_path, map_location="cpu",
                             weights_only=False)["state_dict"]
            vsd = {k.replace("module.", ""): v for k, v in vsd.items()}
            vqa.load_state_dict(vsd, strict=True)
            vqa.eval().to(self._device)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("Zoom-VQA unavailable: state_dict load failed (%s)", e)
            return

        self._iqa = iqa
        self._vqa = vqa
        self._iqa_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.config.get("iqa_rsize", 512)),
            transforms.CenterCrop(self.config.get("iqa_csize", 320)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self._ml_available = True
        self._backend = "real"
        logger.info("Zoom-VQA initialised (real backend, device=%s)",
                    self._device)

    # ---- inference helpers ----------------------------------------------

    def _iqa_score_from_frames(self, frames_rgb) -> Optional[float]:
        """frames_rgb: list of HxWx3 uint8 RGB numpy arrays."""
        from PIL import Image

        if not frames_rgb:
            return None
        batch = torch.stack([self._iqa_tf(Image.fromarray(f))
                             for f in frames_rgb]).to(self._device)
        with torch.no_grad():
            out = self._iqa(batch)  # (N, 1)
        return float(out.mean().item())

    def _run_iqa_image(self, path: str) -> Optional[float]:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._iqa_score_from_frames([rgb])

    def _run_iqa_video(self, path: str) -> Optional[float]:
        import decord

        vr = decord.VideoReader(str(path))
        n = len(vr)
        if n == 0:
            return None
        k = min(int(self.config.get("subsample", 16)), n)
        idxs = np.linspace(0, n - 1, k).astype(int).tolist()
        batch = vr.get_batch(idxs)  # tolerate a leaked global decord torch-bridge
        frames = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.cpu().numpy()  # (k, H, W, 3) RGB
        return self._iqa_score_from_frames([frames[i] for i in range(len(idxs))])

    def _run_vqa(self, path: str) -> Optional[float]:
        import decord

        clip_len = int(self.config.get("vqa_clip_len", 32))
        num_clips = int(self.config.get("vqa_num_clips", 4))
        interval = int(self.config.get("vqa_frame_interval", 2))
        rsize = int(self.config.get("vqa_rsize", 480))
        fsize = int(self.config.get("vqa_patch_size", 6)) * 8

        vr = decord.VideoReader(str(path))
        n = len(vr)
        if n == 0:
            return None
        # Temporal: fsize_t=clip_len, fragments_t=num_clips (per vqa_mine.py)
        sampler = _FragmentSampleFrames(clip_len, num_clips, interval)
        frame_inds = sampler(n)
        batch = vr.get_batch(frame_inds.tolist())  # tolerate a leaked global decord torch-bridge
        frames = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.cpu().numpy()  # (T,H,W,3) RGB
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).float()  # C,T,H,W

        # min-edge resize to rsize
        h, w = video.shape[-2:]
        if h > w:
            t_h, t_w = int(h / (w / rsize)), rsize
        else:
            t_h, t_w = rsize, int(w / (h / rsize))
        video = torch.nn.functional.interpolate(
            video / 255.0, size=(t_h, t_w), mode="bilinear")
        video = (video * 255.0)

        # Spatial fragments
        sampled = _get_spatial_fragments(
            video, fragments_h=7, fragments_w=7, fsize_h=fsize, fsize_w=fsize,
            aligned=clip_len)
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])
        sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        # reshape into clips: (C, num_clips*clip_len, H, W) -> (num_clips, C, clip_len, H, W)
        sampled = sampled.reshape(sampled.shape[0], num_clips, -1,
                                  *sampled.shape[2:]).transpose(0, 1)
        sampled = sampled.to(self._device)
        with torch.no_grad():
            out = self._vqa(sampled)
        return float(out.mean().item())

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "real":
            return sample

        w = float(self.config.get("fusion_iqa_weight", 0.5))
        try:
            if getattr(sample, "is_video", False):
                iqa = self._run_iqa_video(str(sample.path))
                vqa = self._run_vqa(str(sample.path))
                if iqa is not None and vqa is not None:
                    score = w * iqa + (1.0 - w) * vqa
                elif iqa is not None:
                    score = iqa
                elif vqa is not None:
                    score = vqa
                else:
                    score = None
            else:
                # Single image: only the IQA branch is defined.
                score = self._run_iqa_image(str(sample.path))
            if score is not None:
                sample.quality_metrics.zoomvqa_score = float(score)
        except Exception as e:
            logger.warning("Zoom-VQA failed for %s: %s", sample.path, e)
        return sample
