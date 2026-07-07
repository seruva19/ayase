"""TRAJAN — point-trajectory autoencoder for assessing generated videos (ICLR 2025).

"Direct Motion Models for Assessing Generated Videos" (DeepMind, arXiv:2505.00209).
TRAJAN estimates point tracks with BootsTAPIR, then (auto)encodes the trajectory
set with a Perceiver-style transformer that reconstructs held-out query-point
trajectories; the realism/consistency score is the Average Jaccard (AJ) of the
autoencoder's track reconstruction. Higher AJ = motion that the learned motion
prior can faithfully reconstruct (i.e. more physically plausible / consistent).

This is the ONLY computation emitted under the TRAJAN name. A CoTracker/LK
point-track + jerk/velocity-smoothness aggregation is a *different* metric, not
TRAJAN's autoencoder-reconstruction score, so it is not emitted here.

PURE-PYTORCH backend
--------------------
DeepMind ships TRAJAN as JAX/Flax, which does not run on this (Windows) platform.
This module is a faithful **pure-torch reimplementation** — it imports **no jax**
at runtime. It has two vendored, self-contained torch model definitions:

  * BootsTAPIR (point tracker): DeepMind's official torch port of TAPIR, loading
    the official ``bootstapir_checkpoint_v2.pt`` torch checkpoint
    (GCS: ``dm-tapnet/bootstap/``) — fetched to ``models/trajan/``.
  * Track autoencoder: a from-scratch torch reimplementation of the Flax
    ``TrackAutoEncoder`` (Perceiver cross/self-attention encoder + decoder), whose
    weights are converted from the Flax checkpoint ``track_autoencoder_ckpt.npz``
    (mirrored at ``AkaneTendo25/ayase-models``; Flax Dense kernels transposed,
    DenseGeneral qkv/out reshaped, LayerNorm/RMSNorm scale mapped 1:1).

VALIDATION (decisive evidence of faithfulness)
----------------------------------------------
The torch port was cross-checked against the ORIGINAL JAX/Flax model on a fixed
synthetic track tensor (jax[cpu]+flax installed temporarily, never shipped):

  * encoder latents:            max abs diff  8.5e-6
  * decoder tracks (no dither): max abs diff  7.7e-7  (isolates architecture)
  * full end-to-end w/ dither:  max abs diff  7.2e-7  on tracks, 9.1e-6 on logits

All far below the 1e-3 acceptance tolerance, so AJ matches to << 0.01. The one
non-obvious detail is the decoder's ``discretize`` step, which adds a
``jax.random.uniform(PRNGKey(0))`` quantization dither: this is reproduced
bit-for-bit by a numpy Threefry-2x32 implementation (``_threefry_uniform``,
verified byte-identical to jax). A second subtlety: the frame term feeding the
query encoder uses integer floor-division (``//``), which is exactly 0 for all
frames < 150 — replicated here.

If torch is not importable, or the checkpoints cannot be obtained, the module
reports itself unavailable and does NOT populate ``trajan_score`` (no proxy, no
heuristic).

Output field: ``trajan_score`` (Average Jaccard in [0, 1]; real backend only).
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# BootsTAPIR torch checkpoint (point tracker producing support/target tracks).
_BOOTSTAPIR_URL = (
    "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
)
_BOOTSTAPIR_REL = "trajan/bootstapir_checkpoint_v2.pt"

# Track-autoencoder weights (DeepMind Flax params, mirrored). Flat
# "path/to/param" -> array npz, converted to a torch state_dict at load time.
_AE_REPO = "AkaneTendo25/ayase-models"
_AE_FILE = "weights/trajan/track_autoencoder_ckpt.npz"

# Cache for the lazily-built torch backend (classes + helpers).
_BACKEND = None


def _recover_tree(flat_dict: dict) -> dict:
    """Rebuild a nested param tree from ``"a/b/c" -> array`` flat entries."""
    tree: dict = {}
    for k, v in flat_dict.items():
        parts = k.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return tree


# --------------------------------------------------------------------------- #
# jax-faithful Threefry-2x32 dither (reproduces jax.random.uniform(PRNGKey(0))).
# Verified byte-identical to jax 0.4.35 output. Pure numpy — no jax at runtime.
# --------------------------------------------------------------------------- #
_U32 = np.uint32


def _tf_round(v, r):
    v0 = v[0] + v[1]
    v1 = ((v[1] << _U32(r)) | (v[1] >> _U32(32 - r))) ^ v0
    return [v0, v1]


def _threefry_uniform(shape, key=(0, 0)):
    """Pure-numpy equivalent of ``jax.random.uniform(PRNGKey(key), shape)`` (f32)."""
    size = int(np.prod(shape))
    odd = size % 2
    n = size + odd
    c = np.arange(n, dtype=_U32)
    x = [c[: n // 2], c[n // 2:]]
    k0, k1 = _U32(key[0]), _U32(key[1])
    ks = [k0, k1, k0 ^ k1 ^ _U32(0x1BD11BDA)]
    rot = [np.array([13, 15, 26, 6], _U32), np.array([17, 29, 16, 24], _U32)]
    schedule = [(rot[0], 1, 2), (rot[1], 2, 0), (rot[0], 0, 1),
                (rot[1], 1, 2), (rot[0], 2, 0)]
    with np.errstate(over="ignore"):
        x = [x[0] + ks[0], x[1] + ks[1]]
        for i, (rr, a, b) in enumerate(schedule):
            for r in rr:
                x = _tf_round(x, r)
            x[0] = x[0] + ks[a]
            x[1] = x[1] + ks[b] + _U32(i + 1)
        bits = np.concatenate([x[0], x[1]])[:size]
        float_bits = (bits >> _U32(9)) | _U32(0x3F800000)
        floats = float_bits.view(np.float32) - np.float32(1.0)
    return floats.reshape(shape)


# --------------------------------------------------------------------------- #
# TAP-Vid Average Jaccard (vendored numpy port of tapnet compute_tapvid_metrics).
# --------------------------------------------------------------------------- #
def _compute_tapvid_metrics(query_points, gt_occluded, gt_tracks,
                            pred_occluded, pred_tracks, query_mode="strided"):
    summing_axis = (1, 2)
    metrics = {}
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == "first":
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    else:
        query_frame_to_eval_frames = 1 - eye
    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)
        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=summing_axis)
        gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=summing_axis)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics[f"jaccard_{thresh}"] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    return metrics


def _build_backend():
    """Lazily import torch and build the vendored TAPIR + track-autoencoder defs.

    Kept behind a function so the module imports without torch (graceful path).
    Returns a namespace object exposing the model classes + converter + helpers.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    import functools
    import math
    from types import SimpleNamespace
    from typing import NamedTuple, Sequence, Tuple

    import torch
    from torch import nn
    import torch.nn.functional as F

    # ===================== vendored torch BootsTAPIR (TAPIR) ================= #
    def bilinear(x, resolution):
        b, t, h, w, c = x.size()
        x = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
        x = F.interpolate(x, size=resolution, mode="bilinear", align_corners=False)
        b, _, h, w = x.size()
        return x.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)

    def map_coordinates_3d(feats, coordinates):
        x = feats.permute(0, 4, 1, 2, 3)
        y = coordinates[:, :, None, None, :].float()
        y[..., 0] += 0.5
        y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
        y = torch.flip(y, dims=(-1,))
        return (F.grid_sample(x, y, mode="bilinear", align_corners=False,
                              padding_mode="border").squeeze(dim=(3, 4)).permute(0, 2, 1))

    def map_coordinates_2d(feats, coordinates):
        n, t, h, w, c = feats.shape
        x = feats.permute(0, 1, 4, 2, 3).view(n * t, c, h, w)
        n, p, t, s, xy = coordinates.shape
        y = coordinates.permute(0, 2, 1, 3, 4).reshape(n * t, p, s, xy)
        y = 2 * (y / h) - 1
        y = torch.flip(y, dims=(-1,)).float()
        out = F.grid_sample(x, y, mode="bilinear", align_corners=False, padding_mode="zeros")
        _, c, _, _ = out.shape
        return out.permute(0, 2, 3, 1).view(n, t, p, s, c).permute(0, 2, 1, 3, 4)

    def soft_argmax_heatmap_batched(softmax_val, threshold=5):
        b, h, w, d1, d2 = softmax_val.shape
        yy, xx = torch.meshgrid(torch.arange(d1, device=softmax_val.device),
                                torch.arange(d2, device=softmax_val.device), indexing="ij")
        coords = torch.stack([xx + 0.5, yy + 0.5], dim=-1).to(softmax_val.device)
        argmax_pos = torch.argmax(softmax_val.reshape(b, h, w, -1), dim=-1)
        pos = coords.reshape(-1, 2)[argmax_pos]
        valid = (torch.sum(torch.square(coords[None, None, None] - pos[:, :, :, None, None]),
                           dim=-1, keepdims=True) < threshold ** 2)
        weighted = torch.sum(coords[None, None, None] * valid * softmax_val[..., None], dim=(3, 4))
        sw = torch.maximum(torch.sum(valid * softmax_val[..., None], dim=(3, 4)),
                           torch.tensor(1e-12, device=softmax_val.device))
        return weighted / sw

    def heatmaps_to_points(all_pairs_softmax, image_shape, threshold=5, query_points=None):
        out_points = soft_argmax_heatmap_batched(all_pairs_softmax, threshold)
        fgs = all_pairs_softmax.shape[1:]
        out_points = convert_grid_coordinates(out_points.detach(), fgs[3:1:-1], image_shape[3:1:-1])
        if query_points is not None:
            query_frame = convert_grid_coordinates(
                query_points.detach(), image_shape[1:4], fgs[1:4], coordinate_format="tyx")[..., 0:1]
            query_frame = torch.round(query_frame)
            frame_indices = torch.arange(image_shape[1], device=query_frame.device)[None, None, :]
            is_query_point = (query_frame == frame_indices)[:, :, :, None]
            out_points = (out_points * ~is_query_point
                          + torch.flip(query_points[:, :, None], dims=(-1,))[..., 0:2] * is_query_point)
        return out_points

    def is_same_res(r1, r2):
        return all([x == y for x, y in zip(r1, r2)])

    def convert_grid_coordinates(coords, input_grid_size, output_grid_size, coordinate_format="xy"):
        if isinstance(input_grid_size, tuple):
            input_grid_size = torch.tensor(input_grid_size, device=coords.device)
        if isinstance(output_grid_size, tuple):
            output_grid_size = torch.tensor(output_grid_size, device=coords.device)
        return coords * output_grid_size / input_grid_size

    def generate_default_resolutions(full_size, train_size):
        if all([x == y for x, y in zip(train_size, full_size)]):
            return [train_size]
        size_ratio = np.array(full_size) / np.array(train_size)
        num_levels = int(np.ceil(np.max(np.log2(size_ratio))) + 1)
        if num_levels <= 1:
            return [train_size]
        h, w = full_size[0:2]
        ll_h, ll_w = train_size[0:2]
        return [(int(round((ll_h * (h / ll_h) ** (i / (num_levels - 1))) // 8)) * 8,
                 int(round((ll_w * (w / ll_w) ** (i / (num_levels - 1))) // 8)) * 8)
                for i in range(num_levels)]

    class ExtraConvBlock(nn.Module):
        def __init__(self, cd, cm):
            super().__init__()
            self.layer_norm = nn.LayerNorm(cd, elementwise_affine=True, bias=True)
            self.conv = nn.Conv2d(cd, cd * cm, 3, 1, 1)
            self.conv_1 = nn.Conv2d(cd * cm, cd, 3, 1, 1)

        def forward(self, x):
            x = self.layer_norm(x).permute(0, 3, 1, 2)
            x = x + self.conv_1(F.gelu(self.conv(x), approximate="tanh"))
            return x.permute(0, 2, 3, 1)

    class ExtraConvs(nn.Module):
        def __init__(self, num_layers=5, cd=256, cm=4):
            super().__init__()
            self.blocks = nn.ModuleList([ExtraConvBlock(cd, cm) for _ in range(num_layers)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    class ConvChannelsMixer(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.mlp2_up = nn.Linear(c, c * 4)
            self.mlp2_down = nn.Linear(c * 4, c)

        def forward(self, x):
            return self.mlp2_down(F.gelu(self.mlp2_up(x), approximate="tanh"))

    class PIPsConvBlock(nn.Module):
        def __init__(self, c, ks=3, block_idx=None):
            super().__init__()
            self.layer_norm = nn.LayerNorm(c, elementwise_affine=True, bias=False)
            self.mlp1_up = nn.Conv1d(c, c * 4, ks, 1, 1, groups=c)
            self.mlp1_up_1 = nn.Conv1d(c * 4, c * 4, ks, 1, 1, groups=c * 4)
            self.layer_norm_1 = nn.LayerNorm(c, elementwise_affine=True, bias=False)
            self.conv_channels_mixer = ConvChannelsMixer(c)

        def forward(self, x):
            to_skip = x
            x = self.layer_norm(x).permute(0, 2, 1)
            x = F.gelu(self.mlp1_up(x), approximate="tanh")
            x = self.mlp1_up_1(x).permute(0, 2, 1)
            x = x[..., 0::4] + x[..., 1::4] + x[..., 2::4] + x[..., 3::4]
            x = x + to_skip
            return x + self.conv_channels_mixer(self.layer_norm_1(x))

    class PIPSMLPMixer(nn.Module):
        def __init__(self, in_c, out_c, hidden_dim=512, num_blocks=12, ks=3):
            super().__init__()
            self.linear = nn.Linear(in_c, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True, bias=False)
            self.linear_1 = nn.Linear(hidden_dim, out_c)
            self.blocks = nn.ModuleList([PIPsConvBlock(hidden_dim, ks, block_idx=i)
                                         for i in range(num_blocks)])

        def forward(self, x):
            x = self.linear(x)
            for b in self.blocks:
                x = b(x)
            return self.linear_1(self.layer_norm(x))

    class BlockV2(nn.Module):
        def __init__(self, ci, co, stride, use_projection):
            super().__init__()
            self.padding = (1, 1, 1, 1) if stride == 1 else (0, 2, 0, 2)
            self.use_projection = use_projection
            if use_projection:
                self.proj_conv = nn.Conv2d(ci, co, 1, stride, 0, bias=False)
            self.bn_0 = nn.InstanceNorm2d(ci, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            self.conv_0 = nn.Conv2d(ci, co, 3, stride, 0, bias=False)
            self.conv_1 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
            self.bn_1 = nn.InstanceNorm2d(co, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        def forward(self, inputs):
            x = shortcut = inputs
            x = torch.relu(self.bn_0(x))
            if self.use_projection:
                shortcut = self.proj_conv(x)
            x = self.conv_0(F.pad(x, self.padding))
            x = torch.relu(self.bn_1(x))
            return self.conv_1(x) + shortcut

    class BlockGroup(nn.Module):
        def __init__(self, ci, co, num_blocks, stride, use_projection):
            super().__init__()
            self.blocks = nn.ModuleList(
                [BlockV2(ci if i == 0 else co, co, (1 if i else stride), (i == 0 and use_projection))
                 for i in range(num_blocks)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    class ResNet(nn.Module):
        def __init__(self, blocks_per_group, channels_per_group, use_projection, strides):
            super().__init__()
            self.initial_conv = nn.Conv2d(3, channels_per_group[0], (7, 7), 2, 0, bias=False)
            self.block_groups = nn.ModuleList([
                BlockGroup(channels_per_group[i - 1] if i > 0 else 64, channels_per_group[i],
                           blocks_per_group[i], strides[i], use_projection[i])
                for i in range(len(strides))])

        def forward(self, inputs):
            result = {}
            out = self.initial_conv(F.pad(inputs, (2, 4, 2, 4)))
            for block_id, bg in enumerate(self.block_groups):
                out = bg(out)
                result[f"resnet_unit_{block_id}"] = out
            return result

    class FeatureGrids(NamedTuple):
        lowres: Sequence
        hires: Sequence
        resolutions: Sequence

    class QueryFeatures(NamedTuple):
        lowres: Sequence
        hires: Sequence
        resolutions: Sequence

    class TAPIR(nn.Module):
        def __init__(self, pyramid_level=1, softmax_temperature=10.0,
                     initial_resolution=(256, 256), feature_extractor_chunk_size=10,
                     extra_convs=True, num_pips_iter=4):
            super().__init__()
            self.highres_dim = 128
            self.num_pips_iter = num_pips_iter
            self.pyramid_level = pyramid_level
            self.softmax_temperature = softmax_temperature
            self.initial_resolution = tuple(initial_resolution)
            self.feature_extractor_chunk_size = feature_extractor_chunk_size
            self.resnet_torch = ResNet((2, 2, 2, 2), (64, 128, 256, 256),
                                       (True, True, True, True), (1, 2, 2, 1))
            self.torch_cost_volume_track_mods = nn.ModuleDict({
                "hid1": nn.Conv2d(1, 16, 3, 1, 1), "hid2": nn.Conv2d(16, 1, 3, 1, 1),
                "hid3": nn.Conv2d(16, 32, 3, 2, 0), "hid4": nn.Linear(32, 16),
                "occ_out": nn.Linear(16, 2)})
            dim = 4 + 128 + 256
            input_dim = dim + (self.pyramid_level + 2) * 49
            self.torch_pips_mixer = PIPSMLPMixer(input_dim, dim)
            self.extra_convs = ExtraConvs() if extra_convs else None

        def get_feature_grids(self, video):
            refinement_resolutions = generate_default_resolutions(video.shape[2:4], self.initial_resolution)
            all_res = [self.initial_resolution] + list(refinement_resolutions)
            feature_grid, hires_feats, resize_im_shape = [], [], []
            curr_resolution = (-1, -1)
            latent = hires = video_resize = None
            for resolution in all_res:
                if not is_same_res(curr_resolution, resolution):
                    video_resize = video if is_same_res(curr_resolution, video.shape[-3:-1]) \
                        else bilinear(video, resolution)
                    curr_resolution = resolution
                    n, f, h, w, c = video_resize.shape
                    vr = video_resize.view(n * f, h, w, c).permute(0, 3, 1, 2)
                    latent_list, hires_list = [], []
                    cs = self.feature_extractor_chunk_size
                    for s in range(0, vr.shape[0], cs):
                        ro = self.resnet_torch(vr[s:s + cs])
                        latent_list.append(ro["resnet_unit_3"].permute(0, 2, 3, 1))
                        hires_list.append(ro["resnet_unit_1"].permute(0, 2, 3, 1))
                    latent = torch.cat(latent_list, dim=0)
                    hires = torch.cat(hires_list, dim=0)
                    if self.extra_convs:
                        latent = self.extra_convs(latent)
                    latent = latent / torch.sqrt(torch.maximum(
                        torch.sum(torch.square(latent), axis=-1, keepdims=True),
                        torch.tensor(1e-12, device=latent.device)))
                    hires = hires / torch.sqrt(torch.maximum(
                        torch.sum(torch.square(hires), axis=-1, keepdims=True),
                        torch.tensor(1e-12, device=hires.device)))
                    latent = latent.view(n, f, *latent.shape[1:])
                    hires = hires.view(n, f, *hires.shape[1:])
                feature_grid.append(latent)
                hires_feats.append(hires)
                resize_im_shape.append(video_resize.shape[2:4])
            return FeatureGrids(tuple(feature_grid), tuple(hires_feats), tuple(resize_im_shape))

        def get_query_features(self, video, query_points, feature_grids):
            shape = video.shape
            curr_resolution = (-1, -1)
            query_feats, hires_query_feats = [], []
            for i, resolution in enumerate(feature_grids.resolutions):
                if is_same_res(curr_resolution, resolution):
                    query_feats.append(query_feats[-1])
                    hires_query_feats.append(hires_query_feats[-1])
                    continue
                pig = convert_grid_coordinates(query_points, shape[1:4],
                                               feature_grids.lowres[i].shape[1:4], "tyx")
                pigh = convert_grid_coordinates(query_points, shape[1:4],
                                                feature_grids.hires[i].shape[1:4], "tyx")
                query_feats.append(map_coordinates_3d(feature_grids.lowres[i], pig))
                hires_query_feats.append(map_coordinates_3d(feature_grids.hires[i], pigh))
            return QueryFeatures(tuple(query_feats), tuple(hires_query_feats),
                                 tuple(feature_grids.resolutions))

        def forward(self, video, query_points, query_chunk_size=64, feature_grids=None):
            if feature_grids is None:
                feature_grids = self.get_feature_grids(video)
            qf = self.get_query_features(video, query_points, feature_grids)
            traj = self.estimate_trajectories(video.shape[-3:-1], feature_grids, qf,
                                              query_points, query_chunk_size)
            p = self.num_pips_iter
            return dict(
                occlusion=torch.mean(torch.stack(traj["occlusion"][p::p]), dim=0),
                tracks=torch.mean(torch.stack(traj["tracks"][p::p]), dim=0),
                expected_dist=torch.mean(torch.stack(traj["expected_dist"][p::p]), dim=0))

        def estimate_trajectories(self, video_size, feature_grids, query_features,
                                  query_points_in_video, query_chunk_size):
            def train2orig(x):
                return convert_grid_coordinates(x, self.initial_resolution[::-1],
                                                video_size[::-1], "xy")
            num_iters = self.num_pips_iter * (len(feature_grids.lowres) - 1)
            occ_iters = [[] for _ in range(num_iters + 1)]
            pts_iters = [[] for _ in range(num_iters + 1)]
            expd_iters = [[] for _ in range(num_iters + 1)]
            infer = functools.partial(self.tracks_from_cost_volume,
                                      im_shp=feature_grids.lowres[0].shape[0:2] + self.initial_resolution + (3,))
            num_queries = query_features.lowres[0].shape[1]
            perm = torch.randperm(num_queries)
            inv_perm = torch.zeros_like(perm)
            inv_perm[perm] = torch.arange(num_queries)
            for ch in range(0, num_queries, query_chunk_size):
                perm_chunk = perm[ch:ch + query_chunk_size]
                chunk = query_features.lowres[0][:, perm_chunk]
                if query_points_in_video is not None:
                    iqp = query_points_in_video[:, perm[ch:ch + query_chunk_size]]
                    nf = feature_grids.lowres[0].shape[1]
                    iqp = convert_grid_coordinates(iqp, (nf,) + tuple(video_size),
                                                   (nf,) + self.initial_resolution, "tyx")
                else:
                    iqp = None
                points, occlusion, expected_dist = infer(chunk, feature_grids.lowres[0], iqp)
                pts_iters[0].append(train2orig(points))
                occ_iters[0].append(occlusion)
                expd_iters[0].append(expected_dist)
                mixer_feats = None
                for i in range(num_iters):
                    fl = i // self.num_pips_iter + 1
                    queries = [query_features.hires[fl][:, perm_chunk],
                               query_features.lowres[fl][:, perm_chunk]]
                    for _ in range(self.pyramid_level):
                        queries.append(queries[-1])
                    pyramid = [feature_grids.hires[fl], feature_grids.lowres[fl]]
                    for _ in range(self.pyramid_level):
                        pyramid.append(F.avg_pool3d(pyramid[-1], (2, 2, 1), (2, 2, 1), 0))
                    points, occlusion, expected_dist, mixer_feats = self.refine_pips(
                        queries, pyramid, points, occlusion, expected_dist,
                        self.initial_resolution, mixer_feats, feature_grids.resolutions[fl])
                    pts_iters[i + 1].append(train2orig(points))
                    occ_iters[i + 1].append(occlusion)
                    expd_iters[i + 1].append(expected_dist)
                    if (i + 1) % self.num_pips_iter == 0:
                        mixer_feats = None
                        expected_dist = expd_iters[0][-1]
                        occlusion = occ_iters[0][-1]
            occlusion, points, expd = [], [], []
            for i in range(len(occ_iters)):
                occlusion.append(torch.cat(occ_iters[i], dim=1)[:, inv_perm])
                points.append(torch.cat(pts_iters[i], dim=1)[:, inv_perm])
                expd.append(torch.cat(expd_iters[i], dim=1)[:, inv_perm])
            return dict(occlusion=occlusion, tracks=points, expected_dist=expd)

        def refine_pips(self, target_feature, pyramid, pos_guess, occ_guess, expd_guess,
                        orig_hw, last_iter, resize_hw):
            orig_h, orig_w = orig_hw
            resized_h, resized_w = resize_hw
            corrs_pyr = []
            for pyridx, (query, grid) in enumerate(zip(target_feature, pyramid)):
                coords = convert_grid_coordinates(pos_guess, (orig_w, orig_h), grid.shape[-2:-4:-1])
                coords = torch.flip(coords, dims=(-1,))
                last_iter_query = None
                if last_iter is not None:
                    last_iter_query = last_iter[..., :self.highres_dim] if pyridx == 0 \
                        else last_iter[..., self.highres_dim:]
                ctxy, ctxx = torch.meshgrid(torch.arange(-3, 4), torch.arange(-3, 4), indexing="ij")
                ctx = torch.stack([ctxy, ctxx], dim=-1).reshape(-1, 2).to(coords.device)
                coords2 = coords.unsqueeze(3) + ctx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                neighborhood = map_coordinates_2d(grid, coords2)
                if last_iter_query is None:
                    patches = torch.einsum("bnfsc,bnc->bnfs", neighborhood, query)
                else:
                    patches = torch.einsum("bnfsc,bnfc->bnfs", neighborhood, last_iter_query)
                corrs_pyr.append(patches)
            corrs_chunked = torch.concatenate(corrs_pyr, dim=-1)
            occ_guess_input = occ_guess[..., None]
            expd_guess_input = expd_guess[..., None]
            if last_iter is None:
                both = torch.cat([target_feature[0], target_feature[1]], axis=-1)
                mlp_input_features = torch.tile(both.unsqueeze(2), (1, 1, corrs_chunked.shape[-2], 1))
            else:
                mlp_input_features = last_iter
            pos_guess_input = torch.zeros_like(pos_guess)
            mlp_input = torch.cat([pos_guess_input, occ_guess_input, expd_guess_input,
                                   mlp_input_features, corrs_chunked], axis=-1)
            b, n = mlp_input.shape[0], mlp_input.shape[1]
            x = mlp_input.reshape(b * n, mlp_input.shape[2], mlp_input.shape[3])
            res = self.torch_pips_mixer(x.float())
            res = res.reshape(b, n, res.shape[1], res.shape[2])
            pos_update = convert_grid_coordinates(res[..., :2], (resized_w, resized_h), (orig_w, orig_h))
            return (pos_update + pos_guess, res[..., 2] + occ_guess, res[..., 3] + expd_guess,
                    res[..., 4:] + (mlp_input_features if last_iter is None else last_iter))

        def tracks_from_cost_volume(self, interp_feature, feature_grid, query_points, im_shp=None):
            mods = self.torch_cost_volume_track_mods
            cost_volume = torch.einsum("bnc,bthwc->tbnhw", interp_feature, feature_grid)
            t, b, n, h, w = cost_volume.shape
            cost_volume = cost_volume.reshape(t * b * n, h, w, 1).permute(0, 3, 1, 2)
            occlusion = torch.relu(mods["hid1"](cost_volume))
            pos = mods["hid2"](occlusion).permute(0, 2, 3, 1)
            pos = pos.reshape(t, b, n, h, w).permute(1, 2, 0, 3, 4)
            pos_sm = pos.reshape(pos.size(0), pos.size(1), pos.size(2), -1)
            softmaxed = F.softmax(pos_sm * self.softmax_temperature, dim=-1)
            pos = softmaxed.view_as(pos)
            points = heatmaps_to_points(pos, im_shp, query_points=query_points)
            occlusion = F.pad(occlusion, (0, 2, 0, 2))
            occlusion = torch.relu(mods["hid3"](occlusion))
            occlusion = torch.mean(occlusion, dim=(-1, -2))
            occlusion = torch.relu(mods["hid4"](occlusion))
            occlusion = mods["occ_out"](occlusion)
            expected_dist = occlusion[..., 1:2].reshape(t, b, n).permute(1, 2, 0)
            occlusion = occlusion[..., 0:1].reshape(t, b, n).permute(1, 2, 0)
            return points, occlusion, expected_dist

    def preprocess_frames(frames):
        return frames.astype(np.float32) / 255 * 2 - 1

    def postprocess_occlusions(occlusions, expected_dist):
        return (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5

    # ===================== vendored torch track-autoencoder ================= #
    def sinusoidal(inputs, num_frequencies=32):
        scales = torch.tensor([2 ** (i / 3) for i in range(num_frequencies)],
                              dtype=inputs.dtype, device=inputs.device)
        x = inputs[..., None] * scales
        out = torch.sin(torch.cat([x, x + 0.5 * math.pi], dim=-1))
        return out.reshape(*out.shape[:-2], out.shape[-2] * out.shape[-1])

    class LayerNormNoBias(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            mu = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            return (x - mu) / torch.sqrt(var + self.eps) * self.scale

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            ms = torch.mean(x * x, dim=-1, keepdim=True)
            return x * torch.rsqrt(ms + self.eps) * self.scale

    class MHDPAttention(nn.Module):
        def __init__(self, q_dim, kv_dim, out_dim, num_heads, qk_size):
            super().__init__()
            self.h = num_heads
            self.hd = qk_size // num_heads
            self.dense_query = nn.Linear(q_dim, qk_size, bias=False)
            self.dense_key = nn.Linear(kv_dim, qk_size, bias=False)
            self.dense_value = nn.Linear(kv_dim, qk_size, bias=False)
            self.norm_query = RMSNorm(self.hd)
            self.norm_key = RMSNorm(self.hd)
            self.dense_out = nn.Linear(qk_size, out_dim, bias=True)

        def forward(self, inputs_q, inputs_kv, mask=None):
            q = self.dense_query(inputs_q).reshape(*inputs_q.shape[:-1], self.h, self.hd)
            k = self.dense_key(inputs_kv).reshape(*inputs_kv.shape[:-1], self.h, self.hd)
            v = self.dense_value(inputs_kv).reshape(*inputs_kv.shape[:-1], self.h, self.hd)
            q = self.norm_query(q)
            k = self.norm_key(k)
            scale = 1.0 / math.sqrt(self.hd)
            w = torch.einsum("...qhd,...khd->...hqk", q * scale, k)
            if mask is not None:
                w = w.masked_fill(mask == 0, torch.finfo(w.dtype).min)
            w = torch.softmax(w, dim=-1)
            o = torch.einsum("...hqk,...khd->...qhd", w, v)
            o = o.reshape(*o.shape[:-2], self.h * self.hd)
            return self.dense_out(o)

    class TransformerBlock(nn.Module):
        def __init__(self, width, kv_dim, num_heads, qkv_size, mlp_size, has_cross):
            super().__init__()
            self.norm_q = LayerNormNoBias(width)
            self.self_att = MHDPAttention(width, width, width, num_heads, qkv_size)
            self.has_cross = has_cross
            if has_cross:
                self.cross_att = MHDPAttention(width, kv_dim, width, num_heads, qkv_size)
            self.norm_attn = LayerNormNoBias(width)
            self.MLP_in = nn.Linear(width, mlp_size)
            self.MLP_out = nn.Linear(mlp_size, width)

        def forward(self, queries, inputs_kv=None, qq_mask=None, qk_mask=None):
            normed = self.norm_q(queries)
            attn_out = queries + self.self_att(normed, normed, mask=qq_mask)
            if self.has_cross and inputs_kv is not None:
                attn_out = attn_out + self.cross_att(normed, inputs_kv, mask=qk_mask)
            h = F.gelu(self.MLP_in(self.norm_attn(attn_out)), approximate="tanh")
            return attn_out + self.MLP_out(h)

    class ImprovedTransformer(nn.Module):
        def __init__(self, width, kv_dim, qkv_size, num_heads, mlp_size, num_layers, has_cross):
            super().__init__()
            self.layers = nn.ModuleList([
                TransformerBlock(width, kv_dim, num_heads, qkv_size, mlp_size, has_cross)
                for _ in range(num_layers)])
            self.norm_encoder = LayerNormNoBias(width)

        def forward(self, queries, inputs_kv=None, qq_mask=None, qk_mask=None):
            for layer in self.layers:
                if qq_mask is not None and qq_mask.dim() == queries.dim():
                    qq_mask = qq_mask.unsqueeze(-3)
                if qk_mask is not None and inputs_kv is not None and qk_mask.dim() == inputs_kv.dim():
                    qk_mask = qk_mask.unsqueeze(-3)
                queries = layer(queries, inputs_kv, qq_mask=qq_mask, qk_mask=qk_mask)
            return self.norm_encoder(queries)

    class TrackAutoEncoder(nn.Module):
        def __init__(self, num_output_frames=150, num_latent_tokens=128, latent_token_dim=64,
                     num_frequencies=32, track_scale_factor=1.0, time_scale_factor=150.0,
                     track_token_dim=256, encoder_latent_dim=512, decoder_num_channels=1024):
            super().__init__()
            self.num_output_frames = num_output_frames
            self.num_frequencies = num_frequencies
            self.track_scale_factor = track_scale_factor
            self.time_scale_factor = time_scale_factor
            self.state_init = nn.Parameter(torch.zeros(num_latent_tokens, encoder_latent_dim))
            self.track_token_projection = nn.Linear(6 * num_frequencies, track_token_dim)
            self.compressor = nn.Linear(encoder_latent_dim, latent_token_dim)
            self.decompressor = nn.Linear(latent_token_dim, decoder_num_channels - 128)
            self.input_track_transformer = ImprovedTransformer(
                track_token_dim, track_token_dim, 64 * 8, 8, 1024, 2, has_cross=False)
            self.tracks_to_latents = ImprovedTransformer(
                encoder_latent_dim, track_token_dim, 64 * 8, 8, 2048, 6, has_cross=True)
            self.decompress_attn = ImprovedTransformer(
                decoder_num_channels - 128, decoder_num_channels - 128, 64 * 8, 8, 2048, 3, has_cross=False)
            self.track_readout_attn = ImprovedTransformer(
                decoder_num_channels, decoder_num_channels, 64 * 8, 8, 1024, 4, has_cross=False)
            _qe_in = (4 * num_frequencies + 1) * 2 * num_frequencies  # sinusoidal(129) = 8256
            self.query_encoder = nn.Linear(_qe_in, decoder_num_channels)
            self.track_predictor = nn.Linear(decoder_num_channels, num_output_frames * 4)

        def embed_track_pos_visible(self, tracks, visible):
            T = tracks.shape[-2]
            fr = torch.arange(T, dtype=tracks.dtype, device=tracks.device) / T
            fr = fr[None, None, :, None].expand_as(visible)
            tr = torch.cat([tracks, fr], dim=-1)
            return sinusoidal(tr / self.track_scale_factor, self.num_frequencies)

        def encode_tracks(self, tracks, visible, restart):
            tok = self.track_token_projection(self.embed_track_pos_visible(tracks, visible))
            T = visible.shape[2]
            time = torch.arange(T, device=tracks.device)
            partition = (time < restart[..., None, None, None]).float()
            vis = (visible[..., 0] > 0).float()
            vismask = torch.ones_like(vis[..., None]) * vis[..., None, :]
            tok = self.input_track_transformer(tok, qq_mask=partition * vismask)
            vb = vis[..., None]
            return torch.sum(tok * vb, dim=-2) / torch.clamp(torch.sum(vb, dim=-2), min=1.0)

        def encode(self, inputs):
            st = self.encode_tracks(inputs["support_tracks"], inputs["support_tracks_visible"],
                                    inputs["boundary_frame"])
            B = inputs["support_tracks"].shape[0]
            latents = self.state_init[None].expand(B, *self.state_init.shape)
            return self.compressor(self.tracks_to_latents(latents, st))

        def append_time_feat(self, latents, query_frame):
            C = latents.shape[-1]
            offset = (query_frame * 5).long()
            d = torch.arange(128, device=latents.device)
            idx = d[None, None, None, :] + offset[..., None, None]
            idx = idx.expand(*latents.shape[:-1], 128)
            valid = (idx < C)
            to_append = torch.gather(latents, -1, idx.clamp(max=C - 1)) * valid.to(latents.dtype)
            return torch.cat([latents, to_append], dim=-1)

        def encode_point_identities(self, query_points):
            return sinusoidal(query_points / self.track_scale_factor, self.num_frequencies)

        def decode(self, latents, decoder_query, query_frame, discretize=True):
            latents = torch.clamp(latents, -1.0, 1.0)
            if discretize:
                disc = torch.round(latents * 128.0) / 128.0
                noise = torch.from_numpy(_threefry_uniform(tuple(latents.shape))).to(latents)
                latents = disc + noise / 128.0 - 1.0 / 256.0
            latents = self.decompress_attn(self.decompressor(latents))
            # Upstream uses integer floor-division (//): 0 for all frames < 150.
            frame_term = torch.floor(query_frame[..., None] / self.time_scale_factor)
            queries = torch.cat([decoder_query, frame_term], dim=-1)
            pce = self.query_encoder(sinusoidal(queries / self.track_scale_factor, self.num_frequencies))
            latents = latents[..., None, :, :].expand(*latents.shape[:-2], pce.shape[-2],
                                                      latents.shape[-2], latents.shape[-1])
            latents = self.append_time_feat(latents, query_frame)
            latents = torch.cat([pce[..., None, :], latents], dim=2)
            out = self.track_readout_attn(latents)[..., 0, :]
            out = self.track_predictor(out)
            nf = self.num_output_frames
            tracks = torch.stack([out[..., :nf], out[..., nf:2 * nf]], dim=-1)
            return tracks, out[..., 2 * nf:3 * nf, None], out[..., 3 * nf:, None]

        def forward(self, inputs):
            latents = self.encode(inputs)
            qp = inputs["query_points"]
            dq = self.encode_point_identities(qp[..., 1:])
            qf = torch.round(qp[..., 0]).to(torch.int32).float()
            return self.decode(latents, dq, qf)

    # ================= Flax -> torch param conversion (AE) ================== #
    def _lin(kernel, bias=None):
        out = {"weight": torch.from_numpy(np.ascontiguousarray(kernel)).float().t().contiguous()}
        if bias is not None:
            out["bias"] = torch.from_numpy(np.ascontiguousarray(bias)).float()
        return out

    def _dg_qkv(kernel):  # (d_in, H, hd) -> Linear weight (H*hd, d_in)
        d_in, H, hd = kernel.shape
        return torch.from_numpy(np.ascontiguousarray(kernel.reshape(d_in, H * hd))).float().t().contiguous()

    def _dg_out(kernel, bias):  # (H, hd, out) -> Linear weight (out, H*hd)
        H, hd, out = kernel.shape
        return (torch.from_numpy(np.ascontiguousarray(kernel.reshape(H * hd, out))).float().t().contiguous(),
                torch.from_numpy(np.ascontiguousarray(bias)).float())

    def _attn(sd, prefix, flat, fp):
        sd[prefix + ".dense_query.weight"] = _dg_qkv(flat[fp + "/dense_query/kernel"])
        sd[prefix + ".dense_key.weight"] = _dg_qkv(flat[fp + "/dense_key/kernel"])
        sd[prefix + ".dense_value.weight"] = _dg_qkv(flat[fp + "/dense_value/kernel"])
        ow, ob = _dg_out(flat[fp + "/dense_out/kernel"], flat[fp + "/dense_out/bias"])
        sd[prefix + ".dense_out.weight"] = ow
        sd[prefix + ".dense_out.bias"] = ob
        sd[prefix + ".norm_query.scale"] = torch.from_numpy(flat[fp + "/norm_query/scale"]).float()
        sd[prefix + ".norm_key.scale"] = torch.from_numpy(flat[fp + "/norm_key/scale"]).float()

    def _transformer(sd, prefix, flat, fp, num_layers, has_cross):
        for i in range(num_layers):
            lp, lf = f"{prefix}.layers.{i}", f"{fp}/layer_{i}"
            sd[lp + ".norm_q.scale"] = torch.from_numpy(flat[lf + "/norm_q/scale"]).float()
            sd[lp + ".norm_attn.scale"] = torch.from_numpy(flat[lf + "/norm_attn/scale"]).float()
            _attn(sd, lp + ".self_att", flat, lf + "/self_att")
            if has_cross:
                _attn(sd, lp + ".cross_att", flat, lf + "/cross_att")
            for name in ("MLP_in", "MLP_out"):
                d = _lin(flat[f"{lf}/{name}/kernel"], flat[f"{lf}/{name}/bias"])
                sd[f"{lp}.{name}.weight"] = d["weight"]
                sd[f"{lp}.{name}.bias"] = d["bias"]
        sd[prefix + ".norm_encoder.scale"] = torch.from_numpy(flat[fp + "/norm_encoder/scale"]).float()

    def convert_ae_params(flat):
        flat = {k: np.asarray(v) for k, v in flat.items()}
        sd = {"state_init": torch.from_numpy(flat["initializer/state_init"]).float()}
        for name in ("track_token_projection", "compressor", "decompressor",
                     "query_encoder", "track_predictor"):
            d = _lin(flat[name + "/kernel"], flat[name + "/bias"])
            sd[name + ".weight"] = d["weight"]
            sd[name + ".bias"] = d["bias"]
        _transformer(sd, "input_track_transformer", flat, "input_track_transformer", 2, False)
        _transformer(sd, "tracks_to_latents", flat, "tracks_to_latents", 6, True)
        _transformer(sd, "decompress_attn", flat, "decompress_attn", 3, False)
        _transformer(sd, "track_readout_attn", flat, "track_readout_attn", 4, False)
        return sd

    _BACKEND = SimpleNamespace(
        torch=torch, TAPIR=TAPIR, TrackAutoEncoder=TrackAutoEncoder,
        convert_ae_params=convert_ae_params, preprocess_frames=preprocess_frames,
        postprocess_occlusions=postprocess_occlusions,
    )
    return _BACKEND


class TRAJANModule(PipelineModule):
    name = "trajan"
    description = "TRAJAN point-track autoencoder motion realism (ICLR 2025, pure-torch)"
    default_config = {
        # Frames actually fed through the tracker/autoencoder. The autoencoder's
        # episode length is 150; more frames = longer trajectories (and cost).
        "max_frames": 60,
        # Spatial resolution the tracker runs at (square, per the demo).
        "resize": 256,
        # Total query points tracked by BootsTAPIR; split into support+target.
        "num_points": 4096,
        "num_support_tracks": 2048,
        "num_target_tracks": 2048,
        "query_chunk_size": 32,
        "models_dir": "models",
    }
    metric_groups = {
        "trajan_score": "motion",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend: Optional[str] = None
        self._torch = None
        self._tapir = None
        self._ae_model = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            self._init_real_backend()
            self._backend = "port"
            self._ml_available = True
            logger.info("TRAJAN initialised (pure-torch BootsTAPIR + track-autoencoder port)")
            return
        except ImportError as e:
            reason = f"{e}; install torch"
        except Exception as e:  # noqa: BLE001 - any load/checkpoint failure -> unavailable
            reason = str(e)
        self._backend = "unavailable"
        self._ml_available = False
        logger.info("TRAJAN unavailable: %s. trajan_score will not be populated.", reason)

    def _init_real_backend(self) -> None:
        """Load torch, both checkpoints, and build the vendored models. Raises on failure."""
        backend = _build_backend()
        torch = backend.torch
        self._torch = torch

        models_dir = self.config.get("models_dir", "models")
        from ayase.config import download_model_file

        # BootsTAPIR torch checkpoint (official DeepMind torch port).
        tapir_ckpt = download_model_file(_BOOTSTAPIR_REL, _BOOTSTAPIR_URL, models_dir)
        tapir = backend.TAPIR(pyramid_level=1, softmax_temperature=10.0, extra_convs=True)
        sd = torch.load(str(tapir_ckpt), map_location="cpu", weights_only=False)
        missing, unexpected = tapir.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"BootsTAPIR checkpoint key mismatch (missing={len(missing)}, "
                f"unexpected={len(unexpected)})")
        tapir.eval()
        self._tapir = tapir

        # Track-autoencoder Flax checkpoint -> torch state_dict.
        from huggingface_hub import hf_hub_download

        ae_path = hf_hub_download(repo_id=_AE_REPO, filename=_AE_FILE)
        loaded = np.load(ae_path, allow_pickle=False)
        flat = {k: np.asarray(loaded[k]) for k in loaded.files}
        ae = backend.TrackAutoEncoder()
        state = backend.convert_ae_params(flat)
        missing, unexpected = ae.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"track-autoencoder param mapping incomplete (missing={len(missing)}, "
                f"unexpected={len(unexpected)})")
        ae.eval()
        self._ae_model = ae

    # ---------------------------------------------------------------- process
    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "port":
            return sample
        if not sample.is_video:
            return sample
        try:
            score = self._compute_trajan(sample)
            if score is not None:
                sample.quality_metrics.trajan_score = float(score)
        except Exception as e:  # noqa: BLE001
            logger.warning("TRAJAN processing failed: %s", e)
        return sample

    # ------------------------------------------------------------- real score
    def _compute_trajan(self, sample: Sample) -> Optional[float]:
        """Run BootsTAPIR -> track autoencoder -> Average Jaccard reconstruction.

        Faithful to DeepMind's TRAJAN pipeline: tracks are estimated at ``resize``
        resolution, normalised to [0, 1], autoencoded, and the reconstruction is
        scored against the estimated target tracks with TAP-Vid Average Jaccard.
        """
        import cv2

        torch = self._torch
        backend = _build_backend()

        max_frames = int(self.config.get("max_frames", 60))
        resize = int(self.config.get("resize", 256))
        num_points = int(self.config.get("num_points", 4096))
        num_support = int(self.config.get("num_support_tracks", 2048))
        num_target = int(self.config.get("num_target_tracks", 2048))
        chunk_size = int(self.config.get("query_chunk_size", 32))

        raw = sample_frames(sample.path, max_frames=max_frames, color="rgb")
        if len(raw) < 4:
            return None
        video = np.stack([np.ascontiguousarray(f) for f in raw]).astype(np.uint8)
        resized = np.stack([cv2.resize(f, (resize, resize)) for f in video])
        n_frames = resized.shape[0]

        frames = torch.from_numpy(backend.preprocess_frames(resized))[None]  # [1,T,H,W,3]
        with torch.no_grad():
            feature_grids = self._tapir.get_feature_grids(frames)

            rng = np.random.default_rng()
            qt = rng.integers(0, n_frames, (num_points, 1))
            qy = rng.integers(0, resize, (num_points, 1))
            qx = rng.integers(0, resize, (num_points, 1))
            query_points = np.concatenate((qt, qy, qx), axis=-1).astype(np.float32)

            all_tracks, all_vis = [], []
            for c in range(0, query_points.shape[0], chunk_size):
                qp = torch.from_numpy(query_points[c:c + chunk_size])[None]
                out = self._tapir(video=frames, query_points=qp,
                                  query_chunk_size=chunk_size, feature_grids=feature_grids)
                visibles = backend.postprocess_occlusions(out["occlusion"], out["expected_dist"])
                all_tracks.append(out["tracks"][0].cpu().numpy())
                all_vis.append(visibles[0].cpu().numpy())
        tracks = np.concatenate(all_tracks, axis=0)  # [N, T, 2] in resized px (xy)
        visibles = np.concatenate(all_vis, axis=0)   # [N, T] bool

        # Map resized-pixel tracks -> normalised [0, 1] (+0.5 pixel-centre).
        tracks_norm = (tracks + 0.5) / resize
        import einops  # noqa: E402  (used across the ML modules)
        batch = {
            "video": video,
            "tracks": einops.rearrange(tracks_norm, "q t c -> t q c"),
            "visible": einops.rearrange(visibles, "q t -> t q"),
        }

        if num_support + num_target > num_points:
            num_support = num_points // 2
            num_target = num_points - num_support
        batch = self._preprocess_tracks(batch, num_support, num_target)
        batch.pop("tracks", None)

        with torch.no_grad():
            inputs = {
                "support_tracks": torch.from_numpy(batch["support_tracks"]).float(),
                "support_tracks_visible": torch.from_numpy(batch["support_tracks_visible"]).float(),
                "query_points": torch.from_numpy(batch["query_points"]).float(),
                "boundary_frame": torch.from_numpy(np.asarray(batch["boundary_frame"])).long(),
            }
            rec_tracks, rec_vis_logits, rec_cert_logits = self._ae_model(inputs)
            rec_visibles = backend.postprocess_occlusions(rec_vis_logits, rec_cert_logits)

        video_length = video.shape[0]
        reconstructed_tracks = rec_tracks[0].cpu().numpy() * resize  # [Q, 150, 2] px
        reconstructed_visibles = rec_visibles.cpu().numpy()          # [1, Q, 150, 1]
        target_tracks_px = batch["target_points"][0] * resize        # [Q, 150, 2] px

        gt_occluded = 1 - batch["target_tracks_visible"][..., :video_length, 0]  # [1, Q, vl]
        query_pts = np.zeros((reconstructed_visibles.shape[0],
                              batch["target_tracks_visible"].shape[1], 1))
        metrics = _compute_tapvid_metrics(
            query_points=query_pts,
            gt_occluded=gt_occluded,
            gt_tracks=target_tracks_px[None, ..., :video_length, :],
            pred_occluded=reconstructed_visibles[..., :video_length, 0],
            pred_tracks=reconstructed_tracks[None, ..., :video_length, :],
            query_mode="strided",
        )
        return float(np.mean([metrics[f"jaccard_{d}"] for d in (1, 2, 4, 8, 16)]))

    def _preprocess_tracks(self, features: dict, num_support: int, num_target: int) -> dict:
        """Sample support/target tracks + query points (TRAJAN colab preprocessor).

        Pads to the autoencoder's 150-frame episode length, selects disjoint
        support/target track subsets, and derives one query point per target
        track at a randomly chosen visible frame.
        """
        episode_length = 150
        tracks_xy = np.asarray(features["tracks"], np.float32)[..., :2]  # [T, Q, 2]
        visibles = np.asarray(features["visible"], np.float32)  # [T, Q]
        boundary_frame = features["video"].shape[0]

        if episode_length - visibles.shape[0] < 0:
            visibles = visibles[:episode_length]
            tracks_xy = tracks_xy[:episode_length]
        visibles = np.pad(
            visibles, [[0, episode_length - visibles.shape[0]], [0, 0]], constant_values=0.0)
        tracks_xy = np.pad(
            tracks_xy, [[0, episode_length - tracks_xy.shape[0]], [0, 0], [0, 0]], constant_values=0.0)

        num_input = tracks_xy.shape[1]
        if num_input < num_support + num_target:
            raise ValueError(
                f"TRAJAN needs >= {num_support + num_target} tracks, got {num_input}")
        idx = np.arange(num_input)
        np.random.shuffle(idx)
        idx_support = idx[-num_support:]
        idx_target = idx[:num_target]

        support_tracks = np.transpose(tracks_xy[:, idx_support, :], [1, 0, 2])
        support_tracks_visible = np.expand_dims(
            np.transpose(visibles[:, idx_support], [1, 0]), axis=-1)
        target_tracks = np.transpose(tracks_xy[:, idx_target, :], [1, 0, 2])
        target_tracks_visible = np.transpose(visibles[:, idx_target], [1, 0])

        n_tgt = target_tracks_visible.shape[0]
        n_frames = target_tracks_visible.shape[1]
        random_frame = np.zeros(n_tgt, dtype=np.int64)
        for i in range(n_tgt):
            vis_idx = np.where(target_tracks_visible[i] > 0)[0]
            random_frame[i] = np.random.choice(vis_idx) if len(vis_idx) else 0
        onehot = np.eye(n_frames, dtype=np.float32)[random_frame]
        target_queries_xy = np.sum(target_tracks * onehot[..., np.newaxis], axis=1)
        target_queries = np.stack(
            [random_frame.astype(np.float32), target_queries_xy[..., 0], target_queries_xy[..., 1]],
            axis=-1)
        target_tracks_visible = np.expand_dims(target_tracks_visible, axis=-1)

        features.update({
            "support_tracks": support_tracks[None, :],
            "support_tracks_visible": support_tracks_visible[None, :],
            "query_points": target_queries[None, :],
            "target_points": target_tracks[None, :],
            "boundary_frame": np.array([boundary_frame]),
            "target_tracks_visible": target_tracks_visible[None, :],
        })
        return features
