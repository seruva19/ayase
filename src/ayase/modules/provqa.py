"""ProVQA -- Progressive Blind 360° Video Quality Assessment (TIP 2022).

Faithful re-implementation of the published ProVQA predictor of Yang et al.,
"Blind VQA on 360° Video via Progressively Learning from Pixels, Frames and
Video" (IEEE TIP 2022).  code: https://github.com/yanglixiaoshen/ProVQA

The trained generator (upstream ``network_g`` of type ``BVQA360v240``) is a
no-reference omnidirectional-video quality network built from spherical
convolutions:

  * **Spatial branch** (``spatial_module``) — a sphere-conv ResNet with a
    selective-feature-integration (SKFF) block extracts a per-frame quality
    feature from the *center* frame of each temporal triplet.
  * **Motion branch** (``motion_module``) — consumes the concatenated
    forward/backward frame differences (center−left, right−center) and mixes
    them with the spatial feature via channel/spatial attention.
  * **Temporal branch** (``nonlocal_module`` + ``tempotalAgg_module``) — a
    3-D non-local block aggregates the per-triplet features across time and a
    3-D pooling MLP regresses a single quality score.

The network operates on equirectangular (ERP) frames directly (the spherical
convolutions embed the ERP→sphere sampling), so it is fed raw frames exactly as
upstream does.  ProVQA was trained on the VQA-ODV 360° dataset; its score is
only strictly meaningful for omnidirectional / ERP content.  Ayase has no
360°-content signal for this module, so — per the upstream inference path — the
network is simply run on the sampled frames and the caveat is documented here.

The real checkpoint (``net_g_26400.pth``, ~5.3 MB, 503 tensors) is mirrored on
the Hugging Face Hub and loaded verbatim (strict); the whole model definition is
vendored below so no upstream package (basicsr / SphereNet / NonLocalNet) is
required.  The raw network output is a predicted DMOS in ~[0, 1] (higher =
*worse* quality); ``provqa_score`` reports ``1 − DMOS`` so higher = better,
matching the field's convention.

Only the real trained model produces ``provqa_score``.  When the weights or a
required dependency (``torch`` / ``huggingface_hub``) are missing the score is
left ``None`` — no heuristic or proxy substitute (no-heuristic policy).

provqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# HuggingFace mirror of the upstream ProVQA checkpoint.
_HF_REPO = "AkaneTendo25/ayase-runtime-assets"
_HF_FILENAME = "provqa/net_g_26400.pth"

# Hyperparameters from the upstream test config
# (options/test/test_bvqa360_OURs.yaml).  ``num_frame=6`` is load-bearing: it
# fixes the temporal-aggregation FC input dim (mos_in//2 * num_frame = 96) and
# therefore must match the trained weights.
_NUM_FRAME = 6
_ERP_H = 240      # equirectangular frame height used in training
_ERP_W = 480      # equirectangular frame width (2:1 ERP aspect ratio)

# Cache for the lazily-built (torch-dependent) model class.
_MODEL_CLASS = None


# ---------------------------------------------------------------------------
# Vendored model definition (built lazily so importing this module never
# requires torch).  Reconstructs the exact upstream ``BVQA360v240`` generator
# (with its SphereNet / antialias / NonLocal dependencies inlined) so the
# published 503-tensor state_dict loads strictly.
# ---------------------------------------------------------------------------
def _load_model_definitions():
    """Build and return the vendored ``BVQA360v240`` class (cached)."""
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS

    import math
    from functools import lru_cache

    import torch
    from numpy import arcsin, arctan, cos, pi, sin, tan
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.parameter import Parameter

    # ---- SphereNet spherical conv/pool (github.com/ChiWeiHsiao/SphereNet) ----
    @lru_cache(None)
    def get_xy(delta_phi, delta_theta):
        return np.array([
            [(-tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),
             (0, tan(delta_phi)),
             (tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi))],
            [(-tan(delta_theta), 0), (1, 1), (tan(delta_theta), 0)],
            [(-tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),
             (0, -tan(delta_phi)),
             (tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi))],
        ])

    @lru_cache(None)
    def cal_index(h, w, img_r, img_c):
        phi = -((img_r + 0.5) / h * pi - pi / 2)
        theta = (img_c + 0.5) / w * 2 * pi - pi
        delta_phi = pi / h
        delta_theta = 2 * pi / w
        xys = get_xy(delta_phi, delta_theta)
        x = xys[..., 0]
        y = xys[..., 1]
        rho = np.sqrt(x ** 2 + y ** 2)
        v = arctan(rho)
        new_phi = arcsin(cos(v) * sin(phi) + y * sin(v) * cos(phi) / rho)
        new_theta = theta + arctan(
            x * sin(v) / (rho * cos(phi) * cos(v) - y * sin(phi) * sin(v)))
        new_r = (-new_phi + pi / 2) * h / pi - 0.5
        new_c = (new_theta + pi) * w / 2 / pi - 0.5
        new_c = (new_c + w) % w
        new_result = np.stack([new_r, new_c], axis=-1)
        new_result[1, 1] = (img_r, img_c)
        return new_result

    @lru_cache(None)
    def _gen_filters_coordinates(h, w, stride):
        co = np.array([[cal_index(h, w, i, j) for j in range(0, w, stride)]
                       for i in range(0, h, stride)])
        return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))

    def gen_filters_coordinates(h, w, stride=1):
        assert isinstance(h, int) and isinstance(w, int)
        return _gen_filters_coordinates(h, w, stride).copy()

    def gen_grid_coordinates(h, w, stride=1):
        coordinates = gen_filters_coordinates(h, w, stride).copy()
        coordinates[0] = (coordinates[0] * 2 / h) - 1
        coordinates[1] = (coordinates[1] * 2 / w) - 1
        coordinates = coordinates[::-1]
        coordinates = coordinates.transpose(1, 3, 2, 4, 0)
        sz = coordinates.shape
        coordinates = coordinates.reshape(1, sz[0] * sz[1], sz[2] * sz[3], sz[4])
        return coordinates.copy()

    class SphereConv2D(nn.Module):
        """Sphere convolution (3x3 kernel only), for ERP inputs."""

        def __init__(self, in_c, out_c, stride=1, bias=True, mode='bilinear'):
            super().__init__()
            self.in_c = in_c
            self.out_c = out_c
            self.stride = stride
            self.mode = mode
            self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3))
            if bias:
                self.bias = Parameter(torch.Tensor(out_c))
            else:
                self.register_parameter('bias', None)
            self.grid_shape = None
            self.grid = None
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
            if self.bias is not None:
                self.bias.data.zero_()

        def forward(self, x):
            if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
                self.grid_shape = tuple(x.shape[2:4])
                coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
                with torch.no_grad():
                    self.grid = torch.FloatTensor(coordinates).to(x.device)
            with torch.no_grad():
                grid = self.grid.repeat(x.shape[0], 1, 1, 1)
            x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
            x = F.conv2d(x, self.weight, self.bias, stride=3)
            return x

    class SphereMaxPool2D(nn.Module):
        """Sphere max-pool (3x3 kernel only)."""

        def __init__(self, stride=1, mode='bilinear'):
            super().__init__()
            self.stride = stride
            self.mode = mode
            self.grid_shape = None
            self.grid = None
            self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        def forward(self, x):
            if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
                self.grid_shape = tuple(x.shape[2:4])
                coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
                with torch.no_grad():
                    self.grid = torch.FloatTensor(coordinates).to(x.device)
            with torch.no_grad():
                grid = self.grid.repeat(x.shape[0], 1, 1, 1)
            return self.pool(F.grid_sample(x, grid, mode=self.mode, align_corners=False))

    # ---- Antialiased downsample (Adobe antialiased-cnns, blur filt_size=3) ----
    def get_pad_layer(pad_type):
        if pad_type in ('refl', 'reflect'):
            return nn.ReflectionPad2d
        if pad_type in ('repl', 'replicate'):
            return nn.ReplicationPad2d
        return nn.ZeroPad2d

    class Downsample(nn.Module):
        def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
            super().__init__()
            self.filt_size = filt_size
            self.pad_off = pad_off
            self.pad_sizes = [
                int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
            ]
            self.pad_sizes = [p + pad_off for p in self.pad_sizes]
            self.stride = stride
            self.off = int((self.stride - 1) / 2.)
            self.channels = channels
            a = np.array([1., 2., 1.])  # filt_size==3 (the only size used here)
            filt = torch.Tensor(a[:, None] * a[None, :])
            filt = filt / torch.sum(filt)
            self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
            self.pad = get_pad_layer(pad_type)(self.pad_sizes)

        def forward(self, inp):
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

    # ---- Temporal non-local block (github.com/AlexHex7/Non-local_pytorch) ----
    class _NonLocalBlockND_Tem(nn.Module):
        def __init__(self, in_channels, inter_channels=None, dimension=3,
                     sub_sample=True, bn_layer=True):
            super().__init__()
            self.dimension = dimension
            self.sub_sample = sub_sample
            self.in_channels = in_channels
            self.inter_channels = inter_channels
            if self.inter_channels is None:
                self.inter_channels = in_channels // 2
                if self.inter_channels == 0:
                    self.inter_channels = 1
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
            self.g = conv_nd(self.in_channels, self.inter_channels, 1, 1, 0)
            if bn_layer:
                self.W = nn.Sequential(
                    conv_nd(self.inter_channels, self.in_channels, 1, 1, 0),
                    bn(self.in_channels))
                nn.init.constant_(self.W[1].weight, 0)
                nn.init.constant_(self.W[1].bias, 0)
            else:
                self.W = conv_nd(self.inter_channels, self.in_channels, 1, 1, 0)
                nn.init.constant_(self.W.weight, 0)
                nn.init.constant_(self.W.bias, 0)
            self.theta = conv_nd(self.in_channels, self.inter_channels, 1, 1, 0)
            self.phi = conv_nd(self.in_channels, self.inter_channels, 1, 1, 0)
            if sub_sample:
                self.g = nn.Sequential(self.g, max_pool_layer)
                self.phi = nn.Sequential(self.phi, max_pool_layer)

        def forward(self, x):
            batch_size = x.size(0)
            g_x = self.g(x).view(batch_size, self.inter_channels, x.size(2), -1)
            g_x = g_x.permute(3, 0, 1, 2).contiguous().view(-1, g_x.size(2), g_x.size(3))
            theta_x = self.theta(x).view(batch_size, self.inter_channels, x.size(2), -1)
            theta_x = theta_x.permute(3, 0, 1, 2).contiguous().view(-1, theta_x.size(2), theta_x.size(3))
            phi_x = self.phi(x).view(batch_size, self.inter_channels, x.size(2), -1)
            phi_x = phi_x.permute(3, 0, 1, 2).contiguous().view(-1, phi_x.size(2), phi_x.size(3)).permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)
            f_div_C = f / f.size(-1)
            y = torch.matmul(g_x, f_div_C)
            y = y.view(-1, batch_size, y.size(1), y.size(2)).permute(1, 2, 3, 0).contiguous()
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
            return self.W(y) + x

    class NONLocalBlock3D_tem(_NonLocalBlockND_Tem):
        def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
            super().__init__(in_channels, inter_channels, 3, sub_sample, bn_layer)

    # ---- BVQA360v240 architecture (model/archs/bvqa360v240_arch.py) ----
    class ChannelAttention(nn.Module):
        def __init__(self, in_planes, ratio=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            return self.sigmoid(avg_out + max_out)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=3):
            super().__init__()
            self.conv1 = SphereConv2D(2, 1, stride=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            return self.sigmoid(self.conv1(x))

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = SphereConv2D(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = SphereConv2D(planes, planes, stride=1)
            self.bn2 = nn.BatchNorm2d(planes)
            self.ca = ChannelAttention(planes)
            self.sa = SpatialAttention()
            self.downsample = downsample
            self.stride = stride
            self.inplanes = inplanes

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.ca(out) * out
            out = self.sa(out) * out
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            return self.relu(out)

    class SFI(nn.Module):
        """Selective feature integration (SKFF) over 3 same-shape feature maps."""

        def __init__(self, in_channels, height=3, reduction=8, bias=False):
            super().__init__()
            self.height = height
            d = max(int(in_channels / reduction), 4)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())
            self.fcs = nn.ModuleList(
                [nn.Conv2d(d, in_channels, 1, 1, bias=bias) for _ in range(self.height)])
            self.softmax = nn.Softmax(dim=1)

        def forward(self, inp_feats):
            batch_size = inp_feats[0].shape[0]
            n_feats = inp_feats[0].shape[1]
            inp_feats = torch.cat(inp_feats, dim=1)
            inp_feats = inp_feats.view(batch_size, self.height, n_feats,
                                       inp_feats.shape[2], inp_feats.shape[3])
            feats_U = torch.sum(inp_feats, dim=1)
            feats_S = self.avg_pool(feats_U)
            feats_Z = self.conv_du(feats_S)
            attention_vectors = torch.cat([fc(feats_Z) for fc in self.fcs], dim=1)
            attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
            attention_vectors = self.softmax(attention_vectors)
            return torch.sum(inp_feats * attention_vectors, dim=1)

    class ResNet_iqa(nn.Module):
        def __init__(self, in_channel, block, layers, out_channels):
            self.inplanes = in_channel
            super().__init__()
            self.bot1 = nn.Sequential(
                Downsample(channels=self.inplanes, filt_size=3, stride=2),
                nn.Conv2d(self.inplanes, out_channels[0], 1, 1, 0, bias=False))
            self.bot2 = nn.Sequential(
                Downsample(channels=out_channels[0], filt_size=3, stride=2),
                nn.Conv2d(out_channels[0], out_channels[1], 1, 1, 0, bias=False))
            self.bot3 = nn.Sequential(
                Downsample(channels=out_channels[1], filt_size=3, stride=2),
                nn.Conv2d(out_channels[1], out_channels[2], 1, 1, 0, bias=False))
            self.layer1 = self._make_layer(block, out_channels[0], layers[0], stride=2)
            self.layer2 = self._make_layer(block, out_channels[1], layers[1], stride=2)
            self.layer3 = self._make_layer(block, out_channels[2], layers[2], stride=2)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1):
            layers = []
            downsample = None
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes), nn.ReLU(inplace=True))
            self.inplanes = planes
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))
            layers.append(self.conv1x1)
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x1 = self.layer1(x) + self.bot1(x)
            x2 = self.layer2(x1) + self.bot2(x1)
            x3 = self.layer3(x2) + self.bot3(x2)
            return x1, x2, x3

    class SpatialFeatureModule(nn.Module):
        def __init__(self, in_channel=32, out_channels=(64, 64, 64), res_blocks=(3, 4, 6),
                     spa3_in=64, spa3_out=32, comb_in=64):
            super().__init__()
            self.leaky = 0.1
            self.shared_layers1 = nn.Sequential(
                SphereConv2D(3, in_channel, stride=1),
                nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
            self.shared_layers2 = ResNet_iqa(in_channel, BasicBlock, res_blocks, out_channels)
            self.skff_block = SFI(comb_in)
            self.shared_layers3 = nn.Sequential(
                SphereConv2D(spa3_in, spa3_out, stride=1),
                nn.BatchNorm2d(spa3_out), nn.ReLU(inplace=True))
            self.low_downsamp = nn.Sequential(
                Downsample(channels=out_channels[0], filt_size=3, stride=2),
                nn.Conv2d(out_channels[0], comb_in, 1, 1, 0, bias=False))
            self.high_upsamp = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(out_channels[2], comb_in, 1, 1, 0, bias=False))

        def forward(self, x):
            x = self.shared_layers1(x)
            x_low, x_mid, x_high = self.shared_layers2(x)
            x1 = self.skff_block([self.low_downsamp(x_low), x_mid, self.high_upsamp(x_high)])
            return self.shared_layers3(x1)

    class MotionFeatureExtr(nn.Module):
        def __init__(self, motion_comb1=32, motion_comb2=64, motion_in=6, layers=(2, 2, 2, 2)):
            super().__init__()
            self.leaky = 0.1
            self.inplanes = motion_in
            self.inplanes1 = motion_comb2
            self.spatial_layer1 = self._make_layer(BasicBlock, motion_comb2, layers[0], stride=2)
            self.spatial_layer2 = self._make_layer(BasicBlock, motion_comb1, layers[1], stride=2)
            self.sa = SpatialAttention()
            self.spatial_layer3 = self._make_layer(BasicBlock, motion_comb1, layers[2])
            self.ca = ChannelAttention(motion_comb2)
            self.feat_comb_layer = nn.Sequential(
                nn.Conv2d(motion_comb2, motion_comb1, 1, bias=False),
                nn.BatchNorm2d(motion_comb1), nn.ReLU(inplace=True))
            self.spatial_layer4 = self._make_layer1(BasicBlock, motion_comb1, layers[3])

        def _make_layer(self, block, planes, blocks, stride=1):
            layers = []
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def _make_layer1(self, block, planes, blocks, stride=1):
            layers = []
            downsample = None
            if stride != 1 or self.inplanes1 != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes1, planes * block.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))
            layers.append(block(self.inplanes1, planes, stride, downsample))
            self.inplanes1 = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes1, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x_cent = x[0]
            x_motion = x[1]
            x_motion_spat = self.spatial_layer2(self.spatial_layer1(x_motion))
            x_motion_spat = x_motion_spat * self.sa(x_motion_spat)
            x_mix = x_cent * x_motion_spat
            x_mix_temp = torch.cat((x_mix, x_cent), dim=1)
            x_mix1 = self.feat_comb_layer(x_mix_temp * self.ca(x_mix_temp))
            x_top = self.spatial_layer3(x_mix)
            x_mix2 = self.spatial_layer4(torch.cat((x_top, x_mix1), dim=1))
            return x_mix2 + x_mix1

    class TemporalNonLocalModule(nn.Module):
        def __init__(self, nonlocal_in=32):
            super().__init__()
            self.out = NONLocalBlock3D_tem(nonlocal_in, sub_sample=False, bn_layer=True)

        def forward(self, x):
            return self.out(x)

    class TemporalScoresAggModule(nn.Module):
        def __init__(self, mos_in=32, num_frame=5, fc_dim1=20):
            super().__init__()
            self.leaky = 0.1
            self.spatial_layer1 = nn.Sequential(
                nn.Conv3d(mos_in, mos_in // 2, 3, 1, 1, bias=False),
                nn.BatchNorm3d(mos_in // 2), nn.ReLU(inplace=True))
            self.maxpool1 = nn.AdaptiveMaxPool3d((num_frame, 30, 60))
            self.avgpool1 = nn.AdaptiveAvgPool3d((num_frame, 30, 60))
            self.spatial_layer2 = nn.Sequential(
                nn.Conv3d(2 * mos_in // 2, mos_in // 4, 3, 1, 1, bias=False),
                nn.BatchNorm3d(mos_in // 4), nn.ReLU(inplace=True))
            self.maxpool2 = nn.AdaptiveMaxPool3d((num_frame, 15, 30))
            self.avgpool2 = nn.AdaptiveAvgPool3d((num_frame, 15, 30))
            self.pool1 = nn.AdaptiveAvgPool3d((num_frame, 10, 10))
            self.pool2 = nn.AdaptiveAvgPool3d((num_frame, 1, 1))
            self.fc = nn.Sequential(
                nn.Linear(mos_in // 2 * num_frame, fc_dim1, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(fc_dim1, 1, bias=False))

        def forward(self, x):
            x1 = self.spatial_layer1(x)
            x_1 = torch.cat((self.maxpool1(x1), self.avgpool1(x1)), dim=1)
            x2 = self.spatial_layer2(x_1)
            x_2 = torch.cat((self.maxpool2(x2), self.avgpool2(x2)), dim=1)
            x_2 = self.pool1(x_2)
            x_2 = self.pool2(x_2)
            x_2 = x_2.view(x_2.size(0), -1)
            return self.fc(x_2)

    class BVQA360v240(nn.Module):
        def __init__(self, in_channel=32, out_channels=(64, 64, 64), res_blocks=(3, 4, 6),
                     spa3_in=64, spa3_out=32, comb_in=64, motion_comb1=32, motion_comb2=64,
                     motion_in=6, layers=(2, 2, 2, 2), nonlocal_in=32, mos_in=32,
                     num_frame=5, fc_dim1=20):
            super().__init__()
            self.spatial_module = SpatialFeatureModule(
                in_channel, out_channels, res_blocks, spa3_in, spa3_out, comb_in)
            self.motion_module = MotionFeatureExtr(
                motion_comb1, motion_comb2, motion_in, layers)
            self.nonlocal_module = TemporalNonLocalModule(nonlocal_in)
            self.tempotalAgg_module = TemporalScoresAggModule(mos_in, num_frame, fc_dim1)
            # Frame layout per temporal group: [left, center, right] at index 3k+{0,1,2}.
            self.l_index = [i * 3 for i in range(0, num_frame)]
            self.cen_index = [i * 3 + 1 for i in range(0, num_frame)]
            self.r_index = [i * 3 + 2 for i in range(0, num_frame)]

        def forward(self, x):
            # x: (b, num_frame*3, c, h, w)
            x1 = x[:, self.cen_index, :, :, :].contiguous()
            x_spa = x1.view(-1, x.size(2), x.size(3), x.size(4))
            x_l = x[:, self.l_index, :, :, :].contiguous()
            motion_l = x_spa - x_l.view(-1, x.size(2), x.size(3), x.size(4))
            x_r = x[:, self.r_index, :, :, :].contiguous()
            motion_r = x_r.view(-1, x.size(2), x.size(3), x.size(4)) - x_spa
            x_spa = self.spatial_module(x_spa)
            x_motion = [x_spa, torch.cat((motion_l, motion_r), dim=1)]
            x_motion = self.motion_module(x_motion)
            x_motion = x_motion.view(
                x.size(0), len(self.l_index), x_motion.size(1),
                x_motion.size(2), x_motion.size(3))
            x_motion = x_motion.permute(0, 2, 1, 3, 4)
            x_motion = self.nonlocal_module(x_motion)
            return self.tempotalAgg_module(x_motion)

    _MODEL_CLASS = BVQA360v240
    return _MODEL_CLASS


def _build_model(device):
    """Reconstruct BVQA360v240 and load the mirrored weights (strict).

    Handles the BasicSR ``{'params': state_dict}`` / ``{'params_ema': ...}``
    checkpoint nesting.  Raises on any missing dependency / weight mismatch so
    the caller can mark the backend unavailable.
    """
    import torch
    from huggingface_hub import hf_hub_download

    cls = _load_model_definitions()
    net = cls(
        in_channel=32, out_channels=(64, 64, 64), res_blocks=(3, 4, 6),
        spa3_in=64, spa3_out=32, comb_in=64, motion_comb1=32, motion_comb2=64,
        motion_in=6, layers=(2, 2, 2, 2), nonlocal_in=32, mos_in=32,
        num_frame=_NUM_FRAME, fc_dim1=20,
    )

    weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    sd = ckpt
    if isinstance(ckpt, dict):
        for cand in ("params", "params_ema", "state_dict"):
            if cand in ckpt and isinstance(ckpt[cand], dict):
                sd = ckpt[cand]
                break
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    # strict=True: the vendored architecture must exactly match the checkpoint.
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


class ProVQAModule(PipelineModule):
    name = "provqa"
    description = "ProVQA progressive blind 360° VQA (real model only)"
    default_config = {
        "device": "auto",
    }
    metric_groups = {
        "provqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._ml_available = False
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = _build_model(self._device)
            self._backend = "real"
            self._ml_available = True
            logger.info("ProVQA loaded real model (%s) on %s", _HF_FILENAME, self._device)
            return
        except Exception as e:  # missing dep, missing weights, or key mismatch
            logger.warning(
                "ProVQA: real model unavailable (%s: %s); provqa_score left unset.",
                type(e).__name__,
                e,
            )
        self._backend = "unavailable"
        self._ml_available = False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "real" or self._model is None:
            # No trained weights available; do not fabricate a score.
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            score = self._score(sample)
            if score is not None:
                sample.quality_metrics.provqa_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("ProVQA failed on %s: %s", sample.path, e)
        return sample

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _score(self, sample: Sample) -> Optional[float]:
        import torch

        clip = self._build_clip(sample)  # (1, num_frame*3, 3, H, W) or None
        if clip is None:
            return None
        with torch.no_grad():
            dmos = float(self._model(clip.to(self._device)).reshape(-1)[0].item())
        # Raw output is a predicted DMOS in ~[0, 1] (higher = worse quality);
        # report 1 - DMOS so provqa_score is higher = better.
        return 1.0 - dmos

    def _build_clip(self, sample: Sample):
        """Build the (1, num_frame*3, 3, H, W) ERP clip tensor.

        The network consumes ``num_frame`` temporal triplets ordered
        ``[left, center, right]``; ``num_frame*3`` uniformly-spaced frames in
        temporal order reproduce that layout (the center frames land spread
        across the video, each flanked by its temporal neighbours).  Frames are
        fed as RGB in [0, 1] with no mean/std normalisation, exactly as the
        upstream ODV-VQA dataset does.
        """
        import cv2
        import torch

        n_needed = _NUM_FRAME * 3  # 18
        frames = list(sample_frames(sample.path, max_frames=n_needed, color="rgb"))
        if not frames:
            return None
        # Pad/resample to exactly n_needed via uniform index selection over what
        # was decoded (a still image or short clip is tiled deterministically).
        idx = np.linspace(0, len(frames) - 1, n_needed).round().astype(int)

        tensors = []
        for i in idx:
            # cv2.resize returns a fresh writable array — never mutates the
            # read-only frame buffers from sample_frames.
            resized = cv2.resize(frames[int(i)], (_ERP_W, _ERP_H))
            arr = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3, H, W)
        clip = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, n_needed, 3, H, W)
        return clip
