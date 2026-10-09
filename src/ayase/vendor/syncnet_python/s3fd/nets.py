"""S3FD network (port of ``detectors/s3fd/nets.py`` from syncnet_python, MIT).

Layer names are unchanged: ``sfd_face.pth`` is loaded by them.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from .box_utils import Detect, PriorBox

_L2NORM_EPS = 1e-10


class L2Norm(nn.Module):
    """Channel-wise L2 normalisation with a learned scale."""

    def __init__(self, n_channels: int, scale: float) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = _L2NORM_EPS
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        init.constant_(self.weight, self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x


class S3FDNet(nn.Module):
    """Single Shot Scale-invariant Face Detector on a VGG16 trunk."""

    def __init__(self) -> None:
        super().__init__()

        self.vgg = nn.ModuleList(
            [
                nn.Conv2d(3, 64, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, ceil_mode=True),
                nn.Conv2d(256, 512, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 1024, 3, 1, padding=6, dilation=6),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, 1, 1),
                nn.ReLU(inplace=True),
            ]
        )

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        self.extras = nn.ModuleList(
            [
                nn.Conv2d(1024, 256, 1, 1),
                nn.Conv2d(256, 512, 3, 2, padding=1),
                nn.Conv2d(512, 128, 1, 1),
                nn.Conv2d(128, 256, 3, 2, padding=1),
            ]
        )

        self.loc = nn.ModuleList(
            [
                nn.Conv2d(256, 4, 3, 1, padding=1),
                nn.Conv2d(512, 4, 3, 1, padding=1),
                nn.Conv2d(512, 4, 3, 1, padding=1),
                nn.Conv2d(1024, 4, 3, 1, padding=1),
                nn.Conv2d(512, 4, 3, 1, padding=1),
                nn.Conv2d(256, 4, 3, 1, padding=1),
            ]
        )

        self.conf = nn.ModuleList(
            [
                nn.Conv2d(256, 4, 3, 1, padding=1),
                nn.Conv2d(512, 2, 3, 1, padding=1),
                nn.Conv2d(512, 2, 3, 1, padding=1),
                nn.Conv2d(1024, 2, 3, 1, padding=1),
                nn.Conv2d(512, 2, 3, 1, padding=1),
                nn.Conv2d(256, 2, 3, 1, padding=1),
            ]
        )

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()
        # Priors depend on the input size only, which is constant within one video.
        self._priors_key: Optional[Tuple[int, ...]] = None
        self._priors: Optional[torch.Tensor] = None

    def _priors_for(self, size: Tuple[int, int], features_maps: List[List[int]]) -> torch.Tensor:
        key = (*size, *(dim for fmap in features_maps for dim in fmap))
        if self._priors is None or self._priors_key != key:
            self._priors = PriorBox(size, features_maps).forward()
            self._priors_key = key
        return self._priors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return detections ``(N, 2, top_k, 5)`` as ``score, x1, y1, x2, y2`` (image fractions)."""
        size = (int(x.size(2)), int(x.size(3)))
        sources = []
        loc = []
        conf = []

        for k in range(16):
            x = self.vgg[k](x)
        sources.append(self.L2Norm3_3(x))

        for k in range(16, 23):
            x = self.vgg[k](x)
        sources.append(self.L2Norm4_3(x))

        for k in range(23, 30):
            x = self.vgg[k](x)
        sources.append(self.L2Norm5_3(x))

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        for k, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            source = sources[i]
            conf.append(self.conf[i](source).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](source).permute(0, 2, 3, 1).contiguous())

        features_maps = [[o.size(1), o.size(2)] for o in loc]

        loc_flat = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf_flat = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        priors = self._priors_for(size, features_maps).to(device=x.device, dtype=x.dtype)

        return self.detect.forward(
            loc_flat.view(loc_flat.size(0), -1, 4),
            self.softmax(conf_flat.view(conf_flat.size(0), -1, 2)),
            priors,
        )
