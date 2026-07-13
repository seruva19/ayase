"""MC360IQA -- Multi-Channel Blind 360-degree IQA (Sun et al., 2019).

Blind (no-reference) image quality assessment for omnidirectional / 360-degree
equirectangular (ERP) content. The published method projects the ERP image into
six perspective viewports (cubemap-style: front/back/left/right/top/bottom),
runs each through a shared multi-channel ResNet-34 with a hyper-network-style
multi-scale side branch and a spatial-attention head, produces a 10-D feature
per viewport, concatenates the six, and regresses a single quality score.

The real trained weights are mirrored on the Hugging Face Hub in two dataset
variants -- ``OIQA.pkl`` (default; more robust per the upstream repo) and
``CVIQ.pkl`` -- as plain ``torch`` state_dicts (229 tensors, strict-loadable
against the vendored architecture below). Model definition and ERP->viewport
projection are vendored inline from github.com/sunwei925/MC360IQA
(``multi_channel_resnet34_hyper.py`` and ``eq2cm.py``).

Per the project's no-heuristic policy the ``mc360iqa_score`` field is populated
only by the real model; when torch / the checkpoint is unavailable the module
reports itself unavailable and leaves the field unset.

mc360iqa_score -- model-native MOS (higher = better quality); real model only.
    The absolute scale is the training set's MOS scale (OIQA/CVIQ), not 0-1.
"""

import logging

import numpy as np

from ayase.image import load_representative_frame
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

HF_REPO = "AkaneTendo25/ayase-models"
_WEIGHT_FILES = {
    "OIQA": "mc360iqa/OIQA.pkl",
    "CVIQ": "mc360iqa/CVIQ.pkl",
}

# ImageNet normalisation used by the upstream test/train transforms.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Six viewport (u, v) rotations in radians, fov = pi/2, exactly matching
# upstream test.py: BAck, BOttom, Front, Left, Right, Top.
_VIEWPORTS = (
    (np.pi, 0.0),          # BA
    (0.0, np.pi / 2),      # BO
    (0.0, 0.0),            # F
    (-np.pi / 2, 0.0),     # L
    (np.pi / 2, 0.0),      # R
    (0.0, -np.pi / 2),     # T
)


# ---------------------------------------------------------------------------
# Vendored ERP -> perspective-viewport projection (upstream eq2cm.py)
# ---------------------------------------------------------------------------
def _gen_xyz(fov, u, v, out_h, out_w):
    out = np.ones((out_h, out_w, 3), np.float32)
    x_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_w, dtype=np.float32)
    y_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_h, dtype=np.float32)
    out[:, :, :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = np.array([[1, 0, 0], [0, np.cos(v), -np.sin(v)], [0, np.sin(v), np.cos(v)]])
    Ry = np.array([[np.cos(u), 0, np.sin(u)], [0, 1, 0], [-np.sin(u), 0, np.cos(u)]])
    R = np.dot(Ry, Rx)
    return out.dot(R.T)


def _xyz_to_uv(xyz):
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    v = np.arctan2(y, c)
    return np.concatenate([u, v], axis=-1)


def _uv_to_XY(uv, eq_h, eq_w):
    u, v = np.split(uv, 2, axis=-1)
    X = (u / (2 * np.pi) + 0.5) * eq_w - 0.5
    Y = (-v / np.pi + 0.5) * eq_h - 0.5
    return np.concatenate([X, Y], axis=-1)


def _eq_to_pers(eqimg, fov, u, v, out_h, out_w):
    """Project an ERP image (H, W, 3) to a perspective viewport (out_h, out_w, 3)."""
    from scipy import ndimage

    xyz = _gen_xyz(fov, u, v, out_h, out_w)
    uv = _xyz_to_uv(xyz)
    eq_h, eq_w = eqimg.shape[:2]
    XY = _uv_to_XY(uv, eq_h, eq_w)
    X, Y = np.split(XY, 2, axis=-1)
    X = np.reshape(X, (out_h, out_w))
    Y = np.reshape(Y, (out_h, out_w))
    mc0 = ndimage.map_coordinates(eqimg[:, :, 0], [Y, X])
    mc1 = ndimage.map_coordinates(eqimg[:, :, 1], [Y, X])
    mc2 = ndimage.map_coordinates(eqimg[:, :, 2], [Y, X])
    return np.stack([mc0, mc1, mc2], axis=-1)


# ---------------------------------------------------------------------------
# Vendored multi-channel ResNet-34 hyper network (upstream
# multi_channel_resnet34_hyper.py). Built lazily so importing the module never
# requires torch.
# ---------------------------------------------------------------------------
def _build_resnet34():
    import torch
    import torch.nn as nn

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
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

    class ResNet(nn.Module):
        def __init__(self, block, layers):
            self.inplanes = 64
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            self.insert1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
            self.insert2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
            self.insert3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
            self.insert4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.insert5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
            self.insert6 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)

            self.spatial_weights1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.spatial_weights2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
            self.spatial_weights3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)

            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.feature_embedding = nn.Linear(512 * block.expansion, 10)
            self.quality = nn.Linear(10 * 6, 1)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def _forward_impl(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x_insert = self.insert1(x)
            x_insert = self.insert2(x_insert)

            x = self.layer2(x)
            x_insert = self.insert3(x_insert + x)
            x_insert = self.insert4(x_insert)

            x = self.layer3(x)
            x_insert = self.insert5(x_insert + x)
            x_insert = self.insert6(x_insert)

            x = self.layer4(x)
            x = x + x_insert

            x = self.relu(x)
            x_spatial = self.spatial_weights1(x)
            x_spatial = self.relu(x_spatial)
            x_spatial = self.spatial_weights2(x_spatial)
            x_spatial = self.relu(x_spatial)
            x_spatial = self.spatial_weights3(x_spatial)

            x = torch.mul(x, x_spatial)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.feature_embedding(x)
            return x

        def forward(self, x_BA, x_BO, x_F, x_L, x_R, x_T):
            x_BA = self._forward_impl(x_BA)
            x_BO = self._forward_impl(x_BO)
            x_F = self._forward_impl(x_F)
            x_L = self._forward_impl(x_L)
            x_R = self._forward_impl(x_R)
            x_T = self._forward_impl(x_T)
            x = torch.cat([x_BA, x_BO, x_F, x_L, x_R, x_T], 1)
            x = self.quality(x)
            return x

    return ResNet(BasicBlock, [3, 4, 6, 3])


class MC360IQAModule(PipelineModule):
    name = "mc360iqa"
    description = "MC360IQA blind 360 IQA (2019; real model only, disabled if unavailable)"
    default_config = {
        "weights_variant": "OIQA",   # "OIQA" (default, more robust) or "CVIQ"
        "projection_size": 480,       # per-viewport perspective render size
        "input_size": 224,            # network input after resize
        "device": "auto",
    }
    metric_groups = {
        "mc360iqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.variant = str(self.config.get("weights_variant", "OIQA")).upper()
        self.projection_size = int(self.config.get("projection_size", 480))
        self.input_size = int(self.config.get("input_size", 224))
        self._ml_available = False
        self._backend = None
        self._model = None
        self._transform = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch  # noqa: F401
            from torchvision import transforms
            from huggingface_hub import hf_hub_download
            import scipy  # noqa: F401 (used by _eq_to_pers projection)
        except Exception as e:
            self._backend = "unavailable"
            self._ml_available = False
            logger.warning("MC360IQA unavailable: missing dependency (%s)", e)
            return

        weight_file = _WEIGHT_FILES.get(self.variant)
        if weight_file is None:
            self._backend = "unavailable"
            self._ml_available = False
            logger.warning(
                "MC360IQA unavailable: unknown weights_variant %r (expected OIQA or CVIQ)",
                self.variant,
            )
            return

        try:
            from ayase.runtime import resolve_torch_device
            self._device = resolve_torch_device(self.config.get("device", "auto"))
        except Exception:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            import torch
            ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=weight_file)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model = _build_resnet34()
            model.load_state_dict(state_dict, strict=True)
            model.eval().to(self._device)
            self._model = model
            self._transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
            ])
            self._ml_available = True
            self._backend = "real"
            logger.info(
                "MC360IQA initialised (real backend, %s weights, device=%s)",
                self.variant, self._device,
            )
        except Exception as e:
            self._backend = "unavailable"
            self._ml_available = False
            self._model = None
            logger.warning("MC360IQA unavailable: failed to load %s weights (%s)", self.variant, e)

    def _score_erp(self, erp_bgr: np.ndarray):
        """Project an ERP frame (BGR uint8) to 6 viewports and return the MOS."""
        import cv2
        import torch
        from PIL import Image

        s = self.projection_size
        tensors = []
        for u, v in _VIEWPORTS:
            vp = _eq_to_pers(erp_bgr, np.pi / 2, u, v, s, s).astype(np.uint8)
            pil = Image.fromarray(cv2.cvtColor(vp, cv2.COLOR_BGR2RGB))
            tensors.append(self._transform(pil).unsqueeze(0).to(self._device))
        with torch.no_grad():
            out = self._model(*tensors)
        return float(out.item())

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "real":
            return sample

        # Load a representative frame in BGR to match the upstream cv2.imread
        # convention (eq_to_pers channel order + BGR2RGB). Returned frame is
        # read-only; the projection does not mutate it, but copy defensively so
        # scipy never writes back into the shared cache buffer.
        frame = load_representative_frame(sample.path, color="bgr")
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return sample
        erp = np.ascontiguousarray(frame[:, :, :3]).copy()

        try:
            score = self._score_erp(erp)
        except Exception as e:
            logger.warning("MC360IQA failed for %s: %s", sample.path, e)
            return sample

        if np.isfinite(score):
            sample.quality_metrics.mc360iqa_score = score
        return sample
