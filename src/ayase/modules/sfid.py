"""sFID — spatial Fréchet Inception Distance.

Dataset-level distribution metric. Unlike FID (which uses the global 2048-d
InceptionV3 pool features), sFID uses the *intermediate spatial* features of
InceptionV3 — the first 7 channels of the ``Mixed_6e`` 17x17x768 activation
flattened to a 2023-d vector — so the distance is sensitive to local spatial
structure. This mirrors the sFID definition from Nash et al. (2021) /
guided-diffusion, ported onto torchvision's InceptionV3.

Real InceptionV3 spatial features are required: when torch/torchvision are
unavailable the metric is left unset (no heuristic stand-in).

sfid_score — LOWER = better (closer distributions)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import load_representative_frame
from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class SFIDModule(BatchMetricModule):
    name = "sfid"
    description = "sFID spatial Fréchet Inception Distance (InceptionV3 spatial features, lower=better)"
    default_config = {
        "device": "auto",
        "resize": 299,
    }
    models = [
        {
            "id": "torchvision/inception_v3",
            "type": "torchvision",
            "task": "InceptionV3 intermediate spatial features for sFID",
        },
    ]
    metric_info = {
        "sfid": "Spatial Fréchet Inception Distance on InceptionV3 Mixed_6e features (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = self.config.get("device", "auto")
        self.resize = self.config.get("resize", 299)
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch  # noqa: F401
            from torchvision import models, transforms
            from torchvision.models import Inception_V3_Weights
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            self._device = resolve_torch_device(self.device_config)

            def load_inception():
                model = models.inception_v3(
                    weights=Inception_V3_Weights.IMAGENET1K_V1,
                    transform_input=False,
                )
                return model.to(self._device).eval()

            self._model = shared_runtime_resource(
                self,
                ("sfid_inception_v3_imagenet", str(self._device)),
                load_inception,
            )
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            self._backend = "inception_v3_spatial"
            logger.info("sFID initialised with torchvision InceptionV3 spatial features on %s", self._device)
        except ImportError:
            logger.warning("sFID unavailable: requires torch and torchvision; sfid left unset.")
        except Exception as e:
            logger.warning("sFID InceptionV3 setup failed (%s); metric disabled", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        if self._backend != "inception_v3_spatial" or self._model is None:
            return None
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        return self._extract_spatial(frame)

    def _extract_spatial(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch

            tensor = self._transform(np.ascontiguousarray(frame, dtype=np.uint8))
            tensor = tensor.unsqueeze(0).to(self._device)
            with torch.no_grad():
                spatial = self._inception_spatial(tensor)  # (1, 768, 17, 17)
                spatial = spatial[:, :7, :, :]  # first 7 channels -> 7*17*17 = 2023
            return spatial.reshape(-1).detach().cpu().numpy().astype(np.float64)
        except Exception as e:
            logger.debug("sFID spatial extraction failed: %s", e)
            return None

    def _inception_spatial(self, x):
        """Run InceptionV3 forward up to the Mixed_6e intermediate block."""
        m = self._model
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)  # (B, 768, 17, 17)
        return x

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> float:
        gen = np.stack(features).astype(np.float64)
        if reference_features and len(reference_features) >= 2:
            ref = np.stack(reference_features).astype(np.float64)
        else:
            mid = len(gen) // 2
            if mid < 1:
                return float("inf")
            ref = gen[:mid]
            gen = gen[mid:]
        return self._frechet_distance(gen, ref)

    def _frechet_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        mu_x = np.mean(x, axis=0)
        mu_y = np.mean(y, axis=0)
        diff = mu_x - mu_y

        if len(x) < 2 or len(y) < 2:
            return float(diff @ diff)

        cov_x = np.atleast_2d(np.cov(x, rowvar=False))
        cov_y = np.atleast_2d(np.cov(y, rowvar=False))
        try:
            from scipy import linalg

            covmean, _ = linalg.sqrtm(cov_x @ cov_y, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            score = diff @ diff + np.trace(cov_x + cov_y - 2.0 * covmean)
        except Exception:
            score = diff @ diff + np.trace(cov_x) + np.trace(cov_y)
        return float(max(score, 0.0))
