"""FID image distribution metric.

Computes Fréchet distance between generated and reference image feature
distributions. Uses InceptionV3 features when torchvision weights are available,
and falls back to deterministic image-statistics features for lightweight runs.
Lower FID is better.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.image import image_stats_features, load_representative_frame
from ayase.models import Sample

logger = logging.getLogger(__name__)


class FIDModule(BatchMetricModule):
    name = "fid"
    description = "Fréchet Inception Distance for image generation (batch metric)"
    default_config = {
        "device": "auto",
        "resize": 299,
        "use_torchvision_backend": True,
    }
    models = [
        {
            "id": "torchvision/inception_v3",
            "type": "torchvision",
            "task": "InceptionV3 feature extractor for FID",
        },
    ]
    metric_info = {
        "fid": "Fréchet Inception Distance between generated and reference image sets (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = self.config.get("device", "auto")
        self.resize = self.config.get("resize", 299)
        self.use_torchvision_backend = self.config.get("use_torchvision_backend", True)
        self._backend = "image_stats"
        self._model = None
        self._transform = None
        self._device = "cpu"

    def setup(self) -> None:
        if not self.use_torchvision_backend:
            return
        try:
            import torch
            from torchvision import models, transforms
            from torchvision.models import Inception_V3_Weights

            if self.device_config == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device_config

            model = models.inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )
            model.fc = torch.nn.Identity()
            self._model = model.to(self._device).eval()
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            self._backend = "inception_v3"
            logger.info("FID initialised with torchvision InceptionV3 on %s", self._device)
        except ImportError:
            logger.warning("FID Inception backend requires torch and torchvision; using image-stat proxy")
        except Exception as e:
            logger.warning("FID Inception setup failed (%s); using image-stat proxy", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        frame = load_representative_frame(sample.path, color="rgb")
        if frame is None:
            return None
        if self._backend == "inception_v3" and self._model is not None:
            return self._extract_inception(frame)
        return image_stats_features(frame)

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        gen = np.stack(features).astype(np.float64)
        if reference_features:
            ref = np.stack(reference_features).astype(np.float64)
        else:
            mid = len(gen) // 2
            if mid < 1:
                return float("inf")
            ref = gen[:mid]
            gen = gen[mid:]
        return self._frechet_distance(gen, ref)

    def _extract_inception(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch

            tensor = self._transform(frame.astype(np.uint8)).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model(tensor)
                if hasattr(features, "logits"):
                    features = features.logits
            return features.detach().cpu().numpy().reshape(-1).astype(np.float64)
        except Exception as e:
            logger.debug("FID Inception extraction failed: %s", e)
            return None

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
