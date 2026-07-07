"""FMD — Frechet Motion Distance (2022).

Dataset-level metric that measures the Frechet distance between distributions
of learned motion features. ayase has no pretrained motion-feature backend
wired for FMD, so the metric is left unset rather than reported from an
optical-flow proxy (which would not reproduce the published FMD).

fmd_score — lower = better (closer motion distributions).
"""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class FMDModule(BatchMetricModule):
    name = "fmd"
    description = "Frechet Motion Distance for motion generation (batch metric, 2022)"
    default_config = {
        "num_frames": 16,
        "subsample_videos": None,
    }
    metric_info = {
        "fmd": "Frechet Motion Distance between generated and reference motion distributions (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.num_frames = self.config.get("num_frames", 16)
        self.subsample_videos = self.config.get("subsample_videos", None)
        self._processed_count = 0
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        logger.warning(
            "FMD unavailable: no pretrained motion-feature backend for Frechet "
            "Motion Distance is wired; metric disabled."
        )

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """FMD has no real feature extractor wired; no features are produced."""
        return None

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute Frechet distance between motion feature distributions."""
        try:
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                mid = len(features_array) // 2
                if mid < 1:
                    return 0.0
                ref_array = features_array[:mid]
                features_array = features_array[mid:]

            return self._frechet_distance(features_array, ref_array)
        except Exception as e:
            logger.error(f"FMD computation failed: {e}")
            return float("inf")

    def _frechet_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute Frechet distance between two feature sets."""
        mu1 = np.mean(feat1, axis=0)
        mu2 = np.mean(feat2, axis=0)

        if feat1.shape[0] < 2 or feat2.shape[0] < 2:
            return float(np.sum((mu1 - mu2) ** 2))

        sigma1 = np.cov(feat1, rowvar=False)
        sigma2 = np.cov(feat2, rowvar=False)

        if sigma1.ndim == 0:
            sigma1 = np.array([[sigma1]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[sigma2]])

        diff = mu1 - mu2

        try:
            from scipy import linalg
            covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        except ImportError:
            fd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2))

        return float(fd)

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 2:
            logger.info(f"FMD: Not enough samples ({len(self._feature_cache)})")
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
            logger.info(f"FMD: {score:.4f} ({len(self._feature_cache)} samples)")

            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("fmd", score)
        except Exception as e:
            logger.error(f"FMD failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0
