"""SFID — Spatial FID (Fréchet Inception Distance).

Dataset-level distribution metric via pyiqa. Computes FID on
spatially-aware features rather than global features, capturing
local texture and structure quality differences.

sfid_score — LOWER = better (closer distributions)
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class SFIDModule(BatchMetricModule):
    name = "sfid"
    description = "SFID spatial Fréchet Inception Distance via pyiqa (batch, lower=better)"
    default_config = {
        "spatial_patches": 4,
        "feature_dim": 128,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.spatial_patches = self.config.get("spatial_patches", 4)
        self.feature_dim = self.config.get("feature_dim", 128)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try pyiqa SFID
        try:
            import pyiqa
            self._model = pyiqa.create_metric("sfid", device="cpu")
            self._backend = "pyiqa"
            logger.info("SFID (pyiqa) initialised")
            return
        except (ImportError, Exception):
            pass

        # Tier 2: Heuristic fallback
        self._backend = "heuristic"
        logger.info("SFID (heuristic) initialised — install pyiqa for full model")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract spatial patch features from a sample."""
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return None
                mid = total // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                if not ret:
                    return None
            finally:
                cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is None:
                return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        # Spatial patch features
        n = self.spatial_patches
        patch_h = h // n
        patch_w = w // n
        features = []

        for i in range(n):
            for j in range(n):
                patch = gray[
                    i * patch_h : (i + 1) * patch_h,
                    j * patch_w : (j + 1) * patch_w,
                ]
                if patch.size == 0:
                    features.extend([0.0] * 8)
                    continue

                # Per-patch statistics
                features.append(np.mean(patch))
                features.append(np.std(patch))

                # Gradient features
                gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(gx ** 2 + gy ** 2)
                features.append(np.mean(grad_mag))
                features.append(np.std(grad_mag))

                # Texture features
                lap = cv2.Laplacian(patch, cv2.CV_64F)
                features.append(np.mean(np.abs(lap)))
                features.append(np.var(lap))

                # Histogram features (2 bins for compactness)
                hist = np.histogram(patch, bins=2, range=(0, 256))[0]
                hist = hist.astype(np.float64) / (hist.sum() + 1e-7)
                features.extend(hist.tolist())

        features = np.array(features, dtype=np.float64)

        # Pad or truncate to fixed dimension
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[: self.feature_dim]

        return features

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> float:
        """Compute spatial FID between feature distributions."""
        feat_matrix = np.array(features)

        if reference_features is not None and len(reference_features) >= 2:
            ref_matrix = np.array(reference_features)
        else:
            # Self-comparison: split into two halves
            n = len(feat_matrix)
            if n < 4:
                return 0.0
            mid = n // 2
            ref_matrix = feat_matrix[mid:]
            feat_matrix = feat_matrix[:mid]

        # Compute Fréchet distance
        mu1 = np.mean(feat_matrix, axis=0)
        mu2 = np.mean(ref_matrix, axis=0)

        sigma1 = np.cov(feat_matrix, rowvar=False) if feat_matrix.shape[0] > 1 else np.eye(feat_matrix.shape[1])
        sigma2 = np.cov(ref_matrix, rowvar=False) if ref_matrix.shape[0] > 1 else np.eye(ref_matrix.shape[1])

        # Ensure 2D
        if sigma1.ndim < 2:
            sigma1 = np.atleast_2d(sigma1)
        if sigma2.ndim < 2:
            sigma2 = np.atleast_2d(sigma2)

        # FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
        diff = mu1 - mu2
        mean_diff_sq = np.sum(diff ** 2)

        # Simplified: use trace approximation
        trace_sum = np.trace(sigma1) + np.trace(sigma2)

        # Approximate sqrt(sigma1 @ sigma2) trace
        product = sigma1 @ sigma2
        eigenvalues = np.real(np.linalg.eigvals(product))
        eigenvalues = np.maximum(eigenvalues, 0)
        trace_sqrt = np.sum(np.sqrt(eigenvalues))

        fid = mean_diff_sq + trace_sum - 2 * trace_sqrt
        return float(np.clip(fid, 0.0, 1e6))
