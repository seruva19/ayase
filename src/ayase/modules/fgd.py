"""FGD — Frechet Gesture Distance (2020).

Dataset-level metric that measures the distance between distributions
of gesture/motion sequences using feature embeddings and the Frechet
distance formula. Designed for gesture generation evaluation.

fgd_score — lower = better (distribution closer to reference).
"""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class FGDModule(BatchMetricModule):
    name = "fgd"
    description = "Frechet Gesture Distance for motion generation (batch metric, 2020)"
    default_config = {
        "num_frames": 16,
        "subsample_videos": None,
    }
    metric_info = {
        "fgd": "Frechet Gesture Distance between generated and reference motion distributions (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.num_frames = self.config.get("num_frames", 16)
        self.subsample_videos = self.config.get("subsample_videos", None)
        self._processed_count = 0

    def setup(self) -> None:
        logger.info("FGD module initialised (heuristic)")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract motion features from video for gesture distribution comparison."""
        if not sample.is_video:
            return None
        if self.subsample_videos is not None and self._processed_count >= self.subsample_videos:
            return None

        try:
            features = self._extract_motion_features(sample.path)
            if features is not None:
                self._processed_count += 1
            return features
        except Exception as e:
            logger.debug(f"FGD feature extraction failed for {sample.path}: {e}")
            return None

    def _extract_motion_features(self, path: Path) -> Optional[np.ndarray]:
        """Extract temporal motion features via optical flow statistics."""
        cap = cv2.VideoCapture(str(path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2:
                return None

            n_sample = min(self.num_frames, total)
            indices = np.linspace(0, total - 1, n_sample, dtype=int)

            prev_gray = None
            flow_features = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64))

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    # Statistics: mean, std, max of magnitude + angle histogram
                    stats = [
                        mag.mean(), mag.std(), mag.max(),
                        ang.mean(), ang.std(),
                    ]
                    # Angle histogram (8 bins)
                    hist, _ = np.histogram(ang.flatten(), bins=8, range=(0, 2 * np.pi))
                    hist = hist.astype(np.float64)
                    hist /= hist.sum() + 1e-8
                    stats.extend(hist.tolist())

                    flow_features.append(stats)

                prev_gray = gray

            if not flow_features:
                return None

            # Aggregate across frames: mean of all per-frame feature vectors
            feat = np.mean(flow_features, axis=0)
            return feat.astype(np.float64)
        finally:
            cap.release()

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute Frechet distance between gesture feature distributions."""
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
            logger.error(f"FGD computation failed: {e}")
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
            # Fallback without scipy: use trace approximation
            fd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2))

        return float(fd)

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 2:
            logger.info(f"FGD: Not enough samples ({len(self._feature_cache)})")
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
            logger.info(f"FGD: {score:.4f} ({len(self._feature_cache)} samples)")

            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("fgd", score)
        except Exception as e:
            logger.error(f"FGD failed: {e}")
        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0
