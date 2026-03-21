"""FVMD (Fréchet Video Motion Distance) module.

FVMD is a variant of FVD that focuses on motion features instead of appearance.
It extracts optical flow features and computes Fréchet distance on motion statistics.
Lower FVMD = better motion quality in generated videos.

This is a dataset-level metric that compares two distributions of videos.
"""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class FVMDModule(BatchMetricModule):
    name = "fvmd"
    description = "Fréchet Video Motion Distance for motion quality evaluation (batch metric)"
    default_config = {
        # Pure OpenCV optical flow
        "num_frames": 16,  # Number of frames to sample
        "flow_method": "farneback",  # Optical flow method
        "subsample_videos": None,  # Max videos to process
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.num_frames = self.config.get("num_frames", 16)
        self.flow_method = self.config.get("flow_method", "farneback")
        self.subsample_videos = self.config.get("subsample_videos", None)
        self._ml_available = True  # Always available (OpenCV only)
        self._processed_count = 0

    def setup(self) -> None:
        self._processed_count = 0
        logger.info("FVMD module initialized (OpenCV optical flow)")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract motion features (optical flow) from video.

        Args:
            sample: Video sample

        Returns:
            Motion feature vector, or None if extraction failed
        """
        if not sample.is_video:
            return None

        # Check subsample limit
        if self.subsample_videos is not None and self._processed_count >= self.subsample_videos:
            return None

        try:
            # Load video frames
            cap = cv2.VideoCapture(str(sample.path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < self.num_frames:
                cap.release()
                return None

            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    return None

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)

            cap.release()

            # Compute optical flow between consecutive frames
            flow_features = []

            for i in range(len(frames) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # Extract flow statistics as features
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Feature vector: mean/std of magnitude and angle
                flow_feats = [
                    magnitude.mean(),
                    magnitude.std(),
                    magnitude.max(),
                    angle.mean(),
                    angle.std(),
                    # Histogram of magnitudes (4 bins)
                    *np.histogram(magnitude, bins=4, range=(0, 10))[0] / magnitude.size,
                ]

                flow_features.append(flow_feats)

            # Concatenate all flow features
            features = np.concatenate(flow_features)

            self._processed_count += 1
            return features

        except Exception as e:
            logger.debug(f"Failed to extract motion features from {sample.path}: {e}")
            return None

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute Fréchet distance between motion feature distributions.

        Args:
            features: List of motion feature vectors from generated videos
            reference_features: Optional list of motion features from reference videos

        Returns:
            FVMD score (lower is better)
        """
        try:
            from scipy import linalg

            # Convert to numpy array
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                # Split features in half
                mid = len(features_array) // 2
                ref_array = features_array[:mid]
                features_array = features_array[mid:]

            # Guard: need at least 2 samples for covariance
            if features_array.shape[0] < 2:
                return 0.0

            # Compute statistics
            mu1 = np.mean(features_array, axis=0)
            sigma1 = np.cov(features_array, rowvar=False)

            if ref_array.shape[0] < 2:
                return 0.0

            mu2 = np.mean(ref_array, axis=0)
            sigma2 = np.cov(ref_array, rowvar=False)

            # Compute Fréchet distance
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fvmd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

            return float(fvmd)

        except Exception as e:
            logger.error(f"Failed to compute FVMD: {e}")
            return float('inf')

    def on_dispose(self) -> None:
        """Compute FVMD after all samples processed."""
        if len(self._feature_cache) < 2:
            logger.info(f"FVMD: Not enough samples ({len(self._feature_cache)}) for metric computation")
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            fvmd_score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None
            )

            logger.info(
                f"FVMD computed: {fvmd_score:.2f} "
                f"(generated: {len(self._feature_cache)}, "
                f"reference: {len(self._reference_cache)})"
            )

            # Store in pipeline stats if available
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("fvmd", fvmd_score)

        except Exception as e:
            logger.error(f"Failed to compute FVMD: {e}")

        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0
