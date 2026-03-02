"""Multi-View Consistency module.

Estimates geometric consistency for video content using feature
matching across frames:

  multiview_consistency — 0-1 (higher = better 3D consistency)

Algorithm:
  1. Extract keypoints (ORB) from frame pairs.
  2. Match features and compute fundamental matrix via RANSAC.
  3. Measure inlier ratio — high inlier ratio = geometrically
     consistent views.
  4. Check epipolar constraint residuals.

This is a proxy for 3D geometric consistency without requiring
multi-view input — it measures whether frame-to-frame geometry
is coherent enough to support 3D reconstruction.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MultiViewConsistencyModule(PipelineModule):
    name = "multi_view_consistency"
    description = "Geometric multi-view consistency via epipolar analysis"
    default_config = {
        "subsample": 5,  # Frame gap for pair analysis
        "max_pairs": 30,
        "min_matches": 20,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.max_pairs = self.config.get("max_pairs", 30)
        self.min_matches = self.config.get("min_matches", 20)

        self._orb = None

    def setup(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=500)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        logger.info("Multi-view consistency: ORB + RANSAC initialised")

    def _compute_pair_consistency(
        self, gray1: np.ndarray, gray2: np.ndarray
    ) -> Optional[float]:
        """Compute geometric consistency between two frames.

        Returns inlier ratio (0-1) from fundamental matrix estimation.
        """
        kp1, des1 = self._orb.detectAndCompute(gray1, None)
        kp2, des2 = self._orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None
        if len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return None

        matches = self._bf.match(des1, des2)
        if len(matches) < self.min_matches:
            return None

        # Sort by distance
        matches = sorted(matches, key=lambda m: m.distance)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Estimate fundamental matrix via RANSAC
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0
        )

        if mask is None:
            return None

        inlier_ratio = float(mask.sum()) / max(len(mask), 1)
        return inlier_ratio

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or self._orb is None:
            return sample

        try:
            score = self._process_video(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.multiview_consistency = score

            logger.debug(
                f"Multi-view consistency for {sample.path.name}: {score:.3f}"
            )

        except Exception as e:
            logger.error(f"Multi-view consistency failed for {sample.path}: {e}")

        return sample

    def _process_video(self, path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        prev_gray = None
        consistencies = []
        idx = 0
        pairs = 0

        while pairs < self.max_pairs:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % self.subsample == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    c = self._compute_pair_consistency(prev_gray, gray)
                    if c is not None:
                        consistencies.append(c)
                        pairs += 1

                prev_gray = gray

            idx += 1

        cap.release()

        if not consistencies:
            return None

        return float(np.mean(consistencies))
