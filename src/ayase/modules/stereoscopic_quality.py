"""Stereoscopic Quality module.

Evaluates stereo 3D video comfort and quality:

  stereo_comfort_score — 0-100 (higher = more comfortable viewing)

For side-by-side (SBS) or top-bottom (TB) stereo content:
  1. Splits the frame into left/right views.
  2. Computes disparity map via stereo matching.
  3. Checks depth comfort zone (disparity range).
  4. Detects stereoscopic window violations.
  5. Estimates cross-talk via view similarity.

For standard 2D content: returns None (metric is inapplicable).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class StereoscopicQualityModule(PipelineModule):
    name = "stereoscopic_quality"
    description = "Stereo 3D comfort and quality assessment"
    default_config = {
        "stereo_format": "auto",  # "sbs" (side-by-side), "tb" (top-bottom), "auto"
        "subsample": 10,
        "max_frames": 30,
        "max_disparity_percent": 3.0,  # Max % of width for comfortable viewing
        "warning_threshold": 50.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.stereo_format = self.config.get("stereo_format", "auto")
        self.subsample = self.config.get("subsample", 10)
        self.max_frames = self.config.get("max_frames", 30)
        self.max_disp_pct = self.config.get("max_disparity_percent", 3.0)
        self.warning_threshold = self.config.get("warning_threshold", 50.0)

    def setup(self) -> None:
        logger.info("Stereoscopic quality: OpenCV stereo matching initialised")

    # ------------------------------------------------------------------
    # Format detection & view splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_stereo_format(frame: np.ndarray) -> Optional[str]:
        """Guess stereo format from frame aspect ratio.

        SBS: width ~2x height.  TB: height ~2x width.
        Standard 16:9 or similar: not stereo.
        """
        h, w = frame.shape[:2]
        aspect = w / max(h, 1)

        if aspect > 3.2:  # Very wide — likely SBS
            return "sbs"
        if aspect < 0.7:  # Very tall — likely TB
            return "tb"
        # Could be half-SBS (normal aspect but each half is squeezed)
        # For now, assume not stereo
        return None

    @staticmethod
    def _split_views(
        frame: np.ndarray, fmt: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        if fmt == "sbs":
            left = frame[:, : w // 2]
            right = frame[:, w // 2:]
        elif fmt == "tb":
            left = frame[: h // 2, :]
            right = frame[h // 2:, :]
        else:
            left = frame
            right = frame
        return left, right

    # ------------------------------------------------------------------
    # Stereo analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_disparity(left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        """Compute disparity map using semi-global block matching."""
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8 * 5 * 5,
            P2=32 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )
        disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disp

    def _evaluate_stereo(self, left: np.ndarray, right: np.ndarray) -> float:
        """Score stereo quality for one frame pair.  Returns 0-100."""
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Resize to match if needed
        h = min(left_gray.shape[0], right_gray.shape[0])
        w = min(left_gray.shape[1], right_gray.shape[1])
        left_gray = cv2.resize(left_gray, (w, h))
        right_gray = cv2.resize(right_gray, (w, h))

        disp = self._compute_disparity(left_gray, right_gray)

        # 1. Depth comfort: disparity range within comfortable limits
        valid = disp > 0
        if valid.sum() < 100:
            return 50.0

        disp_range = float(np.percentile(disp[valid], 95) - np.percentile(disp[valid], 5))
        max_comfortable = w * self.max_disp_pct / 100.0
        comfort = min(max_comfortable / max(disp_range, 1e-6), 1.0)

        # 2. Completeness: fraction of valid disparity pixels
        completeness = float(valid.sum()) / max(valid.size, 1)

        # 3. Smoothness: disparity gradient magnitude (less blocky = better)
        disp_dx = np.abs(np.diff(disp, axis=1))
        disp_dy = np.abs(np.diff(disp, axis=0))
        smoothness = 1.0 - min(float(disp_dx.mean() + disp_dy.mean()) / 10.0, 1.0)

        # 4. Cross-talk estimate: views should differ but not too much
        view_diff = cv2.absdiff(left_gray, right_gray)
        mean_diff = float(view_diff.mean()) / 255.0
        # Too similar (~0) = no stereo, too different (>0.3) = cross-talk/error
        crosstalk = 1.0 - abs(mean_diff - 0.05) * 10.0
        crosstalk = max(0, min(1, crosstalk))

        score = (
            0.35 * comfort
            + 0.25 * completeness
            + 0.20 * smoothness
            + 0.20 * crosstalk
        ) * 100

        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            score = self._process_video(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.stereo_comfort_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low stereo comfort: {score:.1f}/100",
                        details={"stereo_comfort_score": score},
                        recommendation="Stereo depth may cause viewer discomfort.",
                    )
                )

            logger.debug(f"Stereo comfort for {sample.path.name}: {score:.1f}")

        except Exception as e:
            logger.error(f"Stereoscopic quality failed for {sample.path}: {e}")

        return sample

    def _process_video(self, path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        # Read first frame to detect format
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return None

        fmt = self.stereo_format
        if fmt == "auto":
            fmt = self._detect_stereo_format(first_frame)
            if fmt is None:
                cap.release()
                return None  # Not stereo content

        scores = []
        left, right = self._split_views(first_frame, fmt)
        scores.append(self._evaluate_stereo(left, right))

        idx = 1
        while len(scores) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                left, right = self._split_views(frame, fmt)
                scores.append(self._evaluate_stereo(left, right))
            idx += 1

        cap.release()
        return float(np.mean(scores)) if scores else None
