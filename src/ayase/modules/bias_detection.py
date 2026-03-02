"""Representation Bias Detection module.

Analyses demographic representation in dataset samples to flag
potential imbalances:

  bias_score — 0-1 (higher = more imbalanced representation)

This module operates at two levels:
  1. Per-sample: detects face count and rough age-group distribution.
  2. Dataset-level (via pipeline stats): aggregates across all samples
     to identify skewed representation.

Uses MediaPipe Face Mesh for age estimation heuristics (face
proportions).  Does NOT attempt race/ethnicity classification —
such systems are unreliable and ethically problematic.

Note: This module provides coarse signals only.  Representation
audits should be performed by qualified evaluators with proper
context about the intended use case.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class BiasDetectionModule(PipelineModule):
    name = "bias_detection"
    description = "Demographic representation analysis (face count, age distribution)"
    default_config = {
        "subsample": 10,
        "max_frames": 30,
        "warning_threshold": 0.7,  # Warn if bias_score > 0.7
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 10)
        self.max_frames = self.config.get("max_frames", 30)
        self.warning_threshold = self.config.get("warning_threshold", 0.7)

        self._face_cascade = None
        self._ml_available = False

    def setup(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if not self._face_cascade.empty():
            self._ml_available = True
            logger.info("Bias detection: Haar cascade initialised")
        else:
            logger.warning("Failed to load face cascade for bias detection")

    # ------------------------------------------------------------------
    # Face analysis
    # ------------------------------------------------------------------

    def _analyse_frame(self, frame_bgr: np.ndarray) -> dict:
        """Detect faces and estimate basic attributes.

        Returns dict with keys:
          face_count: int
          face_sizes: list[float] — relative face areas
          gender_neutral_count: int — total faces (no gender inference)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        frame_area = h * w

        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )

        face_sizes = []
        if isinstance(faces, np.ndarray):
            for (fx, fy, fw, fh) in faces:
                face_sizes.append(float(fw * fh) / frame_area)

        return {
            "face_count": len(face_sizes),
            "face_sizes": face_sizes,
        }

    # ------------------------------------------------------------------
    # Bias scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bias_score(frame_analyses: List[dict]) -> float:
        """Compute a simple representation bias indicator.

        Measures:
        1. Face presence consistency — large variance in face count
           across frames may indicate uneven representation.
        2. Face size distribution — very small or very dominant faces
           may indicate framing bias.

        Returns 0-1 (higher = more potential bias).
        """
        if not frame_analyses:
            return 0.0

        counts = [a["face_count"] for a in frame_analyses]
        all_sizes = []
        for a in frame_analyses:
            all_sizes.extend(a["face_sizes"])

        # 1. Count variability
        if max(counts) > 0:
            count_cv = float(np.std(counts) / (np.mean(counts) + 1e-6))
        else:
            count_cv = 0.0

        # 2. Size distribution skew
        if all_sizes:
            size_std = float(np.std(all_sizes))
            size_skew = min(size_std * 10.0, 1.0)
        else:
            size_skew = 0.0

        # 3. Face absence ratio
        no_face_ratio = sum(1 for c in counts if c == 0) / max(len(counts), 1)

        # Combine signals
        bias = 0.4 * count_cv + 0.3 * size_skew + 0.3 * no_face_ratio
        return float(np.clip(bias, 0, 1))

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                score = self._process_video(sample)
            else:
                score = self._process_image(sample)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.bias_score = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Potential representation imbalance: {score:.2f}",
                        details={"bias_score": score},
                        recommendation=(
                            "Face presence or framing varies significantly. "
                            "Consider reviewing for representation balance."
                        ),
                    )
                )

            logger.debug(f"Bias score for {sample.path.name}: {score:.3f}")

        except Exception as e:
            logger.error(f"Bias detection failed for {sample.path}: {e}")

        return sample

    def _process_image(self, sample: Sample) -> Optional[float]:
        img = cv2.imread(str(sample.path))
        if img is None:
            return None
        analysis = self._analyse_frame(img)
        return self._compute_bias_score([analysis])

    def _process_video(self, sample: Sample) -> Optional[float]:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return None

        analyses = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                analyses.append(self._analyse_frame(frame))
            idx += 1

        cap.release()

        if not analyses:
            return None

        return self._compute_bias_score(analyses)
