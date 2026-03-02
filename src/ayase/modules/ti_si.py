"""Temporal Information (TI) and Spatial Information (SI) module.

ITU-T P.910 standard metrics used worldwide for characterising video
content complexity.

SI (Spatial Information):
  For each frame, apply a Sobel filter. SI is the maximum (over all
  frames) of the standard deviation of the filtered luminance frame.
  Higher SI means more spatial detail / edges.

TI (Temporal Information):
  For each pair of consecutive frames, compute the pixel-wise
  difference. TI is the maximum (over all frame-pairs) of the standard
  deviation of that difference frame.
  Higher TI means more temporal change / motion.

Both metrics are unit-less, typical ranges roughly 0-250 but depend on
content and resolution.  No external dependencies beyond OpenCV.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TISIModule(PipelineModule):
    name = "ti_si"
    description = "ITU-T P.910 Temporal & Spatial Information"
    default_config = {
        "max_frames": 300,  # Cap to avoid very long videos
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 300)

    def _compute_si(self, gray: np.ndarray) -> float:
        """Standard deviation of Sobel-filtered luminance frame."""
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.hypot(sobel_x, sobel_y)
        return float(sobel.std())

    def _compute_image(self, sample: Sample) -> Sample:
        """SI only (no TI for a single image)."""
        img = cv2.imread(str(sample.path))
        if img is None:
            logger.warning(f"Cannot read image: {sample.path}")
            return sample

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        si = self._compute_si(gray)

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.spatial_information = si
        logger.debug(f"SI for {sample.path.name}: {si:.2f}")
        return sample

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return self._compute_image(sample)

        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {sample.path}")
            return sample

        try:
            si_values = []
            ti_values = []
            prev_gray: Optional[np.ndarray] = None
            frame_idx = 0

            while frame_idx < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # --- SI ---
                si_values.append(self._compute_si(gray))

                # --- TI ---
                if prev_gray is not None:
                    diff = gray.astype(np.float64) - prev_gray.astype(np.float64)
                    ti_values.append(float(diff.std()))

                prev_gray = gray
                frame_idx += 1

            if not si_values:
                return sample

            # ITU-T P.910 defines SI/TI as the *maximum* over time, but the
            # mean is also commonly reported.  We store the max (standard).
            si = float(np.max(si_values))
            ti = float(np.max(ti_values)) if ti_values else 0.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.spatial_information = si
            sample.quality_metrics.temporal_information = ti

            logger.debug(
                f"TI/SI for {sample.path.name}: SI={si:.2f}, TI={ti:.2f} "
                f"({frame_idx} frames)"
            )

        except Exception as e:
            logger.error(f"TI/SI failed for {sample.path}: {e}")
        finally:
            cap.release()

        return sample
