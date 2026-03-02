"""Professional Production Quality metrics module.

Assesses cinematography-level quality attributes:

  color_grading_score  — colour consistency across the clip (0-100)
  white_balance_score  — accuracy / neutrality of white balance (0-100)
  exposure_consistency — stability of exposure over time (0-100)
  focus_quality        — sharpness / depth-of-field quality (0-100)
  banding_severity     — colour banding in gradients (0-100, lower=better)

All metrics are pure OpenCV + NumPy (no ML).  Only videos are
processed (images get a single-frame assessment where applicable).
"""

import logging
from typing import List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ProductionQualityModule(PipelineModule):
    name = "production_quality"
    description = "Professional production quality (colour, exposure, focus, banding)"
    default_config = {
        "max_frames": 150,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 150)

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _color_grading_score(frames: List[np.ndarray]) -> float:
        """How consistent is the colour palette across the clip?

        We measure the standard deviation of per-frame mean colour
        in LAB space.  Lower deviation → higher score.
        """
        if len(frames) < 2:
            return 100.0

        means = []
        for f in frames:
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
            means.append(lab.mean(axis=(0, 1)))

        means = np.stack(means)
        dev = float(means.std(axis=0).mean())
        # Normalise: dev ~0 → 100, dev ~20+ → 0
        return float(np.clip(100.0 - dev * 5.0, 0, 100))

    @staticmethod
    def _white_balance_score(frame: np.ndarray) -> float:
        """Grey-world assumption: how close is the mean colour to neutral grey?

        Neutral means R≈G≈B.  The further away, the stronger the
        colour cast (worse WB).
        """
        bgr_mean = frame.mean(axis=(0, 1))  # [B, G, R]
        overall = bgr_mean.mean()
        if overall < 1:
            return 50.0

        deviation = np.std(bgr_mean / overall)
        # deviation ~0 → perfect, ~0.3+ → bad
        return float(np.clip(100.0 - deviation * 300.0, 0, 100))

    @staticmethod
    def _exposure_consistency(frames: List[np.ndarray]) -> float:
        """Measures stability of luminance across frames."""
        if len(frames) < 2:
            return 100.0

        lums = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            lums.append(float(gray.mean()))

        cv_val = float(np.std(lums) / (np.mean(lums) + 1e-6))
        # CV ~0 → 100 (stable), CV ~0.2+ → 0 (unstable)
        return float(np.clip(100.0 - cv_val * 500.0, 0, 100))

    @staticmethod
    def _focus_quality(frame: np.ndarray) -> float:
        """Sharpness via Laplacian variance, mapped to 0-100."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # lap_var ~1000+ is very sharp, ~10 is very blurry
        return float(np.clip(lap_var / 10.0, 0, 100))

    @staticmethod
    def _banding_severity(frame: np.ndarray) -> float:
        """Detects colour banding in smooth gradients.

        Method: look at the Y channel, compute horizontal and
        vertical second-order differences.  In smooth regions
        banding creates periodic step-like patterns which have high
        second-derivative amplitude on an otherwise flat region.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Second-order horizontal difference
        d2 = np.abs(gray[:, 2:] - 2 * gray[:, 1:-1] + gray[:, :-2])

        # Focus on smooth regions: where local gradient is small
        grad = np.abs(gray[:, 1:] - gray[:, :-1])
        # Pad grad to match d2 shape
        grad = grad[:, :-1]
        smooth_mask = grad < 5.0  # smooth region

        if smooth_mask.sum() < 100:
            return 0.0

        # Banding score = mean of second derivative in smooth regions
        banding = float(d2[smooth_mask].mean())
        # Scale: banding ~0 → no issue, ~5+ → severe
        return float(np.clip(banding * 20.0, 0, 100))

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        try:
            if sample.is_video:
                return self._process_video(sample)
            else:
                return self._process_image(sample)
        except Exception as e:
            logger.error(f"Production quality failed for {sample.path}: {e}")
            return sample

    def _process_image(self, sample: Sample) -> Sample:
        img = cv2.imread(str(sample.path))
        if img is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.white_balance_score = self._white_balance_score(img)
        sample.quality_metrics.focus_quality = self._focus_quality(img)
        sample.quality_metrics.banding_severity = self._banding_severity(img)

        return sample

    def _process_video(self, sample: Sample) -> Sample:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine which frame indices to keep (uniform subsample if needed)
        if total_frames > 0 and total_frames > self.max_frames:
            keep_indices = set(np.linspace(0, total_frames - 1, self.max_frames, dtype=int))
        else:
            keep_indices = None  # keep all

        frames = []
        wb_scores = []
        focus_scores = []
        banding_scores = []
        frame_idx = 0
        kept_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if keep_indices is None or frame_idx in keep_indices:
                # Keep every sampled frame for temporal metrics (colour/exposure)
                frames.append(frame)
                # Subsample spatial metrics (every 5th kept frame)
                if kept_count % 5 == 0:
                    wb_scores.append(self._white_balance_score(frame))
                    focus_scores.append(self._focus_quality(frame))
                    banding_scores.append(self._banding_severity(frame))
                kept_count += 1

            frame_idx += 1

        cap.release()

        if not frames:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.color_grading_score = self._color_grading_score(frames)
        sample.quality_metrics.exposure_consistency = self._exposure_consistency(frames)
        sample.quality_metrics.white_balance_score = (
            float(np.mean(wb_scores)) if wb_scores else None
        )
        sample.quality_metrics.focus_quality = (
            float(np.mean(focus_scores)) if focus_scores else None
        )
        sample.quality_metrics.banding_severity = (
            float(np.mean(banding_scores)) if banding_scores else None
        )

        logger.debug(
            f"Production for {sample.path.name}: "
            f"cg={sample.quality_metrics.color_grading_score:.0f} "
            f"wb={sample.quality_metrics.white_balance_score:.0f} "
            f"exp={sample.quality_metrics.exposure_consistency:.0f} "
            f"foc={sample.quality_metrics.focus_quality:.0f} "
            f"band={sample.quality_metrics.banding_severity:.0f}"
        )

        return sample
