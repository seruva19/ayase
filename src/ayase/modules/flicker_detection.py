"""Flicker Detection module.

Detects temporal luminance / colour fluctuations that appear as
visible flicker to the human eye.  Common in:
  - poorly compressed video (quantisation flicker),
  - AI-generated video (temporal inconsistency),
  - screen captures with variable refresh rates.

The score is 0-100 where 0 = no flicker, 100 = severe flicker.

Algorithm
---------
1. Compute per-frame mean luminance.
2. Take the first-order difference of that time-series.
3. Detect high-frequency oscillations via the standard deviation of
   the sign-changes (zero-crossings) in the difference signal — a
   simple proxy for "how often does brightness reverse direction?"
4. Combine oscillation rate with the magnitude of changes.

Pure OpenCV + NumPy, no external dependencies.
"""

import logging

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FlickerDetectionModule(PipelineModule):
    name = "flicker_detection"
    description = "Detects temporal luminance flicker"
    default_config = {
        "max_frames": 600,
        "warning_threshold": 30.0,  # Warn if flicker_score > 30
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 600)
        self.warning_threshold = self.config.get("warning_threshold", 30.0)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample  # Flicker is a temporal phenomenon

        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {sample.path}")
            return sample

        try:
            luminances = []
            frame_idx = 0

            while frame_idx < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                luminances.append(float(gray.mean()))
                frame_idx += 1

            if len(luminances) < 3:
                return sample  # Need at least 3 frames

            lum = np.array(luminances)

            # --- 1. First-order difference ---
            diff = np.diff(lum)

            # --- 2. Oscillation rate (zero-crossing rate of diff) ---
            signs = np.sign(diff)
            # Remove zeros so sign changes are clean
            signs[signs == 0] = 1
            sign_changes = np.abs(np.diff(signs)) / 2  # 0 or 1
            zero_crossing_rate = float(sign_changes.mean())  # 0-1

            # --- 3. Magnitude of fluctuation ---
            # Use std of diff normalised by mean luminance
            mean_lum = lum.mean() if lum.mean() > 1 else 1.0
            magnitude = float(np.std(diff) / mean_lum)

            # --- Combine into 0-100 score ---
            # High oscillation rate + high magnitude → high flicker
            # Scale factors chosen so "normal" content stays < 10 and
            # heavily flickering content approaches 100.
            raw = zero_crossing_rate * magnitude * 500.0
            flicker_score = float(np.clip(raw, 0.0, 100.0))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.flicker_score = flicker_score

            if flicker_score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Flicker detected: {flicker_score:.1f}",
                        details={
                            "flicker_score": flicker_score,
                            "zero_crossing_rate": zero_crossing_rate,
                            "magnitude": magnitude,
                        },
                        recommendation=(
                            "Temporal luminance flicker detected. "
                            "May indicate compression artefacts, "
                            "inconsistent lighting, or generation instability."
                        ),
                    )
                )

            logger.debug(
                f"Flicker for {sample.path.name}: {flicker_score:.1f} "
                f"(zcr={zero_crossing_rate:.3f}, mag={magnitude:.4f})"
            )

        except Exception as e:
            logger.error(f"Flicker detection failed for {sample.path}: {e}")
        finally:
            cap.release()

        return sample
