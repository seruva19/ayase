"""Judder and Stutter Detection module.

Judder — irregular temporal sampling, typically caused by frame-rate
conversion (e.g. 24→30 pull-down) or uneven capture intervals.
Detected by measuring variance in the temporal cadence of
frame-to-frame differences.

Stutter — duplicate or near-duplicate frames that produce visible
pauses.  Detected by counting consecutive frames with very low
pixel difference.

Both scores are 0-100 (lower = better / smoother).
Pure OpenCV + NumPy, no ML dependencies.
"""

import logging

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class JudderStutterModule(PipelineModule):
    name = "judder_stutter"
    description = "Detects judder (uneven cadence) and stutter (duplicate frames)"
    default_config = {
        "max_frames": 600,
        "duplicate_threshold": 1.0,  # Max mean-absolute-diff to call a frame "duplicate"
        "warning_threshold": 20.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 600)
        self.dup_thresh = self.config.get("duplicate_threshold", 1.0)
        self.warning_threshold = self.config.get("warning_threshold", 20.0)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        try:
            diffs = []  # mean absolute frame-to-frame difference
            prev_gray = None
            idx = 0

            while idx < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

                if prev_gray is not None:
                    d = float(np.mean(np.abs(gray - prev_gray)))
                    diffs.append(d)

                prev_gray = gray
                idx += 1

            if len(diffs) < 3:
                return sample

            diffs_arr = np.array(diffs)

            # ----- Stutter: fraction of near-duplicate frames -----
            n_duplicates = int(np.sum(diffs_arr < self.dup_thresh))
            stutter_pct = n_duplicates / len(diffs_arr) * 100.0

            # ----- Judder: irregularity of the cadence -----
            # In smooth video the sequence of frame diffs forms a
            # relatively regular pattern.  Judder shows up as high
            # variance in that sequence (after removing near-
            # duplicates which are a different artefact).
            non_dup = diffs_arr[diffs_arr >= self.dup_thresh]

            if len(non_dup) > 2:
                mean_diff = non_dup.mean()
                if mean_diff > 0:
                    cv = float(non_dup.std() / mean_diff)  # coefficient of variation
                else:
                    cv = 0.0
                # Map CV to 0-100.  CV ~0.3 is normal, >1 is severe.
                judder = float(np.clip(cv / 1.0 * 100.0, 0, 100))
            else:
                judder = 0.0

            # Store
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.judder_score = judder
            sample.quality_metrics.stutter_score = stutter_pct

            # Warn on either
            worst = max(judder, stutter_pct)
            if worst > self.warning_threshold:
                parts = []
                if judder > self.warning_threshold:
                    parts.append(f"judder={judder:.1f}")
                if stutter_pct > self.warning_threshold:
                    parts.append(f"stutter={stutter_pct:.1f}%")
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Temporal issues: {', '.join(parts)}",
                        details={
                            "judder_score": judder,
                            "stutter_score": stutter_pct,
                            "duplicate_frames": n_duplicates,
                            "total_frame_pairs": len(diffs_arr),
                        },
                        recommendation=(
                            "Video has temporal smoothness problems. "
                            "High judder may indicate frame-rate conversion "
                            "artefacts; high stutter indicates duplicate or "
                            "dropped frames."
                        ),
                    )
                )

            logger.debug(
                f"Judder/stutter for {sample.path.name}: "
                f"judder={judder:.1f}, stutter={stutter_pct:.1f}% "
                f"(dups={n_duplicates}/{len(diffs_arr)})"
            )

        except Exception as e:
            logger.error(f"Judder/stutter failed for {sample.path}: {e}")
        finally:
            cap.release()

        return sample
