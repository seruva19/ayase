"""Face Landmark Quality module.

Assesses temporal stability and consistency of facial features in
video using MediaPipe Face Mesh (468 landmarks).

face_landmark_jitter       — temporal jitter of landmarks 0-100 (lower=better)
face_expression_smoothness — smoothness of expression changes 0-100 (higher=better)
face_identity_consistency  — identity stability across frames 0-1 (higher=better)

Videos only — images are skipped (single frame has no temporal signal).
Requires ``mediapipe`` package.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FaceLandmarkQualityModule(PipelineModule):
    name = "face_landmark_quality"
    description = "Facial landmark jitter, expression smoothness, identity consistency"
    default_config = {
        "subsample": 2,  # Every Nth video frame
        "max_frames": 300,
        "jitter_warning": 30.0,  # Warn if jitter > 30
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 2)
        self.max_frames = self.config.get("max_frames", 300)
        self.jitter_warning = self.config.get("jitter_warning", 30.0)

        self._face_mesh = None
        self._ml_available = False

    def setup(self) -> None:
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._ml_available = True
            logger.info("Face landmark quality: MediaPipe Face Mesh initialised")
        except ImportError:
            logger.warning("mediapipe not installed. Install with: pip install mediapipe")
        except Exception as e:
            logger.warning(f"Failed to setup face landmark quality: {e}")

    def _extract_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 face landmarks as (N, 2) normalised coordinates."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        coords = np.array(
            [(lm.x, lm.y) for lm in face.landmark],
            dtype=np.float32,
        )
        return coords

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            return self._process_video(sample)
        except Exception as e:
            logger.error(f"Face landmark quality failed for {sample.path}: {e}")
            return sample

    def _process_video(self, sample: Sample) -> Sample:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        landmark_series: List[np.ndarray] = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                lm = self._extract_landmarks(frame)
                if lm is not None:
                    landmark_series.append(lm)
            idx += 1

        cap.release()

        if len(landmark_series) < 3:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        # Compute metrics from the landmark time series
        jitter = self._compute_jitter(landmark_series)
        smoothness = self._compute_expression_smoothness(landmark_series)
        identity = self._compute_identity_consistency(landmark_series)

        sample.quality_metrics.face_landmark_jitter = jitter
        sample.quality_metrics.face_expression_smoothness = smoothness
        sample.quality_metrics.face_identity_consistency = identity

        if jitter > self.jitter_warning:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"High face landmark jitter: {jitter:.1f}/100",
                    details={
                        "face_landmark_jitter": jitter,
                        "face_expression_smoothness": smoothness,
                    },
                    recommendation=(
                        "Facial landmarks are unstable across frames. "
                        "May indicate tracking issues or face flickering."
                    ),
                )
            )

        logger.debug(
            f"Face landmarks for {sample.path.name}: "
            f"jitter={jitter:.1f} smooth={smoothness:.1f} id={identity:.3f}"
        )

        return sample

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_jitter(series: List[np.ndarray]) -> float:
        """Compute temporal jitter of face landmarks.

        Jitter = mean frame-to-frame landmark displacement after
        normalising for global face translation.  Higher = worse.
        Score 0-100.
        """
        displacements = []
        for i in range(1, len(series)):
            prev, curr = series[i - 1], series[i]

            # Remove global translation (center-of-mass shift)
            prev_c = prev - prev.mean(axis=0, keepdims=True)
            curr_c = curr - curr.mean(axis=0, keepdims=True)

            # Per-landmark displacement
            disp = np.linalg.norm(curr_c - prev_c, axis=1)
            displacements.append(float(disp.mean()))

        if not displacements:
            return 0.0

        mean_disp = float(np.mean(displacements))
        # Normalise: mean_disp ~0 → jitter=0, mean_disp ~0.02+ → jitter=100
        # (normalised coords are 0-1, so 0.02 is significant movement)
        return float(np.clip(mean_disp * 5000.0, 0, 100))

    @staticmethod
    def _compute_expression_smoothness(series: List[np.ndarray]) -> float:
        """Measure how smoothly facial expressions change.

        Uses the second derivative (acceleration) of landmark positions.
        Lower acceleration = smoother motion = higher score.
        Score 0-100 (higher=smoother).
        """
        if len(series) < 3:
            return 100.0

        accelerations = []
        for i in range(1, len(series) - 1):
            prev = series[i - 1] - series[i - 1].mean(axis=0, keepdims=True)
            curr = series[i] - series[i].mean(axis=0, keepdims=True)
            nxt = series[i + 1] - series[i + 1].mean(axis=0, keepdims=True)

            accel = nxt - 2 * curr + prev
            accelerations.append(float(np.linalg.norm(accel, axis=1).mean()))

        if not accelerations:
            return 100.0

        mean_accel = float(np.mean(accelerations))
        # Normalise: accel ~0 → smooth=100, accel ~0.01+ → smooth=0
        return float(np.clip(100.0 - mean_accel * 10000.0, 0, 100))

    @staticmethod
    def _compute_identity_consistency(series: List[np.ndarray]) -> float:
        """Measure how consistent the face shape is across frames.

        Compares each frame's centred landmark configuration to the
        mean configuration using Procrustes-like distance.
        Score 0-1 (higher=more consistent).
        """
        # Centre all landmark sets
        centred = []
        for lm in series:
            c = lm - lm.mean(axis=0, keepdims=True)
            # Normalise scale
            scale = np.linalg.norm(c)
            if scale > 0:
                c = c / scale
            centred.append(c)

        # Mean shape
        mean_shape = np.mean(centred, axis=0)
        mean_scale = np.linalg.norm(mean_shape)
        if mean_scale > 0:
            mean_shape = mean_shape / mean_scale

        # Deviation from mean
        deviations = []
        for c in centred:
            dist = float(np.linalg.norm(c - mean_shape))
            deviations.append(dist)

        mean_dev = float(np.mean(deviations))
        # Normalise: dev ~0 → consistency=1, dev ~0.5+ → consistency=0
        return float(np.clip(1.0 - mean_dev * 2.0, 0, 1))
