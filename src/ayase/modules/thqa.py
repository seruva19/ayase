"""THQA — Talking Head Quality Assessment (ICIP 2024).

No-reference metric for evaluating talking head video quality.
Considers face region quality, lip movement smoothness, and overall
temporal coherence.

pip install thqa

thqa_score — higher = better quality.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class THQAModule(PipelineModule):
    name = "thqa"
    description = "THQA talking head quality assessment (ICIP 2024)"
    default_config = {
        "subsample": 16,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.subsample = self.config.get("subsample", 16)

    def setup(self) -> None:
        # Tier 1: thqa package
        try:
            import thqa as thqa_lib
            self._model = thqa_lib
            self._ml_available = True
            self._backend = "thqa"
            logger.info("THQA module initialised (thqa package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"THQA package init failed: {e}")

        # Tier 2: heuristic (face region quality + lip smoothness)
        logger.info("THQA module initialised (heuristic fallback)")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            if self._ml_available and self._backend == "thqa":
                score = self._score_thqa_package(sample.path)
            else:
                score = self._score_heuristic(sample.path)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.thqa_score = score
            logger.debug(f"THQA for {sample.path.name}: {score:.4f}")
        except Exception as e:
            logger.error(f"THQA failed: {e}")
        return sample

    def _score_thqa_package(self, path: Path) -> Optional[float]:
        try:
            score = self._model.evaluate(str(path))
            return float(score)
        except Exception as e:
            logger.debug(f"THQA package scoring failed: {e}")
            return self._score_heuristic(path)

    def _score_heuristic(self, path: Path) -> Optional[float]:
        """Heuristic: face region quality + lip movement smoothness.

        1. Detect face region via Haar cascade.
        2. Compute sharpness (Laplacian variance) of face region.
        3. Compute temporal smoothness of face/lip region motion.
        """
        cap = cv2.VideoCapture(str(path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2:
                return None

            n_sample = min(self.subsample, total)
            indices = np.linspace(0, total - 1, n_sample, dtype=int)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            sharpness_scores = []
            prev_face_region = None
            motion_smoothness = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect face
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    # Use largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face_roi = gray[y:y + h, x:x + w]

                    # Sharpness: Laplacian variance
                    lap = cv2.Laplacian(face_roi, cv2.CV_64F)
                    sharpness = float(lap.var())
                    sharpness_scores.append(sharpness)

                    # Motion smoothness of face region
                    face_resized = cv2.resize(face_roi, (64, 64)).astype(np.float64)
                    if prev_face_region is not None:
                        diff = np.mean(np.abs(face_resized - prev_face_region))
                        motion_smoothness.append(diff)
                    prev_face_region = face_resized
                else:
                    # No face detected: use full frame sharpness
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    sharpness_scores.append(float(lap.var()) * 0.5)

            if not sharpness_scores:
                return None

            # Normalise sharpness (typical range 0-2000 for face regions)
            mean_sharpness = np.mean(sharpness_scores)
            sharpness_norm = min(mean_sharpness / 1000.0, 1.0)

            # Motion smoothness score (lower diff variance = smoother = better)
            if motion_smoothness:
                motion_arr = np.array(motion_smoothness)
                # Smoothness: inverse of motion jerkiness
                jerk = float(np.std(motion_arr))
                smoothness_norm = max(1.0 - jerk / 20.0, 0.0)
            else:
                smoothness_norm = 0.5

            # Combined score
            score = 0.6 * sharpness_norm + 0.4 * smoothness_norm
            return float(np.clip(score, 0.0, 1.0))
        finally:
            cap.release()
