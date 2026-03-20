"""CLiF-VQA — Human Feelings VQA.

2024 — extracts human-feelings features from CLIP to simulate
HVS perceptual process for quality assessment.

clifvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CLiFVQAModule(PipelineModule):
    name = "clifvqa"
    description = "CLiF-VQA human feelings VQA via CLIP (2024)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import clifvqa
            self._model = clifvqa
            self._backend = "native"
            logger.info("CLiF-VQA (native) initialised")
            return
        except ImportError:
            pass
        self._backend = "heuristic"
        logger.info("CLiF-VQA (heuristic)")

    def process(self, sample: Sample) -> Sample:
        try:
            score = (
                float(self._model.predict(str(sample.path)))
                if self._backend == "native"
                else self._process_heuristic(sample)
            )
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.clifvqa_score = score
        except Exception as e:
            logger.warning(f"CLiF-VQA failed for {sample.path}: {e}")
        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: human feeling simulation via perceptual features."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        feeling_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Comfort feeling: brightness + contrast balance
            brightness = gray.mean() / 255.0
            brightness_comfort = 1.0 - abs(brightness - 0.5) * 2.0

            # Clarity feeling: sharpness
            clarity = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)

            # Warmth feeling: color temperature proxy
            b, g, r = frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()
            warmth = min(r / (b + 1e-8), 2.0) / 2.0

            # Vibrancy feeling: saturation
            vibrancy = min(hsv[:, :, 1].astype(float).mean() / 128.0, 1.0)

            # Harmony feeling: color consistency (low std in hue)
            hue = hsv[:, :, 0].astype(float)
            harmony = 1.0 / (1.0 + hue.std() / 30.0)

            feeling = (
                0.25 * brightness_comfort
                + 0.30 * clarity
                + 0.15 * warmth
                + 0.15 * vibrancy
                + 0.15 * harmony
            )
            feeling_scores.append(feeling)

        return float(np.clip(np.mean(feeling_scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames
