"""FAVER — Blind Quality Prediction of Variable Frame Rate Videos.

Signal Processing 2024 — first NR-VQA targeting variable/high
frame rate distortions via bandpass temporal statistics + motion-aware
deep features.

faver_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FAVERModule(PipelineModule):
    name = "faver"
    description = "FAVER blind VQA for variable frame rate videos (2024)"
    default_config = {
        "subsample": 16,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import faver
            self._model = faver
            self._backend = "native"
            logger.info("FAVER (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("FAVER (heuristic) — install faver for full model")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            score = (
                float(self._model.predict(str(sample.path)))
                if self._backend == "native"
                else self._process_heuristic(sample)
            )

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.faver_score = score

        except Exception as e:
            logger.warning(f"FAVER failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: bandpass temporal statistics + frame quality."""
        cap = cv2.VideoCapture(str(sample.path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 1:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
        grays = []
        qualities = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gray = cv2.resize(gray, (320, 240))
            grays.append(gray)

            # Per-frame spatial quality
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            qualities.append(min(lap / 500.0, 1.0))

        cap.release()

        if len(grays) < 2:
            return None

        # Temporal bandpass statistics
        frame_diffs = []
        for i in range(len(grays) - 1):
            diff = np.mean(np.abs(grays[i + 1] - grays[i]))
            frame_diffs.append(diff)

        diff_arr = np.array(frame_diffs)

        # Bandpass: variance of differences (irregular = bad for VFR)
        temporal_regularity = 1.0 / (1.0 + np.var(diff_arr) * 0.01)

        # Motion-aware: mean difference level
        motion_level = np.mean(diff_arr)
        motion_quality = 1.0 / (1.0 + abs(motion_level - 5.0) * 0.05)

        # Frame rate factor
        fps_quality = min(fps / 30.0, 1.0) if fps > 0 else 0.5

        spatial_quality = float(np.mean(qualities))

        score = (
            0.35 * spatial_quality
            + 0.25 * temporal_regularity
            + 0.20 * motion_quality
            + 0.20 * fps_quality
        )

        return float(np.clip(score, 0.0, 1.0))
