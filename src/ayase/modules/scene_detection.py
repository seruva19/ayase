"""Scene stability metric via scene change detection.

Measures how temporally stable a video is — single continuous scenes
score high (1.0), rapid-cut montages score low (→0).

Uses the real TransNetV2 shot-boundary model. When TransNetV2 is not
installed the scene fields are left unset (no threshold-guess fallback).
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SceneDetectionModule(PipelineModule):
    name = "scene_detection"
    description = "Scene stability metric — penalises rapid cuts (0-1, higher=more stable)"
    default_config = {"threshold": 0.5}
    metric_groups = {
        "avg_scene_duration": "scene",
        "scene_stability": "temporal",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._backend = None

    def setup(self) -> None:
        try:
            from transnetv2 import TransNetV2 as TransNet

            self._model = TransNet()
            self._ml_available = True
            self._backend = "transnetv2"
            logger.info("TransNetV2 model loaded")
        except ImportError:
            self._ml_available = False
            self._backend = "unavailable"
            logger.warning(
                "scene_detection: TransNetV2 not installed; scene fields left unset."
            )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video or not self._ml_available:
            return sample

        try:
            cuts, duration = self._detect_transnet(sample)

            if duration > 0:
                cuts_per_second = cuts / duration
                # 1.0 for zero cuts, decays toward 0 as cuts increase
                sample.quality_metrics.scene_stability = 1.0 / (1.0 + cuts_per_second * 5.0)
                segments = cuts + 1
                sample.quality_metrics.avg_scene_duration = duration / segments
            else:
                sample.quality_metrics.scene_stability = 1.0
        except Exception as e:
            logger.warning("Scene stability scoring failed: %s", e)
        return sample

    def _detect_transnet(self, sample: Sample) -> tuple:
        """Returns (num_cuts, duration_seconds)."""
        import cv2

        cap = cv2.VideoCapture(str(sample.path))
        frames = []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
        finally:
            cap.release()

        if not frames:
            return 0, 0.0

        video = np.array(frames)
        predictions, _ = self._model.predict_video(video)
        threshold = self.config.get("threshold", 0.5)
        num_cuts = int(np.sum(predictions > threshold))
        duration = len(frames) / fps
        return num_cuts, duration
