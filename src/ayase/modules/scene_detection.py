"""Scene stability metric via scene change detection.

Measures how temporally stable a video is — single continuous scenes
score high (1.0), rapid-cut montages score low (→0).

Uses TransNetV2 when available, falls back to frame-difference heuristic.
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

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None

    def setup(self) -> None:
        try:
            from transnetv2 import TransNetV2 as TransNet

            self._model = TransNet()
            self._ml_available = True
            logger.info("TransNetV2 model loaded")
        except ImportError:
            self._ml_available = False
            logger.info("TransNetV2 not installed, using frame-difference heuristic")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample

        try:
            if self._ml_available:
                cuts, duration = self._detect_transnet(sample)
            else:
                cuts, duration = self._detect_heuristic(sample)

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
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        cap.release()

        if not frames:
            return 0, 0.0

        video = np.array(frames)
        predictions, _ = self._model.predict_video(video)
        threshold = self.config.get("threshold", 0.5)
        num_cuts = int(np.sum(predictions > threshold))
        duration = len(frames) / fps
        return num_cuts, duration

    def _detect_heuristic(self, sample: Sample) -> tuple:
        """Fallback: detect scene changes via frame difference.
        Returns (num_cuts, duration_seconds).
        """
        import cv2

        cap = cv2.VideoCapture(str(sample.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        diff_threshold = 30.0

        prev_frame = None
        num_cuts = 0

        step = max(1, total_frames // 500)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
                if diff > diff_threshold:
                    num_cuts += 1
            prev_frame = gray
        cap.release()

        duration = total_frames / fps
        return num_cuts, duration
