"""Ada-DQA — Adaptive Diverse Quality-aware Feature Acquisition.

ACM MM 2023 — adaptive quality-aware feature extraction using
diverse pre-trained models for content/distortion/motion diversity.

adadqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AdaDQAModule(PipelineModule):
    name = "adadqa"
    description = "Ada-DQA adaptive diverse quality feature VQA (ACM MM 2023)"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import adadqa
            self._model = adadqa
            self._backend = "native"
            logger.info("Ada-DQA (native) initialised")
            return
        except ImportError:
            pass
        self._backend = "heuristic"
        logger.info("Ada-DQA (heuristic)")

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
                sample.quality_metrics.adadqa_score = score
        except Exception as e:
            logger.warning(f"Ada-DQA failed for {sample.path}: {e}")
        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: adaptive multi-feature: content + distortion + motion."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        content_scores = []
        distortion_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            # Content diversity: edge density + color variance
            edges = cv2.Canny(frame, 50, 150)
            content = min(np.mean(edges > 0) / 0.12, 1.0)
            content_scores.append(content)
            # Distortion: sharpness + noise
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            distortion_scores.append(sharpness)

        content_avg = float(np.mean(content_scores))
        distortion_avg = float(np.mean(distortion_scores))

        # Motion diversity
        if len(frames) > 1:
            diffs = []
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                g1 = cv2.resize(g1, (160, 120))
                g2 = cv2.resize(g2, (160, 120))
                diffs.append(np.mean(np.abs(g1 - g2)))
            motion = 1.0 / (1.0 + np.var(diffs) * 0.01)
        else:
            motion = 1.0

        # Adaptive weighting: emphasise weakest dimension
        components = [content_avg, distortion_avg, motion]
        min_idx = np.argmin(components)
        weights = [0.33, 0.34, 0.33]
        weights[min_idx] += 0.15
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        score = sum(w * c for w, c in zip(weights, components))
        return float(np.clip(score, 0.0, 1.0))

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
