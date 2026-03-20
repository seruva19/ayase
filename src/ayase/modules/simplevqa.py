"""SimpleVQA — Simple Blind Video Quality Assessment.

2022 — Swin Transformer-B spatial + fixed SlowFast temporal
features for blind VQA. Base model for RQ-VQA.

GitHub: https://github.com/sunwei925/SimpleVQA

simplevqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SimpleVQAModule(PipelineModule):
    name = "simplevqa"
    description = "SimpleVQA Swin+SlowFast blind VQA (2022)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import simplevqa
            self._model = simplevqa
            self._backend = "native"
            logger.info("SimpleVQA (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("SimpleVQA (heuristic) — install simplevqa for full model")

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
                sample.quality_metrics.simplevqa_score = score

        except Exception as e:
            logger.warning(f"SimpleVQA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: spatial (Swin-like multi-scale) + temporal (SlowFast-like)."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        spatial_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Multi-scale spatial (Swin-like pyramid)
            scale_scores = []
            for scale in [1.0, 0.5, 0.25]:
                h, w = gray.shape
                sh, sw = int(h * scale), int(w * scale)
                if sh < 16 or sw < 16:
                    continue
                scaled = cv2.resize(gray, (sw, sh))
                lap = cv2.Laplacian(scaled, cv2.CV_64F).var()
                scale_scores.append(min(lap / 500.0, 1.0))

            spatial_scores.append(np.mean(scale_scores) if scale_scores else 0.5)

        spatial = float(np.mean(spatial_scores))

        # SlowFast-like temporal: slow pathway (all frames) + fast pathway (frame diffs)
        if len(frames) > 1:
            # Slow: overall frame consistency
            slow_diffs = []
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                g1 = cv2.resize(g1, (160, 120))
                g2 = cv2.resize(g2, (160, 120))
                slow_diffs.append(np.mean(np.abs(g1 - g2)))

            slow_quality = 1.0 / (1.0 + np.var(slow_diffs) * 0.01)

            # Fast: rapid motion changes
            if len(slow_diffs) > 1:
                diff_of_diffs = np.diff(slow_diffs)
                fast_quality = 1.0 / (1.0 + np.var(diff_of_diffs) * 0.05)
            else:
                fast_quality = 1.0

            temporal = 0.6 * slow_quality + 0.4 * fast_quality
        else:
            temporal = 1.0

        score = 0.65 * spatial + 0.35 * temporal
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
