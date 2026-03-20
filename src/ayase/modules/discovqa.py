"""DisCoVQA — Temporal Distortion-Content Transformers for VQA.

IEEE 2023 — separates temporal distortion extraction from
content-aware temporal attention using transformers.

GitHub: https://github.com/VQAssessment/DisCoVQA

discovqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DisCoVQAModule(PipelineModule):
    name = "discovqa"
    description = "DisCoVQA temporal distortion-content VQA (2023)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import discovqa
            self._model = discovqa
            self._backend = "native"
            logger.info("DisCoVQA (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("DisCoVQA (heuristic) — install discovqa for full model")

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
                sample.quality_metrics.discovqa_score = score

        except Exception as e:
            logger.warning(f"DisCoVQA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: distortion extraction + content-aware temporal attention."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        # Distortion features per frame
        distortion_scores = []
        content_features = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Distortion: blur + noise + compression artifacts
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_dist = 1.0 - min(lap_var / 500.0, 1.0)  # Higher = more distorted

            noise_var = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
            noise_dist = min(noise_var * 0.0001, 1.0)

            distortion = 0.6 * blur_dist + 0.4 * noise_dist
            distortion_scores.append(distortion)

            # Content: edge density as complexity proxy
            edges = cv2.Canny(frame, 50, 150)
            complexity = np.mean(edges > 0)
            content_features.append(complexity)

        # Content-aware temporal attention
        content_arr = np.array(content_features)
        weights = content_arr / (content_arr.sum() + 1e-8)  # Attend more to complex frames

        # Temporal distortion variation
        dist_arr = np.array(distortion_scores)
        weighted_distortion = np.sum(weights * dist_arr)
        temporal_var = np.var(dist_arr)

        # Quality = 1 - distortion, penalise temporal inconsistency
        quality = 1.0 - weighted_distortion
        temporal_penalty = min(temporal_var * 2.0, 0.3)
        score = quality - temporal_penalty

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
