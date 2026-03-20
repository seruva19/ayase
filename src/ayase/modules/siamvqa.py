"""SiamVQA — Siamese Network for High-Resolution VQA.

arXiv 2025 — Siamese network sharing weights between aesthetic
and technical branches. Outperforms DOVER on all datasets.

Paper: https://arxiv.org/html/2503.02330

siamvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SiamVQAModule(PipelineModule):
    name = "siamvqa"
    description = "SiamVQA Siamese high-resolution VQA (2025)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import siamvqa
            self._model = siamvqa
            self._backend = "native"
            logger.info("SiamVQA (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("SiamVQA (heuristic) — install siamvqa for full model")

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
                sample.quality_metrics.siamvqa_score = score

        except Exception as e:
            logger.warning(f"SiamVQA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: dual-branch aesthetic + technical via shared features."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        tech_scores = []
        aes_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Technical branch: sharpness + noise + contrast
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            contrast = min(gray.std() / 65.0, 1.0)
            noise = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
            cleanliness = 1.0 / (1.0 + noise * 0.0001)
            tech_scores.append(0.4 * sharpness + 0.3 * contrast + 0.3 * cleanliness)

            # Aesthetic branch: color harmony + composition + brightness
            sat = hsv[:, :, 1].astype(float)
            color_harmony = min(sat.mean() / 128.0, 1.0)
            brightness = 1.0 - abs(gray.mean() - 127.5) / 127.5

            # Rule of thirds proxy
            h, w = gray.shape
            thirds_h = [h // 3, 2 * h // 3]
            thirds_w = [w // 3, 2 * w // 3]
            center = gray[thirds_h[0]:thirds_h[1], thirds_w[0]:thirds_w[1]]
            composition = min(center.var() / gray.var() if gray.var() > 0 else 1.0, 1.5) / 1.5

            aes_scores.append(0.4 * color_harmony + 0.3 * brightness + 0.3 * composition)

        technical = float(np.mean(tech_scores))
        aesthetic = float(np.mean(aes_scores))

        # Siamese fusion (shared features = average)
        score = 0.55 * technical + 0.45 * aesthetic

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
