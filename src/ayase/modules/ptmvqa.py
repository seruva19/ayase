"""PTM-VQA — Pre-Trained Model fusion VQA.

CVPR 2024 — integrates features from multiple frozen pre-trained
models with ICID loss for quality representation. Processes 1080p
in ~1s via model selection scheme.

Paper: https://arxiv.org/abs/2405.17765

ptmvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PTMVQAModule(PipelineModule):
    name = "ptmvqa"
    description = "PTM-VQA multi-PTM fusion VQA (CVPR 2024)"
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
            import ptmvqa
            self._model = ptmvqa
            self._backend = "native"
            logger.info("PTM-VQA (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("PTM-VQA (heuristic) — install ptmvqa for full model")

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
                sample.quality_metrics.ptmvqa_score = score

        except Exception as e:
            logger.warning(f"PTM-VQA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: multi-feature fusion mimicking diverse PTM features."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        spatial_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Feature 1: Edge/texture richness (ResNet-like)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = min(np.mean(np.sqrt(gx ** 2 + gy ** 2)) / 35.0, 1.0)

            # Feature 2: Sharpness (CLIP-like visual clarity)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)

            # Feature 3: Color distribution (DINOv2-like)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1].astype(float)
            color_richness = min(sat.mean() / 128.0, 1.0)

            # Feature 4: Naturalness (NSS regularity)
            mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
            sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray ** 2, (7, 7), 7 / 6) - mu ** 2) + 1e-7)
            mscn = (gray - mu) / sigma
            naturalness = 1.0 / (1.0 + abs(np.mean(mscn)) + abs(np.var(mscn) - 1.0))

            spatial_scores.append(
                0.30 * sharpness + 0.25 * edge_density + 0.20 * color_richness + 0.25 * naturalness
            )

        spatial = float(np.mean(spatial_scores))

        # Temporal smoothness
        if len(frames) > 1:
            diffs = []
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                diffs.append(np.mean(np.abs(g1 - g2)))
            temporal = 1.0 / (1.0 + np.var(diffs) * 0.005)
        else:
            temporal = 1.0

        return float(np.clip(0.75 * spatial + 0.25 * temporal, 0.0, 1.0))

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
