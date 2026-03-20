"""BVQI — Blind Video Quality Index.

ICME 2023 Oral → TIP — zero-shot VQA that outperforms supervised
methods. Uses CLIP features with quality anchor texts for zero-shot
quality prediction.

GitHub: https://github.com/VQAssessment/BVQI

bvqi_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class BVQIModule(PipelineModule):
    name = "bvqi"
    description = "BVQI zero-shot blind video quality index (ICME 2023)"
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
            import bvqi
            self._model = bvqi
            self._backend = "native"
            logger.info("BVQI (native) initialised")
            return
        except ImportError:
            pass

        try:
            import pyiqa
            self._model = pyiqa.create_metric("bvqi", device="cpu")
            self._backend = "pyiqa"
            logger.info("BVQI (pyiqa) initialised")
            return
        except (ImportError, Exception):
            pass

        self._backend = "heuristic"
        logger.info("BVQI (heuristic) — install bvqi or pyiqa for full model")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            elif self._backend == "pyiqa":
                score = self._process_pyiqa(sample)
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.bvqi_score = score

        except Exception as e:
            logger.warning(f"BVQI failed for {sample.path}: {e}")

        return sample

    def _process_pyiqa(self, sample: Sample) -> Optional[float]:
        import torch
        with torch.no_grad():
            result = self._model(str(sample.path))
        return float(result.item()) if hasattr(result, "item") else float(result)

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: zero-shot quality via spatial+temporal NSS features."""
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return None
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

        if not frames:
            return None

        spatial_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 600.0, 1.0)
            contrast = min(gray.std() / 65.0, 1.0)
            noise_est = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
            cleanliness = 1.0 / (1.0 + noise_est * 0.0001)
            spatial_scores.append(0.4 * sharpness + 0.3 * contrast + 0.3 * cleanliness)

        spatial = float(np.mean(spatial_scores))

        # Temporal consistency
        if len(frames) > 1:
            diffs = []
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                diffs.append(np.mean(np.abs(g1 - g2)))
            temporal = 1.0 / (1.0 + np.var(diffs) * 0.01)
        else:
            temporal = 1.0

        return float(np.clip(0.7 * spatial + 0.3 * temporal, 0.0, 1.0))
