"""ModularBVQA — Modular Blind Video Quality Assessment.

CVPR 2024 — decomposes VQA into base quality predictor + spatial
rectifier + temporal rectifier for resolution/framerate-aware quality.

GitHub: https://github.com/winwinwenwen77/ModularBVQA

modularbvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ModularBVQAModule(PipelineModule):
    name = "modularbvqa"
    description = "ModularBVQA resolution/framerate-aware blind VQA (CVPR 2024)"
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
            import modularbvqa
            self._model = modularbvqa
            self._backend = "native"
            logger.info("ModularBVQA (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("ModularBVQA (heuristic) — install modularbvqa for full model")

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
                sample.quality_metrics.modularbvqa_score = score

        except Exception as e:
            logger.warning(f"ModularBVQA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: base quality + resolution rectifier + temporal rectifier."""
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total <= 0:
                cap.release()
                return None

            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            base_scores = []
            prev_gray = None
            temporal_diffs = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 600.0, 1.0)
                contrast = min(gray.std() / 65.0, 1.0)
                base_scores.append(0.5 * sharpness + 0.5 * contrast)

                if prev_gray is not None:
                    diff = np.mean(np.abs(gray - prev_gray))
                    temporal_diffs.append(diff)
                prev_gray = gray

            cap.release()

            if not base_scores:
                return None

            base_quality = float(np.mean(base_scores))

            # Spatial rectifier: reward higher resolution
            max_dim = max(w, h)
            if max_dim >= 2160:
                res_factor = 1.0
            elif max_dim >= 1080:
                res_factor = 0.95
            elif max_dim >= 720:
                res_factor = 0.85
            elif max_dim >= 480:
                res_factor = 0.70
            else:
                res_factor = 0.55

            # Temporal rectifier: reward higher framerate + temporal consistency
            fps_factor = min(fps / 30.0, 1.0) if fps > 0 else 0.5
            temporal_consistency = (
                1.0 / (1.0 + np.var(temporal_diffs) * 0.01) if temporal_diffs else 0.5
            )
            temporal_factor = 0.5 * fps_factor + 0.5 * temporal_consistency

            score = base_quality * (0.5 + 0.25 * res_factor + 0.25 * temporal_factor)
        else:
            img = cv2.imread(str(sample.path))
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 600.0, 1.0)
            contrast = min(gray.std() / 65.0, 1.0)
            h, w = img.shape[:2]
            max_dim = max(w, h)
            res_factor = min(max_dim / 1080.0, 1.0)
            score = (0.5 * sharpness + 0.5 * contrast) * (0.6 + 0.4 * res_factor)

        return float(np.clip(score, 0.0, 1.0))
