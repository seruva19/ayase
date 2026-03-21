"""Zoom-VQA — Patches, Frames and Clips Integration for VQA.

CVPRW 2023 — multi-level perception of spatiotemporal quality
at patch, frame, and clip levels.

GitHub: https://github.com/k-zha14/Zoom-VQA

zoomvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ZoomVQAModule(PipelineModule):
    name = "zoomvqa"
    description = "Zoom-VQA multi-level patch/frame/clip VQA (CVPRW 2023)"
    default_config = {
        "subsample": 8,
        "patch_size": 64,
        "n_patches": 16,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.patch_size = self.config.get("patch_size", 64)
        self.n_patches = self.config.get("n_patches", 16)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import zoomvqa
            self._model = zoomvqa
            self._backend = "native"
            logger.info("Zoom-VQA (native) initialised")
            return
        except ImportError:
            pass

        self._backend = "heuristic"
        logger.info("Zoom-VQA (heuristic) — install zoomvqa for full model")

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
                sample.quality_metrics.zoomvqa_score = score

        except Exception as e:
            logger.warning(f"Zoom-VQA failed for {sample.path}: {e}")

        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: multi-level quality at patch, frame, and clip levels."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        patch_scores = []
        frame_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape

            # Patch-level quality
            rng = np.random.RandomState(42)
            p_scores = []
            for _ in range(self.n_patches):
                y = rng.randint(0, max(h - self.patch_size, 1))
                x = rng.randint(0, max(w - self.patch_size, 1))
                patch = gray[y : y + self.patch_size, x : x + self.patch_size]
                lap = cv2.Laplacian(patch, cv2.CV_64F).var()
                p_scores.append(min(lap / 500.0, 1.0))
            patch_scores.append(np.mean(p_scores))

            # Frame-level quality
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            contrast = min(gray.std() / 65.0, 1.0)
            frame_scores.append(0.6 * sharpness + 0.4 * contrast)

        patch_quality = float(np.mean(patch_scores))
        frame_quality = float(np.mean(frame_scores))

        # Clip-level quality (temporal consistency)
        if len(frames) > 1:
            diffs = []
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                diffs.append(np.mean(np.abs(g1 - g2)))
            clip_quality = 1.0 / (1.0 + np.var(diffs) * 0.005)
        else:
            clip_quality = 1.0

        # Multi-level fusion
        score = 0.30 * patch_quality + 0.40 * frame_quality + 0.30 * clip_quality

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
