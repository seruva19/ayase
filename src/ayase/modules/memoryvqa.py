"""Memory-VQA — Video Quality Based on Human Memory System.

Neurocomputing 2025 — mimics human memory formation (5 stages:
sensory input, encoding, storage, retrieval, decision) for
quality perception.

memoryvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MemoryVQAModule(PipelineModule):
    name = "memoryvqa"
    description = "Memory-VQA human memory system VQA (Neurocomputing 2025)"
    default_config = {
        "subsample": 12,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import memoryvqa
            self._model = memoryvqa
            self._backend = "native"
            logger.info("Memory-VQA (native) initialised")
            return
        except ImportError:
            pass
        self._backend = "heuristic"
        logger.info("Memory-VQA (heuristic)")

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
                sample.quality_metrics.memoryvqa_score = score
        except Exception as e:
            logger.warning(f"Memory-VQA failed for {sample.path}: {e}")
        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: 5-stage memory model approximation."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        # Stage 1: Sensory input — raw visual features
        sensory_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            brightness = 1.0 - abs(gray.mean() - 127.5) / 127.5
            contrast = min(gray.std() / 65.0, 1.0)
            sensory_scores.append(0.5 * brightness + 0.5 * contrast)
        sensory = np.mean(sensory_scores)

        # Stage 2: Encoding — detail extraction
        encoding_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            edges = cv2.Canny(frame, 50, 150)
            detail = min(np.mean(edges > 0) / 0.15, 1.0)
            encoding_scores.append(0.6 * sharpness + 0.4 * detail)
        encoding = np.mean(encoding_scores)

        # Stage 3: Storage — temporal consistency (stable = well stored)
        if len(frames) > 1:
            diffs = []
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                g1 = cv2.resize(g1, (160, 120))
                g2 = cv2.resize(g2, (160, 120))
                diffs.append(np.mean(np.abs(g1 - g2)))
            storage = 1.0 / (1.0 + np.var(diffs) * 0.01)
        else:
            storage = 1.0

        # Stage 4: Retrieval — distinctiveness (memorable = high quality)
        if len(frames) > 1:
            frame_features = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64)).flatten().astype(float)
                gray = gray / (np.linalg.norm(gray) + 1e-8)
                frame_features.append(gray)
            feat_arr = np.array(frame_features)
            similarity_matrix = feat_arr @ feat_arr.T
            mean_sim = (np.sum(similarity_matrix) - len(frames)) / max(len(frames) * (len(frames) - 1), 1)
            retrieval = (1.0 - min(mean_sim, 1.0)) * 0.5 + 0.5
        else:
            retrieval = 0.8

        # Stage 5: Decision — weighted combination
        score = 0.15 * sensory + 0.30 * encoding + 0.25 * storage + 0.15 * retrieval + 0.15 * (sensory * encoding)

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
