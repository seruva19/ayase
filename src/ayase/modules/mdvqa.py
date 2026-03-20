"""MD-VQA — Multi-Dimensional Quality Assessment for UGC Live Videos.

CVPR 2023 — evaluates semantic, distortion, and motion aspects
separately for UGC live streaming videos.

GitHub: https://github.com/zzc-1998/MD-VQA

mdvqa_semantic, mdvqa_distortion, mdvqa_motion — all higher = better
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MDVQAModule(PipelineModule):
    name = "mdvqa"
    description = "MD-VQA multi-dimensional UGC live VQA (CVPR 2023)"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import mdvqa
            self._model = mdvqa
            self._backend = "native"
            logger.info("MD-VQA (native) initialised")
            return
        except ImportError:
            pass
        self._backend = "heuristic"
        logger.info("MD-VQA (heuristic)")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                result = self._model.predict(str(sample.path))
                semantic = float(result.get("semantic", 0))
                distortion = float(result.get("distortion", 0))
                motion = float(result.get("motion", 0))
            else:
                semantic, distortion, motion = self._process_heuristic(sample)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.mdvqa_semantic = semantic
            sample.quality_metrics.mdvqa_distortion = distortion
            sample.quality_metrics.mdvqa_motion = motion

        except Exception as e:
            logger.warning(f"MD-VQA failed for {sample.path}: {e}")
        return sample

    def _process_heuristic(self, sample: Sample):
        """Heuristic: 3 quality dimensions."""
        frames = self._extract_frames(sample)
        if not frames:
            return 0.5, 0.5, 0.5

        # Semantic: content richness (edge density + color diversity)
        semantic_scores = []
        for frame in frames:
            edges = cv2.Canny(frame, 50, 150)
            edge_density = min(np.mean(edges > 0) / 0.12, 1.0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue_std = hsv[:, :, 0].astype(float).std()
            color_div = min(hue_std / 50.0, 1.0)
            semantic_scores.append(0.5 * edge_density + 0.5 * color_div)
        semantic = float(np.mean(semantic_scores))

        # Distortion: sharpness + noise + compression
        distortion_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            contrast = min(gray.std() / 65.0, 1.0)
            distortion_scores.append(0.6 * sharpness + 0.4 * contrast)
        distortion = float(np.mean(distortion_scores))

        # Motion: temporal smoothness
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
            motion = 0.8

        return (
            float(np.clip(semantic, 0.0, 1.0)),
            float(np.clip(distortion, 0.0, 1.0)),
            float(np.clip(motion, 0.0, 1.0)),
        )

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
