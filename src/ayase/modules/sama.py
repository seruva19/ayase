"""SAMA — Scaling and Masking for Video Quality Assessment.

2024 — patch pyramid with masking strategy for local + global
quality, improving on FAST-VQA.

sama_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SAMAModule(PipelineModule):
    name = "sama"
    description = "SAMA scaling+masking VQA (2024)"
    default_config = {
        "subsample": 8,
        "mask_ratio": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.mask_ratio = self.config.get("mask_ratio", 0.5)
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import sama
            self._model = sama
            self._backend = "native"
            logger.info("SAMA (native) initialised")
            return
        except ImportError:
            pass
        self._backend = "heuristic"
        logger.info("SAMA (heuristic)")

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
                sample.quality_metrics.sama_score = score
        except Exception as e:
            logger.warning(f"SAMA failed for {sample.path}: {e}")
        return sample

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: patch pyramid + masking for multi-scale quality."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        pyramid_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape

            scale_scores = []
            for scale, patch_size in [(1.0, 32), (0.5, 32), (0.25, 32)]:
                sh, sw = int(h * scale), int(w * scale)
                if sh < patch_size or sw < patch_size:
                    continue
                scaled = cv2.resize(gray, (sw, sh))

                # Extract patches with masking
                n_patches_h = sh // patch_size
                n_patches_w = sw // patch_size
                total_patches = n_patches_h * n_patches_w

                if total_patches == 0:
                    continue

                # Random masking (deterministic)
                rng = np.random.RandomState(42)
                n_visible = max(int(total_patches * (1 - self.mask_ratio)), 1)
                patch_indices = rng.choice(total_patches, n_visible, replace=False)

                patch_qualities = []
                for pidx in patch_indices:
                    pi = pidx // n_patches_w
                    pj = pidx % n_patches_w
                    patch = scaled[
                        pi * patch_size : (pi + 1) * patch_size,
                        pj * patch_size : (pj + 1) * patch_size,
                    ]
                    lap = cv2.Laplacian(patch, cv2.CV_64F).var()
                    patch_qualities.append(min(lap / 400.0, 1.0))

                scale_scores.append(np.mean(patch_qualities))

            if scale_scores:
                # Multi-scale fusion (weighted: fine scales more important)
                weights = np.array([0.5, 0.3, 0.2][: len(scale_scores)])
                weights = weights / weights.sum()
                pyramid_scores.append(float(np.dot(weights, scale_scores)))

        if not pyramid_scores:
            return None

        spatial = float(np.mean(pyramid_scores))

        # Temporal consistency
        if len(frames) > 1:
            diffs = []
            for i in range(len(frames) - 1):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float)
                g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float)
                g1 = cv2.resize(g1, (160, 120))
                g2 = cv2.resize(g2, (160, 120))
                diffs.append(np.mean(np.abs(g1 - g2)))
            temporal = 1.0 / (1.0 + np.var(diffs) * 0.005)
        else:
            temporal = 1.0

        return float(np.clip(0.7 * spatial + 0.3 * temporal, 0.0, 1.0))

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
