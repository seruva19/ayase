"""WorldScore — Unified Evaluation for World Generation (ICCV 2025).

GitHub: https://github.com/haoyi-duan/WorldScore
Distribution-level: worldscore
"""
import logging
import cv2
import numpy as np
from typing import List, Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class WorldScoreModule(BatchMetricModule):
    name = "worldscore"
    description = "WorldScore world generation evaluation (ICCV 2025)"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)

    def extract_features(self, sample: Sample) -> Optional[object]:
        """Extract quality + dynamics + controllability features per sample."""
        try:
            if not sample.is_video:
                return None

            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 1:
                    return None
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, f = cap.read()
                    if ret:
                        frames.append(f)
            finally:
                cap.release()

            if len(frames) < 2:
                return None

            # Quality
            q = float(np.mean([
                min(cv2.Laplacian(
                    cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64), cv2.CV_64F
                ).var() / 500, 1)
                for f in frames
            ]))

            # Dynamics
            diffs = [
                np.mean(np.abs(
                    cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(float), (80, 60))
                    - cv2.resize(cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(float), (80, 60))
                ))
                for i in range(len(frames) - 1)
            ]
            dynamics = min(np.mean(diffs) / 10, 1)

            # Controllability proxy: temporal consistency
            ctrl = 1.0 / (1.0 + np.var(diffs) * 0.01)

            return np.array([q, dynamics, ctrl])
        except Exception as e:
            logger.warning(f"WorldScore feature extraction failed: {e}")
            return None

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> float:
        """Compute WorldScore from accumulated features."""
        if not features:
            return 0.0

        feats = np.array(features)
        # Average across all samples: quality, dynamics, controllability
        q = float(feats[:, 0].mean())
        d = float(feats[:, 1].mean())
        c = float(feats[:, 2].mean())
        score = 0.4 * q + 0.3 * d + 0.3 * c

        if hasattr(self, "pipeline") and self.pipeline and hasattr(self.pipeline, "stats"):
            self.pipeline.stats["worldscore"] = score

        return score
