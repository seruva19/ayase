"""STREAM — Spatio-Temporal Evaluation and Analysis Metric (ICLR 2024).

GitHub: https://github.com/pro2nit/STREAM
Distribution-level: stream_spatial, stream_temporal
"""
import logging
import cv2
import numpy as np
from typing import List, Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class STREAMModule(BatchMetricModule):
    name = "stream_metric"
    description = "STREAM spatial/temporal generation eval (ICLR 2024)"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._temporal_features: List = []

    def extract_features(self, sample: Sample) -> Optional[object]:
        """Extract spatial + temporal features from a sample."""
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                try:
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total <= 1:
                        return None
                    indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                    grays = []
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, f = cap.read()
                        if ret:
                            grays.append(cv2.resize(
                                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float), (64, 64)
                            ))
                finally:
                    cap.release()

                if len(grays) < 2:
                    return None

                # Spatial feature: flattened mean frame
                spatial = np.mean([g.flatten() for g in grays], axis=0)

                # Temporal feature: FFT of frame diffs
                diffs = [np.mean(np.abs(grays[i] - grays[i + 1])) for i in range(len(grays) - 1)]
                fft_mag = np.abs(np.fft.fft(diffs))
                self._temporal_features.append(fft_mag[: len(fft_mag) // 2 + 1])

                return spatial
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    gray = cv2.resize(
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float), (64, 64)
                    )
                    return gray.flatten()
                return None
        except Exception as e:
            logger.warning(f"STREAM feature extraction failed: {e}")
            return None

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> float:
        """Compute STREAM spatial and temporal scores."""
        spatial_score = 0.0
        temporal_score = 0.0

        if len(features) >= 2:
            feats = np.array(features)
            mean_feat = feats.mean(axis=0)
            diversity = np.mean(np.std(feats, axis=0))
            fidelity = np.mean(np.linalg.norm(feats - mean_feat, axis=1))
            spatial_score = float(diversity / (fidelity + 1e-8))

        if len(self._temporal_features) >= 2:
            max_len = max(len(t) for t in self._temporal_features)
            padded = [np.pad(t, (0, max_len - len(t))) for t in self._temporal_features]
            t_feats = np.array(padded)
            temporal_score = float(1.0 / (1.0 + np.var(t_feats.mean(axis=1))))

        # Store in pipeline stats if available
        if hasattr(self, "pipeline") and self.pipeline and hasattr(self.pipeline, "stats"):
            self.pipeline.stats["stream_spatial"] = spatial_score
            self.pipeline.stats["stream_temporal"] = temporal_score

        return spatial_score  # Return primary metric
