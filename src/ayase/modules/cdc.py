"""CDC — Color Distribution Consistency for Video Colorization (2024).

No-reference metric that measures temporal consistency of color
distributions across consecutive frames. Uses Jensen-Shannon divergence
of color histograms in LAB color space.

cdc_score — lower = better (more consistent color distribution).
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CDCModule(PipelineModule):
    name = "cdc"
    description = "CDC color distribution consistency for video colorization (2024)"
    default_config = {
        "subsample": 16,
        "hist_bins": 32,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.subsample = self.config.get("subsample", 16)
        self.hist_bins = self.config.get("hist_bins", 32)

    def setup(self) -> None:
        logger.info("CDC module initialised (heuristic)")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            score = self._compute_cdc(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.cdc_score = score
            logger.debug(f"CDC for {sample.path.name}: {score:.6f}")
        except Exception as e:
            logger.error(f"CDC failed: {e}")
        return sample

    def _compute_cdc(self, path: Path) -> Optional[float]:
        """Compute color distribution consistency via JSD in LAB space."""
        cap = cv2.VideoCapture(str(path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2:
                return None

            n_sample = min(self.subsample, total)
            indices = np.linspace(0, total - 1, n_sample, dtype=int)

            prev_hist = None
            jsd_values = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Convert to LAB
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

                # Compute color histogram (A and B channels in LAB)
                hist = self._compute_lab_histogram(lab)

                if prev_hist is not None:
                    jsd = self._jensen_shannon_divergence(prev_hist, hist)
                    jsd_values.append(jsd)

                prev_hist = hist

            if not jsd_values:
                return None

            return float(np.mean(jsd_values))
        finally:
            cap.release()

    def _compute_lab_histogram(self, lab: np.ndarray) -> np.ndarray:
        """Compute normalised 2D histogram of A and B channels in LAB."""
        a_channel = lab[:, :, 1].flatten()
        b_channel = lab[:, :, 2].flatten()

        hist, _, _ = np.histogram2d(
            a_channel.astype(np.float64),
            b_channel.astype(np.float64),
            bins=self.hist_bins,
            range=[[0, 256], [0, 256]],
        )

        # Normalise to probability distribution
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist.flatten()

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        # Ensure non-negative and normalised
        p = np.maximum(p, 0)
        q = np.maximum(q, 0)

        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum > 0:
            p = p / p_sum
        if q_sum > 0:
            q = q / q_sum

        m = 0.5 * (p + q)

        # KL divergence with epsilon for numerical stability
        eps = 1e-12

        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))

        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        return float(max(jsd, 0.0))
