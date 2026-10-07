"""CDC — Color Distribution Consistency for video colorization.

Published definition (Liu et al.; used as the temporal-consistency metric of
the NTIRE 2023 Video Colorization Challenge):

    CDC = mean over consecutive frame pairs and over the R/G/B channels of the
    Jensen–Shannon divergence between the frames' per-channel colour
    histograms.

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
    provenance = "published"
    sources = {
        "cdc_score": "CDC (Liu et al.; NTIRE 2023 Video Colorization Challenge temporal metric) — https://doi.org/10.1109/CVPRW59228.2023.00159",
    }
    deviations = {
        "cdc_score": "the paper does not fix the histogram bin count — 256 is used (the full 8-bit channel resolution)",
    }
    description = "CDC color distribution consistency for video colorization"
    default_config = {
        "hist_bins": 256,
    }
    metric_groups = {
        "cdc_score": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self.hist_bins = int(self.config.get("hist_bins", 256))
        # CDC is defined as the JS-divergence of consecutive-frame per-channel
        # colour histograms; the numpy implementation below IS that metric.
        self._backend = "algorithmic"

    def setup(self) -> None:
        logger.info("CDC module initialised (algorithmic JS-divergence)")

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
        """Compute CDC: mean JS divergence of per-channel histograms over
        consecutive frames."""
        cap = cv2.VideoCapture(str(path))
        try:
            prev_hist = None
            jsd_values = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                hist = self._compute_rgb_histogram(frame)

                if prev_hist is not None:
                    jsd = self._jensen_shannon_divergence(prev_hist, hist)
                    jsd_values.append(jsd)

                prev_hist = hist

            if not jsd_values:
                return None

            return float(np.mean(jsd_values))
        finally:
            cap.release()

    def _compute_rgb_histogram(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Concatenated normalised 1-D histograms of the R, G, B channels."""
        hists = []
        for ch in (2, 1, 0):  # BGR -> R, G, B
            hist, _ = np.histogram(
                frame_bgr[:, :, ch].ravel().astype(np.float64),
                bins=self.hist_bins,
                range=(0, 256),
            )
            total = hist.sum()
            hists.append(hist / total if total > 0 else hist)
        return np.concatenate(hists)

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
