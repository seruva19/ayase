"""VIIDEO — Video Intrinsic Integrity and Distortion Evaluation Oracle.

Mittal et al. 2016 — completely blind NR-VQA using natural scene
statistics (NSS) of frame differences. No training data or human
opinions required; relies on statistical regularity of natural videos.

Backend:
  scikit-video (``skvideo.measure.viideo_score``) — the canonical VIIDEO
  implementation. When scikit-video is not installed the metric is left
  ``None``; an ad-hoc NSS pooling is not substituted for the published
  algorithm's spatio-temporal quality index.

viideo_score — LOWER = better quality (distortion measure)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VIIDEOModule(PipelineModule):
    name = "viideo"
    description = "VIIDEO blind NR-VQA via natural video statistics (Mittal 2016, lower=better)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "viideo_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._skvideo_fn = None
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            from skvideo.measure import viideo_score as skvideo_viideo

            self._skvideo_fn = skvideo_viideo
            self._ml_available = True
            self._backend = "skvideo"
            logger.info("VIIDEO initialised (scikit-video backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "VIIDEO unavailable: install scikit-video for the canonical "
            "implementation (pip install scikit-video)"
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "skvideo":
            return sample
        if not sample.is_video:
            return sample

        try:
            score = self._process_skvideo(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.viideo_score = score
        except Exception as e:
            logger.warning("VIIDEO failed for %s: %s", sample.path, e)

        return sample

    def _process_skvideo(self, sample: Sample) -> Optional[float]:
        """Use scikit-video's viideo_score() (canonical VIIDEO)."""
        import skvideo.io

        video_data = skvideo.io.vread(str(sample.path))
        score = self._skvideo_fn(video_data)
        if isinstance(score, np.ndarray):
            if score.size == 0:
                return None
            score = float(score.flatten()[0])
        return float(score)
