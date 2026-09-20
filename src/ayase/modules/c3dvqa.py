"""Full-reference video quality prediction through an optional C3DVQA backend.

The published model combines distorted and residual-frame spatial features
with 3D convolutions to learn spatiotemporal masking. Ayase ships no trained
implementation or proxy and delegates video paths to external c3dvqa.predict.
Faithful use requires reference_path; without it the adapter makes a
one-argument backend call whose meaning is backend-specific. Returned scores
are not transformed, so no universal direction or range is asserted, and an
unavailable backend leaves the score unset.

Basis: https://arxiv.org/abs/1910.13646
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class C3DVQAModule(PipelineModule):
    name = "c3dvqa"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "C3DVQA 3D-CNN full-reference video quality (Xu et al. 2020)"
    default_config = {
        "clip_length": 16,
        "subsample": 4,
    }
    metric_groups = {
        "c3dvqa_score": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._backend = None

    def setup(self) -> None:
        try:
            import c3dvqa  # type: ignore  # upstream C3DVQA backend

            self._model = c3dvqa
            self._ml_available = True
            self._backend = "c3dvqa"
            logger.info("C3DVQA initialised (c3dvqa backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("C3DVQA unavailable: the trained C3DVQA backend is not installed")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "c3dvqa":
            return sample
        if not sample.is_video:
            return sample

        reference = getattr(sample, "reference_path", None)
        try:
            score = self._score(sample.path, reference)
            if score is not None:
                sample.quality_metrics.c3dvqa_score = float(score)
        except Exception as e:
            logger.warning("C3DVQA failed: %s", e)
        return sample

    def _score(self, sample_path, reference_path) -> Optional[float]:
        """Delegate scoring to the real C3DVQA backend when available."""
        predict = getattr(self._model, "predict", None)
        if predict is None:
            return None
        if reference_path is not None:
            return float(predict(str(sample_path), str(reference_path)))
        return float(predict(str(sample_path)))
