"""C3DVQA (3D CNN Video Quality Assessment) module.

C3DVQA (Xu et al., 2020) is a full-reference video quality metric that uses
3D convolutions over spatiotemporal video volumes to capture spatial and
temporal distortions in a single forward pass.

This module requires the trained C3DVQA backend. When that backend is not
installed the metric is left ``None`` — no proxy network or handcrafted
approximation is substituted for the published metric.
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class C3DVQAModule(PipelineModule):
    name = "c3dvqa"
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
            import c3dvqa  # type: ignore  # official C3DVQA backend

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
