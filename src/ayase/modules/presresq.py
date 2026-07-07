"""PreResQ-R1 -- Fine-Grained Rank-and-Score VQA (2025).

PreResQ-R1 is a trained rank-and-score reasoning model. A CLIP-IQA-style
zero-shot ranking against ordered quality-level text prompts is a proxy in the
CLIP-IQA family — not the trained PreResQ-R1 weights — so it is not emitted
under the PreResQ-R1 name. This module reports itself unavailable until a real
PreResQ-R1 backend is wired in.

Output field: ``presresq_score`` (populated only with a real backend).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PreResQModule(PipelineModule):
    name = "presresq"
    description = "PreResQ-R1 rank+score VQA (2025)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "presresq_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = None
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import presresq  # type: ignore  # official PreResQ-R1 backend

            self._model = presresq
            self._ml_available = True
            self._backend = "presresq"
            logger.info("PreResQ-R1 initialised (presresq backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "PreResQ-R1 unavailable: no trained PreResQ-R1 model is installable; "
            "presresq_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "presresq":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.presresq_score = float(score)
        except Exception as e:
            logger.warning("PreResQ-R1 failed for %s: %s", sample.path, e)
        return sample

    def _compute_clip_rank_score(self, sample: Sample) -> Optional[float]:
        return None
