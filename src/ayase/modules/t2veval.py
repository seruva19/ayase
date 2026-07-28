"""T2VEval --- Text-to-Video Generated Video Evaluation (2025).

T2VEval scores text-video consistency and realness with a trained evaluator.
A CLIP prompt-contrast proxy (real-vs-generated and quality prompt contrasts
plus caption cosine) is not T2VEval, so it is not emitted under the T2VEval
name. This module reports itself unavailable until a real T2VEval backend is
wired in.

Output field: ``t2veval_score`` (populated only with a real backend).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class T2VEvalModule(PipelineModule):
    name = "t2veval"
    description = "T2VEval text-video consistency+realness (2025)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "t2veval_score": "alignment",
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
            import t2veval  # type: ignore  # upstream T2VEval backend

            self._model = t2veval
            self._ml_available = True
            self._backend = "t2veval"
            logger.info("T2VEval initialised (t2veval backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "T2VEval unavailable: no trained T2VEval model is installable; "
            "t2veval_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "t2veval":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            caption = sample.caption.text if sample.caption else None
            score = predict(str(sample.path), caption)
            if score is not None:
                sample.quality_metrics.t2veval_score = float(score)
        except Exception as e:
            logger.warning("T2VEval failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None
