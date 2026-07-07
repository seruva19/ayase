"""MaxVQA — Explainable VQA via Language-Prompted CLIP.

ACM MM 2023 Oral — language-prompted VQA using a modified CLIP head trained on
the MaxWell dataset for explainable quality scoring.

GitHub: https://github.com/VQAssessment/ExplainableVQA

Only the real MaxVQA model produces ``maxvqa_score``. A generic CLIP
quality-prompt cosine similarity is NOT MaxVQA, so it is not used as a
stand-in; when the real backend is unavailable the score is left ``None``.

maxvqa_score — higher = better quality; real model only.
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MaxVQAModule(PipelineModule):
    name = "maxvqa"
    description = "MaxVQA explainable language-prompted VQA (ACM MM 2023; real model only)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "maxvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        # Real MaxVQA package (ExplainableVQA). No proxy fallback.
        try:
            import maxvqa

            self._model = maxvqa
            self._backend = "native"
            self._ml_available = True
            logger.info("MaxVQA (native) initialised")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug("MaxVQA native init failed: %s", e)

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "MaxVQA: real model unavailable (install the ExplainableVQA/maxvqa "
            "package); maxvqa_score left unset."
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "native":
            return sample

        try:
            score = self._process_native(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.maxvqa_score = score
        except Exception as e:
            logger.warning(f"MaxVQA failed for {sample.path}: {e}")

        return sample

    def _process_native(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))
