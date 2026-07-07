"""AIGVQA --- Multi-Dimensional AI-Generated VQA (ICCVW 2025).

GitHub: https://github.com/IntMeGroup/AIGVQA
Weights: https://huggingface.co/IntMeGroup/ICCVW_mos0_8B (8B, InternVL2-based)

AIGVQA is the SJTU-IntMeGroup entry to the VQualA 2025 GenAI-Bench AIGC video
quality challenge. It is not a stock InternVL chat model: the released
checkpoint is a custom two-stream regression network whose forward returns a
predicted MOS (``score1``) --- it does NOT generate a rateable text answer at
inference, and it cannot be loaded by ``transformers`` alone (see REVIVAL NOTES).
No installable/self-contained AIGVQA backend exists, and a CLIP multi-prompt
proxy is not AIGVQA, so nothing is emitted under the AIGVQA name until a real
backend is wired in. This module reports itself unavailable and leaves
``aigvqa_score`` unset.

Output field: ``aigvqa_score`` (populated only with a real backend)."""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AIGVQAModule(PipelineModule):
    name = "aigvqa"
    provisional = True  # no turnkey / self-contained real backend
    description = "AIGVQA multi-dimensional AIGC VQA (ICCVW 2025)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "aigvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = "unavailable"
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # The published AIGVQA checkpoint (IntMeGroup/ICCVW_mos0_8B) is a custom
        # two-stream InternVL2 regression model whose modeling code is NOT shipped
        # with the weights; it needs the GitHub repo pipeline + a separate LOVE
        # temporal.pth. There is no importable/turnkey backend,
        # so the metric stays unset rather than falling back to a proxy.
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "AIGVQA unavailable: IntMeGroup/ICCVW_mos0_8B has no self-contained "
            "loader (custom repo architecture + temporal.pth required); "
            "aigvqa_score will not be populated. See module REVIVAL NOTES."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        # Real-or-none: without the real AIGVQA backend, emit nothing.
        if not self._ml_available or self._backend != "real":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.aigvqa_score = float(score)
        except Exception as e:
            logger.warning("AIGVQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None
