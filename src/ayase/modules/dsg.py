"""DSG — Davidsonian Scene Graph faithfulness (ICLR 2024, Google).

Decomposes a text caption into atomic propositions (questions) and
verifies each one against the generated image/video via visual question
answering.  The final score is the fraction of questions answered
affirmatively.

Backend: official ``dsg-t2i`` package (LLM decomposition + VQA).

dsg_score — higher = better faithfulness (0-1)
Requires a caption (``sample.caption.text`` or ``sample.auto_caption``).
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _get_caption_text(sample: Sample) -> Optional[str]:
    """Extract caption text from sample metadata."""
    if sample.caption is not None and sample.caption.text:
        return sample.caption.text
    qm = sample.quality_metrics
    if qm is not None and qm.auto_caption:
        return qm.auto_caption
    return None


class DSGModule(PipelineModule):
    name = "dsg"
    description = "DSG Davidsonian Scene Graph faithfulness (ICLR 2024, Google)"
    default_config = {
        "subsample": 4,     # frames to sample for video
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import dsg  # noqa: F401
            self._model = dsg
            self._ml_available = True
            logger.info("DSG initialised (native dsg-t2i package)")
        except ImportError:
            logger.warning("DSG: dsg-t2i not installed, module disabled")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption = _get_caption_text(sample)
        if caption is None:
            logger.debug("DSG: no caption for %s, skipping", sample.path.name)
            return sample

        try:
            score = self._process_native(sample, caption)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.dsg_score = score
        except Exception as e:
            logger.warning("DSG failed for %s: %s", sample.path, e)

        return sample

    def _process_native(self, sample: Sample, caption: str) -> Optional[float]:
        """Use the official dsg-t2i package."""
        result = self._model.evaluate(str(sample.path), caption)
        if isinstance(result, dict):
            return float(result.get("score", result.get("dsg_score", 0.0)))
        return float(result)
