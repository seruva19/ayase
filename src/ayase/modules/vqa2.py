"""VQA^2 --- Visual Question Answering for Video Quality Assessment (MM 2025).

GitHub: https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment

Uses a released vision-language model to score video quality via prompted VQA.
This module requires that released VQA^2 model; when it is not installed the
metric is left ``None`` — a different quality model (e.g. Q-Align) or a CLIP
zero-shot proxy is not substituted for the published metric.

vqa2_score --- higher = better (0-1 range)
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VQA2Module(PipelineModule):
    name = "vqa2"
    description = "VQA^2 LMM video quality assessment (MM 2025)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "vqa2_score": "nr_quality",
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
            import vqa2  # type: ignore  # official VQA^2 backend

            self._model = vqa2
            self._ml_available = True
            self._backend = "vqa2"
            logger.info("VQA^2 initialised (vqa2 backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "VQA^2 unavailable: the released VQA^2 model is not installed"
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "vqa2":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.vqa2_score = float(score)
        except Exception as e:
            logger.warning("VQA^2 failed for %s: %s", sample.path, e)
        return sample
