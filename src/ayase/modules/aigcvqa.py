"""AIGC-VQA --- Holistic Perception for AIGC Video Quality (CVPRW 2024).

The published AIGC-VQA model is a trained multi-branch network (technical /
aesthetic / text-video alignment) built on the holistic AIGC perception
dataset. There is no installable AIGC-VQA backend, and a CLIP zero-shot
prompt-contrast comparison is a proxy — not AIGC-VQA — so it is not emitted
under the AIGC-VQA name. This module reports itself unavailable until a real
AIGC-VQA backend is wired in.

Output fields (populated only with a real backend):
    aigcvqa_technical, aigcvqa_aesthetic, aigcvqa_alignment --- 0-1, higher = better

REVIVAL NOTES (requires_external_backend --- no turnkey backend)
Metric: AIGC-VQA (CVPRW/NTIRE 2024).
Category: TRAINING-ONLY.
Why requires_external_backend: No released weights; the 3-branch model must be trained.
To revive: Reimplement the 3-branch arch (ResNet50 + ConvNeXt-3D + BLIP); train on T2VQA-DB
  (public T2V MOS); validate you reproduce the paper's SRCC/PLCC before flipping requires_external_backend=False.
  Effort M; differentiated value (AIGC video).
Source: AIGC-VQA, CVPRW/NTIRE 2024 (no released weights).
"""

import logging
from typing import List, Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AIGCVQAModule(PipelineModule):
    name = "aigcvqa"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "aigcvqa_aesthetic": "nr_quality",
        "aigcvqa_alignment": "alignment",
        "aigcvqa_technical": "nr_quality",
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
            import aigcvqa  # type: ignore  # upstream AIGC-VQA backend

            self._model = aigcvqa
            self._ml_available = True
            self._backend = "aigcvqa"
            logger.info("AIGC-VQA initialised (aigcvqa backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "AIGC-VQA unavailable: no trained AIGC-VQA model is installable; "
            "aigcvqa_technical/aigcvqa_aesthetic/aigcvqa_alignment will not be "
            "populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "aigcvqa":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            scores = predict(str(sample.path))
            if scores is None:
                return sample
            sample.quality_metrics.aigcvqa_technical = float(scores["technical"])
            sample.quality_metrics.aigcvqa_aesthetic = float(scores["aesthetic"])
            if scores.get("alignment") is not None:
                sample.quality_metrics.aigcvqa_alignment = float(scores["alignment"])
        except Exception as e:
            logger.warning("AIGC-VQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_scores(self, sample: Sample) -> Optional[List[float]]:
        return None
