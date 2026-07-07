"""Video Memorability module.

Predicting media memorability (as in the MediaEval / VideoMem / Memento10k
tasks) requires a *trained* memorability regression head on top of visual
features. CLIP/DINOv2 features are real, but a hand-tuned combination of
feature-norm / diversity / uniqueness statistics is not a trained
memorability predictor and its output does not correspond to a memorability
score.

Rather than emit an uncalibrated proxy under the ``video_memorability``
name, this module reports itself unavailable until a real trained
memorability head is wired in.

video_memorability — populated only with a real backend.
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoMemorabilityModule(PipelineModule):
    name = "video_memorability"
    description = "Content memorability approximation (CLIP/DINOv2 feature statistics)"
    default_config = {
        "subsample": 5,
    }
    metric_groups = {
        "video_memorability": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "VideoMemorability unavailable: no trained memorability regression head "
            "is bundled; video_memorability will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

    def _compute_memorability(self, frames) -> Optional[float]:
        return None
