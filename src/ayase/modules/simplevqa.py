"""SimpleVQA — Simple Blind Video Quality Assessment (Sun et al., 2022).

Swin Transformer-B spatial features + temporal difference features for blind
video quality assessment. Base model for RQ-VQA.

Only the real trained SimpleVQA model produces ``simplevqa_score``. Earlier
revisions ran a Swin-B backbone through randomly-initialised spatial and
temporal quality heads — those heads are untrained, so their output is
meaningless and has been removed. No trained SimpleVQA weights are wired up
here, so the score is left ``None``.

GitHub: https://github.com/sunwei925/SimpleVQA

simplevqa_score — higher = better quality
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SimpleVQAModule(PipelineModule):
    name = "simplevqa"
    description = "SimpleVQA Swin+SlowFast blind VQA (real model only)"
    default_config = {
        "slow_frames": 8,
        "fast_frames": 32,
        "frame_size": 224,
        "fast_frame_size": 112,
    }
    metric_groups = {
        "simplevqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.slow_frames = self.config.get("slow_frames", 8)
        self.fast_frames = self.config.get("fast_frames", 32)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("SimpleVQA: real trained model unavailable; simplevqa_score left unset.")

    def process(self, sample: Sample) -> Sample:
        # No trained SimpleVQA weights available; do not fabricate a score.
        return sample
