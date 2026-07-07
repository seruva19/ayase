"""MC360IQA -- Multi-Channel Blind 360-degree IQA (2019).

Blind IQA for omnidirectional/360-degree content. The published approach
extracts features from multiple viewports sampled across the equirectangular
projection with a trained multi-channel CNN + quality regressor.

Ayase does not ship (and cannot currently locate a loadable checkpoint for) the
real MC360IQA weights. Per the project's no-heuristic policy a named-metric
field must be produced by the real model or left ``None`` -- it must never be
fabricated from an ImageNet ResNet backbone bolted to a randomly-initialised,
untrained quality-regression head (which is all the previous implementation
did). Until a genuine MC360IQA checkpoint is wired in, ``mc360iqa_score`` is
left unset.

mc360iqa_score -- higher = better quality (0-1); real model only
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MC360IQAModule(PipelineModule):
    name = "mc360iqa"
    description = "MC360IQA blind 360 IQA (2019; real model only, disabled if unavailable)"
    default_config = {
        "subsample": 8,
        "n_viewports": 10,
        "viewport_size": 224,
    }
    metric_groups = {
        "mc360iqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.n_viewports = self.config.get("n_viewports", 10)
        self.viewport_size = self.config.get("viewport_size", 224)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        # No public, loadable MC360IQA checkpoint is wired in. The previous
        # implementation ran an ImageNet ResNet through an untrained, random-init
        # regression head, so its scores were meaningless. Disable rather than
        # fabricate.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "MC360IQA: no real MC360IQA model/weights available; module disabled "
            "(mc360iqa_score left unset). Wire a genuine MC360IQA checkpoint to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave mc360iqa_score None (graceful).
        return sample
