"""StableVQA — Video Stability Quality Assessment (ACM MM 2023).

The published StableVQA model fuses a learned spatial/temporal backbone with
a motion-stability branch and is trained end-to-end on subjective stability
MOS. The official trained weights are not distributed as an installable
package, and there is no faithful way to reproduce the calibrated
``stablevqa_score`` without them.

A previous revision of this module fabricated the score with a
randomly-initialised quality head on top of ResNet-50 features, which does
not predict anything meaningful. That has been removed: rather than emit a
fabricated number, the module reports the metric as unavailable and leaves
``stablevqa_score`` unset.

GitHub: https://github.com/QMME/StableVQA
"""

import logging
from typing import Optional

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class StableVQAModule(PipelineModule):
    name = "stablevqa"
    description = "StableVQA video stability quality assessment (ACM MM 2023)"
    default_config = {
        "step": 2,
        "max_frames": 120,
        "frame_size": 224,
    }
    metric_groups = {
        "stablevqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        self._ml_available = False
        self._backend = "unavailable"
        logger.info(
            "StableVQA: official trained weights are not publicly available; "
            "the calibrated stability score cannot be reproduced. "
            "Metric reported as unavailable."
        )

    def process(self, sample: Sample) -> Sample:
        # No trained StableVQA backend available -> leave stablevqa_score unset
        # instead of fabricating a value from an untrained head.
        return sample
