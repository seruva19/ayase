"""FAVER -- Blind Quality Prediction of Variable Frame Rate Videos.

Zheng et al. Signal Processing 2024 -- first NR-VQA designed for variable and
high frame rate content. Extracts bandpass temporal natural scene statistics
(NSS) alongside deep spatial features, with frame-rate-aware temporal
aggregation, and regresses quality with a *trained* SVR.

The published FAVER pipeline depends on its trained regressor; there is no
maintained Python package or portable weights for it. A ResNet-feature + NSS
approximation without the trained regressor would not reproduce FAVER's scores,
so the metric is left unavailable until the official model is wired in.

faver_score -- higher = better quality (0-1); left None when unavailable.
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FAVERModule(PipelineModule):
    name = "faver"
    description = "FAVER blind VQA for variable frame rate videos (2024)"
    default_config = {
        "subsample": 16,
    }
    metric_groups = {
        "faver_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        # No maintained package ships the trained FAVER regressor/weights.
        # Refuse to fabricate a score from an untrained heuristic.
        logger.warning(
            "FAVER unavailable: no maintained package provides the trained "
            "FAVER model/regressor (Zheng et al. 2024). faver_score left unset."
        )

    def process(self, sample: Sample) -> Sample:
        return sample
