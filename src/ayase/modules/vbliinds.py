"""V-BLIINDS — Video BLIINDS blind NR-VQA (Saad & Bovik, 2014).

The V-BLIINDS algorithm extracts DCT-domain natural-scene-statistics and
motion-coherency features (~46-d) and maps them to a quality score with a
*trained* SVR fit on human opinion scores (the DMOS regressor released with
the reference MATLAB implementation).

The feature extractor alone is not a quality score: without the trained SVR,
any features-to-score mapping is a hand-tuned heuristic, not V-BLIINDS.
Neither scikit-video nor any pip package ships the trained SVR weights, so
this module reports itself unavailable rather than emit a heuristic proxy.

vbliinds_score — higher = better quality, populated only with a real backend.
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VBLIINDSModule(PipelineModule):
    name = "vbliinds"
    description = "V-BLIINDS blind NR-VQA via DCT-domain GGD + motion coherency (Saad 2014)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "vbliinds_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "V-BLIINDS unavailable: the trained DMOS SVR regressor is not bundled, "
            "so no faithful V-BLIINDS score can be produced; vbliinds_score will "
            "not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

    def _compute(self, sample: Sample) -> Optional[float]:
        return None
