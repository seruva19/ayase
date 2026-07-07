"""pVMAF — Predictive VMAF (35x faster).

2024 research method that predicts VMAF from bitstream and pixel-level
features via a *trained* lightweight regression head, avoiding full
reference decoding.

No public pretrained pVMAF weights are available, so this module does not
emit a score: predicting VMAF through an untrained (randomly initialised)
head would be fabrication. Wire trained pVMAF weights to enable it.

pvmaf_score — 0-100 scale (higher = better)
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class PVMAFModule(ReferenceBasedModule):
    name = "pvmaf"
    description = "Predictive VMAF ~35x faster via bitstream+pixel features (2024, 0-100)"
    metric_field = "pvmaf_score"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "pvmaf_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "unavailable"
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        logger.info(
            "pVMAF unavailable: no pretrained pVMAF weights wired; no score emitted."
        )

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        # No trained pVMAF backend is available; never fabricate a VMAF value.
        return None
