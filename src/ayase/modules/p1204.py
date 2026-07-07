"""ITU-T P.1204.3 -- Bitstream-based NR Video Quality for UHD.

ITU-T Rec. P.1204.3 (2020) -- no-reference bitstream quality model that
predicts Mean Opinion Score from the coded bitstream and decoded frames.

The reference model (https://github.com/Telecommunication-Telemedia-Assessment/
bitstream_mode3_p1204_3) ships trained regression coefficients. No public
pretrained weights are wired into Ayase, so this module reports no score
rather than emitting a value from an untrained regression head (which would
be fabricated). Install and integrate the official ``p1204_3`` package to
enable it.

p1204_mos -- 1-5, higher = better
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class P1204Module(PipelineModule):
    name = "p1204"
    description = "ITU-T P.1204.3 bitstream NR quality (2020)"
    default_config = {
        "subsample": 4,
    }
    metric_groups = {
        "p1204_mos": "codec",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        # The official P.1204.3 model is trained; without its published
        # coefficients any regression head is untrained and its output would
        # be fabricated. Report unavailable until the real model is wired.
        self._backend = "unavailable"
        logger.info(
            "P.1204 unavailable: official bitstream_mode3_p1204_3 model not installed; "
            "no score emitted."
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if not sample.is_video:
            return sample

        # Kept for field discovery; only reached when a real backend is wired.
        mos = self._compute_mos(sample)
        if mos is not None:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.p1204_mos = float(np.clip(mos, 1.0, 5.0))

        return sample

    def _compute_mos(self, sample: Sample) -> Optional[float]:
        # No real P.1204.3 backend is available; never fabricate a MOS.
        return None
