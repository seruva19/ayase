"""VFIPS — Video Frame Interpolation Perceptual Similarity (ECCV 2022).

Full-reference perceptual metric designed for frame interpolation
evaluation, using a *trained* spatiotemporal network (see
github.com/hqqxyy/VFIPS).

No installable VFIPS backend / weights are available here. A hand-rolled
SSIM-like spatial distance is not VFIPS and would misrepresent the metric,
so this module reports itself unavailable rather than emit a proxy value.

vfips_score — lower = better (perceptual distance), populated only with a
real backend.
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class VFIPSModule(ReferenceBasedModule):
    name = "vfips"
    description = "VFIPS frame interpolation perceptual similarity (ECCV 2022, FR)"
    metric_field = "vfips_score"
    default_config = {"subsample": 8}
    metric_groups = {
        "vfips_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.subsample = self.config.get("subsample", 8)

    def setup(self) -> None:
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "VFIPS unavailable: the trained VFIPS spatiotemporal network is not "
            "bundled; vfips_score will not be populated by this module."
        )

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        return None
