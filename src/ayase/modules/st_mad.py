"""ST-MAD — Spatiotemporal Most Apparent Distortion (TIP 2012).

MAD (Larson & Chandler, "Most apparent distortion", JEI 2010) and its
spatiotemporal extension ST-MAD are two-stage full-reference metrics: a
detection stage using a contrast-sensitivity-weighted local-contrast model in
the near-threshold regime, and an appearance stage using log-Gabor subband
statistics, combined by a distortion-level-dependent weighting.

A previous revision approximated this with a contrast-masking-weighted mean
absolute error (``|dist - ref|`` weighted by an inverse-Laplacian mask). That
is not the MAD algorithm — it shares neither the CSF detection stage nor the
log-Gabor appearance stage — so it does not measure "most apparent
distortion". There is no faithful ST-MAD reference implementation bundled, so
the metric is reported as unavailable rather than emitting a proxy value.
"""
import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class STMADModule(ReferenceBasedModule):
    name = "st_mad"
    description = "ST-MAD spatiotemporal MAD (TIP 2012)"
    metric_field = "st_mad"
    default_config = {"subsample": 8}
    metric_groups = {
        "st_mad": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        self._backend = "unavailable"
        logger.info(
            "ST-MAD: no faithful implementation of the MAD two-stage "
            "(CSF detection + log-Gabor appearance) algorithm is available; "
            "metric reported as unavailable."
        )

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        # No faithful ST-MAD backend -> leave st_mad unset rather than emit a
        # contrast-masked MAE proxy that misrepresents the published metric.
        return None
