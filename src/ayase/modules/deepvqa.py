"""DeepVQA -- Deep Video Quality Assessor with Spatiotemporal Masking.

Kim et al. ECCV 2018 -- full-reference VQA using deep features with
spatiotemporal visual sensitivity masking.

DeepVQA is a *trained* model: the spatiotemporal sensitivity maps and the
quality regressor are learned end-to-end on subjective VQA data. There is no
maintained Python package or published set of portable weights for it, so this
module has no real backend to run. Rather than approximate DeepVQA with generic
VGG features and hand-tuned masking (which would not reproduce the paper's
scores), the metric is left unavailable until the official model is wired in.

deepvqa_score -- higher = better quality (0-1); left None when unavailable.
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class DeepVQAModule(ReferenceBasedModule):
    name = "deepvqa"
    description = "DeepVQA spatiotemporal masking FR-VQA (ECCV 2018)"
    metric_field = "deepvqa_score"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "deepvqa_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return
        # No maintained package ships the trained DeepVQA model/weights. Refuse
        # to fabricate a score from an untrained proxy; leave the metric None.
        logger.warning(
            "DeepVQA unavailable: no maintained package provides the trained "
            "DeepVQA model (Kim et al. ECCV 2018). deepvqa_score left unset."
        )

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        return None
