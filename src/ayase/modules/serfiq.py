"""SER-FIQ -- Stochastic Embedding Robustness for Face Image Quality (CVPR 2020).

Terhoerst et al. "SER-FIQ: Unsupervised Estimation of Face Image Quality Based
on Stochastic Embedding Robustness" -- quality is the robustness of a face
embedding under stochastic (dropout-enabled) forward passes of the *face
recognition* network.

Only the real SER-FIQ procedure produces ``serfiq_score``. Earlier revisions
used either (a) an ImageNet ResNet-50 with injected random dropout — which is
not a face-recognition model and whose embeddings are not identity embeddings —
or (b) InsightFace embeddings perturbed by input Gaussian noise, which is not
the published dropout-based robustness. Both are stand-ins and have been
removed. A true SER-FIQ backend (a dropout-enabled ArcFace face model) is not
wired up here, so the score is left ``None``.

serfiq_score -- higher = better quality
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SERFIQModule(PipelineModule):
    name = "serfiq"
    description = "SER-FIQ face quality via embedding robustness (real model only)"
    default_config = {
        "subsample": 4,
        "face_model": "buffalo_l",
        "n_forward_passes": 10,
        "det_size": 640,
        "dropout_rate": 0.1,
    }
    metric_groups = {
        "serfiq_score": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "SER-FIQ: real dropout-enabled ArcFace backend unavailable; serfiq_score left unset."
        )

    def process(self, sample: Sample) -> Sample:
        # No dropout-enabled face-recognition backbone available; do not
        # fabricate a proxy score.
        return sample
