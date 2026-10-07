"""Behavioral embedding — static+temporal biometric signature (external backend).

Agarwal et al. (arXiv:2004.14491) verify identity with a learned embedding
combining static appearance and temporal behavioural features of a person of
interest, compared against reference recordings of that person.

The upstream model/weights were not released, so the module stays registered
but marked ``requires_external_backend`` until a compatible embedding is
wired in. The video-only sibling metric is ``id_reveal``.

behavioral_embedding_distance -- embedding distance to reference recordings (lower=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class BehavioralEmbeddingModule(PipelineModule):
    name = "behavioral_embedding"
    description = "Static+temporal behavioural embedding distance (Agarwal 2020, external backend)"
    provenance = "published"
    sources = {
        "behavioral_embedding_distance": "behavioral biometric embedding, Agarwal et al. (https://arxiv.org/abs/2004.14491)",
    }
    requires_external_backend = True  # no released model/weights
    requires_reference = True
    default_config = {"fps": 25}
    metric_info = {
        "behavioral_embedding_distance": "Behavioural-embedding distance vs reference recordings (lower=better)",
    }
    metric_groups = {"behavioral_embedding_distance": "face"}

    def setup(self) -> None:
        logger.warning(
            "behavioral_embedding unavailable: no released model or weights; "
            "behavioral_embedding_distance left unset."
        )

    def process(self, sample):
        return sample
