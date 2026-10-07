"""SRGR — semantic-weighted PCK of gestures (external backend).

BEAT (Liu et al., ECCV 2022, arXiv:2203.05297) scores gesture correctness with
SRGR: a PCK-style joint-accuracy measure where each joint is weighted by its
semantic-relevance score to the speech content, produced by the dataset's
semantic annotation pipeline.

SRGR needs per-gesture semantic labels from the BEAT annotation pipeline,
which is not bundled, so the module stays registered but marked
``requires_external_backend`` until that pipeline is wired in.

srgr -- semantic-relevance-weighted joint accuracy (0-1, higher=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SRGRModule(PipelineModule):
    name = "srgr"
    description = "SRGR: semantic-weighted gesture PCK (BEAT, external backend)"
    provenance = "published"
    sources = {
        "srgr": "SRGR, BEAT (Liu et al., ECCV 2022, arXiv:2203.05297) — https://github.com/PantoMatrix/BEAT",
    }
    requires_external_backend = True  # needs BEAT semantic annotation pipeline
    requires_reference = True
    default_config = {"pck_threshold": 0.1}
    metric_info = {
        "srgr": "Semantic-weighted joint accuracy vs ground truth (0-1, higher=better)",
    }
    metric_groups = {"srgr": "pose"}

    def setup(self) -> None:
        logger.warning(
            "srgr unavailable: needs BEAT semantic gesture annotations (not "
            "bundled); srgr left unset."
        )

    def process(self, sample):
        return sample
