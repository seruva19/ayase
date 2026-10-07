"""FDD — Facial Deformation Distance on upper-face 3D dynamics (external backend).

CodeTalker (Xing et al., CVPR 2023, arXiv:2301.02379) evaluates generated
facial motion with FDD: the mean distance between the temporal dynamics of
upper-face mesh vertices of generated and ground-truth sequences — the
motion-pattern difference on FLAME vertex streams.

A FLAME-based vertex decoder is required to produce the 3D vertex sequences;
FLAME assets are license-gated and not bundled, so the module stays
registered but marked ``requires_external_backend`` until a licensed decoder
is wired in.

fdd -- mean upper-face vertex-motion distance (lower=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FDDModule(PipelineModule):
    name = "fdd"
    description = "FDD: upper-face vertex-motion distance (CodeTalker, external backend)"
    provenance = "published"
    sources = {
        "fdd": "FDD, CodeTalker (Xing et al., CVPR 2023, arXiv:2301.02379) — https://github.com/Doubiiu/CodeTalker",
    }
    requires_external_backend = True  # FLAME vertex decoder is license-gated
    requires_reference = True
    default_config = {"fps": 30}
    metric_info = {
        "fdd": "Mean distance of upper-face vertex motion vs reference (lower=better)",
    }
    metric_groups = {"fdd": "face"}

    def setup(self) -> None:
        logger.warning(
            "fdd unavailable: needs a FLAME vertex decoder (license-gated, "
            "not bundled); fdd left unset."
        )

    def process(self, sample):
        return sample
